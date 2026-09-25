import { Fn, If, float, instanceIndex, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type {
  HashGrid,
  ParticleRange,
  ParticleSystem,
  SphKernelUniforms,
} from '../../core/index.js';
import { emitForEachNeighbor, emitPoly6FromRSq } from '../../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildSmokeAdvectKernelArgs {
  /** Gas-owned smoke positions, sized to the GasSystem's `capacity`. */
  readonly smokePositions: StorageBufferNode<'vec4'>;
  /** Gas-owned alive flags. `1` = active smoke particle, `0` = free slot. */
  readonly smokeAlive: StorageBufferNode<'uint'>;
  /** Gas-owned ages (seconds since emission). Lifetime gate writes here. */
  readonly smokeAge: StorageBufferNode<'float'>;
  /**
   * Optional gas-owned interpolated velocities buffer. When present, the
   * SPH-interpolated `v(x_s)` per Macklin 2014 eq. 28 is written here too,
   * so a renderer or downstream kernel (e.g. a future motion-blur pass)
   * can read the smoke's drift velocity without recomputing the walk.
   */
  readonly smokeVelocities?: StorageBufferNode<'vec4'>;
  /** Total slots in the gas-owned buffers. */
  readonly smokeCapacity: number;
  /**
   * The fluid `ParticleSystem` whose post-solve velocity field drives
   * advection. `positions` and `velocities` are read; `boundaryVolume`
   * is read to gate non-fluid neighbours out of the SPH sum.
   */
  readonly fluidParticles: ParticleSystem;
  /** Fluid range within {@link fluidParticles}. Foreign neighbours skipped. */
  readonly fluidRange: ParticleRange;
  /** Hash grid built over `fluidParticles` (already rebuilt by SimLoop). */
  readonly hashGrid: HashGrid;
  /** SPH kernel uniforms (`h`, `hSq`, `poly6Coef`). MUST match FluidSystem.sph. */
  readonly sph: SphKernelUniforms;
  /** Substep `Δt` from the shared XPBD uniforms. */
  readonly dt: UniformNode<'float', number>;
  /** Lifetime in seconds; smoke older than this is killed (alive ← 0). */
  readonly lifetime: UniformNode<'float', number>;
}

/**
 * Passive smoke-particle advection kernel — Macklin 2014 §7.2.1 eq. 28.
 *
 * Per smoke particle `s` (one thread per slot in the gas-owned buffer):
 *
 *   `v(x_s) = Σ_j v_j · W(x_s − x_j, h) / Σ_j W(x_s − x_j, h)`
 *
 * The sum walks fluid neighbours via the same hash grid `FluidSystem`
 * uses. Smoke is NOT a generator in that grid (gas-owned buffer is
 * separate from `ParticleSystem`); the walk is performed by passing
 * `smokePositions[s].xyz` into `emitForEachNeighbor` as a *foreign*
 * query position. Phase 07a's `emitForEachNeighbor` accepts an arbitrary
 * `vec3` for exactly this case.
 *
 * Neighbour candidates outside the fluid range are silently skipped so
 * boundary / softbody / rigid particles in the same `ParticleSystem`
 * don't contaminate the velocity interpolation. (Boundary particles
 * have well-defined velocities, but the paper specifies advection
 * against fluid velocities — eq. 28's "fluid particles" — so the
 * interpolation must be range-restricted.)
 *
 * After the walk:
 *   - if `wSum > epsilon`, `v_s = vSum / wSum` (the SPH-weighted mean);
 *     otherwise `v_s = 0` (no fluid neighbours within `h` — paper does
 *     not specify; we Assume zero, matching "drift halts when isolated").
 *   - `smokePositions[s] += v_s · Δt`
 *   - `smokeAge[s] += Δt`; if the new age ≥ `lifetime`, set
 *     `smokeAlive[s] = 0` (slot freed for next emission, no compaction
 *     needed for MVP — the kernel skips dead slots at the head).
 *
 * Substep slot — `Material.postAdvectKernels`. Paper §7.2.1: "we use
 * the velocity calculated immediately after the constraint solve to
 * ensure the velocity used for advection is consistent with the
 * incompressibility constraints (step 24 in Algorithm 1)." Post-advect
 * is exactly that point: `velocities[j] = (x*_j − x_j) / Δt` reflects
 * the divergence-free constraint solve.
 */
export function buildSmokeAdvectKernel(args: BuildSmokeAdvectKernelArgs): ComputeNode {
  const {
    smokePositions,
    smokeAlive,
    smokeAge,
    smokeVelocities,
    smokeCapacity,
    fluidParticles,
    fluidRange,
    hashGrid,
    sph,
    dt,
    lifetime,
  } = args;

  if (
    !Number.isInteger(fluidRange.start) ||
    !Number.isInteger(fluidRange.count) ||
    fluidRange.start < 0 ||
    fluidRange.count <= 0 ||
    fluidRange.start + fluidRange.count > fluidParticles.capacity
  ) {
    throw new Error(
      `buildSmokeAdvectKernel: invalid fluidRange start=${fluidRange.start} count=${fluidRange.count} capacity=${fluidParticles.capacity}`,
    );
  }
  if (!Number.isInteger(smokeCapacity) || smokeCapacity <= 0) {
    throw new Error(
      `buildSmokeAdvectKernel: smokeCapacity must be a positive integer, got ${smokeCapacity}`,
    );
  }

  // Range bounds are constructor-set and treated as immutable; bake
  // them as compile-time `uint(literal)` constants rather than uniforms
  // so we don't have to wrestle TSL's `uniform()` overloads (which
  // declare only `'float'` despite the backend accepting `'uint'`).
  const fluidStartConst = fluidRange.start;
  const fluidEndConst = fluidRange.start + fluidRange.count;

  return Fn(() => {
    const s: Any = instanceIndex;
    const alive: Any = smokeAlive.element(s);

    // Skip dead slots — no compaction in MVP, so dead slots persist
    // until `emit()` reuses them. Single-branch gate, no else.
    If(alive.greaterThan(uint(0)), () => {
      const pos: Any = smokePositions.element(s).xyz.toVar();
      const age: Any = smokeAge.element(s);

      // Lifetime gate. The paper §7.2.2 mentions "after a pre-defined
      // lifetime" as a removal criterion alongside leaving the AABB and
      // having no fluid neighbours within radius. MVP implements the
      // lifetime path; AABB is post-MVP.
      const newAge: Any = age.add(dt as Any);
      If(newAge.greaterThanEqual(lifetime as Any), () => {
        smokeAlive.element(s).assign(uint(0));
      }).Else(() => {
        // SPH velocity interpolation walk.
        const vSum: Any = vec3(0, 0, 0).toVar();
        const wSum: Any = float(0).toVar();

        emitForEachNeighbor({
          queryPosXyz: pos,
          hashOrigin: hashGrid.hashOriginUniform,
          cellSize: hashGrid.cellSizeUniform,
          hashTableSize: hashGrid.hashTableSize,
          cellStart: hashGrid.cellStart,
          cellEnd: hashGrid.cellEnd,
          sortedIndices: hashGrid.sortedIndices,
          onCandidate: (j: Any) => {
            // Range-gate to fluid slots. Smoke is not in the hash grid
            // as a generator (separate buffer), but the same
            // `ParticleSystem` may host softbody / rigid / boundary
            // particles. Eq. 28's sum is over fluid particles only.
            const inRange: Any = j
              .greaterThanEqual(uint(fluidStartConst))
              .and(j.lessThan(uint(fluidEndConst)));
            If(inRange, () => {
              const xj: Any = fluidParticles.positions.element(j).xyz;
              const diff: Any = pos.sub(xj);
              const rSq: Any = diff.dot(diff);
              // Poly6 self-clamps to 0 outside `h` via the
              // `max(hSq − rSq, 0)` form, so no within-h check needed.
              const w: Any = emitPoly6FromRSq(rSq, sph);
              const vj: Any = fluidParticles.velocities.element(j).xyz;
              vSum.addAssign(vj.mul(w));
              wSum.addAssign(w);
            });
          },
        });

        // Normalise. Below epsilon, the smoke drifts on its own
        // momentum — but smoke has no momentum buffer in MVP (no
        // self-advection), so the fallback is zero-velocity hold-in-
        // place. Paper §7.2.2 mentions "advect them with the
        // procedural background velocity field" as an alternative;
        // procedural fields are post-MVP.
        const EPS: Any = float(1e-12);
        const vSmoke: Any = wSum.greaterThan(EPS).select(vSum.div(wSum.max(EPS)), vec3(0, 0, 0));

        // Commit position + age. Velocity buffer is optional.
        const newPos: Any = pos.add(vSmoke.mul(dt as Any));
        smokePositions.element(s).assign(vec4(newPos, float(0)));
        smokeAge.element(s).assign(newAge);
        if (smokeVelocities !== undefined) {
          smokeVelocities.element(s).assign(vec4(vSmoke, float(0)));
        }
      });
    });
  })().compute(smokeCapacity);
}
