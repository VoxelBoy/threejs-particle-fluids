import { Continue, Fn, If, Return, float, instanceIndex, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { ContactAccumulator, ParticleRange, ParticleSystem } from '../../core/index.js';
import { emitAccumulateDelta, emitForEachPair } from '../../core/index.js';

import { emitSpikyGrad, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/*
 * Phase 11 — fluid → solid Newton-3 reaction (PBF density coupling).
 *
 * The Phase 08 PBF density-correction kernels (`density.ts`, `lambda.ts`,
 * `positionDelta.ts`) dispatch over fluid particles only. Boundary
 * particles registered via `FluidSystem.registerBoundaryParticles` appear
 * in the fluid's `ψ_j = ρ_0 · V_j` density sum (Akinci 2012 eq. 5) but
 * never receive the equal-and-opposite Δp. The result is one-sided
 * coupling: fluid pressure is balanced around a submerged body but no
 * force transfers to the body. This kernel ports the missing reaction.
 *
 *
 * **Per-fluid-i scatter formula:**
 *   Δp_j = − w_j · (λ_i / ρ_0) · ψ_j · ∇W_spiky(x_i − x_j, h)
 *
 * Provenance line by line:
 *   `−`              from Macklin 2013 eq. 8 (k=j case): the constraint
 *                    gradient at x_j is the negative of the gradient at x_i.
 *   `w_j`            mass-weighted PBD update (Macklin 2014 eq. 4 + §7.1.1).
 *   `λ_i / ρ_0`      the fluid's Lagrange multiplier scaled by 1/ρ_0 — same
 *                    `(λ / ρ_0)` factor `positionDelta.ts:138` applies on
 *                    the fluid side of the same constraint.
 *   `ψ_j`            from differentiating `ρ_i = Σ_k ψ_k W` wrt x_j: only
 *                    the k=j term survives, with weight ψ_j = ρ_0 · V_j
 *                    for boundary particles (Akinci eq. 5).
 *   `∇W_spiky`       paper's gradient kernel for force computation (Müller
 *                    2003 + Macklin 2013 §3 last paragraph); same kernel
 *                    `positionDelta.ts` uses on the fluid side.
 *
 * **Mechanism choice — scatter, not gather** (plan §"Mechanism choice"):
 *   - The fluid side of the constraint already iterates the per-substep
 *     pair list (`pairList.ts`). Building a separate gather pass over
 *     [fluid ∪ boundary] would mean a second hash-grid walk for boundary
 *     threads.
 *   - The shared `ContactAccumulator` (`src/core/contact/
 *     accumulator.ts`) is already wired into SimLoop's per-iter chain
 *     (loop.ts line ~589: `applyToPredicted` drains it into
 *     predictedPositions at iter-end). Scattering into it requires no new
 *     apply kernel and no new buffer.
 *   - i32 atomicAdd is associative + commutative + exact, so the
 *     per-particle Δp sum is independent of thread execution order — the
 *     kernel is G4 tier-1 bit-deterministic, matching contact / collider /
 *     SDF (ARCHITECTURE.md §Guardrails).
 *
 * **Newton-3 conservation note** (Verified at M-2 entry):
 *   Strict mass-weighted PBD conservation requires w_i on the fluid side
 *   too. `positionDelta.ts` uses Macklin 2013 eq. 12's equal-mass form
 *   (no explicit w_i — absorbed into the equal-mass simplification). The
 *   resulting per-pair momentum residual for fluid-boundary pairs scales
 *   as `(m_i · m_fluid · V_b) · (λ_i / ρ_0) · ∇W`, which at MVP scales
 *   (m_fluid ≈ 0.015 kg, V_b ≈ spacing³ ≈ 1.5e-5 m³, ρ_0 = 1000) is
 *   ~2e-7 · λ_i per pair — well inside the `1e-5 · max(|v_i|)` test
 *   tolerance the plan specifies. Macklin 2014 §7.1.1 prescribes this
 *   exact mechanism and reports it produces Figure-12 buoyancy; we
 *   follow the paper.
 *
 * Boundary range filtering: the kernel accepts ALL non-fluid neighbours
 * via the `boundaryVolume_j > 0` branch (same convention `positionDelta`,
 * `lambda`, and `density` use). Boundary ranges are registered through
 * `FluidSystem.registerBoundaryParticles`; this kernel does not need
 * range storage of its own.
 *
 * Boundary kinematic check: a boundary particle with `invMass = 0` is
 * pinned (immovable). The scatter must skip it — applying Δp_j to a
 * pinned slot would shift a kinematic boundary by accumulator-quantum
 * noise, and (per `applyAccumulator` semantics) the apply kernel writes
 * unconditionally. Skip via `w_j > 0`.
 */

export interface BuildSolidReactionScatterKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  /** `ρ_0` — rest density, kg/m³. */
  readonly restDensity: UniformNode<'float', number>;
  /** `λ_i` buffer, populated by `buildLambdaKernel` earlier in the same iter. */
  readonly lambda: StorageBufferNode<'float'>;
  /** Shared position-domain scatter accumulator (see file-level docstring). */
  readonly contactAccumulator: ContactAccumulator;
  readonly fluidParticles: ParticleRange;
  /**
   * Per-substep neighbor pair list — same list `density` / `lambda` /
   * `positionDelta` consume. Confirmed at M-2 entry (`pairList.ts:217`)
   * to include boundary neighbours: the build kernel walks via
   * `emitForEachNeighbor` and filters only by `|r|² < h²`, so any
   * within-h boundary particle is in the list.
   */
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Build the per-iter fluid → solid Newton-3 reaction scatter kernel.
 *
 * Dispatch: `fluidParticles.count` threads, one per fluid `i`. For each
 * `i` with `λ_i ≠ 0`, walks the cached pair list, filters to boundary
 * neighbours with non-zero invMass, computes the per-pair `Δp_j`, and
 * scatters into the shared {@link ContactAccumulator} via
 * `emitAccumulateDelta`. The existing per-iter `applyToPredicted` kernel
 * drains the accumulator into `predictedPositions[j]` at iter-end.
 *
 * Ordering in the per-iter chain (set up in `FluidSystem.perIterKernels`):
 *   density → lambda → positionDelta → **solidReactionScatter** →
 *   applyDelta → **applyAccumulator (drain solid-reaction accumulator)**
 *
 * The scatter is BEFORE `applyDelta` (rather than after) so it reads
 * the same `predictedPositions` that `positionDelta` read on the fluid
 * side — both compute `∇W(x_i − x_j)` against pre-update positions.
 * Placing the scatter after applyDelta would have it use post-update
 * fluid positions while positionDelta used pre-update; the resulting
 * gradient mismatch breaks per-pair Newton-3 momentum conservation
 * (the boundary-side Δp would no longer cancel the fluid-side Δp at
 * the FP-noise level — measured 5–10% residual in the m=1 unit test).
 *
 * `applyAccumulator` is the existing core
 * `buildApplyAccumulatorToPredictedKernel` — placed AFTER applyDelta is
 * fine because it dispatches over all particles but only boundary slots
 * have non-zero accumulator contributions (this kernel filters fluid
 * neighbours), so fluid x* is unaffected by the drain.
 */
export function buildSolidReactionScatterKernel(
  args: BuildSolidReactionScatterKernelArgs,
): ComputeNode {
  const {
    particles,
    sph,
    restDensity,
    lambda,
    contactAccumulator,
    fluidParticles,
    pairList,
    pairCount,
  } = args;

  if (
    !Number.isInteger(fluidParticles.start) ||
    !Number.isInteger(fluidParticles.count) ||
    fluidParticles.start < 0 ||
    fluidParticles.count <= 0 ||
    fluidParticles.start + fluidParticles.count > particles.capacity
  ) {
    throw new Error(
      `buildSolidReactionScatterKernel: invalid fluidParticles range start=${fluidParticles.start} count=${fluidParticles.count} capacity=${particles.capacity}`,
    );
  }
  if (contactAccumulator.particles !== particles) {
    throw new Error(
      'buildSolidReactionScatterKernel: contactAccumulator must wrap the same ParticleSystem as fluidParticles',
    );
  }

  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const lambdaI: Any = lambda.element(i).toVar();
    // Unilateral clamp (Macklin 2014 §7 eq. 26): `lambda.ts:143-146`
    // assigns λ_i = 0 when C_i ≤ 0. Skip the whole walk in that case —
    // expansion regions produce no reaction.
    If(lambdaI.equal(float(0.0)), () => {
      Return();
    });

    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const invRho0: Any = float(1.0).div(restDensity as Any);
    // Per-thread scalar coefficient: `−(λ_i / ρ_0)` is constant across
    // every neighbour `j` of `i`, so factor it out of the loop.
    const lambdaOverRho0Neg: Any = lambdaI.mul(invRho0).negate().toVar();

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        // Boundary-only filter. Fluid-fluid Δp is handled by
        // `positionDelta.ts` (Macklin 2013 eq. 12 symmetric pair form);
        // adding it here would double-apply.
        const Vj: Any = particles.boundaryVolume.element(j).toVar();
        If(Vj.lessThanEqual(float(0.0)), () => {
          Continue();
        });

        // Pinned-boundary skip. `w_j == 0` means the slot is kinematic;
        // applying any Δp would move a particle the scene script wants
        // immovable. The contact pipeline handles pinned-boundary
        // collisions via the analytic primitive / SDF paths instead.
        const wj: Any = particles.invMass.element(j).toVar();
        If(wj.lessThanEqual(float(0.0)), () => {
          Continue();
        });

        const xj: Any = particles.predictedPositions.element(j).xyz;
        const diff: Any = xi.sub(xj).toVar();
        // Pair list is filtered to `|r|² < h²` at build time. Self
        // (`j == i`) cannot be a boundary (boundaryVolume = 0 for fluid
        // slots), so the boundary filter above already excludes it.

        const psiJ: Any = (restDensity as Any).mul(Vj);
        // ∇W is the same Spiky gradient `positionDelta.ts:133` uses on
        // the fluid side — taken with respect to x_i. The reaction at
        // x_j (Macklin 2013 eq. 8 k=j case) is `−∇_{x_i} W`, which the
        // leading `−` in `lambdaOverRho0Neg` already captures.
        const gradW: Any = emitSpikyGrad(diff, sph);
        // Δp_j = −w_j · (λ_i / ρ_0) · ψ_j · ∇W(x_i − x_j)
        const dxJ: Any = gradW.mul(lambdaOverRho0Neg).mul(psiJ).mul(wj);
        emitAccumulateDelta(contactAccumulator, j, dxJ);
      },
    });
  })().compute(fluidParticles.count);
}
