import { instancedArray, uniform } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type {
  HashGrid,
  Material,
  ParticleRange,
  ParticleSystem,
  XpbdUniforms,
} from '../../core/index.js';
import { createSphKernelUniforms, type SphKernelUniforms } from '../../core/index.js';

import { buildSmokeAdvectKernel } from './smokeAdvect.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const DEFAULT_LIFETIME_SEC = 5;

export interface GasSystemOptions {
  /**
   * Maximum number of simultaneously-alive smoke particles. Sizes the
   * gas-owned buffers (`smokePositions`, `smokeVelocities`, `smokeAge`,
   * `smokeAlive`). Fixed at construction; no runtime grow.
   */
  readonly capacity: number;
  /**
   * The fluid `ParticleSystem` whose post-solve velocity field drives
   * smoke advection. Read-only from the gas package's perspective —
   * smoke does NOT register itself in `fluidParticles` or its hash grid.
   */
  readonly fluidParticles: ParticleSystem;
  /**
   * Fluid range within {@link fluidParticles}. The SPH walk filters by
   * this range so other materials (boundary / softbody / rigid) sharing
   * the same `ParticleSystem` don't contribute to smoke advection.
   */
  readonly fluidRange: ParticleRange;
  /**
   * Hash grid built over `fluidParticles`. MUST be the same instance
   * `SimLoop` rebuilds at the head of each substep — the gas advect
   * kernel reads `cellStart` / `cellEnd` / `sortedIndices` directly.
   */
  readonly hashGrid: HashGrid;
  /**
   * SPH smoothing length `h` (m). MUST equal the partner `FluidSystem.h`
   * so eq. 28's interpolation has the same support as the fluid density
   * solve. The gas package builds its own `SphKernelUniforms` (Poly6
   * coefficients depend on `h`); a future post-MVP revision could share
   * the FluidSystem's uniforms instance.
   */
  readonly h: number;
  /**
   * Shared XPBD uniforms — typically the ones owned by the `SimLoop`
   * this gas participates in. Must be the *same instance* the loop
   * constructs with; the `dt` uniform is shared by identity, not value.
   */
  readonly xpbd: XpbdUniforms;
  /** Smoke lifetime in seconds. Default 5. */
  readonly lifetime?: number;
}

/**
 * `src/gas` — passive smoke-particle advection (Macklin 2014
 * §7.2.1 eq. 28). Implements the core {@link Material} interface so it
 * registers via `SimLoopOptions.materials` alongside the partner
 * `FluidSystem` (the gas IS that fluid, with reduced gravity; smoke is
 * the visual tracer that lets the eye see the otherwise-invisible
 * flow).
 *
 * **Material registration order.** Macklin 2014 Algorithm 1 puts
 * "advect diffuse particles" (line 25) BEFORE "apply f_drag, f_vort"
 * (line 26). `SimLoop` dispatches `materialsPostAdvectKernels` in the
 * order materials appear in `SimLoopOptions.materials`, so callers that
 * partner gas with fluid MUST register gas first:
 *   `materials: [gas, fluid]`.
 * Otherwise smoke samples post-vorticity/drag velocity instead of the
 * post-constraint-solve velocity the paper specifies in §7.2.1.
 *
 *
 * Lifecycle: emission writes into free slots via `emit()`; advection
 * runs in `postAdvectKernels` once per substep, which both interpolates
 * SPH velocity and ages particles past their lifetime. The GPU kernel
 * is the authoritative killer (sets `smokeAlive[s] = 0` once
 * `age[s] ≥ lifetime`); the CPU mirrors that prediction inside
 * {@link emit} when the caller passes the current `simTime`, so slot
 * reuse stays in sync without a per-frame GPU readback. Without
 * `simTime` the CPU mirror is conservative (slots are seen as alive
 * forever once written, until {@link reset} or a fresh `simTime` is
 * supplied) and long-running emitters can hit `capacity`.
 */
export class GasSystem implements Material {
  readonly capacity: number;
  readonly fluidParticles: ParticleSystem;
  readonly fluidRange: ParticleRange;
  readonly hashGrid: HashGrid;
  readonly h: number;
  readonly lifetime: number;

  /** Per-smoke-slot position (xyz + pad). Read by renderer; written by advect. */
  readonly smokePositions: StorageBufferNode<'vec4'>;
  /** Per-smoke-slot SPH-interpolated drift velocity (xyz + pad). */
  readonly smokeVelocities: StorageBufferNode<'vec4'>;
  /** Per-smoke-slot age (seconds since emission). */
  readonly smokeAge: StorageBufferNode<'float'>;
  /** Per-smoke-slot alive flag (1 = active, 0 = free slot). */
  readonly smokeAlive: StorageBufferNode<'uint'>;

  /** Gas-owned SPH uniforms; must match the partner FluidSystem's `h`. */
  readonly sph: SphKernelUniforms;
  readonly lifetimeUniform: UniformNode<'float', number>;

  /** {@link Material} interface — single advect kernel runs per substep. */
  readonly postAdvectKernels: readonly ComputeNode[];

  /**
   * CPU-side mirror of the simTime at which each slot was last emitted,
   * used to predict the GPU's lifetime kill so {@link emit} can reuse
   * dead slots without a per-frame GPU readback. Updated by {@link emit};
   * read by {@link emit} when the caller supplies a fresh `simTime`.
   * Slots that have never been emitted hold `0` — those slots are dead
   * (`smokeAlive[s] = 0`) by construction so the value is irrelevant
   * for them.
   */
  private readonly emitSimTimes: Float32Array;
  /**
   * Largest `simTime` ever passed into {@link emit}. Lets the CPU reap
   * dead slots once at the start of each emit() call rather than every
   * iteration, and lets {@link reset} advance the threshold past every
   * historic emit timestamp.
   */
  private lastEmitSimTime = 0;

  constructor(options: GasSystemOptions) {
    const { capacity, fluidParticles, fluidRange, hashGrid, h, xpbd, lifetime } = options;

    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`GasSystem: capacity must be a positive integer, got ${capacity}`);
    }
    if (!Number.isFinite(h) || h <= 0) {
      throw new Error(`GasSystem: h must be a positive finite number, got ${h}`);
    }
    if (
      !Number.isInteger(fluidRange.start) ||
      !Number.isInteger(fluidRange.count) ||
      fluidRange.start < 0 ||
      fluidRange.count <= 0 ||
      fluidRange.start + fluidRange.count > fluidParticles.capacity
    ) {
      throw new Error(
        `GasSystem: invalid fluidRange start=${fluidRange.start} count=${fluidRange.count} capacity=${fluidParticles.capacity}`,
      );
    }
    if (hashGrid.particles !== fluidParticles) {
      throw new Error(
        'GasSystem: hashGrid must be constructed against the same ParticleSystem as fluidParticles',
      );
    }
    if (hashGrid.cellSize < h) {
      throw new Error(
        `GasSystem: hashGrid.cellSize (${hashGrid.cellSize}) must be ≥ h (${h}) — Phase 08 invariant, applies equally to gas advection`,
      );
    }
    const lifetimeValue = lifetime ?? DEFAULT_LIFETIME_SEC;
    if (!Number.isFinite(lifetimeValue) || lifetimeValue <= 0) {
      throw new Error(`GasSystem: lifetime must be a positive finite number, got ${lifetimeValue}`);
    }

    this.capacity = capacity;
    this.fluidParticles = fluidParticles;
    this.fluidRange = fluidRange;
    this.hashGrid = hashGrid;
    this.h = h;
    this.lifetime = lifetimeValue;
    this.emitSimTimes = new Float32Array(capacity);

    // --- Storage (gas-package-owned, not on ParticleSystem) ---
    this.smokePositions = instancedArray(capacity, 'vec4');
    this.smokeVelocities = instancedArray(capacity, 'vec4');
    this.smokeAge = instancedArray(capacity, 'float');
    this.smokeAlive = instancedArray(capacity, 'uint');

    // --- Uniforms ---
    this.sph = createSphKernelUniforms(h);
    this.lifetimeUniform = uniform(lifetimeValue, 'float');

    // --- Kernels ---
    const advect = buildSmokeAdvectKernel({
      smokePositions: this.smokePositions,
      smokeAlive: this.smokeAlive,
      smokeAge: this.smokeAge,
      smokeVelocities: this.smokeVelocities,
      smokeCapacity: capacity,
      fluidParticles,
      fluidRange,
      hashGrid,
      sph: this.sph,
      dt: xpbd.dt,
      lifetime: this.lifetimeUniform,
    });
    this.postAdvectKernels = [advect];
  }

  /**
   * Emit `count` smoke particles at position `x` with initial velocity
   * `v`. Scans the alive flags from index 0 and reuses the first
   * `count` free slots. Returns the actual number emitted (less than
   * `count` if the gas-owned buffer is saturated).
   *
   * `simTime` (optional): the harness-side simulation time in seconds.
   * When supplied, the CPU mirror reaps slots whose
   * `simTime − emitSimTimes[s] ≥ lifetime` before scanning, mirroring
   * the GPU kernel's lifetime kill so reused slots stay in sync with
   * GPU state. Without it the mirror is conservative — slots stay
   * "alive" CPU-side forever once written, and a long-running emitter
   * silently saturates at `capacity` even though the GPU has freed
   * most slots. The harness-supplied `simTime` is monotonic per
   * `runDemo` (see `runDemo.ts:simTime += dt`) and advances at the
   * same rate as GPU-side `smokeAge` (both step by substep `dt`), so
   * the CPU prediction is exact modulo round-off.
   *
   * Writes use `addUpdateRange` per the ARCH "Compute path" partial-
   * write gotcha — without it, `needsUpdate = true` triggers a full-
   * buffer re-upload that clobbers any other slot the GPU has been
   * mutating since the last upload.
   *
   * CPU-side scan: O(capacity) worst case. Acceptable at MVP smoke
   * counts (~few thousand). A free-list ring buffer is the obvious
   * optimisation if a future scene shows this in profiles.
   */
  emit(
    x: readonly [number, number, number],
    v: readonly [number, number, number],
    count: number,
    simTime?: number,
  ): number {
    if (!Number.isInteger(count) || count <= 0) {
      throw new Error(`GasSystem.emit: count must be a positive integer, got ${count}`);
    }

    const aliveArr = (this.smokeAlive.value as Any).array as Uint32Array;
    const posArr = (this.smokePositions.value as Any).array as Float32Array;
    const velArr = (this.smokeVelocities.value as Any).array as Float32Array;
    const ageArr = (this.smokeAge.value as Any).array as Float32Array;

    // CPU-side reap, mirroring the GPU kernel's lifetime kill. Without
    // this, every emit() write would mark a slot CPU-alive forever even
    // though the GPU has long since killed it on age, and emit() would
    // silently fill to `capacity` and then return 0 forever after. With
    // it, the scan finds the same dead slots the GPU has freed.
    if (simTime !== undefined) {
      if (!Number.isFinite(simTime) || simTime < 0) {
        throw new Error(
          `GasSystem.emit: simTime must be a non-negative finite number, got ${simTime}`,
        );
      }
      this.lastEmitSimTime = Math.max(this.lastEmitSimTime, simTime);
      const threshold = simTime - this.lifetime;
      let reapFirst = -1;
      let reapLast = -1;
      for (let s = 0; s < this.capacity; s++) {
        if (aliveArr[s] === 0) continue;
        if (this.emitSimTimes[s]! <= threshold) {
          aliveArr[s] = 0;
          if (reapFirst === -1) reapFirst = s;
          reapLast = s;
        }
      }
      if (reapFirst !== -1) {
        (this.smokeAlive.value as Any).addUpdateRange(reapFirst, reapLast - reapFirst + 1);
        (this.smokeAlive.value as Any).needsUpdate = true;
      }
    }

    let emitted = 0;
    let firstSlot = -1;
    let lastSlot = -1;
    for (let s = 0; s < this.capacity && emitted < count; s++) {
      if (aliveArr[s] !== 0) continue;
      aliveArr[s] = 1;
      posArr[s * 4 + 0] = x[0];
      posArr[s * 4 + 1] = x[1];
      posArr[s * 4 + 2] = x[2];
      posArr[s * 4 + 3] = 0;
      velArr[s * 4 + 0] = v[0];
      velArr[s * 4 + 1] = v[1];
      velArr[s * 4 + 2] = v[2];
      velArr[s * 4 + 3] = 0;
      ageArr[s] = 0;
      this.emitSimTimes[s] = simTime ?? this.lastEmitSimTime;
      if (firstSlot === -1) firstSlot = s;
      lastSlot = s;
      emitted++;
    }

    if (emitted > 0) {
      const slotRange = lastSlot - firstSlot + 1;
      // `addUpdateRange(start, count)` is in **typed-array elements**,
      // not slot indices. For vec4 buffers (itemSize = 4) we have to
      // multiply by 4 — passing slot-indexed args silently uploads
      // only the X-component of each slot and leaves Y/Z stale, which
      // shows up as smoke spawning at `(x, 0, 0)` instead of the
      // actual click point.
      // (See `node_modules/three/src/core/BufferAttribute.js`'s
      // `addUpdateRange` jsdoc: "start - Position at which to start
      // update", "count - The number of components to update.")
      (this.smokeAlive.value as Any).addUpdateRange(firstSlot, slotRange);
      (this.smokePositions.value as Any).addUpdateRange(firstSlot * 4, slotRange * 4);
      (this.smokeVelocities.value as Any).addUpdateRange(firstSlot * 4, slotRange * 4);
      (this.smokeAge.value as Any).addUpdateRange(firstSlot, slotRange);
      (this.smokeAlive.value as Any).needsUpdate = true;
      (this.smokePositions.value as Any).needsUpdate = true;
      (this.smokeVelocities.value as Any).needsUpdate = true;
      (this.smokeAge.value as Any).needsUpdate = true;
    }

    return emitted;
  }
}
