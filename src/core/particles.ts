import { instancedArray } from 'three/tsl';
import type { WebGPURenderer } from 'three/webgpu';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

/**
 * CPU-side snapshot returned by {@link ParticleSystem.readback}. Fields are
 * freshly-read `Float32Array` views of GPU memory — owned by the caller and
 * safe to mutate.
 *
 *
 * Phase 02 scope: only the buffers the predict/advect kernels actually
 * touch appear here. `phase`, `flags`, and `boundaryVolume` are declared on
 * `ParticleSystem` for later phases but are never written by a compute
 * kernel in Phase 02 — their GPU-side allocation does not exist yet and
 * `renderer.getArrayBufferAsync` throws on never-allocated storage. When a
 * later phase first writes one of those buffers on GPU, extend
 * `ParticleSnapshot` to include it.
 *
 * Phase 12 (Mueller 2011 §4.1): `rotation`, `predictedRotation`,
 * `angularVelocity` are written by the always-on prologue/epilogue dispatched
 * from `SimLoop`, so they are present on every snapshot. `rotation` and
 * `predictedRotation` are vec4 unit quaternions `(x, y, z, w)`;
 * `angularVelocity` is vec3 (xyz, w padding).
 */
export interface ParticleSnapshot {
  readonly capacity: number;
  readonly positions: Float32Array;
  readonly predictedPositions: Float32Array;
  readonly velocities: Float32Array;
  readonly invMass: Float32Array;
  readonly rotation: Float32Array;
  readonly predictedRotation: Float32Array;
  readonly angularVelocity: Float32Array;
}

/**
 * A contiguous range of particle slots within a {@link ParticleSystem},
 * `[start, start + count)`. Used by material modules to declare which
 * slots they own (e.g. `FluidSystem.fluidParticles`) and to register
 * other modules' particles as SPH boundaries (e.g.
 * `FluidSystem.registerBoundaryParticles(clothRange)` in Phase 20, post-MVP).
 *
 */
export interface ParticleRange {
  readonly start: number;
  readonly count: number;
}

/**
 * Per-particle state supplied to {@link ParticleSystem.uploadParticles}. All
 * fields are required. Arrays shorter than `capacity` leave trailing slots
 * zero-initialized — with `invMass = 0` (kinematic / inert) those slots do
 * not participate in integration.
 */
export interface ParticleInit {
  readonly position: readonly [number, number, number];
  readonly velocity: readonly [number, number, number];
  readonly invMass: number;
  readonly phase: number;
}

/**
 * Shared particle store for all of Three.js Particle Fluids — the foundation every later
 * phase reads and writes.
 *
 *
 * The buffer node fields are deliberately public-readonly so kernel authors
 * in `integrate.ts` (and later `neighbors.ts`, `solve.ts`, etc.) can
 * reference them inside `Fn().compute(...)` bodies. External callers should
 * treat them as opaque — mutating via `.value.array` outside of
 * {@link uploadParticles} will work but bypasses the version-tracking
 * contract (`needsUpdate`) that three.js uses to decide whether to
 * re-upload.
 */
export class ParticleSystem {
  readonly renderer: WebGPURenderer;
  readonly capacity: number;
  /**
   * Scene-global contact radius in metres. Paper-faithful to Macklin 2014
   * §3: "We restrict ourselves to a fixed particle radius per scene in
   * order to leverage efficient collision detection based on uniform
   * grids." Not per-particle, not per-material. Modules that expose a
   * radius parameter must assert it matches this value.
   */
  readonly particleRadius: number;

  readonly positions: StorageBufferNode<'vec4'>;
  readonly predictedPositions: StorageBufferNode<'vec4'>;
  readonly velocities: StorageBufferNode<'vec4'>;
  readonly invMass: StorageBufferNode<'float'>;
  readonly phase: StorageBufferNode<'uint'>;
  readonly flags: StorageBufferNode<'uint'>;
  readonly boundaryVolume: StorageBufferNode<'float'>;
  /**
   * Phase 15a — per-particle inverse mass consumed by the contact-solve
   * helper `emitContactSolveCorrection`. Decoupled from {@link invMass}
   * so material modules can override the contact-time mass without
   * disturbing predict/advect or the constraint scheduler.
   *
   * Lifecycle: a per-substep copy kernel
   * (`buildCopyContactInvMassKernel`) initialises every slot to
   * `invMass[i]` at the head of the contact preIter block, before any
   * material's `preIterKernels` run. Materials that need to override
   * (currently only `src/softbody`'s stiff-stack mass scaling for
   * rigid particles, paper §5.2 eq. 21) write into this buffer from
   * their own preIter kernel — those writes survive into the iter loop
   * because materials' preIter runs after the copy.
   *
   * Stabilize (paper §4.4) and the velocity-friction pass (Macklin 2020
   * §3.6) deliberately read {@link invMass}, not this buffer: §4.4 is
   * pre-iter interpenetration recovery before any stack-stability
   * scaling can be meaningful, and §3.6 normalises by reduced mass to
   * keep the tangential clamp dimensionally consistent — which the
   * unscaled mass produces.
   */
  readonly contactInvMass: StorageBufferNode<'float'>;
  /**
   * Phase 12 oriented-particle state (Mueller 2011 §4.1). Per-particle unit
   * quaternion `q`, identity-initialized `(0, 0, 0, 1)`. Read by the §4.1
   * prologue (Eq. 11) to predict `qp` and re-read by the §5.1 implicit-
   * shape-matching kernel as the substep-start orientation that feeds the
   * `Aᵢ = (1/5)·m·r²·Rᵢ` term (Eq. 8). Overwritten by the §4.1 post-solve
   * epilogue (Eq. 15) at the end of every substep.
   */
  readonly rotation: StorageBufferNode<'vec4'>;
  /**
   * Phase 12 — `qp` slot (analog of `predictedPositions`). Written by the
   * §4.1 prologue (Eq. 11) at substep entry, then optionally overwritten by
   * the §5.1 kernel's Pass 4 (`qp_i ← shorterEquivalent(quatFromMat3(R_i))`)
   * for particles in implicit-mode softbodies. Consumed by the §4.1
   * epilogue (Eq. 14) to finite-difference the angular velocity.
   */
  readonly predictedRotation: StorageBufferNode<'vec4'>;

  readonly angularVelocity: StorageBufferNode<'vec4'>;

  private disposed = false;

  constructor(renderer: WebGPURenderer, capacity: number, particleRadius: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`ParticleSystem: capacity must be a positive integer, got ${capacity}`);
    }
    if (!Number.isFinite(particleRadius) || particleRadius <= 0) {
      throw new Error(
        `ParticleSystem: particleRadius must be a positive finite number, got ${particleRadius}`,
      );
    }
    this.renderer = renderer;
    this.capacity = capacity;
    this.particleRadius = particleRadius;

    this.positions = instancedArray(capacity, 'vec4');
    this.predictedPositions = instancedArray(capacity, 'vec4');
    this.velocities = instancedArray(capacity, 'vec4');
    this.invMass = instancedArray(capacity, 'float');
    this.phase = instancedArray(capacity, 'uint');
    this.flags = instancedArray(capacity, 'uint');
    this.boundaryVolume = instancedArray(capacity, 'float');
    this.contactInvMass = instancedArray(capacity, 'float');
    this.rotation = instancedArray(capacity, 'vec4');
    this.predictedRotation = instancedArray(capacity, 'vec4');
    this.angularVelocity = instancedArray(capacity, 'vec4');

    // Identity-init both quaternion buffers. `instancedArray` zero-initialises
    // its backing typed array, which would be the zero quaternion `(0,0,0,0)`
    // — not a unit quaternion. The §4.1 prologue assumes `|q| = 1` on first
    // use; an unnormalised zero would propagate NaN through the first
    // `qp · q⁻¹` finite-difference in Eq. 14. Filling once at construction
    // is cheaper than gating with a "first run" flag inside the kernel.
    const identityQuat = this.rotation.value.array as Float32Array;
    const predictedIdentityQuat = this.predictedRotation.value.array as Float32Array;
    for (let i = 0; i < capacity; i++) {
      const base = i * 4;
      identityQuat[base + 3] = 1.0;
      predictedIdentityQuat[base + 3] = 1.0;
    }
    this.rotation.value.needsUpdate = true;
    this.predictedRotation.value.needsUpdate = true;
  }

  /**
   * Seed the particle state. Writes into the already-allocated ceiling-sized
   * buffers — does not grow them. Trailing slots beyond `data.length` are
   * left at their current values (zero-initialized at construction, which
   * means `invMass = 0` and the slot is inert).
   *
   * Throws if `data.length > capacity` so the caller sees silent truncation
   * as a loud error.
   */
  uploadParticles(data: readonly ParticleInit[]): void {
    this.assertAlive();
    if (data.length > this.capacity) {
      throw new Error(
        `ParticleSystem.uploadParticles: ${data.length} particles exceeds capacity ${this.capacity}`,
      );
    }
    const positions = this.positions.value.array as Float32Array;
    const predictedPositions = this.predictedPositions.value.array as Float32Array;
    const velocities = this.velocities.value.array as Float32Array;
    const invMass = this.invMass.value.array as Float32Array;
    const contactInvMass = this.contactInvMass.value.array as Float32Array;
    const phase = this.phase.value.array as Uint32Array;
    for (let i = 0; i < data.length; i++) {
      const p = data[i]!;
      const base = i * 4;
      positions[base + 0] = p.position[0];
      positions[base + 1] = p.position[1];
      positions[base + 2] = p.position[2];
      positions[base + 3] = 0;
      predictedPositions[base + 0] = p.position[0];
      predictedPositions[base + 1] = p.position[1];
      predictedPositions[base + 2] = p.position[2];
      predictedPositions[base + 3] = 0;
      velocities[base + 0] = p.velocity[0];
      velocities[base + 1] = p.velocity[1];
      velocities[base + 2] = p.velocity[2];
      velocities[base + 3] = 0;
      invMass[i] = p.invMass;
      contactInvMass[i] = p.invMass;
      phase[i] = p.phase;
    }
    this.positions.value.needsUpdate = true;
    this.predictedPositions.value.needsUpdate = true;
    this.velocities.value.needsUpdate = true;
    this.invMass.value.needsUpdate = true;
    this.contactInvMass.value.needsUpdate = true;
    this.phase.value.needsUpdate = true;
  }

  /**
   * Read every particle buffer back from the GPU into CPU-side typed
   * arrays. Debug / test only — a full readback for `capacity = 10k` is
   * ~1 MB round-trip and will stall the render loop.
   */
  async readback(): Promise<ParticleSnapshot> {
    this.assertAlive();
    const [
      positions,
      predictedPositions,
      velocities,
      invMass,
      rotation,
      predictedRotation,
      angularVelocity,
    ] = await Promise.all([
      this.renderer.getArrayBufferAsync(this.positions.value),
      this.renderer.getArrayBufferAsync(this.predictedPositions.value),
      this.renderer.getArrayBufferAsync(this.velocities.value),
      this.renderer.getArrayBufferAsync(this.invMass.value),
      this.renderer.getArrayBufferAsync(this.rotation.value),
      this.renderer.getArrayBufferAsync(this.predictedRotation.value),
      this.renderer.getArrayBufferAsync(this.angularVelocity.value),
    ]);
    return {
      capacity: this.capacity,
      positions: new Float32Array(positions),
      predictedPositions: new Float32Array(predictedPositions),
      velocities: new Float32Array(velocities),
      invMass: new Float32Array(invMass),
      rotation: new Float32Array(rotation),
      predictedRotation: new Float32Array(predictedRotation),
      angularVelocity: new Float32Array(angularVelocity),
    };
  }

  /**
   *
   *
   * After `destroy()` any subsequent `uploadParticles` / `readback` call
   * throws.
   */
  destroy(): void {
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) {
      throw new Error('ParticleSystem has been destroyed');
    }
  }
}
