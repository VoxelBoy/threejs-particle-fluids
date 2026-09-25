import { Fn, float, instanceIndex, instancedArray, uniform, uniformArray } from 'three/tsl';
import type { Object3D, Vector3 } from 'three';
import { Matrix4, Quaternion, Vector3 as Vector3Ctor } from 'three';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleSystem } from '../particles.js';
import { FLAG_INVERT, KIND_BOX, KIND_CAPSULE, KIND_PLANE, KIND_SPHERE } from './primitives.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Numerical epsilon used to detect non-unit scale on a Matrix4 passed
 * into `addBox`. Non-unit scale on an oriented box is out of scope — the
 * box's half-extents already carry the size, so a non-unit Matrix4 scale
 * component is almost certainly a caller mistake rather than intent. A
 * rejection throw with this epsilon catches `1.000001`-style round-trip
 * noise through `Object3D.matrixWorld` without accepting a real 1.1×
 * scale.
 */
const SCALE_EPS = 1e-4;

export interface PrimitiveSetOptions {
  /**
   * Maximum number of colliders the set can hold. Chosen at construction
   * and never resized — same ceiling-allocate convention the particle
   * system uses, for the same reason (U-15 — three.js r184
   * `StorageBufferAttribute` has no per-buffer `dispose()`).
   */
  readonly capacity: number;
}

/**
 * Per-collider friction coefficients. Stored per-slot so different
 * surfaces in the same scene can have different friction (e.g., a rubber
 * floor and a steel tank wall).
 */
export interface ColliderFrictionOptions {
  /** Static Coulomb coefficient μ_s. Defaults to 0.5. */
  readonly muS?: number;
  /** Kinetic / dynamic Coulomb coefficient μ_k = μ_d. Defaults to 0.4. */
  readonly muK?: number;
}

export interface PlaneOptions extends ColliderFrictionOptions {
  /**
   * Linear velocity of the plane (m/s). Used by the Macklin 2020 §3.6
   * velocity-friction pass to compute relative motion at the contact
   * point. Default `(0, 0, 0)`.
   */
  readonly linearVelocity?: Vector3;
}

export interface SphereOptions extends ColliderFrictionOptions {
  /**
   * If `true`, the sphere's interior is the valid region (bowl / container).
   * `phi` and the outward normal are negated at kernel time — see
   * `primitives.ts::emitColliderSdf` invert handling. Default `false`.
   */
  readonly invert?: boolean;
  readonly linearVelocity?: Vector3;
}

export interface BoxOptions extends ColliderFrictionOptions {
  /** Invert flag — as for {@link SphereOptions.invert}. Default `false`. */
  readonly invert?: boolean;
  readonly linearVelocity?: Vector3;
  /**
   * World-space orientation. Default identity. Takes effect only when
   * {@link PrimitiveSet.addBox} is called with a `Vector3` center — the
   * `Matrix4` overload extracts rotation from the matrix's 3×3 block
   * instead and this field is ignored. Resolves U-21.
   */
  readonly rotation?: Quaternion;
}

export interface CapsuleOptions extends ColliderFrictionOptions {
  readonly linearVelocity?: Vector3;
}

/**
 * CPU-side record kept per collider slot so `attachToObject3D` can pull
 * world position + orientation from a three.js `Object3D` every frame.
 *
 * Kinematic linear velocity is derived on the CPU as `(positionCurr -
 * positionPrev) / dt` at each `updateKinematics(dt)` call. Rotation is
 * read from the attached `Object3D` as a fresh world-space quaternion
 * and written to the collider's rotation buffer. Angular velocity is
 * NOT yet derived — the collider's friction pipeline only uses linear
 * velocity for contact-point relative motion; a rotated kinematic box
 * slides correctly around its center but its friction ignores the
 * tangential motion a rotating surface would impart. Post-MVP if an
 * artist-scene calls for it.
 */
interface AttachmentRecord {
  readonly slot: number;
  readonly object: Object3D;
  readonly prevPosition: Vector3;
}

/**
 * Storage and authoring for analytic primitive colliders (plane, sphere,
 * axis-aligned box, capsule) — the Phase 6 deliverable that replaces the
 * Phase 5 kinematic-sphere-floor surrogate.
 *
 * Data layout (parallel arrays, SoA, fixed capacity, packed to stay under
 * WebGPU's 11 storage-buffers-per-stage ceiling — see `renderer.ts`
 * `maxStorageBuffersPerShaderStage` comment):
 *
 *   packed    : u32[capacity]       // bits: (kind << 16) | flags
 *   data0     : vec4[capacity]      // per-kind .xyz; .w = radius / spare
 *   data1     : vec4[capacity]      // per-kind .xyz; .w = friction μ_s
 *   linVel    : vec4[capacity]      // .xyz = linear velocity; .w = μ_k
 *   rotation  : vec4[capacity]      // unit quaternion (x, y, z, w); identity for
 *                                   // plane / sphere / capsule; U-21 resolution
 *   lambdaNT  : f32[2 * particleCap * capacity]   // [λ_n, λ_t] interleaved
 *
 * The λ buffer is f32 (non-atomic) because the per-particle solve kernel
 * assigns each thread exclusive write access to its own `(p, 0..N)` slice
 * — no atomicity needed. Reset each substep via
 * {@link PrimitiveSet.resetLambdaKernel}. The λ_n / λ_t interleave
 * (`[λ_n_00, λ_t_00, λ_n_01, λ_t_01, …]`) lets both reads for a single
 * contact share one storage binding instead of two.
 *
 * Rotation (Phase 07 U-21 resolution): oriented boxes supported via a
 * unit quaternion stored in the `rotation` buffer. `addBox` accepts a
 * `Matrix4` (rotation decomposed via `Matrix4.decompose()` with
 * non-unit scale rejected) or a `Vector3` with optional `rotation:
 * Quaternion` option. Plane / sphere / capsule store identity
 * quaternion — the kernel's quat-rotate is a no-op on identity.
 *
 * Determinism: the per-particle solve kernel is gather-mode (one thread
 * per particle, no atomics on position or λ state), so the collider solve
 * is G4 tier-1 bit-exact by construction.
 */
export class PrimitiveSet {
  readonly particles: ParticleSystem;
  readonly capacity: number;

  /** `(kind << 16) | flags` per slot. Extract in kernel with bit ops. */
  readonly packed: StorageBufferNode<'uint'>;
  readonly data0: StorageBufferNode<'vec4'>;
  /** `.xyz` = per-kind secondary data; `.w` = μ_s. */
  readonly data1: StorageBufferNode<'vec4'>;
  /** `.xyz` = linear velocity (m/s); `.w` = μ_k. */
  readonly linVel: StorageBufferNode<'vec4'>;
  /**
   * Per-slot unit quaternion `(x, y, z, w)`. Identity `(0, 0, 0, 1)` for
   * plane / sphere / capsule — the kernel's quat-rotate is a no-op on
   * identity, so non-box colliders pay two cross products per sample
   * but return the same answer they would without the rotation path.
   * For box, defines world orientation. Resolves U-21.
   *
   * Stored as a **uniform-buffer-backed array** (`three/tsl::uniformArray`)
   * rather than a storage buffer — the Phase 06 solve already binds 10
   * storage buffers per stage (see `renderer.ts` comment), which is the
   * ceiling on Apple M1 Pro + Chrome 147. Routing rotation through the
   * uniform-binding budget (default `maxUniformBuffersPerShaderStage = 12`)
   * keeps the device-limit request at 10 without touching existing
   * adapter compatibility. `uniformArray` updates per render from the JS
   * array, so mutating `rotationArray[slot].set(...)` takes effect on
   * the next dispatch without an explicit `upload()` flag.
   */
  readonly rotation: Any;

  /**
   * Per-(particle, collider) accumulated Lagrange multipliers, interleaved:
   * `lambdaNT[2 · (p · capacity + c) + 0] = λ_n`
   * `lambdaNT[2 · (p · capacity + c) + 1] = λ_t`
   * Units: kg·m (XPBD Lagrange-multiplier convention, matching Phase 05a).
   */
  readonly lambdaNT: StorageBufferNode<'float'>;

  /** Number of populated collider slots. Kernels early-out past this. */
  readonly numColliders: UniformNode<'uint', number>;

  /** Reset kernel for the λ buffer — dispatch once per substep. */
  readonly resetLambdaKernel: ComputeNode;

  /** CPU-side mirror of every GPU buffer — updated by authoring calls. */
  private readonly cpuPacked: Uint32Array;
  private readonly cpuData0: Float32Array;
  private readonly cpuData1: Float32Array;
  private readonly cpuLinVel: Float32Array;
  /**
   * Mutable Quaternion array backing the `rotation` uniform array.
   * `three/tsl::uniformArray` reads `.x`/`.y`/`.z`/`.w` per frame; we
   * call `.set(x, y, z, w)` on the per-slot Quaternion to update.
   */
  private readonly rotationArray: Quaternion[];

  /** Active slot count. Next `addX` writes to `count` and increments. */
  private count = 0;

  private readonly attachments: AttachmentRecord[] = [];
  private disposed = false;

  constructor(particles: ParticleSystem, options: PrimitiveSetOptions) {
    if (!Number.isInteger(options.capacity) || options.capacity <= 0) {
      throw new Error(`PrimitiveSet: capacity must be a positive integer, got ${options.capacity}`);
    }
    this.particles = particles;
    this.capacity = options.capacity;

    this.packed = instancedArray(options.capacity, 'uint');
    this.data0 = instancedArray(options.capacity, 'vec4');
    this.data1 = instancedArray(options.capacity, 'vec4');
    this.linVel = instancedArray(options.capacity, 'vec4');
    this.lambdaNT = instancedArray(2 * particles.capacity * options.capacity, 'float');

    this.cpuPacked = this.packed.value.array as Uint32Array;
    this.cpuData0 = this.data0.value.array as Float32Array;
    this.cpuData1 = this.data1.value.array as Float32Array;
    this.cpuLinVel = this.linVel.value.array as Float32Array;

    // Rotation — uniform-buffer-backed array of Quaternions. Default to
    // identity so any slot that hasn't been authored (or uses a kind
    // other than box) leaves the kernel's quat-rotate as a no-op.
    this.rotationArray = new Array(options.capacity);
    for (let s = 0; s < options.capacity; s++) {
      this.rotationArray[s] = new Quaternion();
    }
    this.rotation = uniformArray(this.rotationArray, 'vec4');

    // @types/three r184 `uniform(value, type)` overloads don't expose a
    // `'uint'` type string (only `'float'`). The runtime supports it —
    // `three/src/nodes/gpgpu/ComputeNode.js:180` uses exactly this call
    // shape — so the cast below bridges the typed surface without
    // forking three's types.
    this.numColliders = uniform(0, 'uint' as unknown as 'float') as unknown as UniformNode<
      'uint',
      number
    >;

    // λ reset — one thread per float slot (both λ_n and λ_t). Plain
    // `.assign` because the buffer is non-atomic; this still emits a legal
    // WGSL store because `instancedArray(..., 'float')` without `.toAtomic()`
    // exposes readWrite access.
    const lambdaNT = this.lambdaNT;
    const total = 2 * particles.capacity * options.capacity;
    this.resetLambdaKernel = Fn(() => {
      const i: Any = instanceIndex;
      lambdaNT.element(i).assign(float(0.0));
    })().compute(total);
  }

  /** Number of currently populated colliders. */
  get colliderCount(): number {
    return this.count;
  }

  /**
   * Append a plane to the set. `normal` must be unit-length; the call
   * throws if it is not (tolerance `1e-4`). Plan §Primitives:
   * `phi = n · (x - p)`.
   *
   * Returns the slot index so the caller can later call
   * {@link attachToObject3D}.
   */
  addPlane(normal: Vector3, point: Vector3, options: PlaneOptions = {}): number {
    this.assertCapacity();
    const n2 = normal.lengthSq();
    if (Math.abs(n2 - 1.0) > 1e-4) {
      throw new Error(`PrimitiveSet.addPlane: normal must be unit-length; |n|² = ${n2}`);
    }
    const slot = this.count++;
    const { muS, muK } = resolveFriction(options);
    this.cpuPacked[slot] = packKindFlags(KIND_PLANE, 0);
    this.writeVec4(this.cpuData0, slot, normal.x, normal.y, normal.z, 0);
    this.writeVec4(this.cpuData1, slot, point.x, point.y, point.z, muS);
    this.writeLinearVelocityWithMuK(slot, options.linearVelocity, muK);
    return slot;
  }

  /**
   * Append a sphere. `phi = |x - c| - R`. With `invert = true`, the sphere
   * acts as a bowl/container (interior is the valid region).
   */
  addSphere(center: Vector3, radius: number, options: SphereOptions = {}): number {
    this.assertCapacity();
    if (!(radius > 0)) {
      throw new Error(`PrimitiveSet.addSphere: radius must be positive, got ${radius}`);
    }
    const slot = this.count++;
    const { muS, muK } = resolveFriction(options);
    this.cpuPacked[slot] = packKindFlags(KIND_SPHERE, options.invert ? FLAG_INVERT : 0);
    this.writeVec4(this.cpuData0, slot, center.x, center.y, center.z, radius);
    this.writeVec4(this.cpuData1, slot, 0, 0, 0, muS);
    this.writeLinearVelocityWithMuK(slot, options.linearVelocity, muK);
    return slot;
  }

  /**
   * Append an oriented box. `transform` may be a `Vector3` (taken as
   * center, combined with `options.rotation` if provided) or a `Matrix4`
   * (decomposed into translation + rotation via `Matrix4.decompose()`;
   * the scale component must be unit — U-21 resolution). The box's
   * half-extents carry the size; the Matrix4 path ignores scale.
   */
  addBox(transform: Vector3 | Matrix4, halfExtents: Vector3, options: BoxOptions = {}): number {
    this.assertCapacity();
    if (!(halfExtents.x > 0 && halfExtents.y > 0 && halfExtents.z > 0)) {
      throw new Error(
        `PrimitiveSet.addBox: halfExtents must be strictly positive, got (${halfExtents.x}, ${halfExtents.y}, ${halfExtents.z})`,
      );
    }
    const { center, rotation } = extractBoxTransform(transform, options.rotation);
    const slot = this.count++;
    const { muS, muK } = resolveFriction(options);
    this.cpuPacked[slot] = packKindFlags(KIND_BOX, options.invert ? FLAG_INVERT : 0);
    this.writeVec4(this.cpuData0, slot, center.x, center.y, center.z, 0);
    this.writeVec4(this.cpuData1, slot, halfExtents.x, halfExtents.y, halfExtents.z, muS);
    this.writeLinearVelocityWithMuK(slot, options.linearVelocity, muK);
    this.rotationArray[slot]!.copy(rotation);
    return slot;
  }

  /**
   * Append a capsule with endpoints `a` and `b` and radius `R`. Degenerate
   * capsules (a ≈ b) reduce to a sphere at `a` with radius `R` — allowed,
   * but the caller should just use `addSphere` for that case.
   */
  addCapsule(a: Vector3, b: Vector3, radius: number, options: CapsuleOptions = {}): number {
    this.assertCapacity();
    if (!(radius > 0)) {
      throw new Error(`PrimitiveSet.addCapsule: radius must be positive, got ${radius}`);
    }
    const slot = this.count++;
    const { muS, muK } = resolveFriction(options);
    this.cpuPacked[slot] = packKindFlags(KIND_CAPSULE, 0);
    this.writeVec4(this.cpuData0, slot, a.x, a.y, a.z, radius);
    this.writeVec4(this.cpuData1, slot, b.x, b.y, b.z, muS);
    this.writeLinearVelocityWithMuK(slot, options.linearVelocity, muK);
    return slot;
  }

  /**
   * Attach an `Object3D` to a previously-added collider slot. The slot's
   * position, orientation, and linear velocity are driven from the object
   * each call to {@link updateKinematics}. Angular velocity is not yet
   * derived — the collider friction pipeline uses linear velocity only
   * for contact-point relative motion; a rotated kinematic box slides
   * correctly around its center but its friction ignores the tangential
   * motion a rotating surface would impart (post-MVP extension).
   *
   * The same slot cannot be attached twice — repeated calls replace the
   * previous attachment.
   */
  attachToObject3D(slot: number, object: Object3D): void {
    this.assertAlive();
    if (!Number.isInteger(slot) || slot < 0 || slot >= this.count) {
      throw new Error(`PrimitiveSet.attachToObject3D: invalid slot ${slot} (count=${this.count})`);
    }
    // Replace any existing attachment for this slot.
    for (let i = this.attachments.length - 1; i >= 0; i--) {
      if (this.attachments[i]!.slot === slot) this.attachments.splice(i, 1);
    }
    const pos = new Vector3Ctor();
    object.getWorldPosition(pos);
    this.attachments.push({ slot, object, prevPosition: pos.clone() });
    // Seed the slot's position + orientation from the object immediately
    // so a caller that attaches before stepping has correct initial state.
    this.setColliderCenter(slot, pos);
    object.getWorldQuaternion(this.rotationArray[slot]!);
    this.invalidate(slot);
  }

  /**
   * Pull current world positions + orientations from every attached
   * `Object3D`, push the updated center and quaternion into the slot's
   * buffers, and derive a fresh linear velocity as
   * `(posCurr - posPrev) / dt`.
   *
   * The caller invokes this once per frame before `SimLoop.step`. `dt`
   * must match the frame delta used in the step call (NOT the substep
   * dt) — the derived velocity is the per-frame average, which is the
   * velocity the contact-friction pass will compute relative to.
   */
  updateKinematics(dt: number): void {
    this.assertAlive();
    if (!Number.isFinite(dt) || dt <= 0) {
      throw new Error(`PrimitiveSet.updateKinematics: dt must be positive finite, got ${dt}`);
    }
    const curr = new Vector3Ctor();
    const invDt = 1 / dt;
    for (const rec of this.attachments) {
      rec.object.getWorldPosition(curr);
      rec.object.getWorldQuaternion(this.rotationArray[rec.slot]!);
      const vx = (curr.x - rec.prevPosition.x) * invDt;
      const vy = (curr.y - rec.prevPosition.y) * invDt;
      const vz = (curr.z - rec.prevPosition.z) * invDt;
      this.setColliderCenter(rec.slot, curr);
      this.writeVec4(this.cpuLinVel, rec.slot, vx, vy, vz, this.cpuLinVel[rec.slot * 4 + 3]!);
      rec.prevPosition.copy(curr);
      this.invalidate(rec.slot);
    }
  }

  /**
   * Update a slot's orientation directly (without attaching an
   * `Object3D`). Useful for live UI sliders or scripted animations that
   * drive rotation without a corresponding scene object.
   */
  setColliderRotation(slot: number, rotation: Quaternion): void {
    this.assertAlive();
    if (!Number.isInteger(slot) || slot < 0 || slot >= this.count) {
      throw new Error(
        `PrimitiveSet.setColliderRotation: invalid slot ${slot} (count=${this.count})`,
      );
    }
    this.rotationArray[slot]!.copy(rotation);
    this.invalidate(slot);
  }

  /**
   * Flush the CPU mirror to the GPU. Called once after authoring is
   * complete, and again whenever {@link updateKinematics} changes a slot.
   * Internally calls `.value.needsUpdate = true` on each buffer and
   * refreshes the `numColliders` uniform.
   */
  upload(): void {
    this.assertAlive();
    this.packed.value.needsUpdate = true;
    this.data0.value.needsUpdate = true;
    this.data1.value.needsUpdate = true;
    this.linVel.value.needsUpdate = true;
    // `rotation` is a `uniformArray` — its update() is invoked per
    // render from `rotationArray`, no manual flag needed.
    this.numColliders.value = this.count;
  }

  destroy(): void {
    this.disposed = true;
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  private writeVec4(
    target: Float32Array,
    slot: number,
    x: number,
    y: number,
    z: number,
    w: number,
  ): void {
    const base = slot * 4;
    target[base + 0] = x;
    target[base + 1] = y;
    target[base + 2] = z;
    target[base + 3] = w;
  }

  /** Pack a Vector3 linear velocity + scalar μ_k into `linVel[slot]`. */
  private writeLinearVelocityWithMuK(slot: number, v: Vector3 | undefined, muK: number): void {
    if (v === undefined) {
      this.writeVec4(this.cpuLinVel, slot, 0, 0, 0, muK);
    } else {
      this.writeVec4(this.cpuLinVel, slot, v.x, v.y, v.z, muK);
    }
  }

  /**
   * Patch a collider's "primary point" (plane point, sphere center, box
   * center, capsule endpoint a) in place. The layout differs by kind
   * because some primitives pack the radius into `data0.w`.
   */
  private setColliderCenter(slot: number, center: Vector3): void {
    const kind = unpackKind(this.cpuPacked[slot]!);
    const base = slot * 4;
    if (kind === KIND_PLANE) {
      // `setColliderCenter` on a plane repositions the plane's reference
      // point, which lives in `data1.xyz`. (The normal stays in data0.)
      const base1 = slot * 4;
      this.cpuData1[base1 + 0] = center.x;
      this.cpuData1[base1 + 1] = center.y;
      this.cpuData1[base1 + 2] = center.z;
    } else if (kind === KIND_SPHERE || kind === KIND_BOX) {
      this.cpuData0[base + 0] = center.x;
      this.cpuData0[base + 1] = center.y;
      this.cpuData0[base + 2] = center.z;
      // radius (sphere) or zero (box) in .w — leave as is.
    } else if (kind === KIND_CAPSULE) {
      // Translating a capsule translates both endpoints rigidly. We
      // treat the "center" of a capsule as the midpoint of (a, b) and
      // shift both endpoints by the delta from the previous midpoint.
      // Simple: no rotation.
      const ax = this.cpuData0[base + 0]!;
      const ay = this.cpuData0[base + 1]!;
      const az = this.cpuData0[base + 2]!;
      const bx = this.cpuData1[base + 0]!;
      const by = this.cpuData1[base + 1]!;
      const bz = this.cpuData1[base + 2]!;
      const midX = 0.5 * (ax + bx);
      const midY = 0.5 * (ay + by);
      const midZ = 0.5 * (az + bz);
      const dx = center.x - midX;
      const dy = center.y - midY;
      const dz = center.z - midZ;
      this.cpuData0[base + 0] = ax + dx;
      this.cpuData0[base + 1] = ay + dy;
      this.cpuData0[base + 2] = az + dz;
      this.cpuData1[base + 0] = bx + dx;
      this.cpuData1[base + 1] = by + dy;
      this.cpuData1[base + 2] = bz + dz;
    }
  }

  private invalidate(_slot: number): void {
    // Per-buffer needsUpdate is set in `upload()`; this hook exists so
    // authoring can mark a buffer dirty per-slot if a future optimization
    // wants to batch partial uploads.
  }

  private assertCapacity(): void {
    this.assertAlive();
    if (this.count >= this.capacity) {
      throw new Error(
        `PrimitiveSet: capacity ${this.capacity} exhausted; construct with a larger capacity`,
      );
    }
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('PrimitiveSet has been destroyed');
  }
}

/**
 * Pack `kind` into the high 16 bits and `flags` into the low 16 bits of a
 * u32. Matches the GPU-side unpack in `primitives.ts`:
 *   `kind  = packed >> 16`
 *   `flags = packed & 0xFFFF`
 * 16 bits per field is well beyond what MVP needs — there are 4 kinds and
 * 1 flag bit — and the packing saves one storage-buffer binding (critical
 * for staying under WebGPU's 10-per-stage default, see `renderer.ts`).
 */
function packKindFlags(kind: number, flags: number): number {
  return ((kind & 0xffff) << 16) | (flags & 0xffff);
}

/** Inverse of {@link packKindFlags} for the kind field. */
function unpackKind(packed: number): number {
  return (packed >>> 16) & 0xffff;
}

/**
 * Resolve per-collider friction options against Phase 05-compatible
 * defaults (μ_s = 0.5, μ_k = 0.4) and validate non-negativity.
 */
function resolveFriction(options: ColliderFrictionOptions): {
  readonly muS: number;
  readonly muK: number;
} {
  const muS = options.muS ?? 0.5;
  const muK = options.muK ?? 0.4;
  if (!(muS >= 0) || !(muK >= 0)) {
    throw new Error(`PrimitiveSet: μ_s and μ_k must be non-negative (got μ_s=${muS}, μ_k=${muK})`);
  }
  return { muS, muK };
}

/**
 * Extract a box center and orientation from either a `Vector3` (with
 * optional separate `rotation` argument) or a `Matrix4` (decomposed into
 * translation + rotation via `Matrix4.decompose()`). The Matrix4 scale
 * component must be unit within {@link SCALE_EPS} — boxes carry their
 * size in `halfExtents`, so a non-unit Matrix4 scale is almost certainly
 * caller confusion and is rejected with a thrown diagnostic.
 *
 * U-21 resolution: rotation is now first-class. Identity rotation is
 * always valid (defaults to identity when the caller passes a plain
 * `Vector3` without a `rotation` option).
 */
function extractBoxTransform(
  transform: Vector3 | Matrix4,
  rotationOption: Quaternion | undefined,
): { center: Vector3; rotation: Quaternion } {
  const v = transform as Vector3;
  if (typeof v.x === 'number' && typeof v.y === 'number' && typeof v.z === 'number') {
    return {
      center: v,
      rotation: rotationOption ? rotationOption.clone() : new Quaternion(),
    };
  }
  const m = transform as Matrix4;
  const translation = new Vector3Ctor();
  const rotation = new Quaternion();
  const scale = new Vector3Ctor();
  m.decompose(translation, rotation, scale);
  const scaleErr = Math.abs(scale.x - 1) + Math.abs(scale.y - 1) + Math.abs(scale.z - 1);
  if (scaleErr > SCALE_EPS) {
    throw new Error(
      `PrimitiveSet.addBox: Matrix4 scale must be unit (got ${scale.x}, ${scale.y}, ${scale.z}). ` +
        'The box carries its size in `halfExtents`; non-unit matrix scale is a caller mistake.',
    );
  }
  return { center: translation, rotation };
}
