import {
  Fn,
  If,
  Return,
  float,
  instanceIndex,
  instancedArray,
  texture3D,
  uniform,
  vec3,
} from 'three/tsl';
import {
  Data3DTexture,
  HalfFloatType,
  LinearFilter,
  ClampToEdgeWrapping,
  Matrix3,
  Matrix4,
  Quaternion,
  RedFormat,
  Vector3,
} from 'three';
import { toHalfFloat } from 'three/src/extras/DataUtils.js';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

// `UniformNode` is imported as a value alias via the type-only import above;
// the reference keeps it active under verbatimModuleSyntax even though TS
// collapses it during emit. Needed for the builder-function `dt` param
// types on the exported `buildSdfSolveKernel`/`buildSdfFrictionVelocityKernel`.

import type { ParticleSystem } from '../particles.js';
import { type ContactAccumulator, emitAccumulateDelta } from '../contact/accumulator.js';
import {
  type VelocityAccumulator,
  emitAccumulateVelocityDelta,
} from '../contact/velocityAccumulator.js';
import { emitSampleSdf, type SdfFields } from './sdf.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Per-collider friction options. Mirrors `ColliderFrictionOptions` in
 * `PrimitiveSet.ts` but re-declared here so `SDFCollider` callers don't
 * have to import from the primitive-set module.
 */
export interface SDFColliderFriction {
  readonly muS?: number;
  readonly muK?: number;
}

export interface SDFColliderOptions extends SDFColliderFriction {
  /**
   * World-space position of the mesh's local origin (the coordinate the
   * baker was fed). `Vector3(0, 0, 0)` places the mesh at world origin.
   * Resolves U-23.
   *
   * Alias `translation` is kept in the same slot for back-compat with
   * the Phase 07 initial API; only one of {position, translation}
   * should be passed.
   */
  readonly position?: Vector3;
  /** Deprecated alias for {@link position}. */
  readonly translation?: Vector3;
  /**
   * World-space orientation of the mesh. Defaults to identity. Applied
   * around the mesh's local origin (i.e. around `position` in world
   * space). Resolves U-23.
   */
  readonly rotation?: Quaternion;
  /**
   * Uniform scale factor applied around the mesh's local origin. Only
   * **uniform** scale is supported — non-uniform scaling breaks the
   * SDF's `|∇φ| = 1` eikonal invariant and would require re-baking
   * anyway. Defaults to `1`. Must be strictly positive.
   */
  readonly scale?: number;
}

/** Normalized SDF data, compatible with the output of `src/sdf/bake.ts`. */
export interface SDFData {
  readonly data: Float32Array;
  readonly resolution: readonly [number, number, number];
  readonly origin: readonly [number, number, number];
  readonly voxelSize: readonly [number, number, number];
}

/**
 * Runtime SDF collider — one instance per baked mesh.
 *
 * Unlike `PrimitiveSet` (which packs up to `capacity` analytic primitives
 * into shared storage buffers), each SDF collider is its own object and
 * its own compute pipeline. The reason is a WebGPU binding constraint:
 * `texture_3d<f32>` bindings are compile-time in WGSL, so a scene with
 * `N` SDFs compiles `N` distinct collider kernels. Runtime indexing into
 * a "binding array" of 3D textures would require the `binding_array`
 * extension which TSL does not currently surface.
 *
 * Scope in v1:
 *   - Static SDFs only (no kinematic motion — no `linearVelocity`, no
 *     `attachToObject3D`). Plan title: "Phase 07 — Static SDF collision".
 *   - Translated-only transform (no rotation, no scale). A rotated SDF
 *     collider is filed as an UNKNOWN at Phase 07 exit, same shape as
 *     U-21 for rotated boxes.
 *   - `r16float` voxel storage — picked over `r32float` because the
 *     latter needs the optional `float32-filterable` WebGPU feature and
 *     would silently fall back to point-sampling on adapters that lack
 *     it (see `node_modules/three/src/renderers/webgpu/nodes/WGSLNodeBuilder.js:727`
 *     — the `isUnfilterable` branch routes `r32float` through
 *     `textureLoad`, which is nearest only). The half-float precision
 *     (~1e-3 relative) is well below the per-voxel quantization of the
 *     baker's grid, so the test gate `|φ − analytical| ≤ 0.5 · voxelSize`
 *     is not tightened by it.
 *
 * Construction is in two steps, mirroring `PrimitiveSet`'s pattern with
 * `ContactBuffer` in Phase 05:
 *   1. `new SDFCollider(particles, sdfData, options)` — allocates the
 *      `Data3DTexture`, uniforms, and λ state; builds the reset kernel.
 *      Can be called before `SimLoop` because no accumulator is needed.
 *   2. `buildSdfSolveKernel` + `buildSdfFrictionVelocityKernel` (exported
 *      below) — called by `SimLoop` once the position and velocity
 *      accumulators exist. Same builder-function pattern as
 *      `buildColliderSolveKernel` in `collision/solve.ts`.
 *
 * Determinism (G4 tier 1): the solve kernel is gather-mode (one thread
 * per particle, no atomics on λ state — each thread writes its own
 * `(p,)` slot), and the Δx correction is scattered via the shared
 * {@link ContactAccumulator}'s i32 `atomicAdd` — same bit-exact path as
 * Phase 06.
 */
export class SDFCollider {
  readonly particles: ParticleSystem;
  readonly resolution: readonly [number, number, number];
  readonly voxelSize: readonly [number, number, number];

  /**
   * Per-collider state exposed as uniforms so demos and live artist tools
   * can change them mid-sim without rebuilding kernels. Writing to `.value`
   * on any of these propagates to the next dispatch; the ergonomic setters
   * ({@link SDFCollider.muS}, {@link SDFCollider.setRotation}, etc.) are
   * the intended surface.
   *
   * The kernel interprets them as: `x_local = invRotation · (x − position) ·
   * invScale`, then samples the grid starting at `bakedOrigin` (an immutable
   * record of where the baker placed the grid's min corner in mesh-local
   * space). On the way out, `φ_world = scale · φ_local` and `∇φ_world =
   * rotation · ∇φ_local`. See `collision/sdf.ts::emitSampleSdf`.
   */
  readonly muSUniform: UniformNode<'float', number>;
  readonly muKUniform: UniformNode<'float', number>;
  readonly positionUniform: UniformNode<'vec3', Vector3>;
  readonly rotationUniform: UniformNode<'mat3', Matrix3>;
  readonly invRotationUniform: UniformNode<'mat3', Matrix3>;
  readonly scaleUniform: UniformNode<'float', number>;
  readonly invScaleUniform: UniformNode<'float', number>;
  readonly bakedOriginUniform: UniformNode<'vec3', Vector3>;

  /** Grid origin as authored by the baker — never mutated. */
  private readonly bakedOrigin: readonly [number, number, number];
  /** Current world-space position of the mesh's local origin. */
  private readonly currentPosition: Vector3;
  /** Current rotation (unit quaternion). */
  private readonly currentRotation: Quaternion;
  /** Scratch Matrix3 reused by `setRotation` to avoid per-call allocation. */
  private readonly scratchMatrix3: Matrix3;
  private readonly scratchInvMatrix3: Matrix3;

  /** f32 byte count of the CPU-side voxel payload. For U-07 logging. */
  readonly cpuByteLength: number;
  /** GPU-side byte count after `f32 → f16` conversion. For U-07 logging. */
  readonly gpuByteLength: number;

  readonly texture: Data3DTexture;
  /** `Texture3DNode` — TSL-side handle on the texture. */
  readonly textureNode: Any;

  /** TSL-side struct-of-uniforms consumed by `emitSampleSdf`. */
  readonly fields: SdfFields;

  /**
   * Per-particle accumulated Lagrange multipliers, interleaved:
   * `lambdaNT[2·p + 0] = λ_n`, `lambdaNT[2·p + 1] = λ_t`. Units kg·m
   * (XPBD convention). One pair per particle; no per-collider slot
   * because each `SDFCollider` owns its own buffer.
   */
  readonly lambdaNT: StorageBufferNode<'float'>;

  /** Dispatched once per substep, before the iter loop. */
  readonly resetLambdaKernel: ComputeNode;

  private disposed = false;

  constructor(particles: ParticleSystem, sdf: SDFData, options: SDFColliderOptions = {}) {
    const muS = options.muS ?? 0.5;
    const muK = options.muK ?? 0.4;
    if (!(muS >= 0) || !(muK >= 0)) {
      throw new Error(`SDFCollider: μ_s and μ_k must be non-negative (got μ_s=${muS}, μ_k=${muK})`);
    }
    const [resX, resY, resZ] = sdf.resolution;
    if (
      !Number.isInteger(resX) ||
      resX < 4 ||
      !Number.isInteger(resY) ||
      resY < 4 ||
      !Number.isInteger(resZ) ||
      resZ < 4
    ) {
      throw new Error(
        `SDFCollider: resolution must be integers ≥ 4, got [${resX}, ${resY}, ${resZ}]`,
      );
    }
    const voxelCount = resX * resY * resZ;
    if (sdf.data.length !== voxelCount) {
      throw new Error(
        `SDFCollider: data length ${sdf.data.length} does not match resolution ${resX}·${resY}·${resZ}=${voxelCount}`,
      );
    }
    this.particles = particles;

    if (options.position !== undefined && options.translation !== undefined) {
      throw new Error(
        'SDFCollider: pass only one of `position` / `translation` (translation is the deprecated alias for position).',
      );
    }
    const initialPosition = options.position ?? options.translation ?? new Vector3(0, 0, 0);
    const initialRotation = options.rotation ?? new Quaternion();
    const initialScale = options.scale ?? 1;
    if (!(initialScale > 0) || !Number.isFinite(initialScale)) {
      throw new Error(`SDFCollider: scale must be a positive finite number, got ${initialScale}`);
    }

    this.bakedOrigin = [sdf.origin[0], sdf.origin[1], sdf.origin[2]];
    this.currentPosition = initialPosition.clone();
    this.currentRotation = initialRotation.clone();
    this.scratchMatrix3 = new Matrix3();
    this.scratchInvMatrix3 = new Matrix3();
    this.resolution = [resX, resY, resZ];
    this.voxelSize = [sdf.voxelSize[0], sdf.voxelSize[1], sdf.voxelSize[2]];

    // f32 → f16 conversion. three's `DataUtils.toHalfFloat` returns a u16
    // bit-pattern per f32 input; we pack into a Uint16Array that
    // `Data3DTexture` accepts with `HalfFloatType`. r16float is filterable
    // on every WebGPU adapter (unlike r32float — see class doc).
    const halfData = new Uint16Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      halfData[i] = toHalfFloat(sdf.data[i]!);
    }
    this.cpuByteLength = sdf.data.byteLength;
    this.gpuByteLength = halfData.byteLength;

    const texture = new Data3DTexture(halfData, resX, resY, resZ);
    texture.format = RedFormat;
    texture.type = HalfFloatType;
    texture.minFilter = LinearFilter;
    texture.magFilter = LinearFilter;
    texture.wrapS = ClampToEdgeWrapping;
    texture.wrapT = ClampToEdgeWrapping;
    texture.wrapR = ClampToEdgeWrapping;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;
    this.texture = texture;
    this.textureNode = texture3D(texture);

    // Uniforms. `uniform(Vector3)` without a second arg hits the Vector3
    // overload and derives the `vec3` node type automatically; same for
    // `uniform(Matrix3)`.
    const positionVec = this.currentPosition.clone();
    const bakedOriginVec = new Vector3(
      this.bakedOrigin[0],
      this.bakedOrigin[1],
      this.bakedOrigin[2],
    );
    const voxelVec = new Vector3(this.voxelSize[0], this.voxelSize[1], this.voxelSize[2]);
    const invVoxelVec = new Vector3(
      1 / this.voxelSize[0],
      1 / this.voxelSize[1],
      1 / this.voxelSize[2],
    );
    const resVec = new Vector3(resX, resY, resZ);
    const rotMat = new Matrix3().setFromMatrix4(
      new Matrix4().makeRotationFromQuaternion(this.currentRotation),
    );
    const invRotMat = rotMat.clone().transpose();

    this.positionUniform = uniform(positionVec);
    const voxelSizeUniform = uniform(voxelVec);
    const invVoxelSizeUniform = uniform(invVoxelVec);
    const resolutionUniform = uniform(resVec);
    this.bakedOriginUniform = uniform(bakedOriginVec);
    this.rotationUniform = uniform(rotMat);
    this.invRotationUniform = uniform(invRotMat);
    this.scaleUniform = uniform(initialScale, 'float');
    this.invScaleUniform = uniform(1 / initialScale, 'float');
    this.muSUniform = uniform(muS, 'float');
    this.muKUniform = uniform(muK, 'float');

    this.fields = {
      texture: this.textureNode,
      position: this.positionUniform,
      bakedOrigin: this.bakedOriginUniform,
      voxelSize: voxelSizeUniform,
      invVoxelSize: invVoxelSizeUniform,
      resolution: resolutionUniform,
      rotation: this.rotationUniform,
      invRotation: this.invRotationUniform,
      scale: this.scaleUniform,
      invScale: this.invScaleUniform,
    };

    // λ state — one (λ_n, λ_t) pair per particle.
    this.lambdaNT = instancedArray(2 * particles.capacity, 'float');
    const lambdaNT = this.lambdaNT;
    this.resetLambdaKernel = Fn(() => {
      const i: Any = instanceIndex;
      lambdaNT.element(i).assign(float(0.0));
    })().compute(2 * particles.capacity);
  }

  /** Static Coulomb coefficient — set triggers uniform update next dispatch. */
  get muS(): number {
    return this.muSUniform.value;
  }
  set muS(v: number) {
    if (!(v >= 0)) throw new Error(`SDFCollider.muS: must be non-negative, got ${v}`);
    this.muSUniform.value = v;
  }

  /** Kinetic Coulomb coefficient — set triggers uniform update next dispatch. */
  get muK(): number {
    return this.muKUniform.value;
  }
  set muK(v: number) {
    if (!(v >= 0)) throw new Error(`SDFCollider.muK: must be non-negative, got ${v}`);
    this.muKUniform.value = v;
  }

  /** Current world-space position of the mesh's local origin. */
  get position(): Vector3 {
    return this.currentPosition.clone();
  }

  /** Current world-space orientation as a unit quaternion. */
  get rotation(): Quaternion {
    return this.currentRotation.clone();
  }

  /** Current uniform scale factor. */
  get scale(): number {
    return this.scaleUniform.value;
  }

  /**
   * Move the collider in world space. The `position` uniform is
   * rewritten so the next dispatch samples at the new location. Safe
   * to call every frame.
   */
  setPosition(position: Vector3): void {
    this.currentPosition.copy(position);
    this.positionUniform.value.copy(position);
  }

  /**
   * Deprecated alias for {@link setPosition}. Kept so the Phase 07 demo
   * and early-adopter code continue to compile after the U-23 rename;
   * prefer `setPosition` in new code.
   */
  setTranslation(translation: Vector3): void {
    this.setPosition(translation);
  }

  /**
   * Rotate the collider around its mesh-local origin. Updates both the
   * forward and inverse rotation uniforms — the kernel uses the inverse
   * to map world → local at sample time and the forward to map the
   * gradient back.
   */
  setRotation(rotation: Quaternion): void {
    this.currentRotation.copy(rotation);
    this.scratchMatrix3.setFromMatrix4(new Matrix4().makeRotationFromQuaternion(rotation));
    this.scratchInvMatrix3.copy(this.scratchMatrix3).transpose();
    this.rotationUniform.value.copy(this.scratchMatrix3);
    this.invRotationUniform.value.copy(this.scratchInvMatrix3);
  }

  /**
   * Uniformly scale the collider around its mesh-local origin. Must be
   * strictly positive. Scaled SDFs preserve the `|∇φ| = 1` eikonal
   * invariant: `φ_world(x) = scale · φ_local((x − position)/scale)`
   * yields gradient magnitudes unchanged.
   */
  setScale(scale: number): void {
    if (!(scale > 0) || !Number.isFinite(scale)) {
      throw new Error(`SDFCollider.setScale: scale must be a positive finite number, got ${scale}`);
    }
    this.scaleUniform.value = scale;
    this.invScaleUniform.value = 1 / scale;
  }

  /**
   * Apply a full `Matrix4` transform at once. The matrix is decomposed
   * into (position, rotation, scale) via `Matrix4.decompose()`. Non-
   * uniform scale (scale.x ≠ scale.y ≠ scale.z) breaks the SDF's
   * eikonal invariant and throws.
   */
  setTransform(transform: Matrix4): void {
    const translation = new Vector3();
    const rotation = new Quaternion();
    const scaleVec = new Vector3();
    transform.decompose(translation, rotation, scaleVec);
    const scaleErr = Math.abs(scaleVec.x - scaleVec.y) + Math.abs(scaleVec.x - scaleVec.z);
    if (scaleErr > 1e-4 * Math.abs(scaleVec.x)) {
      throw new Error(
        `SDFCollider.setTransform: non-uniform scale (${scaleVec.x}, ${scaleVec.y}, ${scaleVec.z}) ` +
          'breaks the SDF eikonal invariant and is not supported. Use `setScale(uniform)`.',
      );
    }
    this.setPosition(translation);
    this.setRotation(rotation);
    this.setScale(scaleVec.x);
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.texture.dispose();
  }
}

/**
 * Build the per-particle SDF solve kernel — Macklin 2014 §6.1 eq. (22)
 * normal projection + Macklin 2020 §3.5 proactive static-friction cone
 * gate. Specializes `collision/solve.ts` to a single static SDF (no
 * `Loop` over collider slots; no `linVel · dt` subtraction because a
 * static SDF has collider Δp = 0).
 */
export function buildSdfSolveKernel(args: {
  readonly sdf: SDFCollider;
  readonly accumulator: ContactAccumulator;
  /** Shared with the XPBD / velocity-friction dt uniform. */
  readonly dt: UniformNode<'float', number>;
}): ComputeNode {
  const { sdf, accumulator } = args;
  const particles = sdf.particles;
  const r = particles.particleRadius;
  const muSUniform: Any = sdf.muSUniform;
  const fields = sdf.fields;
  const lambdaNT = sdf.lambdaNT;

  return Fn(() => {
    const p: Any = instanceIndex;
    const w: Any = particles.invMass.element(p).toVar();
    If(w.lessThanEqual(float(0.0)), () => {
      Return();
    });

    const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
    const x0: Any = particles.positions.element(p).xyz.toVar();
    const dp: Any = xStar.sub(x0).toVar();

    const phi: Any = float(0.0).toVar();
    const grad: Any = vec3(float(0.0), float(0.0), float(0.0)).toVar();
    emitSampleSdf(fields, xStar, phi, grad);

    If(phi.lessThan(float(r)), () => {
      const d: Any = float(r).sub(phi).toVar();

      // Normalize gradient — for an SDF `|∇φ| ≈ 1`, but central
      // differences introduce measurable error (plan §Validation G1
      // gradient sanity bound `|∇φ| ∈ [0.8, 1.2]`). Normalizing keeps
      // the correction magnitude consistent with the projection depth.
      const gradLen: Any = grad.length().toVar();
      If(gradLen.lessThanEqual(float(1e-6)), () => {
        Return();
      });
      const n: Any = grad.div(gradLen).toVar();

      // ---- Normal projection (Macklin 2014 §6.1 eq. (22)) ----
      const dxN: Any = n.mul(d).toVar();

      // Accumulate λ_n — interleaved `[λ_n, λ_t]` per particle.
      const lambdaNIdx: Any = p.mul(2);
      const lambdaTIdx: Any = p.mul(2).add(1);
      const dLambdaN: Any = d.div(w).toVar();
      const prevLambdaN: Any = lambdaNT.element(lambdaNIdx).toVar();
      const lambdaNNow: Any = prevLambdaN.add(dLambdaN).toVar();
      lambdaNT.element(lambdaNIdx).assign(lambdaNNow);

      // ---- Static friction (Macklin 2020 §3.5 eqs. 26-28) ----
      // Static collider: Δp_rel = dp (no `linVel · dt` subtraction).
      const tangential: Any = dp.sub(n.mul(dp.dot(n))).toVar();
      const tanLen: Any = tangential.length().toVar();

      const prevLambdaT: Any = lambdaNT.element(lambdaTIdx).toVar();
      const correctionLambdaT: Any = tanLen.div(w).toVar();
      const coneCapacity: Any = muSUniform.mul(lambdaNNow).sub(prevLambdaT).toVar();
      const tanValid: Any = tanLen.greaterThan(float(1e-10));
      const withinStatic: Any = correctionLambdaT.lessThanEqual(coneCapacity);
      const applyStatic: Any = tanValid.and(withinStatic);

      const dxF: Any = tangential.negate();
      const totalDx: Any = applyStatic.select(dxN.add(dxF), dxN).toVar();
      emitAccumulateDelta(accumulator, p, totalDx);

      const dLambdaT: Any = applyStatic.select(tanLen.div(w), float(0.0)).toVar();
      lambdaNT.element(lambdaTIdx).assign(prevLambdaT.add(dLambdaT));
    });
  })().compute(particles.capacity);
}

/**
 * Build the per-particle SDF velocity-level dynamic-friction kernel —
 * Macklin 2020 §3.6 eqs. (29)-(30) & (33). Specializes
 * `collision/frictionVelocity.ts` to a single static SDF (no collider
 * Loop; no `linVel` subtraction because a static SDF has collider
 * contact-point velocity = 0).
 */
export function buildSdfFrictionVelocityKernel(args: {
  readonly sdf: SDFCollider;
  readonly velocityAccumulator: VelocityAccumulator;
  readonly dt: UniformNode<'float', number>;
}): ComputeNode {
  const { sdf, velocityAccumulator, dt } = args;
  const particles = sdf.particles;
  const muKUniform: Any = sdf.muKUniform;
  const fields = sdf.fields;
  const lambdaNT = sdf.lambdaNT;

  return Fn(() => {
    const p: Any = instanceIndex;
    const w: Any = particles.invMass.element(p).toVar();
    If(w.lessThanEqual(float(0.0)), () => {
      Return();
    });

    const lambdaNIdx: Any = p.mul(2);
    const lambdaN: Any = lambdaNT.element(lambdaNIdx).toVar();
    If(lambdaN.lessThanEqual(float(0.0)), () => {
      Return();
    });

    const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
    const v: Any = particles.velocities.element(p).xyz.toVar();

    const phi: Any = float(0.0).toVar();
    const grad: Any = vec3(float(0.0), float(0.0), float(0.0)).toVar();
    emitSampleSdf(fields, xStar, phi, grad);
    const gradLen: Any = grad.length().toVar();
    If(gradLen.lessThanEqual(float(1e-6)), () => {
      Return();
    });
    const n: Any = grad.div(gradLen).toVar();

    // Static SDF: collider contact-point velocity = 0 → v_rel = v.
    const vN: Any = n.dot(v);
    const vT: Any = v.sub(n.mul(vN)).toVar();
    const vTLen: Any = vT.length().toVar();
    If(vTLen.lessThanEqual(float(1e-6)), () => {
      Return();
    });

    // Eq. 30 threshold with explicit reduced-mass factor (same
    // dimensional-consistency note as `collision/frictionVelocity.ts`).
    const threshold: Any = muKUniform
      .mul(lambdaN)
      .mul(w)
      .div(dt as Any)
      .toVar();
    const deltaVMag: Any = threshold.min(vTLen).toVar();
    const deltaV: Any = vT.div(vTLen).mul(deltaVMag).negate().toVar();
    emitAccumulateVelocityDelta(velocityAccumulator, p, deltaV);
  })().compute(particles.capacity);
}
