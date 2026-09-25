import {
  Fn,
  atomicAdd,
  atomicLoad,
  atomicMax,
  atomicStore,
  float,
  instanceIndex,
  int,
  uint,
  uniform,
  vec4,
} from 'three/tsl';
import { instancedArray } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleSystem } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Ceiling for the fixed-point velocity accumulator's magnitude. Same 2^30
 * headroom convention as the position accumulator — one atomicAdd of any
 * valid scaled contribution cannot push the stored i32 past 0x7FFFFFFF.
 */
export const VELOCITY_ACCUMULATOR_HEADROOM = 1 << 30; // 2^30

/**
 * Default upper bound on per-particle velocity magnitude (m/s) used to derive
 * the velocity accumulator's fixed-point scale. 50 m/s is well above any
 * MVP-scale particle velocity (gravity over a few seconds reaches at most
 * ~30 m/s without driving forces; a single substep's Δv cannot come close).
 */
export const DEFAULT_MAX_VELOCITY = 50;

/**
 * Derive the fixed-point scale factor (units: ticks per m/s) from the
 * expected maximum per-particle velocity magnitude. One i32 tick then
 * represents `1 / scale` m/s of physical velocity; the accumulator can hold
 * a signed sum up to `VELOCITY_ACCUMULATOR_HEADROOM / scale = maxVelocity`
 * m/s before saturating. Overflowing that bound requires a physically
 * implausible velocity which should not occur under MVP scene parameters;
 * if it does, the apply-kernel's overflow flag surfaces the condition.
 *
 * Precision at typical maxVelocity = 50 m/s:
 *   `scale ≈ 2.15e7 / (m/s)` → 1 tick ≈ 4.7e-8 m/s
 * More than two orders of magnitude below the f32 ULP envelope of the
 * velocity-friction reductions, so quantization is not a physical-accuracy
 * concern.
 */
export function deriveVelocityAccumulatorScale(maxVelocity: number): number {
  if (!Number.isFinite(maxVelocity) || maxVelocity <= 0) {
    throw new Error(
      `deriveVelocityAccumulatorScale: maxVelocity must be positive, got ${maxVelocity}`,
    );
  }
  return Math.floor(VELOCITY_ACCUMULATOR_HEADROOM / maxVelocity);
}

/**
 * Per-particle fixed-point Δv accumulator for the Macklin 2020 §3.6
 * velocity-level dynamic friction pass.
 *
 * Same shape as {@link import('./accumulator.js').ContactAccumulator} but
 * scaled for velocity units instead of position units, and applied to
 * {@link ParticleSystem.velocities} instead of predictedPositions. The
 * overflow flag is independent from the position accumulator's flag so
 * callers can distinguish which buffer saturated (if either).
 *
 * Determinism: i32 atomicAdd is associative, commutative, and exact, so the
 * per-particle Δv sum is independent of thread execution order. The velocity
 * friction pass is therefore G4 tier-1 bit-exact, matching the position
 * scatter contact solve.
 */
export class VelocityAccumulator {
  readonly particles: ParticleSystem;
  readonly capacity: number;
  readonly scaleTicksPerMetersPerSecond: number;
  readonly scale: UniformNode<'float', number>;
  readonly invScale: UniformNode<'float', number>;

  /** i32[3·capacity], atomic. */
  readonly delta: StorageBufferNode<'int'>;
  /** u32[1], atomic; set to 1 if any axis exceeded VELOCITY_ACCUMULATOR_HEADROOM. */
  readonly overflowFlag: StorageBufferNode<'uint'>;

  constructor(particles: ParticleSystem, scaleTicksPerMetersPerSecond: number) {
    if (!Number.isFinite(scaleTicksPerMetersPerSecond) || scaleTicksPerMetersPerSecond <= 0) {
      throw new Error(
        `VelocityAccumulator: scale must be positive, got ${scaleTicksPerMetersPerSecond}`,
      );
    }
    this.particles = particles;
    this.capacity = particles.capacity;
    this.scaleTicksPerMetersPerSecond = scaleTicksPerMetersPerSecond;
    this.scale = uniform(scaleTicksPerMetersPerSecond, 'float');
    this.invScale = uniform(1 / scaleTicksPerMetersPerSecond, 'float');
    this.delta = instancedArray(3 * particles.capacity, 'int').toAtomic();
    this.overflowFlag = instancedArray(1, 'uint').toAtomic();
  }

  /** Read the CPU-visible overflow flag. 1 = overflow detected last frame. */
  async readbackOverflow(): Promise<number> {
    const buf = await this.particles.renderer.getArrayBufferAsync(this.overflowFlag.value);
    return new Uint32Array(buf)[0]!;
  }
}

/**
 * Reset the velocity accumulator slots to zero. One thread per axis slot,
 * dispatched over 3·capacity. Called once per substep after the position
 * solve and before the velocity-friction scatter kernel.
 */
export function buildResetVelocityAccumulatorKernel(accumulator: VelocityAccumulator): ComputeNode {
  return Fn(() => {
    const i: Any = instanceIndex;
    atomicStore(accumulator.delta.element(i), int(0));
  })().compute(3 * accumulator.capacity);
}

/**
 * Zero the velocity accumulator's overflow flag. Single-thread dispatch,
 * called alongside the accumulator reset.
 */
export function buildResetVelocityOverflowFlagKernel(
  accumulator: VelocityAccumulator,
): ComputeNode {
  return Fn(() => {
    atomicStore(accumulator.overflowFlag.element(uint(0)), uint(0));
  })().compute(1);
}

/**
 * Apply the accumulated Δv to {@link ParticleSystem.velocities} and zero the
 * accumulator slots for the next substep. One thread per particle. Paper
 * reference: Macklin 2020 §3.6 eq. 33 (velocity update, particle-simplified
 * without the rotational terms).
 */
export function buildApplyVelocityAccumulatorKernel(accumulator: VelocityAccumulator): ComputeNode {
  const velocities = accumulator.particles.velocities;
  return Fn(() => {
    const p: Any = instanceIndex;
    const base: Any = p.mul(uint(3));
    const dxTicksLoaded: Any = atomicLoad(accumulator.delta.element(base));
    const dyTicksLoaded: Any = atomicLoad(accumulator.delta.element(base.add(uint(1))));
    const dzTicksLoaded: Any = atomicLoad(accumulator.delta.element(base.add(uint(2))));
    const dxTicks: Any = dxTicksLoaded.toVar();
    const dyTicks: Any = dyTicksLoaded.toVar();
    const dzTicks: Any = dzTicksLoaded.toVar();
    emitVelocityOverflowCheck(accumulator, dxTicks, dyTicks, dzTicks);
    const inv: Any = accumulator.invScale;
    const dvx: Any = dxTicks.toFloat().mul(inv);
    const dvy: Any = dyTicks.toFloat().mul(inv);
    const dvz: Any = dzTicks.toFloat().mul(inv);
    const v: Any = velocities.element(p).xyz.toVar();
    const newV: Any = v.add(vec4(dvx, dvy, dvz, float(0.0)).xyz);
    velocities.element(p).assign(vec4(newV, float(0.0)));
    atomicStore(accumulator.delta.element(base), int(0));
    atomicStore(accumulator.delta.element(base.add(uint(1))), int(0));
    atomicStore(accumulator.delta.element(base.add(uint(2))), int(0));
  })().compute(accumulator.capacity);
}

/**
 * TSL helper: scatter a 3-vector Δv (m/s) into the velocity accumulator for
 * particle p via three atomicAdd calls on i32 slots. Overflow check is
 * deferred to the apply kernel for the same reason documented in the
 * position accumulator (see Phase 5 Finding 3: single-cache-line atomicMax
 * contention dominates scatter cost at high contact counts).
 *
 * Must be called inside a TSL Fn body with deltaXyz a vec3 node.
 */
export function emitAccumulateVelocityDelta(
  accumulator: VelocityAccumulator,
  pNode: Any,
  deltaXyz: Any,
): void {
  const base: Any = pNode.mul(uint(3));
  const scale: Any = accumulator.scale;
  const dxTicks: Any = deltaXyz.x.mul(scale).toInt();
  const dyTicks: Any = deltaXyz.y.mul(scale).toInt();
  const dzTicks: Any = deltaXyz.z.mul(scale).toInt();
  atomicAdd(accumulator.delta.element(base), dxTicks);
  atomicAdd(accumulator.delta.element(base.add(uint(1))), dyTicks);
  atomicAdd(accumulator.delta.element(base.add(uint(2))), dzTicks);
}

/**
 * TSL helper: check the magnitude of the current accumulator slots for
 * particle p against VELOCITY_ACCUMULATOR_HEADROOM; set the overflow flag if
 * any axis exceeded. Called from the apply kernel.
 *
 * Must be called inside a TSL Fn body.
 */
function emitVelocityOverflowCheck(
  accumulator: VelocityAccumulator,
  dxTicks: Any,
  dyTicks: Any,
  dzTicks: Any,
): void {
  const headroom: Any = int(VELOCITY_ACCUMULATOR_HEADROOM);
  const overX: Any = dxTicks.abs().greaterThanEqual(headroom);
  const overY: Any = dyTicks.abs().greaterThanEqual(headroom);
  const overZ: Any = dzTicks.abs().greaterThanEqual(headroom);
  const any: Any = overX.or(overY).or(overZ);
  const flagVal: Any = any.select(uint(1), uint(0));
  atomicMax(accumulator.overflowFlag.element(uint(0)), flagVal);
}
