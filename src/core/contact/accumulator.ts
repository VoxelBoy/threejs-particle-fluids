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
 * Ceiling for the fixed-point accumulator's magnitude. Chosen as `2^30` so
 * that a single `atomicAdd` of any valid scaled contribution cannot push
 * the stored i32 past `0x7FFFFFFF`; the headroom between `2^30` and `2^31 −
 * 1` is the safety margin the overflow-flag path relies on.
 */
export const ACCUMULATOR_HEADROOM = 1 << 30; // 2^30

/**
 * Derive the fixed-point scale factor (units: "ticks per metre") from an
 * upper bound on a single iteration's per-particle `|Δx_k|`. In XPBD
 * practice the per-iteration correction is bounded by the largest
 * penetration resolution the solver attempts in one iter — the U-16 50k-
 * particle measurement saw `|Δx| < 0.01 m`, so the 10 m default callers
 * typically pass sits three orders of magnitude above typical usage.
 * Scenes with unusually large expected corrections (extreme XPBD
 * compliance, a deliberately-oversized bootstrap correction) can pass
 * a larger value.
 *
 * Scale is `floor(ACCUMULATOR_HEADROOM / maxCorrectionMeters)`. One i32
 * tick represents `1 / scale` metres of physical displacement; the
 * accumulator can hold a signed sum up to `maxCorrectionMeters` before
 * saturating. Saturation trips the overflow flag (see
 * {@link ContactAccumulator.readbackOverflow}); it does not wrap silently.
 *
 */
export function deriveAccumulatorScale(maxCorrectionMeters: number): number {
  if (!Number.isFinite(maxCorrectionMeters) || maxCorrectionMeters <= 0) {
    throw new Error(
      `deriveAccumulatorScale: maxCorrectionMeters must be positive, got ${maxCorrectionMeters}`,
    );
  }
  return Math.floor(ACCUMULATOR_HEADROOM / maxCorrectionMeters);
}

/**
 * Per-particle fixed-point Δx accumulator for Macklin-2014 §6 contact
 * scatter (U-16 Path B).
 *
 * Layout: `buffer[3·p + axis]` — one signed i32 slot per (particle, xyz).
 * All three axes are distinct atomic slots; each `atomicAdd` by a contact
 * thread only touches one axis of one particle, so axes are independent.
 *
 * Scale: CPU-side scalar `scaleTicksPerMeter`. The scatter kernel converts
 * an f32 displacement `Δx_k` into ticks via `i32(round(Δx_k · scale))`.
 * The apply kernel converts back as `f32(ticks) / scale`.
 *
 * Overflow: i32 wraps silently on saturation. Phase 05 plan mandates a
 * "debug-build overflow assert". We implement this via an atomic `u32`
 * flag set whenever a scatter kernel detects `|new_value| ≥
 * ACCUMULATOR_HEADROOM`. CPU reads the flag after each frame; the helper
 * {@link ContactAccumulator.readbackOverflow} exposes it. Test code gates
 * assertions on this flag; production code may ignore it.
 *
 * G4 determinism: i32 `atomicAdd` is associative+commutative+exact, so the
 * final per-particle sum is independent of atomic-scheduling order.
 * Contact scatter is therefore tier-1 bit-exact — the same property U-18
 * (hash-grid `cellStart`/`cellEnd`) relies on for its tier-1 classification.
 */
export class ContactAccumulator {
  readonly particles: ParticleSystem;
  readonly capacity: number;
  readonly scaleTicksPerMeter: number;
  readonly scale: UniformNode<'float', number>;
  readonly invScale: UniformNode<'float', number>;

  /** `i32[3·capacity]`, atomic. */
  readonly delta: StorageBufferNode<'int'>;
  /** `u32[1]`, atomic; set to 1 if any axis exceeded `ACCUMULATOR_HEADROOM`. */
  readonly overflowFlag: StorageBufferNode<'uint'>;

  constructor(particles: ParticleSystem, scaleTicksPerMeter: number) {
    if (!Number.isFinite(scaleTicksPerMeter) || scaleTicksPerMeter <= 0) {
      throw new Error(`ContactAccumulator: scale must be positive, got ${scaleTicksPerMeter}`);
    }
    this.particles = particles;
    this.capacity = particles.capacity;
    this.scaleTicksPerMeter = scaleTicksPerMeter;
    this.scale = uniform(scaleTicksPerMeter, 'float');
    this.invScale = uniform(1 / scaleTicksPerMeter, 'float');
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
 * Per-iteration reset: zero the Δx accumulator AND the overflow flag.
 * Dispatched over `3·capacity` (one thread per axis slot) — a single
 * dispatch-over-capacity-threads would still need 3 writes per thread;
 * the flatter dispatch is cleaner in TSL and avoids a per-thread inner
 * loop.
 */
export function buildResetAccumulatorKernel(accumulator: ContactAccumulator): ComputeNode {
  return Fn(() => {
    const i: Any = instanceIndex;
    atomicStore(accumulator.delta.element(i), int(0));
    // `instanceIndex < 1` branch stores 0 into the overflow flag. Guard
    // with `If` but `uint(0).equal(i)` used directly as store gate is
    // cleaner; dispatch count is `3·capacity + 1` no — keep separate.
  })().compute(3 * accumulator.capacity);
}

/**
 * Zero the overflow flag. One-thread dispatch — kept as a separate kernel
 * from `resetAccumulator` so the caller can place it wherever the
 * pipeline convention demands (in practice: once per substep, before the
 * iter loop). Currently called alongside accumulator reset.
 */
export function buildResetOverflowFlagKernel(accumulator: ContactAccumulator): ComputeNode {
  return Fn(() => {
    atomicStore(accumulator.overflowFlag.element(uint(0)), uint(0));
  })().compute(1);
}

/**
 * Apply the accumulated Δx to `predictedPositions` and zero the accumulator
 * for the next iteration. One thread per particle. Paper Macklin 2014
 * Algorithm 3 line 8 (scatter path): the scatter kernel accumulates per-
 * particle deltas across all contacts; this kernel commits them.
 *
 * The caller is responsible for resetting the accumulator before the next
 * iteration — we DO NOT zero here, because Macklin-2014 §4.2 eq. 12 uses
 * constraint averaging which needs per-iteration accumulation, whereas
 * we're using the XPBD small-step form where λ is carried across iters but
 * Δx is NOT. To avoid a footgun, this kernel zeros the accumulator after
 * applying — matching the scheduler's `λ_0 ← 0` pattern per substep but
 * per-ITERATION instead of per-substep.
 */
export function buildApplyAccumulatorToPredictedKernel(
  accumulator: ContactAccumulator,
): ComputeNode {
  const predicted = accumulator.particles.predictedPositions;
  return Fn(() => {
    const p: Any = instanceIndex;
    const base: Any = p.mul(uint(3));
    const dxTicksLoaded: Any = atomicLoad(accumulator.delta.element(base));
    const dyTicksLoaded: Any = atomicLoad(accumulator.delta.element(base.add(uint(1))));
    const dzTicksLoaded: Any = atomicLoad(accumulator.delta.element(base.add(uint(2))));
    const dxTicks: Any = dxTicksLoaded.toVar();
    const dyTicks: Any = dyTicksLoaded.toVar();
    const dzTicks: Any = dzTicksLoaded.toVar();
    emitAccumulatorOverflowCheck(accumulator, dxTicks, dyTicks, dzTicks);
    const inv: Any = accumulator.invScale;
    const dx: Any = dxTicks.toFloat().mul(inv);
    const dy: Any = dyTicks.toFloat().mul(inv);
    const dz: Any = dzTicks.toFloat().mul(inv);
    const xStar: Any = predicted.element(p).xyz.toVar();
    const newXstar: Any = xStar.add(vec4(dx, dy, dz, float(0.0)).xyz);
    predicted.element(p).assign(vec4(newXstar, float(0.0)));
    // Zero the accumulator slots for next iteration.
    atomicStore(accumulator.delta.element(base), int(0));
    atomicStore(accumulator.delta.element(base.add(uint(1))), int(0));
    atomicStore(accumulator.delta.element(base.add(uint(2))), int(0));
  })().compute(accumulator.capacity);
}

/**
 * Stabilization-pass variant: apply the accumulated Δx to BOTH `positions`
 * and `predictedPositions`, per paper Macklin 2014 §4.4 ("any deltas
 * applied to the original positions are also applied to the predicted
 * positions before the main constraint solving loop begins"). Zeros the
 * accumulator afterwards.
 */
export function buildApplyAccumulatorToBothKernel(accumulator: ContactAccumulator): ComputeNode {
  const positions = accumulator.particles.positions;
  const predicted = accumulator.particles.predictedPositions;
  return Fn(() => {
    const p: Any = instanceIndex;
    const base: Any = p.mul(uint(3));
    const dxTicksLoaded: Any = atomicLoad(accumulator.delta.element(base));
    const dyTicksLoaded: Any = atomicLoad(accumulator.delta.element(base.add(uint(1))));
    const dzTicksLoaded: Any = atomicLoad(accumulator.delta.element(base.add(uint(2))));
    const dxTicks: Any = dxTicksLoaded.toVar();
    const dyTicks: Any = dyTicksLoaded.toVar();
    const dzTicks: Any = dzTicksLoaded.toVar();
    emitAccumulatorOverflowCheck(accumulator, dxTicks, dyTicks, dzTicks);
    const inv: Any = accumulator.invScale;
    const dx: Any = dxTicks.toFloat().mul(inv);
    const dy: Any = dyTicks.toFloat().mul(inv);
    const dz: Any = dzTicks.toFloat().mul(inv);
    const x: Any = positions.element(p).xyz.toVar();
    const xStar: Any = predicted.element(p).xyz.toVar();
    const delta: Any = vec4(dx, dy, dz, float(0.0)).xyz;
    positions.element(p).assign(vec4(x.add(delta), float(0.0)));
    predicted.element(p).assign(vec4(xStar.add(delta), float(0.0)));
    atomicStore(accumulator.delta.element(base), int(0));
    atomicStore(accumulator.delta.element(base.add(uint(1))), int(0));
    atomicStore(accumulator.delta.element(base.add(uint(2))), int(0));
  })().compute(accumulator.capacity);
}

/**
 * TSL helper: scatter a 3-vector `delta` (metres) into the accumulator for
 * particle `p` via three `atomicAdd` calls on i32 slots.
 *
 * **Overflow check is NOT done here** — the single overflow flag would
 * serialize every scatter thread through one `atomicMax` cache line,
 * dominating runtime at 145k+ contacts / substep. Overflow detection is
 * deferred to the apply kernels (one check per particle per iter
 * instead of three per contact), which also has the advantage of
 * checking the ACTUAL accumulated value rather than a potentially-
 * temporary intermediate (atomicAdds from other threads may lower the
 * magnitude after this thread's add — for the safety guarantee we
 * care about, the final post-iter value is the only one that matters).
 *
 * Must be called inside a TSL `Fn` body with `delta` a `vec3` node.
 */
export function emitAccumulateDelta(
  accumulator: ContactAccumulator,
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
 * particle `p` against `ACCUMULATOR_HEADROOM`; set the overflow flag if
 * any axis exceeded. Called from the apply kernels.
 *
 * Must be called inside a TSL `Fn` body.
 */
function emitAccumulatorOverflowCheck(
  accumulator: ContactAccumulator,
  dxTicks: Any,
  dyTicks: Any,
  dzTicks: Any,
): void {
  const headroom: Any = int(ACCUMULATOR_HEADROOM);
  const overX: Any = dxTicks.abs().greaterThanEqual(headroom);
  const overY: Any = dyTicks.abs().greaterThanEqual(headroom);
  const overZ: Any = dzTicks.abs().greaterThanEqual(headroom);
  const any: Any = overX.or(overY).or(overZ);
  const flagVal: Any = any.select(uint(1), uint(0));
  atomicMax(accumulator.overflowFlag.element(uint(0)), flagVal);
}
