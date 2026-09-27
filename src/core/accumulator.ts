import {
  Fn,
  If,
  atomicAdd,
  atomicLoad,
  atomicMax,
  atomicStore,
  instanceIndex,
  instancedArray,
  int,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import type { ParticleRange, ParticleSystem } from './particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** A buffer to apply an accumulated sum to, with an optional scale. */
export interface ApplyTarget {
  readonly buffer: StorageBufferNode<'vec4'>;
  readonly scale?: Any;
}

/** Largest per-axis tick count before a sum is flagged as saturated (2^30). */
const HEADROOM = 1 << 30;

/**
 * Per-particle vec3 sums built from integer atomics.
 *
 * Scatter kernels (one thread per contact or neighbor pair) cannot add floats
 * atomically, so each contribution is converted to fixed point and added with
 * `atomicAdd`. Integer addition is exact and order-independent, which keeps
 * the result identical from run to run. An apply kernel then converts the sum
 * back to floats, adds it to its target buffers, and clears it.
 */
export class Accumulator {
  readonly particles: ParticleSystem;
  /** Fixed-point ticks per unit (metres for positions, m/s for velocities). */
  readonly scale: number;
  /** `3 · capacity` atomic i32 values, xyz per particle. */
  readonly delta: StorageBufferNode<'int'>;
  /** Atomic flag set to 1 when any sum reached the headroom limit. */
  readonly overflowFlag: StorageBufferNode<'uint'>;

  /** @param maxMagnitude Largest per-axis sum expected per apply; sets the fixed-point scale. */
  constructor(particles: ParticleSystem, maxMagnitude: number) {
    if (!Number.isFinite(maxMagnitude) || maxMagnitude <= 0) {
      throw new Error(`Accumulator: maxMagnitude must be positive, got ${maxMagnitude}`);
    }
    this.particles = particles;
    this.scale = Math.floor(HEADROOM / maxMagnitude);
    this.delta = instancedArray(3 * particles.capacity, 'int').toAtomic();
    this.overflowFlag = instancedArray(1, 'uint').toAtomic();
  }

  /** Emit TSL that adds `value` (vec3) to particle `index`'s sum. */
  add(index: Any, value: Any): void {
    const base: Any = index.mul(uint(3));
    // Round rather than truncate, so small contributions aren't biased toward zero.
    atomicAdd(this.delta.element(base), value.x.mul(this.scale).round().toInt());
    atomicAdd(this.delta.element(base.add(uint(1))), value.y.mul(this.scale).round().toInt());
    atomicAdd(this.delta.element(base.add(uint(2))), value.z.mul(this.scale).round().toInt());
  }

  /** Zero every sum and the overflow flag. */
  buildResetKernel(): ComputeNode {
    return Fn(() => {
      const i: Any = instanceIndex;
      atomicStore(this.delta.element(i), int(0));
      If(i.equal(uint(0)), () => {
        atomicStore(this.overflowFlag.element(uint(0)), uint(0));
      });
    })().compute(3 * this.particles.capacity);
  }

  /**
   * Add each particle's sum to the xyz of every target buffer, then clear the
   * sum. A target can scale the sum first, e.g. `{ buffer, scale: dt }` to
   * turn a velocity change into a position change.
   *
   * @param range Particles to apply; defaults to all of them.
   */
  buildApplyKernel(
    targets: readonly (StorageBufferNode<'vec4'> | ApplyTarget)[],
    range: ParticleRange = { start: 0, count: this.particles.capacity },
  ): ComputeNode {
    const inv = 1 / this.scale;
    return Fn(() => {
      const p: Any = instanceIndex.add(uint(range.start));
      const base: Any = p.mul(uint(3));
      const ticks: Any = vec3(
        (atomicLoad(this.delta.element(base)) as Any).toFloat(),
        (atomicLoad(this.delta.element(base.add(uint(1)))) as Any).toFloat(),
        (atomicLoad(this.delta.element(base.add(uint(2)))) as Any).toFloat(),
      ).toVar();
      const magnitude: Any = ticks.abs();
      const saturated: Any = magnitude.x
        .max(magnitude.y)
        .max(magnitude.z)
        .greaterThanEqual(HEADROOM);
      atomicMax(this.overflowFlag.element(uint(0)), saturated.select(uint(1), uint(0)));
      const change: Any = ticks.mul(inv).toVar();
      for (const target of targets) {
        const { buffer, scale } =
          'buffer' in target ? target : { buffer: target, scale: undefined };
        const value: Any = buffer.element(p);
        value.assign(vec4(value.xyz.add(scale ? change.mul(scale) : change), value.w));
      }
      atomicStore(this.delta.element(base), int(0));
      atomicStore(this.delta.element(base.add(uint(1))), int(0));
      atomicStore(this.delta.element(base.add(uint(2))), int(0));
    })().compute(range.count);
  }

  /** 1 if a sum saturated since the last reset, else 0. Stalls on the GPU. */
  async readbackOverflow(): Promise<number> {
    const buffer = await this.particles.renderer.getArrayBufferAsync(this.overflowFlag.value);
    return new Uint32Array(buffer)[0]!;
  }
}
