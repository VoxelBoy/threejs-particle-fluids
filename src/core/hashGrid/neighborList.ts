import { Fn, If, Loop, atomicStore, instanceIndex, instancedArray, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { assertRange, type ParticleRange, type ParticleSystem } from '../particles.js';
import type { HashGrid } from './HashGrid.js';
import { emitForEachNeighbor } from './query.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Most neighbors stored per particle. Extra neighbors are dropped and flagged. */
export const MAX_NEIGHBORS = 64;

/**
 * Neighbors within a radius for every particle in a range, gathered once per
 * substep so kernels that visit neighbors several times (density, pressure,
 * viscosity, …) don't each walk the grid (Macklin & Müller 2013, §6).
 *
 * Stored column-major: neighbor `k` of local particle `i` lives at
 * `k · count + i`, so a warp reading its `k`th neighbors reads contiguously.
 */
export class NeighborList {
  readonly particles: ParticleSystem;
  readonly range: ParticleRange;
  readonly indices: StorageBufferNode<'uint'>;
  readonly counts: StorageBufferNode<'uint'>;
  /** Atomic flag set when some particle had more than {@link MAX_NEIGHBORS} neighbors. */
  readonly overflowFlag: StorageBufferNode<'uint'>;

  constructor(particles: ParticleSystem, range: ParticleRange) {
    assertRange(particles, range, 'NeighborList');
    this.particles = particles;
    this.range = range;
    this.indices = instancedArray(range.count * MAX_NEIGHBORS, 'uint');
    this.counts = instancedArray(range.count, 'uint');
    this.overflowFlag = instancedArray(1, 'uint').toAtomic();
  }

  /**
   * Kernels that rebuild the list from `grid`, keeping neighbors closer than
   * `sqrt(radiusSq)`: one clears the overflow flag, the next fills the list.
   */
  buildKernels(grid: HashGrid, radiusSq: UniformNode<'float', number>): ComputeNode[] {
    const { particles, range } = this;
    // A separate dispatch, so no thread can clear a flag another has already set.
    const resetOverflow = Fn(() => {
      atomicStore(this.overflowFlag.element(uint(0)), uint(0));
    })().compute(1);
    const build = Fn(() => {
      const local: Any = instanceIndex.toVar();
      const xi: Any = particles.predictedPositions
        .element(local.add(uint(range.start)))
        .xyz.toVar();
      const count: Any = uint(0).toVar();
      emitForEachNeighbor(grid, xi, (j: Any, slot: Any) => {
        // The grid's sorted copy of the positions keeps these reads coalesced.
        const offset: Any = xi.sub(grid.sortedPredictedPositions.element(slot).xyz);
        If(offset.dot(offset).lessThan(radiusSq), () => {
          If(count.lessThan(uint(MAX_NEIGHBORS)), () => {
            this.indices.element(count.mul(uint(range.count)).add(local)).assign(j);
            count.addAssign(uint(1));
          }).Else(() => {
            atomicStore(this.overflowFlag.element(uint(0)), uint(1));
          });
        });
      });
      this.counts.element(local).assign(count);
    })().compute(range.count);
    return [resetOverflow, build];
  }

  /** Emit TSL that calls `onNeighbor(j)` for every stored neighbor of particle `i`. */
  forEach(i: Any, onNeighbor: (j: Any) => void): void {
    const local: Any = i.sub(uint(this.range.start)).toVar();
    Loop(
      { start: uint(0), end: this.counts.element(local), type: 'uint', condition: '<' },
      ({ i: k }: { i: Any }) => {
        onNeighbor(this.indices.element(k.mul(uint(this.range.count)).add(local)));
      },
    );
  }

  /** True if some particle's list was truncated during the last rebuild. Stalls on the GPU. */
  async readbackOverflow(): Promise<boolean> {
    const buffer = await this.particles.renderer.getArrayBufferAsync(this.overflowFlag.value);
    return new Uint32Array(buffer)[0] !== 0;
  }
}
