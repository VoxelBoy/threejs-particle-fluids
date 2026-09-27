import { Vector3 } from 'three';
import { instancedArray } from 'three/tsl';
import type { WebGPURenderer } from 'three/webgpu';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import type { ParticleSystem } from '../particles.js';
import { buildCellIndexKernels } from './cellIndex.js';
import { allocateSortedPositionsBuffer, buildSortedPositionsKernel } from './sortedPositions.js';
import {
  MAX_CELLS_SINGLE_LEVEL_SCAN,
  SCAN_WORKGROUP_SIZE,
  buildCountSortKernels,
  padToScanWorkgroup,
} from './sort.js';

/**
 * Round `n` up to the next power of two. Used to pick the default
 * `hashTableSize` from `2 · capacity`.
 */
function nextPow2(n: number): number {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export interface HashGridOptions {
  /**
   * Cell edge length. A neighbor query visits the 27 cells around a point,
   * so this must be at least the largest query radius.
   */
  readonly cellSize: number;
  /**
   * Hash buckets; a power of two. Default: the next power of two above
   * `2 × capacity`, which keeps collisions between distant cells rare.
   * At most 1,048,576.
   */
  readonly hashTableSize?: number;
  /**
   * Offset subtracted from positions before they are quantized to cells.
   * Keeps cell indices precise for scenes far from the origin. Default
   * `(0, 0, 0)`.
   */
  readonly hashOrigin?: Vector3;
}

/**
 * The grid's buffers read back to the CPU, for tests and debugging. `counts`,
 * `cellStart`, and `cellEnd` are padded past `hashTableSize` with zeros.
 * Particle order within a bucket may differ from run to run.
 */
export interface HashGridSnapshot {
  readonly capacity: number;
  readonly hashTableSize: number;
  readonly hashTableSizePadded: number;
  readonly cellIndex: Uint32Array;
  readonly counts: Uint32Array;
  readonly cellStart: Uint32Array;
  readonly cellEnd: Uint32Array;
  readonly sortedIndices: Uint32Array;
}

/**
 * Spatial hash for neighbor search (Teschner et al. 2003), rebuilt on the GPU
 * with a counting sort:
 *
 * 1. hash each particle's cell (Morton order) and count particles per bucket;
 * 2. prefix-sum the counts (Blelloch scan) into each bucket's start and end;
 * 3. scatter particle indices into bucket order;
 * 4. copy predicted positions into that order too, so neighbor walks read
 *    memory contiguously.
 *
 * The space is unbounded: distant cells share buckets, and queries filter
 * the extra candidates by distance. {@link SimLoop} owns and rebuilds a grid
 * for you; build one directly only for custom tools.
 */
export class HashGrid {
  readonly renderer: WebGPURenderer;
  readonly particles: ParticleSystem;
  readonly cellSize: number;
  readonly hashOrigin: Vector3;
  /** Number of hash buckets, a power of two. */
  readonly hashTableSize: number;
  /** `hashTableSize` rounded up to a multiple of `SCAN_WORKGROUP_SIZE`. */
  readonly hashTableSizePadded: number;

  readonly cellIndex: StorageBufferNode<'uint'>;
  readonly counts: StorageBufferNode<'uint'>;
  readonly cellStart: StorageBufferNode<'uint'>;
  readonly cellEnd: StorageBufferNode<'uint'>;
  readonly sortedIndices: StorageBufferNode<'uint'>;
  /** `predictedPositions` in bucket order, refreshed by every rebuild. */
  readonly sortedPredictedPositions: StorageBufferNode<'vec4'>;
  /** Atomic flag set to 1 when a particle's cell coordinate saturated during a rebuild. */
  readonly overflowFlag: StorageBufferNode<'uint'>;

  readonly hashOriginUniform: UniformNode<'vec3', Vector3>;
  readonly cellSizeUniform: UniformNode<'float', number>;

  private readonly blockSums: StorageBufferNode<'uint'>;
  private readonly writeCursor: StorageBufferNode<'uint'>;
  private readonly pipeline: ComputeNode[];

  private disposed = false;

  constructor(particles: ParticleSystem, options: HashGridOptions) {
    const { cellSize } = options;

    if (!Number.isFinite(cellSize) || cellSize <= 0) {
      throw new Error(`HashGrid: cellSize must be a positive finite number, got ${cellSize}`);
    }

    const hashTableSize = options.hashTableSize ?? nextPow2(2 * particles.capacity);
    if ((hashTableSize & (hashTableSize - 1)) !== 0 || hashTableSize < 1) {
      throw new Error(
        `HashGrid: hashTableSize must be a positive power of two, got ${hashTableSize}`,
      );
    }
    const hashTableSizePadded = padToScanWorkgroup(hashTableSize);

    if (hashTableSizePadded > MAX_CELLS_SINGLE_LEVEL_SCAN) {
      throw new Error(
        `HashGrid: hashTableSize=${hashTableSize} padded to ${hashTableSizePadded} ` +
          `exceeds the single-level Blelloch scan cap (${MAX_CELLS_SINGLE_LEVEL_SCAN} = ` +
          `SCAN_WORKGROUP_SIZE²). Use a smaller hashTableSize.`,
      );
    }

    const hashOrigin = (options.hashOrigin ?? new Vector3(0, 0, 0)).clone();

    this.renderer = particles.renderer;
    this.particles = particles;
    this.cellSize = cellSize;
    this.hashOrigin = hashOrigin;
    this.hashTableSize = hashTableSize;
    this.hashTableSizePadded = hashTableSizePadded;

    this.cellIndex = instancedArray(particles.capacity, 'uint');
    this.counts = instancedArray(hashTableSizePadded, 'uint').toAtomic();
    this.cellStart = instancedArray(hashTableSizePadded, 'uint');
    this.cellEnd = instancedArray(hashTableSizePadded, 'uint');
    this.sortedIndices = instancedArray(particles.capacity, 'uint');
    this.sortedPredictedPositions = allocateSortedPositionsBuffer(particles);
    this.overflowFlag = instancedArray(1, 'uint').toAtomic();
    this.blockSums = instancedArray(SCAN_WORKGROUP_SIZE, 'uint');
    this.writeCursor = instancedArray(hashTableSizePadded, 'uint').toAtomic();

    // Bin by predicted positions: every query runs after prediction, at predicted positions.
    const cellIndexKernels = buildCellIndexKernels(
      particles.predictedPositions,
      this.cellIndex,
      this.counts,
      this.overflowFlag,
      particles.capacity,
      hashTableSize,
      hashTableSizePadded,
      this.hashOrigin,
      cellSize,
    );
    this.hashOriginUniform = cellIndexKernels.hashOrigin;
    this.cellSizeUniform = cellIndexKernels.cellSize;

    const sort = buildCountSortKernels(
      this.counts,
      this.cellStart,
      this.cellEnd,
      this.blockSums,
      this.writeCursor,
      this.cellIndex,
      this.sortedIndices,
      particles.capacity,
      hashTableSizePadded,
    );

    const sortedPositions = buildSortedPositionsKernel({
      particles,
      sortedIndices: this.sortedIndices,
      sortedPredictedPositions: this.sortedPredictedPositions,
    });

    this.pipeline = [
      cellIndexKernels.resetCounts,
      cellIndexKernels.resetOverflowFlag,
      cellIndexKernels.cellIndexAndHistogram,
      sort.blockScan,
      sort.blockSumScan,
      sort.finalizeCellRanges,
      sort.scatter,
      sortedPositions,
    ];
  }

  /** Rebuild the grid from the current `particles.predictedPositions`. */
  async rebuild(): Promise<void> {
    this.assertAlive();
    await this.renderer.computeAsync(this.pipeline);
  }

  /** The rebuild's kernels, for batching into a larger dispatch. Don't modify it. */
  get rebuildPipeline(): readonly ComputeNode[] {
    return this.pipeline;
  }

  async readback(): Promise<HashGridSnapshot> {
    this.assertAlive();
    const [cellIndex, counts, cellStart, cellEnd, sortedIndices] = await Promise.all([
      this.renderer.getArrayBufferAsync(this.cellIndex.value),
      this.renderer.getArrayBufferAsync(this.counts.value),
      this.renderer.getArrayBufferAsync(this.cellStart.value),
      this.renderer.getArrayBufferAsync(this.cellEnd.value),
      this.renderer.getArrayBufferAsync(this.sortedIndices.value),
    ]);
    return {
      capacity: this.particles.capacity,
      hashTableSize: this.hashTableSize,
      hashTableSizePadded: this.hashTableSizePadded,
      cellIndex: new Uint32Array(cellIndex),
      counts: new Uint32Array(counts),
      cellStart: new Uint32Array(cellStart),
      cellEnd: new Uint32Array(cellEnd),
      sortedIndices: new Uint32Array(sortedIndices),
    };
  }

  /** 1 if a particle was too far out for its cell coordinate during the last rebuild. Stalls on the GPU. */
  async readbackOverflow(): Promise<number> {
    this.assertAlive();
    const buf = await this.renderer.getArrayBufferAsync(this.overflowFlag.value);
    return new Uint32Array(buf)[0]!;
  }

  dispose(): void {
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('HashGrid has been disposed');
  }
}
