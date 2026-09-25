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
   *
   *
   * For paper §9 expanded-radius queries (Macklin 2014 "catches particles
   * that move into range during the constraint solve"), the caller inflates
   * this by `(1 + ε)` at construction time and applies the actual query
   * radius as a per-pair distance filter inside the `onCandidate` callback.
   */
  readonly cellSize: number;
  /**
   * Number of buckets in the Teschner §4.1 hash table. MUST be a power of
   * two (the bitmask `& (hashTableSize − 1)` implements `mod n`). If
   * omitted, defaults to `nextPow2(2 · particles.capacity)`, matching
   * Teschner Fig. 3 / Fig. 4's "flattened" load-factor regime (`n ≳
   * 2 × occupied-cell count`). Raising `n` reduces hash collisions and
   * the narrow-phase false-positive rate at the cost of buffer memory
   * (`counts`, `cellStart`, `cellEnd`, `writeCursor` are each sized to
   * `padToScanWorkgroup(hashTableSize) · 4 bytes`).
   *
   * Capped at `MAX_CELLS_SINGLE_LEVEL_SCAN` (1,048,576). Larger values
   * require a multi-level scan (deferred — file a new UNKNOWN at the
   * requesting phase).
   */
  readonly hashTableSize?: number;
  /**
   * Optional scene-centering offset subtracted before the f32→i32 cell-
   * coordinate quantization. Preserves sub-millimetre cell-index precision
   * for scenes centered far from the world origin. Defaults to `(0, 0, 0)`
   * — the standard precision-optimal choice when particles cluster near
   * the origin. This is NOT a domain declaration: particles may sit
   * arbitrarily far from the origin without being clamped.
   */
  readonly hashOrigin?: Vector3;
}

/**
 * CPU-side snapshot returned by {@link HashGrid.readback}. Debug / test only.
 *
 * Layout:
 *   `cellIndex[p]`      — per-particle hash-bucket index, length `capacity`.
 *   `counts[c]`         — particles in bucket `c` after the histogram pass.
 *   `cellStart[c]`      — first index into `sortedIndices` belonging to
 *                         bucket `c`; = exclusive prefix of `counts`.
 *   `cellEnd[c]`        — one past the last index for bucket `c`;
 *                         = `cellStart[c] + counts[c]`.
 *   `sortedIndices[k]`  — particle indices sorted by bucket. Length
 *                         `capacity`. The **sequence within a bucket** is
 *                         not guaranteed stable across runs (see
 *                         `sort.ts` "Stability (non-)guarantee").
 *
 * `counts`, `cellStart`, `cellEnd` are each `hashTableSizePadded`-sized;
 * callers that iterate up to `hashTableSize` (the semantically-valid
 * prefix) must bring their own bound — padded tail entries are
 * zero-filled and benign.
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
 * Unbounded spatial-hash neighbor search (Teschner et al. 2003) for
 * Three.js Particle Fluids core.
 *
 *
 * Pipeline per `rebuild()`:
 *   1. resetCounts              — `counts[c] = 0`
 *   2. resetOverflowFlag        — `overflowFlag = 0`
 *   3. cellIndexAndHistogram    — `cellIndex[p]` + `atomicAdd(counts, 1)`
 *   4. blockScan                — Blelloch per-block exclusive scan
 *   5. blockSumScan             — scan of block totals (single workgroup)
 *   6. finalizeCellRanges       — fused: addPrefix + resetWriteCursor +
 *                                 cellBounds. After this, `cellStart[c]`
 *                                 is the global exclusive prefix of
 *                                 `counts`, `writeCursor[c] = cellStart[c]`,
 *                                 and `cellEnd[c] = cellStart[c] + counts[c]`.
 *                                 (Phase Perf-17.)
 *   7. scatter                  — `sortedIndices[atomicAdd(cursor)] = p`
 *   8. sortedPositions          — `sortedPredictedPositions[k] =
 *                                  predictedPositions[sortedIndices[k]]`
 *                                  (Phase Perf-09; consumed by pair-list build).
 *
 * Steps 4–6 are the standard three-pass extension of the Blelloch scan
 * (Probe 2, `tests/_helpers/probes/scan.ts`) with the post-scan finalize pass fused
 * (Phase Perf-17). Steps 1 + 2 + 3 + 7 are straight bookkeeping. Step 8
 * materializes a Morton-permuted shadow of `predictedPositions` for
 * coalesced reads in the pair-list build (see `sortedPositions.ts`). All
 * eight run in a single `renderer.computeAsync([...])` dispatch so
 * command-encoder ordering is sufficient — same pattern validated by
 * Phase 01 Probe 5 (`_probe/kernelChain.ts`).
 */
export class HashGrid {
  readonly renderer: WebGPURenderer;
  readonly particles: ParticleSystem;
  readonly cellSize: number;
  readonly hashOrigin: Vector3;
  /** Hash-table size (Teschner §4.1 `n`). Power of two. */
  readonly hashTableSize: number;
  /** `hashTableSize` rounded up to a multiple of `SCAN_WORKGROUP_SIZE`. */
  readonly hashTableSizePadded: number;

  readonly cellIndex: StorageBufferNode<'uint'>;
  readonly counts: StorageBufferNode<'uint'>;
  readonly cellStart: StorageBufferNode<'uint'>;
  readonly cellEnd: StorageBufferNode<'uint'>;
  readonly sortedIndices: StorageBufferNode<'uint'>;
  /**
   * Morton-permuted shadow copy of `particles.predictedPositions`.
   * Rebuilt once per substep as the last stage of {@link rebuildPipeline}.
   * `sortedPredictedPositions[k].xyz =
   *  predictedPositions[sortedIndices[k]].xyz` for `k ∈ [0, capacity)`.
   * Read by {@link buildPairListKernel} so the inner cell-bucket loop's
   * candidate-position reads coalesce across adjacent `k` values
   * instead of scattering through the original layout. See
   * `sortedPositions.ts` for the design rationale.
   */
  readonly sortedPredictedPositions: StorageBufferNode<'vec4'>;
  /** `u32[1]`, atomic; set to 1 if any particle's cell coord saturated. */
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
          `SCAN_WORKGROUP_SIZE²). Reduce hashTableSize (at the cost of higher hash- ` +
          `collision rate) or file an UNKNOWN for recursive multi-level scan.`,
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

    const cellIndexKernels = buildCellIndexKernels(
      particles.positions,
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

  /**
   * Dispatch the full eight-stage rebuild pipeline for the current
   * `particles.positions`. Awaitable so callers can chain `predict →
   * rebuild → contacts → solve → advect` at Phase 04+.
   */
  async rebuild(): Promise<void> {
    this.assertAlive();
    await this.renderer.computeAsync(this.pipeline);
  }

  /**
   * Read-only access to the ordered kernel chain that {@link rebuild}
   * dispatches. Exposed so {@link SimLoop} can splice the grid rebuild into
   * its larger per-substep `computeAsync` call instead of doing one
   * `computeAsync` per stage. The array is the same one the instance holds
   * internally — callers MUST NOT mutate it.
   */
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

  /** Read the CPU-visible overflow flag. 1 = any particle's cell coord was clamped last rebuild. */
  async readbackOverflow(): Promise<number> {
    this.assertAlive();
    const buf = await this.renderer.getArrayBufferAsync(this.overflowFlag.value);
    return new Uint32Array(buf)[0]!;
  }

  destroy(): void {
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('HashGrid has been destroyed');
  }
}
