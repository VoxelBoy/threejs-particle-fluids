import {
  Fn,
  If,
  atomicAdd,
  atomicLoad,
  atomicStore,
  globalId,
  instanceIndex,
  localId,
  uint,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

// TSL's @types surface many GPGPU nodes as bare `Node`. See Phase 02
// `integrate.ts` and Phase 01 `tests/_helpers/probes/scan.ts` for the same escape hatch.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Workgroup size for the prefix-scan. Matches the Phase 01 Blelloch probe at
 * `tests/_helpers/probes/scan.ts` — the `createParticleRenderer` helper
 * requests `maxComputeInvocationsPerWorkgroup: 1024` so this size is always
 * available before dispatch.
 */
export const SCAN_WORKGROUP_SIZE = 1024;

/**
 * Hard cap on the number of grid cells the count-sort pipeline can scan in a
 * single rebuild. Implied by `SCAN_WORKGROUP_SIZE² = 1,048,576`: the block-sum
 * scan runs in a single workgroup of `W` threads over up to `W` block sums, so
 * `paddedN = numBlocks · W ≤ W · W`. Raising this ceiling requires a recursive
 * (multi-level) scan; file a fresh UNKNOWN with measurements if a real scene
 * needs it.
 */
export const MAX_CELLS_SINGLE_LEVEL_SCAN = SCAN_WORKGROUP_SIZE * SCAN_WORKGROUP_SIZE;

/**
 * Round `n` up to the next multiple of `SCAN_WORKGROUP_SIZE`. The Blelloch
 * scan operates on a W-aligned buffer; trailing padding entries carry the
 * histogram's zero initialization through unchanged.
 */
export function padToScanWorkgroup(n: number): number {
  const W = SCAN_WORKGROUP_SIZE;
  return Math.ceil(n / W) * W;
}

/**
 * The count-sort kernel chain.
 *
 *

 *
 */
export interface CountSortKernels {
  /** Dispatched over `nCellsPadded` after the histogram kernel. */
  readonly blockScan: ComputeNode;
  /** Dispatched as a single W-thread workgroup. Reads/writes `blockSums`. */
  readonly blockSumScan: ComputeNode;
  /**
   * Dispatched over `nCellsPadded`. Fuses three same-shape passes that
   * each finalize per-cell range bookkeeping after the block scans:
   *   (1) addPrefix       : fold each block's scanned prefix back onto
   *                         `cellStart[i]`.
   *   (2) resetWriteCursor: initialize the per-cell scatter cursor from
   *                         the finalized `cellStart[i]`.
   *   (3) cellBounds      : derive `cellEnd[i] = cellStart[i] + counts[i]`.
   *
   */
  readonly finalizeCellRanges: ComputeNode;
  /** Dispatched over `capacity`. Fills `sortedIndices`. */
  readonly scatter: ComputeNode;
}

export function buildCountSortKernels(
  counts: StorageBufferNode<'uint'>,
  cellStart: StorageBufferNode<'uint'>,
  cellEnd: StorageBufferNode<'uint'>,
  blockSums: StorageBufferNode<'uint'>,
  writeCursor: StorageBufferNode<'uint'>,
  cellIndex: StorageBufferNode<'uint'>,
  sortedIndices: StorageBufferNode<'uint'>,
  capacity: number,
  nCellsPadded: number,
): CountSortKernels {
  const W = SCAN_WORKGROUP_SIZE;

  // Pass 1 — per-block Blelloch exclusive scan of `counts` into `cellStart`;
  // writes each block's total to `blockSums` before the down-sweep clears it.
  // Algorithm mirrors _probe/scan.ts; the S-02 loop-variable-capture hazard
  // is handled the same way (snapshot `offset` into
  // `off` inside each JS loop iteration before the `If()` callback).
  const blockScan = Fn(() => {
    const tid: Any = localId.x;
    const wg: Any = workgroupId.x;
    const gi: Any = globalId.x;

    const s: Any = workgroupArray('uint', W);
    // `counts` is atomic — a plain `.element(gi)` read would generate
    // invalid WGSL in the shader backend. `atomicLoad` returns the current
    // value as a plain u32.
    s.element(tid).assign(atomicLoad(counts.element(gi)));
    workgroupBarrier();

    let offset = 1;
    for (let d = W >> 1; d > 0; d >>= 1) {
      const off = offset;
      If(tid.lessThan(uint(d)), () => {
        const ai: Any = tid.mul(uint(off * 2)).add(uint(off - 1));
        const bi: Any = ai.add(uint(off));
        s.element(bi).assign(s.element(bi).add(s.element(ai)));
      });
      workgroupBarrier();
      offset *= 2;
    }

    If(tid.equal(uint(0)), () => {
      blockSums.element(wg).assign(s.element(uint(W - 1)));
      s.element(uint(W - 1)).assign(uint(0));
    });
    workgroupBarrier();

    for (let d = 1; d < W; d *= 2) {
      offset >>= 1;
      const off = offset;
      If(tid.lessThan(uint(d)), () => {
        const ai: Any = tid.mul(uint(off * 2)).add(uint(off - 1));
        const bi: Any = ai.add(uint(off));
        const t: Any = s.element(ai).toVar();
        s.element(ai).assign(s.element(bi));
        s.element(bi).assign(s.element(bi).add(t));
      });
      workgroupBarrier();
    }

    cellStart.element(gi).assign(s.element(tid));
  })().compute(nCellsPadded, [W]);

  // Pass 2 — single-workgroup Blelloch scan over `blockSums`. `blockSums`
  // is always sized W entries (unused trailing entries are zero from their
  // initial allocation, which is the scan identity — so the scan produces
  // the correct exclusive prefix for the first `numBlocks` entries).
  const blockSumScan = Fn(() => {
    const tid: Any = localId.x;

    const s: Any = workgroupArray('uint', W);
    s.element(tid).assign(blockSums.element(tid));
    workgroupBarrier();

    let offset = 1;
    for (let d = W >> 1; d > 0; d >>= 1) {
      const off = offset;
      If(tid.lessThan(uint(d)), () => {
        const ai: Any = tid.mul(uint(off * 2)).add(uint(off - 1));
        const bi: Any = ai.add(uint(off));
        s.element(bi).assign(s.element(bi).add(s.element(ai)));
      });
      workgroupBarrier();
      offset *= 2;
    }

    If(tid.equal(uint(0)), () => {
      s.element(uint(W - 1)).assign(uint(0));
    });
    workgroupBarrier();

    for (let d = 1; d < W; d *= 2) {
      offset >>= 1;
      const off = offset;
      If(tid.lessThan(uint(d)), () => {
        const ai: Any = tid.mul(uint(off * 2)).add(uint(off - 1));
        const bi: Any = ai.add(uint(off));
        const t: Any = s.element(ai).toVar();
        s.element(ai).assign(s.element(bi));
        s.element(bi).assign(s.element(bi).add(t));
      });
      workgroupBarrier();
    }

    blockSums.element(tid).assign(s.element(tid));
  })().compute(W, [W]);

  // Pass 3 — fused finalize. Three same-shape kernels collapsed into one
  // dispatch over `nCellsPadded`:
  //   (a) addPrefix        : fold blockSums prefix back onto cellStart.
  //   (b) resetWriteCursor : initialize per-cell scatter cursor.
  //   (c) cellBounds       : cellEnd = cellStart + counts.
  // All three operate on the same `i = instanceIndex` and only touch
  // i-indexed buffers — no cross-thread sync needed. The relative
  // ordering of the cellEnd write vs the scatter write changes (cellEnd
  // is now produced before scatter runs); this is observably equivalent
  // because cellEnd reads `counts[i]` (frozen by histogram) and the
  // finalized `cellStart[i]` (just computed by this thread), neither of
  // which scatter writes. Phase Perf-17.
  const finalizeCellRanges = Fn(() => {
    const i: Any = instanceIndex;
    const wg: Any = i.div(uint(W));

    // (a) addPrefix
    const final: Any = cellStart.element(i).add(blockSums.element(wg)).toVar();
    cellStart.element(i).assign(final);

    // (b) resetWriteCursor — `writeCursor` is atomic (for the `atomicAdd`
    //     in `scatter`), so initialization goes through `atomicStore`.
    atomicStore(writeCursor.element(i), final);

    // (c) cellBounds — `counts` is atomic; reads go through `atomicLoad`
    //     to generate valid WGSL.
    const cnt: Any = atomicLoad(counts.element(i));
    cellEnd.element(i).assign(final.add(cnt));
  })().compute(nCellsPadded);

  // Scatter per particle. Each thread atomically bumps the cell's write
  // cursor, landing in a unique slot inside that cell's run. Per Green 2010
  // §"Building the Grid using Atomic Operations" final paragraph: "write
  // them to contiguous locations in the grid array using the results of the
  // scan."
  const scatter = Fn(() => {
    const p: Any = instanceIndex;
    const c: Any = cellIndex.element(p);
    const slot: Any = atomicAdd(writeCursor.element(c), uint(1));
    sortedIndices.element(slot).assign(p);
  })().compute(capacity);

  return { blockScan, blockSumScan, finalizeCellRanges, scatter };
}
