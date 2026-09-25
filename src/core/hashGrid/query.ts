import { Vector3 } from 'three';
import { If, Loop, atomicAdd, int, uint } from 'three/tsl';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import { mortonBucketUnmasked } from './mortonHash.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Emit the 27-cell neighbor iteration into the surrounding TSL `Fn` body.
 *
 *

 *
 *   Bucket function: 3D Morton (Z-curve) encoding via `mortonHash.ts`,
 *   shared with `cellIndex.ts` so the histogram and walk sides produce
 *   bit-identical bucket indices. Replaces the Teschner XOR-mix hash used
 *   in prior phases. Morton bucketing maps adjacent cells in 3D to nearby
 *   bucket indices so the 27-cell walk's `cellStart` / `cellEnd` lookups
 *   cluster instead of scattering — measured 1.98×–2.78× speedup on the
 *   reference platform vs Teschner XOR-mix (locality probe at
 *   `tests/perf/_probe/zsort-locality.gpu.perf.ts`). Out-of-range cells
 *   wrap-alias gracefully; the histogram side flags them via `overflowFlag`.
 *
 *   Hash-collision tax: distinct cells can still map to the same bucket
 *   under Morton (lower-bit collisions for cells differing in higher bits,
 *   plus mask-collisions when `hashTableSize` is smaller than the encoding
 *   range). Every caller's `onCandidate` distance filter rejects the
 *   resulting false-positive pairs at a small per-pair cost. The collision-
 *   rate curve is pinned by
 *   `tests/analytical/hashgrid/hashgrid-collision-rate.gpu.test.ts`
 *   (re-baselined for Morton's profile in the Z-sort phase).
 *
 * Bucket-dedup inside the 27-cell walk: two different `(dx, dy, dz)`
 * triples can hash to the same bucket when the bucket function's output
 * collapses (Morton lower-bit collisions, or mask-induced when
 * `hashTableSize` is smaller than the encoding range), and that bucket's
 * particles would then be iterated twice — double-counting each candidate
 * for the same query. We dedup by comparing each bucket against all
 * buckets computed earlier in the same query's walk and gating the
 * particle-loop on the `not-yet-seen` predicate. The compile-time unroll
 * produces at most `27·26/2 = 351` equality checks, which is cheap on a
 * GPU compared to the actual particle-body work each hit kicks off.
 *
 * Loop-variable-capture hazard: the three outer
 * `for (let d*; ...)` headers use `let`, which rebinds per iteration, so
 * each loop-body callback closes over its own `(dx, dy, dz)` binding. Do
 * not convert these to a single outer `let dz; dz++` pattern — that would
 * reintroduce the hazard.
 */
export function emitForEachNeighbor(args: {
  /** vec3 TSL node — typically `positions.element(p).xyz`. */
  readonly queryPosXyz: Any;
  readonly hashOrigin: UniformNode<'vec3', Vector3>;
  readonly cellSize: UniformNode<'float', number>;
  readonly hashTableSize: number;
  readonly cellStart: StorageBufferNode<'uint'>;
  readonly cellEnd: StorageBufferNode<'uint'>;
  readonly sortedIndices: StorageBufferNode<'uint'>;
  /**
   * Invoked per candidate neighbor particle index. The caller filters by
   * actual distance before applying forces. This callback runs at TSL-build
   * time and must emit TSL operations into the surrounding shader stack.
   */
  readonly onCandidate: (neighborIdx: Any) => void;
  /**
   * Phase Perf — optional candidate-visit counter. When provided alongside
   * {@link queryIdx}, the macro emits `atomicAdd(_perfCounter[queryIdx], 1)`
   * once per candidate visited, after bucket-dedup, before the
   * `onCandidate` callback. When omitted (default — the production path),
   * no counter code is emitted at all, so production callers carry zero
   * runtime overhead.
   *
   * Counter run is separate from timing run because the atomicAdd itself
   * perturbs timing — the harness does one instrumented frame per
   * measurement run (plan §"Step 4 — neighbor-candidate counting").
   */
  readonly _perfCounter?: StorageBufferNode<'uint'>;
  /**
   * Phase Perf — query particle index. Required if and only if
   * {@link _perfCounter} is provided; it is the index into the counter
   * buffer where the per-particle candidate count accumulates. Typically
   * passed as `instanceIndex` (or whatever scalar index the caller's
   * outer dispatch threads use).
   */
  readonly queryIdx?: Any;
}): void {
  const {
    queryPosXyz,
    hashOrigin,
    cellSize,
    hashTableSize,
    cellStart,
    cellEnd,
    sortedIndices,
    onCandidate,
    _perfCounter,
    queryIdx,
  } = args;

  if (_perfCounter !== undefined && queryIdx === undefined) {
    throw new Error(
      'emitForEachNeighbor: _perfCounter requires queryIdx (the index into the counter buffer)',
    );
  }

  if ((hashTableSize & (hashTableSize - 1)) !== 0) {
    throw new Error(
      `emitForEachNeighbor: hashTableSize must be a power of two, got ${hashTableSize}`,
    );
  }
  const bucketMask = hashTableSize - 1;

  // Query cell coords in signed-int space so `dx ∈ {-1, 0, 1}` shifts stay
  // well-defined at integer extremes. `.toVar()` materializes each component
  // once so the 27-cell unroll reads from a variable instead of rebuilding
  // the floor/sub/div chain 27 times.
  const rel: Any = queryPosXyz.sub(hashOrigin as Any).div(cellSize as Any);
  const qcx: Any = rel.x.floor().toInt().toVar();
  const qcy: Any = rel.y.floor().toInt().toVar();
  const qcz: Any = rel.z.floor().toInt().toVar();

  const previousBuckets: Any[] = [];
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const ncx: Any = qcx.add(int(dx));
        const ncy: Any = qcy.add(int(dy));
        const ncz: Any = qcz.add(int(dz));

        // Morton (Z-curve) bucket encoding — bit-identical to the
        // `cellIndex.ts` histogram side via the shared `mortonHash.ts`
        // helper. No bounds guard here because the histogram side already
        // sets the overflow flag for cells outside the Morton range; the
        // walk safely processes the wrapped buckets (false-positive
        // candidates get filtered by the per-pair distance check inside
        // `onCandidate`).
        const mortonCode: Any = mortonBucketUnmasked(ncx, ncy, ncz);
        const bucket: Any = mortonCode.bitAnd(uint(bucketMask)).toVar();

        // Dedup: skip this bucket if any earlier (dx, dy, dz) triple in
        // this same query hashed to it. See the macro-level JSDoc above
        // for why this check exists.
        let alreadySeen: Any = null;
        for (const prev of previousBuckets) {
          const eq: Any = bucket.equal(prev);
          alreadySeen = alreadySeen === null ? eq : alreadySeen.or(eq);
        }

        const walkBucket = (): void => {
          Loop(
            {
              start: cellStart.element(bucket),
              end: cellEnd.element(bucket),
              type: 'uint',
              condition: '<',
            },
            ({ i }: { i: Any }) => {
              const neighborIdx: Any = sortedIndices.element(i);
              if (_perfCounter !== undefined) {
                // Phase Perf — count this candidate visit. Runs after
                // bucket-dedup (the `If(alreadySeen.not(), walkBucket)`
                // gate above) so a bucket that two `(dx, dy, dz)` triples
                // hash to the same place is counted only once.
                atomicAdd(_perfCounter.element(queryIdx), uint(1));
              }
              onCandidate(neighborIdx);
            },
          );
        };

        if (alreadySeen === null) {
          // First bucket — nothing to compare against.
          walkBucket();
        } else {
          If(alreadySeen.not(), walkBucket);
        }

        previousBuckets.push(bucket);
      }
    }
  }
}
