// Per-substep neighbor pair list (Macklin & Müller 2013 "Position Based
// Fluids" Algorithm 1 line 6 amortization + §6 paragraph 2 prescription:
// "We recompute particle neighborhoods once per-step and recalculate
// distance and constraint values each solver iteration"). Each fluid
// thread builds its own list once per substep; the per-iter density /
// lambda / positionDelta kernels then iterate that list instead of
// re-walking the hash grid every iteration.
//
// Paper acknowledges (§6 paragraph 2) that this can produce density
// underestimates when a particle separates from its initial neighbor
// set: "In PCISPH this can cause serious problems… Our algorithm
// considers only the current particle positions (not accumulated
// pressure), so this does not occur." PBF is robust to staleness
// because each iteration recomputes density from current positions
// and the cached pairs — no accumulated pressure error.
//

import { Fn, If, Loop, atomicStore, instanceIndex, instancedArray, int, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { HashGrid } from './HashGrid.js';
import { mortonBucketUnmasked } from './mortonHash.js';
import type { ParticleSystem } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Maximum neighbors per fluid particle stored in the pair list.
 *
 * Phase 08 measurements at PBF spacing `h = 0.04 m`, `particleSpacing
 * = 0.025 m` give roughly 30 within-`h` neighbors per particle. Setting
 * the cap at 64 gives ~2× headroom for compressed configurations
 * (high-velocity splash, dense piles, gravity-loaded columns).
 *
 * Memory cost: `fluidCount × 64 × 4 bytes` per FluidSystem. At 50k
 * fluid particles = 12.8 MB; at 100k = 25.6 MB. Well within the
 * `maxBufferSize: 256 MB` measured by H4 on Apple Silicon metal-3.
 *
 * If a real scene exceeds this, `pairOverflowFlag` fires and the
 * particle's list is truncated to the first 64 neighbors discovered
 * during the 27-cell walk. The simulation continues — the truncated
 * particle will see a density underestimate proportional to the
 * fraction of neighbors dropped — but the overflow flag makes the
 * situation observable from CPU. Raise `MAX_NEIGHBORS` (and re-baseline
 * memory cost) when this becomes load-bearing.
 */
export const MAX_NEIGHBORS = 64;

/**
 * Allocate the storage triple a {@link buildPairListKernel} writes into.
 * Convenience helper so callers don't have to remember the layout
 * conventions. Three return values, in dispatch order:
 *
 *   `pairList`         — `u32[MAX_NEIGHBORS × fluidCount]`. Column-major
 *                        per neighbor slot; `pairList[k × fluidCount +
 *                        localIdx]` for `k ∈ [0, pairCount[localIdx])`
 *                        holds the global particle index of the k-th
 *                        neighbor of the fluid particle at local index
 *                        `localIdx = i − fluidStart`. Layout chosen so
 *                        adjacent threads in a warp at the same `k`
 *                        write (and read) adjacent memory addresses,
 *                        which the GPU coalesces into a single memory
 *                        transaction (Winkler, Rezavand & Rauch 2018
 *                        "Neighbour lists for SPH on GPUs" CPC 225,
 *                        140–148, §3.3 + §4 — coalesced VL storage).
 *   `pairCount`        — `u32[fluidCount]`. Number of valid entries
 *                        for each particle. Bounded by
 *                        {@link MAX_NEIGHBORS}.
 *   `pairOverflowFlag` — `atomic<u32>[1]`. Set to 1 if any particle had
 *                        more than {@link MAX_NEIGHBORS} candidates
 *                        within `h` during the build pass; otherwise 0.
 *
 * Lifetime / reset: the build kernel zeroes `pairCount` and the
 * overflow flag itself on every dispatch (each fluid thread starts
 * its own `count = 0` and the overflow flag is `atomicStore(0)` on
 * the first dispatch's first thread — see {@link buildPairListKernel}).
 * Callers do NOT need to issue a reset kernel before each build.
 */
export function allocatePairListStorage(fluidCount: number): {
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
  readonly pairOverflowFlag: StorageBufferNode<'uint'>;
} {
  if (!Number.isInteger(fluidCount) || fluidCount <= 0) {
    throw new Error(
      `allocatePairListStorage: fluidCount must be a positive integer, got ${fluidCount}`,
    );
  }
  return {
    pairList: instancedArray(fluidCount * MAX_NEIGHBORS, 'uint'),
    pairCount: instancedArray(fluidCount, 'uint'),
    pairOverflowFlag: instancedArray(1, 'uint').toAtomic(),
  };
}

export interface BuildPairListKernelArgs {
  readonly particles: ParticleSystem;
  readonly hashGrid: HashGrid;
  /** `h²` — the within-radius squared cutoff. Typically `sph.hSq`. */
  readonly hSq: UniformNode<'float', number>;
  /**
   * Range of fluid particles this list covers. The pair list is sized
   * to `fluidParticles.count × MAX_NEIGHBORS`; the build kernel
   * dispatches `fluidParticles.count` threads.
   */
  readonly fluidParticles: { readonly start: number; readonly count: number };
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
  readonly pairOverflowFlag: StorageBufferNode<'uint'>;
}

/**
 * Build the per-substep pair list for one fluid particle range.
 *
 * Algorithmic outline (one thread per fluid particle, `localIdx =
 * instanceIndex`, global `i = fluidStart + localIdx`):
 *   1. Read `xi = predictedPositions[i].xyz`. Matches the read source
 *      of every per-iter caller — Phase 08 ψ-in-gradient finding pinned
 *      `predictedPositions` as the canonical PBF read source.
 *   2. Walk the 27-cell neighborhood (inlined, mirroring
 *      {@link emitForEachNeighbor} for the dedup logic and bucket
 *      iteration order). Self-candidate (`j == i`) is naturally
 *      included — paper density formula sums over self with
 *      `W_poly6(0, h) = poly6Coef · h⁶`.
 *   3. For each cell `k` in `[cellStart[bucket], cellEnd[bucket])`:
 *        a. Read `xj = sortedPredictedPositions[k].xyz` — the
 *           Morton-permuted shadow buffer rebuilt by
 *           {@link buildSortedPositionsKernel} once per substep at
 *           the end of `HashGrid.rebuildPipeline`. Reads coalesce
 *           within each bucket because `k` advances sequentially,
 *           replacing the uncoalesced `predictedPositions[
 *           sortedIndices[k]]` indirection of prior phases.
 *        b. Compute `rSq = |xi − xj|²`.
 *        c. Skip if `rSq ≥ h²` (paper's within-`h` cutoff).
 *        d. If the local `count < MAX_NEIGHBORS`: resolve `j =
 *           sortedIndices[k]` (delayed to here so the indirection
 *           fires only on accepted pairs — ~7× less than today at
 *           MVP density), write `pairList[count · fluidCount +
 *           localIdx] = j` (column-major; adjacent warp lanes at the
 *           same `count` write adjacent addresses, which coalesces —
 *           Winkler 2018 §3.3 + §4) and increment `count`.
 *        e. Otherwise: `atomicStore(pairOverflowFlag, 1)` (race-free
 *           because writers all write the same value 1) and discard
 *           the pair.
 *   4. Write `pairCount[localIdx] = count`.
 *
 * The local `count` is a per-thread `var` — no cross-thread atomic
 * traffic. Each fluid thread writes only its own column of `pairList`
 * (slots `{0, 1, …, count-1} · fluidCount + localIdx`) and its own
 * slot of `pairCount`, so the build is gather-mode and tier-1 G4
 * bit-deterministic given the hash grid's tier-1 stable
 * `sortedIndices` output.
 *
 * Per-particle pair order is the 27-cell walk order (Z-curve cluster,
 * with bucket-dedup) followed by `sortedIndices` order within each
 * cell — byte-identical to the prior row-major build, so downstream
 * summation (density, lambda numerator, positionDelta) preserves its
 * tier-2 bounded-error envelope and the
 * `density-pair-list-cpu-parity.gpu.test.ts` parity gate stays at
 * unchanged tolerance. Only the storage layout rotated 90°: slot `k`
 * for particle `localIdx` moved from `localIdx · MAX_NEIGHBORS + k`
 * (row-major) to `k · fluidCount + localIdx` (column-major).
 *
 * `pairOverflowFlag` is reset to zero on every dispatch by the
 * single thread `instanceIndex == 0`. Production scenes with healthy
 * particle distributions (Phase 08 measurements: ~30 neighbors at
 * standard PBF spacing) leave the flag at zero. Compressed regimes
 * may trip it; the simulation continues with a truncated list and
 * the flag is observable from CPU via a readback if a caller wants
 * to assert.
 *
 * Why the walk is inlined instead of using
 * {@link emitForEachNeighbor}: the build kernel needs the sorted
 * index `k` (for the coalesced `sortedPredictedPositions[k]` read)
 * AND the original index `j = sortedIndices[k]` (for the pair-list
 * output). `emitForEachNeighbor` only exposes `j` to its callback.
 * Splitting that helper to expose both indices would change a public
 * surface used by `contact.generate` and other callers, where the
 * current single-`j` callback is exactly right. Inlining the walk
 * here keeps the optimization local to the one kernel that benefits.
 */
export function buildPairListKernel(args: BuildPairListKernelArgs): ComputeNode {
  const { particles, hashGrid, hSq, fluidParticles, pairList, pairCount, pairOverflowFlag } = args;

  if (
    !Number.isInteger(fluidParticles.start) ||
    !Number.isInteger(fluidParticles.count) ||
    fluidParticles.start < 0 ||
    fluidParticles.count <= 0 ||
    fluidParticles.start + fluidParticles.count > particles.capacity
  ) {
    throw new Error(
      `buildPairListKernel: invalid fluidParticles range start=${fluidParticles.start} count=${fluidParticles.count} capacity=${particles.capacity}`,
    );
  }

  if ((hashGrid.hashTableSize & (hashGrid.hashTableSize - 1)) !== 0) {
    throw new Error(
      `buildPairListKernel: hashTableSize must be a power of two, got ${hashGrid.hashTableSize}`,
    );
  }

  const startIdx = fluidParticles.start;
  const fluidCount = fluidParticles.count;
  const maxN = MAX_NEIGHBORS;
  const bucketMask = hashGrid.hashTableSize - 1;

  return Fn(() => {
    const localIdx: Any = (instanceIndex as Any).toVar();
    const i: Any = localIdx.add(uint(startIdx)).toVar();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();

    // Per-thread local counter — no atomic traffic because each fluid
    // thread owns its own column of `pairList` (slots `{0, …, count-1}
    // · fluidCount + localIdx`) and its own slot of `pairCount`.
    // Initialized to 0 every dispatch.
    const count: Any = uint(0).toVar();

    // The single thread `localIdx == 0` resets the overflow flag at
    // the start of each dispatch so the next substep starts clean.
    // `atomicStore` is race-free with concurrent writers writing the
    // same value 1 later in the dispatch (the flag is monotonic — it
    // can flip 0→1 but never back; resets only fire from one thread).
    If(localIdx.equal(uint(0)), () => {
      atomicStore(pairOverflowFlag.element(uint(0)) as Any, uint(0));
    });

    // Query cell coords in signed-int space (mirrors `query.ts`). The
    // 27-cell unroll reads from these vars instead of rebuilding the
    // floor/sub/div chain 27 times.
    const rel: Any = xi.sub(hashGrid.hashOriginUniform as Any).div(hashGrid.cellSizeUniform as Any);
    const qcx: Any = rel.x.floor().toInt().toVar();
    const qcy: Any = rel.y.floor().toInt().toVar();
    const qcz: Any = rel.z.floor().toInt().toVar();

    // 27-cell walk with bucket-dedup. The dedup OR-chain is intact
    // because the `pairlist-build-variants` probe (commit `5cc445f`)
    // measured cheaper-dedup variants regress the kernel by 4.6–9.5%
    // on p50 — the chain is saving more time than it costs.
    const previousBuckets: Any[] = [];
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ncx: Any = qcx.add(int(dx));
          const ncy: Any = qcy.add(int(dy));
          const ncz: Any = qcz.add(int(dz));

          const mortonCode: Any = mortonBucketUnmasked(ncx, ncy, ncz);
          const bucket: Any = mortonCode.bitAnd(uint(bucketMask)).toVar();

          let alreadySeen: Any = null;
          for (const prev of previousBuckets) {
            const eq: Any = bucket.equal(prev);
            alreadySeen = alreadySeen === null ? eq : alreadySeen.or(eq);
          }

          const walkBucket = (): void => {
            Loop(
              {
                start: hashGrid.cellStart.element(bucket),
                end: hashGrid.cellEnd.element(bucket),
                type: 'uint',
                condition: '<',
              },
              ({ i: k }: { i: Any }) => {
                // Coalesced sequential read across the bucket — this
                // is the optimization. Replaces
                // `predictedPositions[sortedIndices[k]]` (one cache-
                // friendly + one random-access read) with one
                // sequential read into the Morton-permuted shadow.
                const xj: Any = hashGrid.sortedPredictedPositions.element(k).xyz;
                const diff: Any = xi.sub(xj).toVar();
                const rSq: Any = diff.dot(diff).toVar();
                // Paper's within-h cutoff. Pairs outside h are NOT
                // included in the list, so downstream callers can
                // drop their own redundant rSq < hSq filter.
                If(rSq.lessThan(hSq as Any), () => {
                  If(count.lessThan(uint(maxN)), () => {
                    // Resolve the original index only on accepted
                    // pairs (~7× less often than visiting candidates).
                    // Pair list stores ORIGINAL `j` so per-iter
                    // consumers (lambda, positionDelta, …) read
                    // `predictedPositions[j]` exactly as before.
                    const j: Any = hashGrid.sortedIndices.element(k);
                    // Column-major: adjacent warp lanes at the same
                    // `count` write adjacent addresses, which the GPU
                    // coalesces into one transaction (Winkler 2018 §3.3).
                    const slotIdx: Any = count.mul(uint(fluidCount)).add(localIdx);
                    pairList.element(slotIdx).assign(j);
                    count.assign(count.add(uint(1)));
                  }).Else(() => {
                    // Discard the pair; raise the overflow flag.
                    atomicStore(pairOverflowFlag.element(uint(0)) as Any, uint(1));
                  });
                });
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

    pairCount.element(localIdx).assign(count);
  })().compute(fluidParticles.count);
}

export interface EmitForEachPairArgs {
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
  /**
   * Global particle index `i = fluidStart + localIdx`. The macro
   * subtracts `fluidStart` to recover the local index used to address
   * the pair list.
   */
  readonly queryIdx: Any;
  readonly fluidStart: number;
  /**
   * Number of fluid particles in this range. Required for the column-
   * major slot address `pairList[k · fluidCount + localIdx]` — should
   * equal the `fluidParticles.count` that was passed to the matching
   * {@link buildPairListKernel} call.
   */
  readonly fluidCount: number;
  /**
   * Invoked per pair `(i, j)` already filtered by within-`h` distance
   * at build time. Callers do NOT need to refilter (and SHOULD remove
   * any redundant `rSq < hSq` check that was historically present in
   * the {@link emitForEachNeighbor} version of the same kernel).
   */
  readonly onCandidate: (neighborIdx: Any) => void;
}

/**
 * Iterate the cached pairs for one fluid particle.
 *
 *
 * Like {@link emitForEachNeighbor}, this is a JS macro — it emits
 * TSL operations into the surrounding `Fn(() => { ... })` body. The
 * inner `Loop` is a real TSL `LoopNode` so per-row pair iteration
 * runs as a single GPU loop, not a JS unroll.
 */
export function emitForEachPair(args: EmitForEachPairArgs): void {
  const { pairList, pairCount, queryIdx, fluidStart, fluidCount, onCandidate } = args;

  // localIdx = queryIdx - fluidStart. Materialised so it can be reused
  // unchanged on every loop iteration as the column-major slot offset.
  const localIdx: Any = (queryIdx as Any).sub(uint(fluidStart)).toVar();
  const count: Any = pairCount.element(localIdx).toVar();

  // Dynamic loop bound (`end: count`) measured faster than a fixed
  // `end: MAX_NEIGHBORS` + inner `If(k >= count) Break()` on Apple
  // Silicon Metal-3 (Verified 2026-04-27, A/B in
  // memory/perf_pair_loop_bound_2026_04_27.md): the fixed-bound +
  // break form regressed every fluid pair-list kernel by 11–63% on
  // p50, with no scene showing improvement. The TSL/WGSL backend on
  // this stack does not unroll or partial-unroll the fixed-bound
  // form, and the extra inner branch is pure overhead.
  Loop(
    {
      start: uint(0),
      end: count,
      type: 'uint',
      condition: '<',
    },
    ({ i: k }: { i: Any }) => {
      // Column-major read: adjacent warp lanes at the same `k` read
      // adjacent addresses, which the GPU coalesces into a single
      // memory transaction. Mirrors the column-major write in
      // `buildPairListKernel`. See {@link allocatePairListStorage} +
      // Winkler 2018 §3.3.
      const slotIdx: Any = k.mul(uint(fluidCount)).add(localIdx);
      const neighborIdx: Any = pairList.element(slotIdx);
      onCandidate(neighborIdx);
    },
  );
}
