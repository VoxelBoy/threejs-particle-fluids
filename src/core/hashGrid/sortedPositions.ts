// Per-substep Morton-permuted shadow copy of `predictedPositions`.
//

//
// Why this exists. `buildPairListKernel` walks ~216 candidates per
// fluid query and reads `predictedPositions[sortedIndices[k]]` for
// every one. The second indirection is uncoalesced because adjacent
// `k` values (Morton-bucket-adjacent particles) decode to widely-
// scattered original-index `j` values in the original layout. The
// dedup-variants probe (`tests/perf/_probe/pairlist-build-variants
// .gpu.perf.ts`, commit `5cc445f`) refuted the dedup-chain hypothesis
// and pinned the remaining cost on the per-pair random-access read.
//

import { Fn, instanceIndex, instancedArray } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import type { ParticleSystem } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Allocate the sorted-positions shadow buffer.
 *
 * Layout: `vec4` per particle slot, sized to `particles.capacity`. Same
 * stride as `particles.predictedPositions` so a single read writes the
 * full 16 B record. At 100k particles = 1.6 MB; at 1M = 16 MB.
 *
 * Lifetime mirrors the hash grid's other scratch buffers: allocated
 * once at construction, no growth, no `dispose()` (three.js r184
 * `StorageBufferAttribute` has no dispose path — see U-15 archive).
 */
export function allocateSortedPositionsBuffer(
  particles: ParticleSystem,
): StorageBufferNode<'vec4'> {
  return instancedArray(particles.capacity, 'vec4');
}

export interface BuildSortedPositionsKernelArgs {
  readonly particles: ParticleSystem;
  /**
   * The hash grid's existing permutation: `sortedIndices[i_sorted] =
   * j_original` for `i_sorted ∈ [0, capacity)`. Output of the count-
   * sort scatter pass (see `HashGrid` rebuild pipeline step 8).
   */
  readonly sortedIndices: StorageBufferNode<'uint'>;
  /**
   * Output. Allocated by {@link allocateSortedPositionsBuffer}. After
   * dispatch: `sortedPredictedPositions[i_sorted].xyz =
   * predictedPositions[sortedIndices[i_sorted]].xyz`.
   */
  readonly sortedPredictedPositions: StorageBufferNode<'vec4'>;
}

/**
 * Materialize a Morton-permuted shadow copy of
 * `particles.predictedPositions` into `sortedPredictedPositions`,
 * indexed by the hash grid's sorted-particle order.
 *
 * Algorithmic outline (one thread per particle slot, `i_sorted =
 * instanceIndex`):
 *   1. `j = sortedIndices[i_sorted]`.
 *   2. `sortedPredictedPositions[i_sorted] = predictedPositions[j]`.
 *
 * Cost: one full-array sequential write + one random-access read per
 * particle. At 100k = ~1.6 MB write + ~1.6 MB read = ~3.2 MB total,
 * dominated by sequential write bandwidth. Estimated ~0.3 ms p50 on
 * Apple Silicon Metal-3 (plan §Leverage).
 *
 * Determinism: per-thread copy with no atomics, no scatters. Output
 * is a deterministic function of `predictedPositions` and
 * `sortedIndices`, both already tier-1 G4 bit-deterministic per
 * `memory/phase_03_exit_2026_04_21.md`.
 *
 * Dispatch placement: appended to `HashGrid.rebuildPipeline` AFTER
 * the `cellBounds` kernel so `sortedIndices` is final.
 * `buildPairListKernel` is the only consumer; it runs ONCE per
 * substep at the start of `FluidSystem`'s pre-iter pipeline, AFTER
 * the hash-grid rebuild, so the shadow is fresh when read.
 */
export function buildSortedPositionsKernel(args: BuildSortedPositionsKernelArgs): ComputeNode {
  const { particles, sortedIndices, sortedPredictedPositions } = args;

  return Fn(() => {
    const i: Any = instanceIndex;
    const j: Any = sortedIndices.element(i);
    sortedPredictedPositions.element(i).assign(particles.predictedPositions.element(j));
  })().compute(particles.capacity);
}
