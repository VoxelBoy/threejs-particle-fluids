// Neighbor walks visit particles in bucket order but would read their
// positions in the original order, which scatters memory reads across the
// whole buffer. Copying the predicted positions into bucket order once per
// rebuild makes those reads contiguous.

import { Fn, instanceIndex, instancedArray } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import type { ParticleSystem } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** A `vec4` buffer for positions in bucket order, one per particle. */
export function allocateSortedPositionsBuffer(
  particles: ParticleSystem,
): StorageBufferNode<'vec4'> {
  return instancedArray(particles.capacity, 'vec4');
}

export interface BuildSortedPositionsKernelArgs {
  readonly particles: ParticleSystem;
  /** Particle index at each sorted slot. */
  readonly sortedIndices: StorageBufferNode<'uint'>;
  /** Output: `predictedPositions[sortedIndices[k]]` at slot `k`. */
  readonly sortedPredictedPositions: StorageBufferNode<'vec4'>;
}

/** Copy predicted positions into bucket order. Runs as the last step of a grid rebuild. */
export function buildSortedPositionsKernel(args: BuildSortedPositionsKernelArgs): ComputeNode {
  const { particles, sortedIndices, sortedPredictedPositions } = args;

  return Fn(() => {
    const i: Any = instanceIndex;
    const j: Any = sortedIndices.element(i);
    sortedPredictedPositions.element(i).assign(particles.predictedPositions.element(j));
  })().compute(particles.capacity);
}
