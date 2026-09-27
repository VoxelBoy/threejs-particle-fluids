import { Continue, Fn, If, float, instanceIndex, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import {
  emitForEachNeighbor,
  emitPoly6FromRSq,
  type HashGrid,
  type ParticleRange,
  type ParticleSystem,
  type SphKernelUniforms,
} from '../../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Boundary volume `V_i = 1 / Σ_k W(x_i − x_k)` for each particle in `range`
 * (Akinci et al. 2012, eq. 4), summing only over the same range. Densely
 * packed boundary particles get smaller volumes, so an unevenly sampled
 * solid still pushes on the fluid evenly.
 *
 * @param positions Which positions to use: the committed ones for a one-time
 *   setup pass, or the predicted ones when the boundary moves every substep.
 */
export function buildBoundaryVolumeKernel(args: {
  readonly particles: ParticleSystem;
  readonly grid: HashGrid;
  readonly sph: SphKernelUniforms;
  readonly range: ParticleRange;
  readonly positions: StorageBufferNode<'vec4'>;
}): ComputeNode {
  const { particles, grid, sph, range, positions } = args;
  const end = range.start + range.count;

  return Fn(() => {
    const i: Any = instanceIndex.add(uint(range.start)).toVar();
    const xi: Any = positions.element(i).xyz.toVar();
    const sum: Any = float(0).toVar();
    emitForEachNeighbor(grid, xi, (j: Any) => {
      If(j.lessThan(uint(range.start)).or(j.greaterThanEqual(uint(end))), () => {
        Continue();
      });
      const offset: Any = xi.sub(positions.element(j).xyz);
      const rSq: Any = offset.dot(offset);
      If(rSq.greaterThanEqual(sph.hSq), () => {
        Continue();
      });
      sum.addAssign(emitPoly6FromRSq(rSq, sph));
    });
    particles.boundaryVolume.element(i).assign(float(1).div(sum.max(1e-12)));
  })().compute(range.count);
}
