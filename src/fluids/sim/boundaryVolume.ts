import { Continue, Fn, If, float, instanceIndex, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { HashGrid, ParticleRange, ParticleSystem } from '../../core/index.js';
import { emitForEachNeighbor } from '../../core/index.js';

import { emitPoly6FromRSq, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Which position buffer the kernel reads to compute `Σ_k W_{ik}`.
 *
 * - `committed` — reads `particles.positions`. Use for the one-shot
 *   registration-time seed: at registration the committed and predicted
 *   buffers are identical (`uploadParticles` populates both), so this is
 *   the right source for scenes that ship the static simplification.
 * - `predicted` — reads `particles.predictedPositions`. Use for the per-
 *   substep dynamic kernel: inside the loop `predictedPositions` is the
 *   working configuration the fluid density will project against, so
 *   `V_i` must be computed at `x*`, not at the last committed `x`.
 */
export type BoundaryVolumePositionSource = 'committed' | 'predicted';

export interface BuildBoundaryVolumeKernelArgs {
  readonly particles: ParticleSystem;
  readonly hashGrid: HashGrid;
  readonly sph: SphKernelUniforms;
  /**
   * Range of boundary particles to compute volumes for. The kernel writes
   * `particles.boundaryVolume[start..start+count)`. The sum in eq. 4
   * restricts neighbors to particles *within the same range* — boundary
   * particles couple to each other, not to fluid neighbors (per Akinci 2012
   * §2).
   */
  readonly boundaryParticles: ParticleRange;
  /** Defaults to `committed` (legacy static seed). */
  readonly positionSource?: BoundaryVolumePositionSource;
}

/**
 * Build the Akinci 2012 boundary-volume kernel.
 *
 *
 * Multi-range registration: MVP supports one contiguous range per call.
 * If a future scene registers multiple boundary ranges (e.g. two cloth
 * patches with different per-patch sampling densities), each range's `V`
 * is computed only from its own members — boundary particles in range A
 * will not "see" boundary particles in range B even if they are spatially
 * adjacent. File a new UNKNOWN when the first multi-range scene lands.
 *
 * The "same-range" restriction is implemented by a range gate in the
 * neighbor callback (`j ∉ [start, start + count)` → skip). The gate is
 * cheap — a pair of unsigned comparisons per candidate — and sidesteps
 * needing a separate per-particle flag bit (which would require a core
 * API change).
 *
 * Static vs dynamic (2026-04-24 decision; U-17 reopened):
 *   Akinci §2.2 (last paragraph) prescribes per-substep recomputation of
 *   `V_i` for moving/deforming boundaries ("for moving boundary particles
 *   and all neighboring boundary particles, the represented particle
 *   volumes are recomputed"). The caller selects the regime by passing
 *   `positionSource`:
 *     - `committed` — one-shot kernel run at registration; writes the
 *       static seed. Correct for boundaries that never move (container
 *       walls, confirmed-static rigid props) and used by `FluidSystem
 *       .registerBoundaryParticles` for the seed pass in every case.
 *     - `predicted` — per-substep kernel wired into `FluidSystem
 *       .preIterKernels`; re-reads `x*` each substep so `V_i` tracks the
 *       current deformed / translated configuration. Required for soft-
 *       body, rigid-in-proximity, and cloth boundary ranges — the paper's
 *       moving-boundary regime.
 *
 * Dispatch shape: `boundaryParticles.count` threads.
 */
export function buildBoundaryVolumeKernel(args: BuildBoundaryVolumeKernelArgs): ComputeNode {
  const { particles, hashGrid, sph, boundaryParticles } = args;
  const positionSource: BoundaryVolumePositionSource = args.positionSource ?? 'committed';

  if (
    !Number.isInteger(boundaryParticles.start) ||
    !Number.isInteger(boundaryParticles.count) ||
    boundaryParticles.start < 0 ||
    boundaryParticles.count <= 0 ||
    boundaryParticles.start + boundaryParticles.count > particles.capacity
  ) {
    throw new Error(
      `buildBoundaryVolumeKernel: invalid boundaryParticles range start=${boundaryParticles.start} count=${boundaryParticles.count} capacity=${particles.capacity}`,
    );
  }

  const startIdx = boundaryParticles.start;
  const endIdx = startIdx + boundaryParticles.count;
  const positionBuffer: StorageBufferNode<'vec4'> =
    positionSource === 'predicted' ? particles.predictedPositions : particles.positions;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = positionBuffer.element(i).xyz.toVar();
    const sumW: Any = float(0.0).toVar();

    emitForEachNeighbor({
      queryPosXyz: xi,
      hashOrigin: hashGrid.hashOriginUniform,
      cellSize: hashGrid.cellSizeUniform,
      hashTableSize: hashGrid.hashTableSize,
      cellStart: hashGrid.cellStart,
      cellEnd: hashGrid.cellEnd,
      sortedIndices: hashGrid.sortedIndices,
      onCandidate: (j: Any) => {
        // Same-range gate: only boundary particles in this registration
        // contribute to the sum.
        If(j.lessThan(uint(startIdx)).or(j.greaterThanEqual(uint(endIdx))), () => {
          Continue();
        });

        const xj: Any = positionBuffer.element(j).xyz;
        const diff: Any = xi.sub(xj).toVar();
        const rSq: Any = diff.dot(diff);
        If(rSq.greaterThanEqual(sph.hSq as Any), () => {
          Continue();
        });

        const w: Any = emitPoly6FromRSq(rSq, sph);
        sumW.addAssign(w);
      },
    });

    // V_i = 1 / Σ_k W. Guard against `sumW = 0` (an isolated boundary
    // particle with no neighbors; would produce `inf`). The `1e-12` floor
    // clamps to a very small positive — the resulting huge `V_i` would
    // only matter if that isolated boundary particle later gets a fluid
    // neighbor, in which case the overly-strong push is a graceful
    // degradation compared to NaN. Not expected in well-sampled scenes.
    const invSum: Any = float(1.0).div(sumW.max(float(1e-12)));
    particles.boundaryVolume.element(i).assign(invSum);
  })().compute(boundaryParticles.count);
}
