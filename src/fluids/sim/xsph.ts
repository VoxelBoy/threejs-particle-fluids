import { Fn, float, instanceIndex, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ParticleRange, ParticleSystem } from '../../core/index.js';
import { emitForEachPair } from '../../core/index.js';

import { emitPoly6FromRSq, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildXsphComputeKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  /** XSPH mixing coefficient `c`. Paper-typical `0.01`. */
  readonly c: UniformNode<'float', number>;
  /**
   * Per-fluid-particle rest volume `V = m / ρ_0 = spacing³`. Uniform
   * because Phase 08 treats particle spacing as scene-constant. See
   * `FluidSystem.particleVolumeUniform`.
   */
  readonly particleVolume: UniformNode<'float', number>;
  /**
   * Per-particle velocity correction output, sized to `particles.capacity`.
   * Consumed by {@link buildXsphApplyKernel} in the next dispatch.
   * Two-phase to avoid the read-modify-write race on `velocities` — the
   * compute kernel reads `velocities[j]` from neighbors, which would
   * alias with another thread's in-dispatch write to `velocities[i]`.
   */
  readonly deltaV: StorageBufferNode<'vec4'>;
  readonly fluidParticles: ParticleRange;

  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * XSPH viscosity compute kernel — Phase 1 of the two-pass velocity
 * smoothing (Macklin 2013 §5 eq. 17, neighbor-walk amortized via
 * Algorithm 1 line 6 / §6 paragraph 2 per-substep pair list).
 *
 * Paper (unit-mass convention): `v_i^new = v_i + c·Σ_j v_{ij}·W(|x_i − x_j|, h)`.
 *
 * Our dimensional SPH form adds the Monaghan-canonical `m_j/ρ_j` weight
 * that paper absorbs under its unit-mass assumption:
 *
 *   `Δv_i = c · Σ_j (m_j/ρ_j) · (v_j − v_i) · W_{ij}`
 *
 * For fluid neighbors `m_j/ρ_j ≈ particleVolume = spacing³`. For Akinci
 * boundary neighbors we use the stored `boundaryVolume[j]` (Akinci 2012
 * eq. 4). Dropping this factor (as paper-verbatim transcription does)
 * produces a velocity correction `1/V ≈ 64 000×` too large for our
 * spacing = 0.025 m / ρ_0 = 1000 kg/m³ scene — observable as instant
 * velocity divergence on any settle. Verified 2026-04-23 in the Phase
 * 08 post-exit debugging pass.
 *
 * Two-phase split: this kernel writes `deltaV[i]`; the apply kernel
 * ({@link buildXsphApplyKernel}) reads `deltaV[i]` and writes
 * `velocities[i]`. Without the split, thread `i` reads `velocities[j]`
 * during the neighbor walk while thread `j` (running concurrently)
 * writes its own `velocities[j]` at the same time. In-dispatch read/write
 * races on the same storage buffer are implementation-defined in WebGPU.
 *
 * Pair-list iteration: pairs were filtered to `|r|² < h²` at build time
 * (against `predictedPositions` pre-iter), so no within-h check is
 * needed here. The Poly6 weight self-clamps to zero outside `h` even if
 * pre-iter→post-advect motion has carried a pair across the boundary,
 * which bounds the staleness error to a few percent near the cutoff.
 *
 * Gather mode — each thread writes only `deltaV[i]`.
 */
export function buildXsphComputeKernel(args: BuildXsphComputeKernelArgs): ComputeNode {
  const { particles, sph, c, particleVolume, deltaV, fluidParticles, pairList, pairCount } = args;
  validateRange('buildXsphComputeKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.positions.element(i).xyz.toVar();
    const vi: Any = particles.velocities.element(i).xyz.toVar();
    const accum: Any = vec3(0, 0, 0).toVar();

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        const xj: Any = particles.positions.element(j).xyz;
        const diff: Any = xi.sub(xj).toVar();
        const rSq: Any = diff.dot(diff);

        // Monaghan `m_j/ρ_j` weight. For fluid: particleVolume (spacing³)
        // as an at-rest approximation; for Akinci boundary: stored V_b.
        const Vj_boundary: Any = particles.boundaryVolume.element(j);
        const isBoundary: Any = Vj_boundary.greaterThan(float(0.0));
        const Vj: Any = isBoundary.select(Vj_boundary, particleVolume as Any);

        const vj: Any = particles.velocities.element(j).xyz;
        const dv: Any = vj.sub(vi);
        const w: Any = emitPoly6FromRSq(rSq, sph);
        accum.addAssign(dv.mul(w).mul(Vj));
      },
    });

    deltaV.element(i).assign(vec4(accum.mul(c as Any), float(0.0)));
  })().compute(fluidParticles.count);
}

export interface BuildXsphApplyKernelArgs {
  readonly particles: ParticleSystem;
  readonly deltaV: StorageBufferNode<'vec4'>;
  readonly fluidParticles: ParticleRange;
}

/**
 * Phase 2 of XSPH: commit `velocities[i] += deltaV[i]`. Runs in a
 * separate dispatch from the compute kernel so the cross-thread reads
 * of `velocities[j]` in the compute kernel see consistent state.
 */
export function buildXsphApplyKernel(args: BuildXsphApplyKernelArgs): ComputeNode {
  const { particles, deltaV, fluidParticles } = args;
  validateRange('buildXsphApplyKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const existing: Any = particles.velocities.element(i).toVar();
    const dv: Any = deltaV.element(i).xyz;
    const newV: Any = existing.xyz.add(dv);
    particles.velocities.element(i).assign(vec4(newV, existing.w));
  })().compute(fluidParticles.count);
}

function validateRange(kernelName: string, particles: ParticleSystem, range: ParticleRange): void {
  if (
    !Number.isInteger(range.start) ||
    !Number.isInteger(range.count) ||
    range.start < 0 ||
    range.count <= 0 ||
    range.start + range.count > particles.capacity
  ) {
    throw new Error(
      `${kernelName}: invalid fluidParticles range start=${range.start} count=${range.count} capacity=${particles.capacity}`,
    );
  }
}
