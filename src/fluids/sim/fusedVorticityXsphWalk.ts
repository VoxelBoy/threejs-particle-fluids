import { Fn, float, instanceIndex, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ParticleRange, ParticleSystem } from '../../core/index.js';
import { emitForEachPair } from '../../core/index.js';

import { emitPoly6FromRSq, emitSpikyGrad, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildFusedVorticityXsphWalkKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  /** Per-fluid-particle `V = m/ρ_0`. See `FluidSystem.particleVolumeUniform`. */
  readonly particleVolume: UniformNode<'float', number>;
  /** XSPH mixing coefficient `c`. Paper-typical `0.01`. */
  readonly c: UniformNode<'float', number>;
  /** Vorticity output buffer (vec3 packed into vec4 with a zero `.w`). */
  readonly omega: StorageBufferNode<'vec4'>;
  /** Vorticity-magnitude output, consumed by Pass 2 for `η = ∇|ω|`. */
  readonly omegaMag: StorageBufferNode<'float'>;
  /** XSPH velocity-correction output, consumed by the fused tail. */
  readonly xsphDeltaV: StorageBufferNode<'vec4'>;
  readonly fluidParticles: ParticleRange;

  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Fused vorticity-Pass-1 + XSPH-compute walk — one pair iteration that
 * accumulates both the vorticity vector `ω_i` (with magnitude) and the
 * XSPH velocity correction `Δv_i` from a single read of each pair's
 * `positions[j]`, `velocities[j]`, and `boundaryVolume[j]`.
 *
 *
 * Per-pair math (unchanged from each existing kernel; see their JSDocs
 * for derivation and dimensional commentary):
 *
 *   `ω_i = Σ_j V_j · (v_i − v_j) × ∇_{p_i} W_spiky(p_i − p_j, h)`
 *     — Macklin 2013 §5 eq. 15, Monaghan-canonical form.
 *
 *   `Δv_i = c · Σ_j V_j · (v_j − v_i) · W_poly6(|p_i − p_j|, h)`
 *     — Macklin 2013 §5 eq. 17, Monaghan-canonical form.
 *
 * `V_j = boundaryVolume[j] > 0 ? boundaryVolume[j] : particleVolume` —
 * Akinci 2012 boundary-volume gate (Phase 08 Finding 8).
 *
 * Sign-convention note: vorticity uses `(v_i − v_j)` × ∇_{p_i}W per the
 * `vorticity.ts` file-level docstring's two-flip identity proof; xsph
 * uses `(v_j − v_i)` per its existing kernel — the two `vij` orientations
 * are correct as-is and are kept independent (no shared variable).
 *
 * Substep-internal-ordering deviation from paper §5: paper sequences
 * vorticity-confinement-then-XSPH; this fused walk computes both halves
 * from a common pre-confinement velocity snapshot, so XSPH reads the
 * pre-confinement field instead of the post-confinement field. Bounded
 * at `5 · ε · c · dt` per substep by `tests/analytical/fluids/vorticity-
 * xsph-order-bound.gpu.test.ts`. See plan §"Substep-internal ordering"
 * for the linear-coupling analysis.
 *
 * Pair-list iteration: pairs were filtered to `|r|² < h²` at build time,
 * so no within-h check is needed here. Both Poly6 and Spiky self-clamp
 * to zero outside `h`.
 *
 * Gather mode — each thread writes only its own `omega[i]`, `omegaMag[i]`,
 * and `xsphDeltaV[i]`. No atomics, Tier-1 bit-exact preserved.
 */
export function buildFusedVorticityXsphWalkKernel(
  args: BuildFusedVorticityXsphWalkKernelArgs,
): ComputeNode {
  const {
    particles,
    sph,
    particleVolume,
    c,
    omega,
    omegaMag,
    xsphDeltaV,
    fluidParticles,
    pairList,
    pairCount,
  } = args;
  validateRange('buildFusedVorticityXsphWalkKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.positions.element(i).xyz.toVar();
    const vi: Any = particles.velocities.element(i).xyz.toVar();
    const omegaSum: Any = vec3(0, 0, 0).toVar();
    const xsphSum: Any = vec3(0, 0, 0).toVar();

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        // Shared reads — each fetched once per pair.
        const xj: Any = particles.positions.element(j).xyz;
        const vj: Any = particles.velocities.element(j).xyz;
        const Vj_boundary: Any = particles.boundaryVolume.element(j);

        const diff: Any = xi.sub(xj).toVar();
        const isBoundary: Any = Vj_boundary.greaterThan(float(0.0));
        const Vj: Any = isBoundary.select(Vj_boundary, particleVolume as Any);

        // Vorticity Pass 1 half — `(v_i − v_j) × ∇_{p_i}W_spiky · V_j`.
        const g: Any = emitSpikyGrad(diff, sph);
        const vij_vorticity: Any = vi.sub(vj);
        omegaSum.addAssign(vij_vorticity.cross(g).mul(Vj));

        // XSPH half — `(v_j − v_i) · W_poly6 · V_j`.
        const rSq: Any = diff.dot(diff);
        const w: Any = emitPoly6FromRSq(rSq, sph);
        const vij_xsph: Any = vj.sub(vi);
        xsphSum.addAssign(vij_xsph.mul(w).mul(Vj));
      },
    });

    omega.element(i).assign(vec4(omegaSum, float(0.0)));
    omegaMag.element(i).assign(omegaSum.length());
    xsphDeltaV.element(i).assign(vec4(xsphSum.mul(c as Any), float(0.0)));
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
