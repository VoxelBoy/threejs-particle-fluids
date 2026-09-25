import { Fn, float, instanceIndex, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ParticleRange, ParticleSystem } from '../../core/index.js';
import { emitForEachPair } from '../../core/index.js';

import { emitSpikyGrad, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildPositionDeltaKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  /** `ρ_0` — rest density, kg/m³. */
  readonly restDensity: UniformNode<'float', number>;
  /** λ_i buffer, populated by {@link buildLambdaKernel}. */
  readonly lambda: StorageBufferNode<'float'>;
  /**
   * Per-particle Δx output, sized to `particles.capacity`. Written by
   * this kernel, read + applied to `predictedPositions` by a subsequent
   * apply-kernel dispatch. Two-phase to avoid the read-modify-write race
   * on `predictedPositions` within a single dispatch.
   */
  readonly deltaX: StorageBufferNode<'vec4'>;
  readonly fluidParticles: ParticleRange;
  /**
   * Per-substep neighbor pair list (Macklin & Müller 2013 Algorithm 1
   * line 6 + §6 paragraph 2). Built once per substep; iterated each
   * positionDelta iteration in place of the 27-cell hash-grid walk.
   */
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Build the per-fluid-particle gather-mode position-delta kernel (Phase 1
 * of the paper's two-pass Δp update).
 *
 *

 *
 * Boundary neighbors contribute `λ_j = 0` (their slot in `lambda` is
 * never written). Self (`j == i`) contributes zero via the `r = 0`
 * zero-gradient convention inside `emitSpikyGrad`.
 *
 * **Mass-weighted Δp (Phase 11, 2026-04-27).** The full XPBD update
 * (Macklin 2016 eq. 17) is `Δx_l = w_l · ∇_l C · Δλ`. Phase 08 dropped
 * the `w_i` per Macklin 2013's equal-mass simplification — fine for
 * fluid-only scenes (constant `w_fluid`, absorbed into a uniform
 * rescale) but incorrect once Phase 11's `solidReaction.ts` scatters
 * `Δp_j = w_j · …` onto boundary particles with `w_j ≠ w_fluid`. Per-
 * pair Newton-3 conservation requires both sides multiply by their
 * respective `w`; without `w_i` here the equal-mass-simplification
 * residual on fluid-boundary pairs scales as `(m_i − 1) · …` per
 * pair, which the bunny-swimming `bunnyInvMass = 100` regression
 * exposed at ~5 % of |Δp_j|. Multiplying the final `dx` by `w_i`
 * restores per-pair conservation (verified at FP-noise on the m = 1
 * unit test) and pairs cleanly with the matching `w_j` factor in
 * `solidReaction.ts`.
 *
 */
export function buildPositionDeltaKernel(args: BuildPositionDeltaKernelArgs): ComputeNode {
  const { particles, sph, restDensity, lambda, deltaX, fluidParticles, pairList, pairCount } = args;

  if (
    !Number.isInteger(fluidParticles.start) ||
    !Number.isInteger(fluidParticles.count) ||
    fluidParticles.start < 0 ||
    fluidParticles.count <= 0 ||
    fluidParticles.start + fluidParticles.count > particles.capacity
  ) {
    throw new Error(
      `buildPositionDeltaKernel: invalid fluidParticles range start=${fluidParticles.start} count=${fluidParticles.count} capacity=${particles.capacity}`,
    );
  }

  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const lambdaI: Any = lambda.element(i).toVar();
    // i's own inverse mass — read once outside the loop. Used after the
    // pair-walk to mass-weight the final Δp per the XPBD form
    // `Δx_i = w_i · ∇_i C · Δλ` (Macklin 2016 eq. 17). See class-level
    // docstring "Mass-weighted Δp".
    const wi: Any = particles.invMass.element(i).toVar();
    const accum: Any = vec3(0, 0, 0).toVar();

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        const xj: Any = particles.predictedPositions.element(j).xyz;
        const diff: Any = xi.sub(xj).toVar();
        // Pair list is filtered to `|r|² < h²` at build time. Self
        // (`j == i`) contributes zero via the `r = 0` zero-gradient
        // convention inside `emitSpikyGrad`.

        // Akinci ψ_j — matches the density / λ kernels. See
        // `lambda.ts` class-level docstring for the dimensional-
        // consistency derivation that justifies keeping ψ here.
        const Vj: Any = particles.boundaryVolume.element(j).toVar();
        const wj: Any = particles.invMass.element(j).toVar();
        const isBoundary: Any = Vj.greaterThan(float(0.0));
        const psiBoundary: Any = (restDensity as Any).mul(Vj);
        const hasMass: Any = wj.greaterThan(float(0.0));
        const massJ: Any = hasMass.select(float(1.0).div(wj), float(0.0));
        const psiJ: Any = isBoundary.select(psiBoundary, massJ);

        const lambdaJ: Any = lambda.element(j);
        const coef: Any = lambdaI.add(lambdaJ).mul(psiJ);
        const g: Any = emitSpikyGrad(diff, sph);
        accum.addAssign(g.mul(coef));
      },
    });

    const invRho0: Any = float(1.0).div(restDensity as Any);
    // Mass-weighted XPBD update: `Δx_i = w_i · (1/ρ_0) · accum`. The
    // matching `w_j` factor lives in `solidReaction.ts` for the
    // fluid → solid Newton-3 reaction; for fluid-fluid pairs (constant
    // w_fluid) this is a uniform rescale that absorbs into α̃-tuning.
    const dx: Any = accum.mul(invRho0).mul(wi);
    deltaX.element(i).assign(vec4(dx, float(0.0)));
  })().compute(fluidParticles.count);
}

export interface BuildApplyDeltaKernelArgs {
  readonly particles: ParticleSystem;
  readonly deltaX: StorageBufferNode<'vec4'>;
  readonly fluidParticles: ParticleRange;
}

/**
 * Phase 2 of the paper's two-pass Δp update: add each fluid particle's
 * precomputed `Δx` into its `predictedPositions`. One dispatch per iter,
 * runs directly after {@link buildPositionDeltaKernel}.
 *
 * Correctness note: this kernel IS a safe read-modify-write on
 * `predictedPositions` — each thread touches only its own slot, and
 * no other thread's read depends on this thread's write within this
 * dispatch.
 */
export function buildApplyDeltaKernel(args: BuildApplyDeltaKernelArgs): ComputeNode {
  const { particles, deltaX, fluidParticles } = args;

  if (
    !Number.isInteger(fluidParticles.start) ||
    !Number.isInteger(fluidParticles.count) ||
    fluidParticles.start < 0 ||
    fluidParticles.count <= 0 ||
    fluidParticles.start + fluidParticles.count > particles.capacity
  ) {
    throw new Error(
      `buildApplyDeltaKernel: invalid fluidParticles range start=${fluidParticles.start} count=${fluidParticles.count} capacity=${particles.capacity}`,
    );
  }

  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const existing: Any = particles.predictedPositions.element(i).toVar();
    const dx: Any = deltaX.element(i).xyz;
    const newXyz: Any = existing.xyz.add(dx);
    particles.predictedPositions.element(i).assign(vec4(newXyz, existing.w));
  })().compute(fluidParticles.count);
}
