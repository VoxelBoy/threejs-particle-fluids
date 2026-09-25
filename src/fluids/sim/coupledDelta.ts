import { Fn, If, float, instanceIndex, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ContactAccumulator, ParticleRange, ParticleSystem } from '../../core/index.js';
import { emitAccumulateDelta, emitForEachPair } from '../../core/index.js';

import { emitSpikyGrad, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildCoupledDeltaKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  readonly restDensity: UniformNode<'float', number>;
  readonly lambda: StorageBufferNode<'float'>;
  readonly deltaX: StorageBufferNode<'vec4'>;
  readonly contactAccumulator: ContactAccumulator;
  readonly fluidParticles: ParticleRange;
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Phase Perf-15 — fused position-delta + solid-reaction-scatter kernel.
 *
 * Combines two consecutive per-iter kernels that both walk the same
 * per-fluid pair list:
 *   - {@link buildPositionDeltaKernel} (`src/fluids/sim/positionDelta.ts`)
 *     — gather mode: computes `Δp_i = w_i · (1/ρ_0) · Σ_j (λ_i + λ_j) · ψ_j ·
 *     ∇W(x_i − x_j)` for fluid `i`, writes `deltaX[i]` (paper Macklin & Müller
 *     2013 §3 eq. 12 with mass-weighted XPBD form per Macklin 2016 eq. 17).
 *   - {@link buildSolidReactionScatterKernel} (`src/fluids/sim/solidReaction.ts`)
 *     — scatter mode: for each boundary neighbour `j` of fluid `i`, scatters
 *     `Δp_j = − w_j · (λ_i / ρ_0) · ψ_j · ∇W(x_i − x_j)` into the shared
 *     {@link ContactAccumulator} (paper Macklin 2014 §7.1.1 mass-weighted PBD
 *     reaction; Akinci 2012 §2.2 eq. 5 boundary ψ).
 *
 * Both passes consume `xi`, `λ_i`, and per-pair `xj`, `V_j`, `w_j`, `λ_j`.
 * Both compute `∇W(x_i − x_j)` per pair. Fusing collapses two passes into
 * one — eliminates one inter-pass barrier (~100–300 µs / dispatch on Apple
 * Silicon Metal-3, × 6 dispatches/frame at S=3 / I=2 = 0.6–1.8 ms / frame),
 * one pipeline-state-object switch, one encoder pass, and the redundant
 * per-pair fetches.
 *
 *
 * **Per-pair body.** Mirrors the legacy two-kernel decomposition exactly:
 *   1. Read `xj`, `V_j`, `w_j`, `λ_j`.
 *   2. ∇W = emitSpikyGrad(xi − xj).
 *   3. **Always** (gather): accumulate `(λ_i + λ_j) · ψ_j · ∇W` into a
 *      thread-local sum. `ψ_j = ρ_0 · V_j` for boundary, `ψ_j = 1/w_j`
 *      for fluid (mirrors `positionDelta.ts:130–134`).
 *   4. **Conditional** (scatter): if `V_j > 0` and `w_j > 0` and `λ_i ≠ 0`,
 *      atomically scatter `Δp_j = − w_j · (λ_i / ρ_0) · (ρ_0 · V_j) · ∇W`
 *      into the boundary's accumulator slot. Mirrors
 *      `solidReaction.ts:222–243`.
 *
 * After the walk, write `deltaX[i] = w_i · accum / ρ_0` (mirrors
 * `positionDelta.ts:140`).
 *
 * **Why the unilateral-clamp early-return moves from the kernel level to
 * the per-pair scatter branch**: today's `solidReactionScatter` does
 * `If(λ_i == 0) Return()` at kernel entry (paper §7 eq. 26). The fused
 * kernel can't `Return()` because the gather term still needs to fire
 * (gather contribution survives even when `λ_i = 0`, because `λ_j` may
 * be non-zero). The scatter branch instead conditions on `λ_i ≠ 0` per
 * pair. The total scatter work in the sub-rest case is then one V_j
 * read + one w_j read + one λ_i compare-against-0 per pair — same as
 * today's early-return path.
 *
 * **Determinism.** The fluid-side reduction is a per-thread local f32 sum
 * over a deterministic pair-list iteration order — G4 tier-2 bounded-
 * error (matches today's `positionDelta`). The boundary-side scatter
 * is i32 atomicAdd — G4 tier-1 bit-deterministic across thread orderings
 * (matches today's `solidReactionScatter`). Fusion preserves both tiers
 * by construction; same `emitAccumulateDelta` + same per-thread reduction.
 */
export function buildCoupledDeltaKernel(args: BuildCoupledDeltaKernelArgs): ComputeNode {
  const {
    particles,
    sph,
    restDensity,
    lambda,
    deltaX,
    contactAccumulator,
    fluidParticles,
    pairList,
    pairCount,
  } = args;

  if (
    !Number.isInteger(fluidParticles.start) ||
    !Number.isInteger(fluidParticles.count) ||
    fluidParticles.start < 0 ||
    fluidParticles.count <= 0 ||
    fluidParticles.start + fluidParticles.count > particles.capacity
  ) {
    throw new Error(
      `buildCoupledDeltaKernel: invalid fluidParticles range start=${fluidParticles.start} count=${fluidParticles.count} capacity=${particles.capacity}`,
    );
  }
  if (contactAccumulator.particles !== particles) {
    throw new Error(
      'buildCoupledDeltaKernel: contactAccumulator must wrap the same ParticleSystem as fluidParticles',
    );
  }

  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const lambdaI: Any = lambda.element(i).toVar();
    const wi: Any = particles.invMass.element(i).toVar();

    // Per-thread scalar coefficient for the boundary scatter — `−(λ_i / ρ_0)`
    // is constant across every neighbour. Factored out per the original
    // `solidReaction.ts:215` optimisation.
    const invRho0: Any = float(1.0).div(restDensity as Any);
    const lambdaOverRho0Neg: Any = lambdaI.mul(invRho0).negate().toVar();
    // Unilateral-clamp gate (paper §7 eq. 26). Mirrors `solidReaction.ts:204`
    // — a sub-rest fluid particle (`λ_i = 0`) produces no reaction.
    const scatterEnabled: Any = lambdaI.notEqual(float(0.0)).toVar();

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
        const Vj: Any = particles.boundaryVolume.element(j).toVar();
        const wj: Any = particles.invMass.element(j).toVar();
        const lambdaJ: Any = lambda.element(j);

        // ψ_j resolution. Verbatim from `positionDelta.ts:130–134`.
        const isBoundary: Any = Vj.greaterThan(float(0.0));
        const psiBoundary: Any = (restDensity as Any).mul(Vj);
        const hasMassJ: Any = wj.greaterThan(float(0.0));
        const massJ: Any = hasMassJ.select(float(1.0).div(wj), float(0.0));
        const psiJ: Any = isBoundary.select(psiBoundary, massJ);

        // Spiky gradient — the same ∇W both legacy kernels compute.
        const gradW: Any = emitSpikyGrad(diff, sph);

        // Gather term: `(λ_i + λ_j) · ψ_j · ∇W` accumulates into the
        // per-thread `accum` for the per-`i` write below. Verbatim from
        // `positionDelta.ts:135–138`.
        const coefGather: Any = lambdaI.add(lambdaJ).mul(psiJ);
        accum.addAssign(gradW.mul(coefGather));

        // Scatter term: only fires for movable boundary neighbours when
        // the unilateral clamp permits it. Verbatim from
        // `solidReaction.ts:222–243` (with `psiBoundary` replacing the
        // re-computed `ρ_0 · V_j`, and the `λ_i ≠ 0` early-return moved
        // from kernel entry to per-pair condition — see kernel docstring
        // "Why the unilateral-clamp early-return moves").
        const scatterFires: Any = isBoundary.and(hasMassJ).and(scatterEnabled);
        If(scatterFires, () => {
          const dxJ: Any = gradW.mul(lambdaOverRho0Neg).mul(psiBoundary).mul(wj);
          emitAccumulateDelta(contactAccumulator, j, dxJ);
        });
      },
    });

    // `Δx_i = w_i · (1/ρ_0) · accum`. Verbatim from `positionDelta.ts:140`.
    const dx: Any = accum.mul(invRho0).mul(wi);
    deltaX.element(i).assign(vec4(dx, float(0.0)));
  })().compute(fluidParticles.count);
}
