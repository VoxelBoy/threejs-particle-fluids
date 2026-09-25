import { Fn, If, Return, float, instanceIndex, uint, vec3 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ParticleRange, ParticleSystem } from '../../core/index.js';
import { emitForEachPair } from '../../core/index.js';

import { emitPoly6FromRSq, emitSpikyGrad, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildLambdaKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  /** `ρ_0` — scene rest density, kg/m³. */
  readonly restDensity: UniformNode<'float', number>;
  /** XPBD substep timestep (from `SimLoop.xpbd.dt`). */
  readonly dt: UniformNode<'float', number>;
  /** XPBD compliance `α`. `α̃ = α / dt²` is computed on-GPU per dispatch. */
  readonly compliance: UniformNode<'float', number>;
  /**
   * Per-particle density `ρ_i`, sized to `particles.capacity`. Phase
   * Perf-08 fused the standalone density kernel into this one — the
   * fused kernel writes `density[i]` after the per-pair walk, before
   * the unilateral-clamp early return, so downstream pre-iter consumers
   * (cohesion, adhesion) continue to see the latest iter's value.
   */
  readonly density: StorageBufferNode<'float'>;
  /** λ_i output; sized to `particles.capacity`. */
  readonly lambda: StorageBufferNode<'float'>;
  /** Which particle slots this fluid owns. */
  readonly fluidParticles: ParticleRange;
  /**
   * Per-substep neighbor pair list (Macklin & Müller 2013 Algorithm 1
   * line 6 + §6 paragraph 2). Built once per substep; iterated each
   * lambda iteration in place of the 27-cell hash-grid walk.
   */
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Build the fused per-fluid-particle density + λ kernel.
 *
 *
 * Algorithmic outline (one thread per fluid particle `i = start +
 * instanceIndex`). Phase Perf-08 collapsed the prior `density →
 * lambda` kernel pair into this single fused walk. The per-pair memory
 * loads (`predictedPositions[j]`, `boundaryVolume[j]`, `invMass[j]`)
 * are now amortized across both consumers — F-PG4 measured ~90% of
 * per-particle kernel cost as walk overhead, and this fusion cuts the
 * per-iter walk count from 3 to 2.
 *   1. Iterate the per-substep pair list via {@link emitForEachPair}.
 *      The list contains only pairs with `|r| < h` (filtered at build
 *      time per Macklin & Müller 2013 §6 paragraph 2), so no within-h
 *      check is needed inside the iter loop. For each cached `j`:
 *        - Compute `ψ_j` (Akinci branch above).
 *        - `ρ_i += ψ_j · W_poly6(|r|², h)` (density-side accumulator).
 *        - `g_ij = ∇_{p_i} W_spiky(x_i − x_j, h) · ψ_j` via
 *          {@link emitSpikyGrad} (gradient-side accumulator).
 *        - `gradSum += g_ij` (collects `∇_{p_i} C_i · ρ_0` sans the
 *          `1/ρ_0` factor — applied once after the loop).
 *        - `normSqSum += w_j · |g_ij|²` (collects the mass-weighted
 *          `k = j` terms `w_j · |∇_{p_j} C_i|²`, one per neighbor; the
 *          `k = i` term is `w_i · |gradSum|²` that lands after the loop).
 *      `j == i` falls through naturally: r = 0 → Poly6 contributes
 *      `ψ_i · poly6Coef · h⁶` (the paper-standard self contribution),
 *      and `emitSpikyGrad` returns vec3(0).
 *   2. Write `density[i] = ρ_i`. ALWAYS — performed before the
 *      unilateral-clamp early-return so pre-iter consumers (cohesion,
 *      adhesion) continue to see the latest iter's value at the start
 *      of the next substep (Phase Perf-08 read-contract preservation).
 *   3. C_i = ρ_i/ρ_0 − 1.
 *   4. Unilateral gate (Macklin 2014 §7 eq. 26): if C_i ≤ 0, write λ_i =
 *      0 and early-return. Expansion regions generate no corrective λ.
 *      Sub-rest particles now pay the walk cost; pre-fusion they
 *      short-circuited before the walk. The accepted tradeoff is the
 *      saved memory traffic vs. paying the walk for sub-rest particles
 *      (see plan §Design §Leverage).
 *   5. `sumGradSq = (w_i · |gradSum|² + normSqSum) / ρ_0²` — the full
 *      paper eq. 9 denominator with the Macklin 2016 mass weighting.
 *      `k = j` terms are `|∇_{p_j} C_i|² = |−ψ_j · ∇W(r)/ρ_0|² = ψ_j² ·
 *      |∇W(r)|²/ρ_0²`, i.e. the same magnitude as the `∇_{p_i}` single-
 *      term summand modulo the sign flip (Macklin 2013 eq. 8), so the
 *      flip has no effect on the denominator.
 *   6. `α̃ = α/dt²` computed inline (per-substep dt → per-dispatch α̃).
 *   7. `λ_i = -C_i / (sumGradSq + α̃)`; write to `lambda[i]`.
 *
 * Mass-weight handling. Macklin 2013 eq. 9 / eq. 11 drop the per-neighbor
 * mass factor `m_j` under the paper's "we treat all particles as having
 * equal mass and will drop this term" simplification. Our density sum
 * keeps `ψ_j` (the Akinci 2012 extension carries `ψ = ρ_0 · V` for
 * boundary particles, which is NOT the fluid mass), so to stay
 * dimensionally consistent with the density side we must ALSO keep `ψ_j`
 * inside the gradient accumulator. Concretely:
 *   `∇_{p_i} ρ_i = Σ_j ψ_j · ∇W_ij`,    `∇_{p_j} ρ_i = ψ_j · (−∇W_ij)`.
 * Dropping ψ (as the paper does, under equal-mass) and keeping ψ in the
 * density sum produces a 1/m² discrepancy in the denominator that
 * catastrophically under-scales `Δx` when `m = ρ_0·spacing³ ≪ 1 kg`.
 * Measured 2026-04-23: dropping ψ gives `Δx ≈ 26 μm / iter` at 50%
 * over-compression — 65× less than gravity's displacement per substep —
 * which causes fluid columns to collapse. Keeping ψ recovers
 * `Δx ≈ 1.7 mm / iter` at the same compression, consistent with the
 * equal-mass paper-verbatim form when `ρ_0 · spacing³` is absorbed.
 * See the Phase 08 exit report §Findings for the full derivation.
 *
 * **Mass-weighted denominator (Phase 11, 2026-04-27).** The full
 * XPBD form (Macklin 2016 eq. 17) is
 *   `λ = −C / (∇C · M⁻¹ · ∇C^T + α̃) = −C / (Σ_k w_k · |∇_k C|² + α̃)`
 * where `M⁻¹ = diag(w_1, w_2, …, w_n)`. The previous Phase 08 code
 * dropped the `w_k` factors per Macklin 2013 §3's equal-mass
 * simplification — a constant rescale for fluid-only scenes (every
 * `w_k = w_fluid`), but catastrophically incorrect when boundary
 * particles with `w_b ≠ w_fluid` enter the constraint (Phase 11's
 * fluid → solid Newton-3 reaction; Macklin 2014 §7.1.1 ¶2 explicitly
 * prescribes this fix to keep buoyancy corrections conservative
 * under mass disparity).
 *
 *
 * Implementation: each pair contributes `w_j · |g_j|²` (where `g_j =
 * ψ_j · ∇W_ij`) to `normSqSum`; the `|∇_i C|²` term picks up `w_i`
 * after the loop. Pure-fluid scenes (every `w_k = w_fluid`) see a
 * uniform `w_fluid` rescale of the gradient term, partially absorbed
 * by `α̃`; the net Δp magnitude shifts predictably and Phase 08 + 09
 * tests' compliance values are retuned to match the paper-default
 * regime (see Phase 11 exit report).
 *
 * Dispatch shape: `fluidParticles.count` threads. Gather-mode (each
 * thread writes only `lambda[i]` for its own i), so no atomics required.
 */
export function buildLambdaKernel(args: BuildLambdaKernelArgs): ComputeNode {
  const {
    particles,
    sph,
    restDensity,
    dt,
    compliance,
    density,
    lambda,
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
      `buildLambdaKernel: invalid fluidParticles range start=${fluidParticles.start} count=${fluidParticles.count} capacity=${particles.capacity}`,
    );
  }

  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    // i's own inverse mass — read once outside the loop. Used after the
    // pair-walk to weight the `|∇_i C|²` term in the mass-weighted
    // denominator (Macklin 2016 XPBD eq. 17, M⁻¹ = diag(w_k)).
    const wi: Any = particles.invMass.element(i).toVar();

    // Phase Perf-08 fused accumulators: density (ρ_i, paper eq. 2) +
    // gradient (∇_{p_i} C_i, paper eqs. 7–8) + mass-weighted gradient
    // norm (Macklin 2016 eq. 17 denominator's k=j sum). One pair walk,
    // three accumulators, two writes after the loop.
    const rho: Any = float(0.0).toVar();
    const gradSum: Any = vec3(0, 0, 0).toVar();
    const normSqSum: Any = float(0.0).toVar();

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        const xj: Any = particles.predictedPositions.element(j).xyz;
        const diff: Any = xi.sub(xj).toVar();
        const rSq: Any = diff.dot(diff).toVar();
        // `j == i` falls through naturally: r = 0 → `emitSpikyGrad`
        // returns vec3(0) and Poly6 contributes the paper-standard
        // self-density term `ψ_i · poly6Coef · h⁶`. The pair list is
        // already filtered to `|r|² < h²` at build time.

        // Akinci ψ_j: shared branch for both density and gradient sides.
        // Keeping ψ in the gradient sum preserves dimensional consistency
        // with the density side — see class-level docstring "Mass-weight
        // handling".
        const Vj: Any = particles.boundaryVolume.element(j).toVar();
        const wj: Any = particles.invMass.element(j).toVar();
        const isBoundary: Any = Vj.greaterThan(float(0.0));
        const psiBoundary: Any = (restDensity as Any).mul(Vj);
        const hasMass: Any = wj.greaterThan(float(0.0));
        const massJ: Any = hasMass.select(float(1.0).div(wj), float(0.0));
        const psiJ: Any = isBoundary.select(psiBoundary, massJ);

        // Density side: ρ_i += ψ_j · W_poly6(|r|², h). Byte-for-byte
        // identical to the prior `buildDensityKernel` body before this
        // fusion (Phase Perf-08).
        const w: Any = emitPoly6FromRSq(rSq, sph);
        rho.addAssign(psiJ.mul(w));

        // Gradient side: g_ij = ψ_j · ∇W_spiky(x_i − x_j, h). Mass-
        // weighted XPBD denominator: each `|∇_j C|²` term is weighted
        // by w_j (Macklin 2016 eq. 17). For fluid neighbours w_j =
        // w_fluid (constant); for boundary neighbours w_j is the
        // boundary's actual inverse mass — proper weighting is what
        // makes Phase 11's buoyancy stable under mass disparity per
        // Macklin 2014 §7.1.1 ¶2 (see class-level docstring).
        const g: Any = emitSpikyGrad(diff, sph).mul(psiJ).toVar();
        gradSum.addAssign(g);
        normSqSum.addAssign(g.dot(g).mul(wj));
      },
    });

    // Density write — ALWAYS, before the unilateral-clamp early return.
    // Pre-iter consumers (`cohesion.ts`, `adhesion.ts`) read the latest
    // iter's `density[i]` at the start of the next substep; the early-
    // return path on `Ci ≤ 0` must still publish the freshly-computed
    // value. See class-level docstring algorithmic outline step 2.
    density.element(i).assign(rho);

    // Paper eq. 1: C_i = ρ_i/ρ_0 − 1. Unilateral clamp (Macklin 2014 §7
    // eq. 26): fluids resist compression only, so `C_i ≤ 0` yields zero λ.
    const invRho0: Any = float(1.0)
      .div(restDensity as Any)
      .toVar();
    const Ci: Any = rho.mul(invRho0).sub(float(1.0)).toVar();
    If(Ci.lessThanEqual(float(0.0)), () => {
      lambda.element(i).assign(float(0.0));
      Return();
    });

    // Mass-weighted denominator: `w_i · |∇_i C|² + Σ_j w_j · |∇_j C|²`.
    // With the `1/ρ_0` factor that scales every gradient, the
    // physical denominator is `(w_i · |gradSum|² + normSqSum) / ρ_0²`.
    // Factor out `1/ρ_0²` to one mul.
    const sumGradSq: Any = gradSum.dot(gradSum).mul(wi).add(normSqSum).toVar();
    const invRho0Sq: Any = invRho0.mul(invRho0);
    const denomPhys: Any = sumGradSq.mul(invRho0Sq);

    // XPBD mapping: α̃ = α / dt² replaces Macklin 2013 eq. 11's ε. α is a
    // scene-constant compliance uniform; dt comes from the substep-level
    // `SimLoop.xpbd.dt` that every material shares.
    const invDtSq: Any = float(1.0).div((dt as Any).mul(dt as Any));
    const alphaTilde: Any = (compliance as Any).mul(invDtSq);

    const denom: Any = denomPhys.add(alphaTilde).toVar();
    // `denom` is bounded below by `α̃ > 0` (plan asserts compliance > 0 at
    // FluidSystem construction; if α̃ = 0 the paper's eq. 9 is recovered
    // and a particle with a singleton neighborhood can produce a
    // divergent λ — we log this case via `compliance >= 0` check and a
    // diagnostic, but guard in-kernel with a small epsilon below to stop
    // the shader from producing `inf` while the scene-tuning pass
    // diagnoses it on CPU).
    const denomSafe: Any = denom.max(float(1e-20));
    const lambdaI: Any = Ci.negate().div(denomSafe);
    lambda.element(i).assign(lambdaI);
  })().compute(fluidParticles.count);
}
