import {
  Continue,
  Fn,
  If,
  atomicLoad,
  atomicStore,
  float,
  instanceIndex,
  int,
  uint,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ParticleRange, ParticleSystem, VelocityAccumulator } from '../../core/index.js';
import { emitAccumulateVelocityDelta, emitForEachPair } from '../../core/index.js';

import { emitSpikyGrad, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/*
 * Akinci, Akinci, Teschner 2013 "Versatile Surface Tension and Adhesion
 * for SPH Fluids" §2 — cohesion + surface-area-minimization (curvature)
 * forces.
 *
 * **Paper §2.3 + §4 mandate per-pair scatter, NOT per-particle gather.**
 * §2.3 last paragraph: "Note that the terms in (5) are fully symmetrized,
 * and the total force is applied to the particle pairs. This is,
 * however, not the case in previous surface area minimization techniques
 * […], as they apply the forces to the particles as external forces."
 * §4 first sentence repeats: "…the forces are computed from the
 * particles and are directly applied to the neighboring pairs."
 *
 * Phase 09 bring-up (U-33 archive) confirmed that the gather / external-
 * force pattern paper rejects really does break momentum conservation
 * once the `K_ij` correction factor (eq. 4) is in the formula — only
 * 22/512 particles ended with non-zero impulse on the cube diagnostic
 * and `Σ impulse ≈ (−1.33, −1.33, −1.33)` instead of FP noise. Paper's
 * scatter is the only architecture that conserves momentum by
 * construction; each pair (i, j) produces one `F_st` that is applied as
 * `+F` to i and `−F` to j in one atomic step, so Newton-3 is enforced
 * independently of thread timing.
 *
 * Pipeline (inside FluidSystem.preIterKernels, in order):
 *   1. Reset velocity accumulator (zero the per-fluid-particle Δv ticks).
 *   2. `buildColorFieldNormalKernel` — gather, writes n_i. Not a pair
 *      force, gather is correct here.
 *   3. `buildSurfaceTensionScatterKernel` — per-pair scatter. One thread
 *      per fluid particle i; visits neighbours j with j > i (canonical
 *      ordering, prevents double-application), computes `F^{st}_{i←j}`
 *      once, atomically accumulates `+Δv = (F/m)·Δt` into slot i and
 *      `−Δv` into slot j via `emitAccumulateVelocityDelta`.
 *   4. `buildApplyVelocityImpulseKernel` — per fluid particle, reads
 *      accumulated Δv, writes `velocities += Δv` AND `predictedPositions
 *      += Δv·Δt`. The x\* update is the paper Alg. 1 line 3 term
 *      equivalent to integrating F_st into predict.
 *
 * Only one accumulator is needed. Δx\*_i = Δv_i · Δt is computed per-
 * particle at apply time (not per-pair at scatter time) because Δt is a
 * uniform — no per-pair data. Halves the atomic op count vs a dual
 * position+velocity accumulator design.
 *
 * **Kernel choice for the color-field gradient (∇W).** Paper uses
 * Monaghan 2005 cubic spline; this module uses Müller 2003 Spiky (Phase
 * 08 convention). The cohesion/curvature combination drives `n_i − n_j`,
 * so a *consistent* kernel choice across particles is what matters;
 * absolute magnitude is absorbed by γ. Flagged **Assumed** at M-2.
 */

export interface CohesionUniforms {
  readonly cohesionCoef: UniformNode<'float', number>;
  readonly cohesionOffset: UniformNode<'float', number>;
  readonly gamma: UniformNode<'float', number>;
  readonly setCohesionH: (newH: number) => void;
  readonly setGamma: (newGamma: number) => void;
}

function cohesionCoefFor(h: number): number {
  return 32 / (Math.PI * Math.pow(h, 9));
}

function cohesionOffsetFor(h: number): number {
  return Math.pow(h, 6) / 64;
}

export function createCohesionUniforms(h: number, gamma: number): CohesionUniforms {
  if (!Number.isFinite(h) || h <= 0) {
    throw new Error(`createCohesionUniforms: h must be a positive finite number, got ${h}`);
  }
  if (!Number.isFinite(gamma)) {
    throw new Error(`createCohesionUniforms: gamma must be a finite number, got ${gamma}`);
  }
  const coefU = uniform(cohesionCoefFor(h), 'float');
  const offsetU = uniform(cohesionOffsetFor(h), 'float');
  const gammaU = uniform(gamma, 'float');
  return {
    cohesionCoef: coefU,
    cohesionOffset: offsetU,
    gamma: gammaU,
    setCohesionH(newH: number): void {
      if (!Number.isFinite(newH) || newH <= 0) {
        throw new Error(`setCohesionH: h must be a positive finite number, got ${newH}`);
      }
      (coefU as Any).value = cohesionCoefFor(newH);
      (offsetU as Any).value = cohesionOffsetFor(newH);
    },
    setGamma(newGamma: number): void {
      if (!Number.isFinite(newGamma)) {
        throw new Error(`setGamma: gamma must be a finite number, got ${newGamma}`);
      }
      (gammaU as Any).value = newGamma;
    },
  };
}

/**
 * Emit Akinci 2013 eq. 2 cohesion spline `C(r)` as a scalar TSL node.
 *
 * `C(r) = (32 / (π · h⁹)) · { (h − r)³ · r³              for 2r > h ∧ r ≤ h
 *                             2·(h − r)³·r³ − h⁶ / 64    for r > 0 ∧ 2r ≤ h
 *                             0                          otherwise }`
 *
 * Shape (paper Figure 3): attractive peak at r = h/2, zero at r = h,
 * negative (repulsive) below r ≈ 0.27·h, clamped close-range so the
 * force doesn't blow up at small separations (unlike a pure Lennard-
 * Jones attractor).
 */
function emitCohesionSpline(r: Any, sph: SphKernelUniforms, coh: CohesionUniforms): Any {
  const hMinusR: Any = (sph.h as Any).sub(r).max(float(0.0));
  const hmr3: Any = hMinusR.mul(hMinusR).mul(hMinusR);
  const r3: Any = r.mul(r).mul(r);
  const shared: Any = hmr3.mul(r3);
  const twoR: Any = r.mul(float(2.0));
  const useFirst: Any = twoR.greaterThan(sph.h as Any);
  const firstBranch: Any = shared;
  const secondBranch: Any = shared.mul(float(2.0)).sub(coh.cohesionOffset as Any);
  const value: Any = useFirst.select(firstBranch, secondBranch);
  const inRange: Any = r.greaterThan(float(0.0)).and(r.lessThanEqual(sph.h as Any));
  return inRange.select(value.mul(coh.cohesionCoef as Any), float(0.0));
}

export interface BuildColorFieldNormalKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  readonly particleVolume: UniformNode<'float', number>;
  readonly colorFieldNormal: StorageBufferNode<'vec4'>;
  readonly fluidParticles: ParticleRange;
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Pass 1 — per-fluid-particle smoothed color-field gradient `n_i`
 * (Akinci 2013 §2.2, unnumbered eq. just above eq. 3; Macklin & Müller
 * 2013 Algorithm 1 line 6 amortized neighbour iteration).
 *
 * `n_i = h · Σ_j (m_j / ρ_j) · ∇W(|x_i − x_j|, h)` — the per-particle
 * surface-normal indicator used by eq. 3. Gather-mode per fluid
 * particle; each thread writes only its own `colorFieldNormal[i]`. Not
 * a pair force, so gather is correct (Newton-3 doesn't apply to
 * per-particle state like n_i).
 *
 * Reads positions from `particles.predictedPositions` to match the
 * pair list's build-time read source (`buildPairListKernel` walks
 * `predictedPositions`). The un-migrated `emitForEachNeighbor` version
 * read `particles.positions` because the hash grid is indexed on
 * `positions` and predicted-positions queries broke pair-symmetric
 * finding in the 27-cell walk; the pair list sidesteps that — `j ∈
 * pair-list(i) ⟺ |x*_i − x*_j| < h ⟺ i ∈ pair-list(j)` by construction.
 * The within-`h` cutoff is applied at pair-list build time, so this
 * kernel drops the redundant `rSq ≥ hSq` early-exit.
 */
export function buildColorFieldNormalKernel(args: BuildColorFieldNormalKernelArgs): ComputeNode {
  const { particles, sph, particleVolume, colorFieldNormal, fluidParticles, pairList, pairCount } =
    args;
  validateRange('buildColorFieldNormalKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const nAccum: Any = vec3(0, 0, 0).toVar();

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        const xj: Any = particles.predictedPositions.element(j).xyz;
        const diff: Any = xi.sub(xj);
        const Vj_boundary: Any = particles.boundaryVolume.element(j);
        const isBoundary: Any = Vj_boundary.greaterThan(float(0.0));
        const Vj: Any = isBoundary.select(Vj_boundary, particleVolume as Any);
        const g: Any = emitSpikyGrad(diff, sph);
        nAccum.addAssign(g.mul(Vj));
      },
    });

    const nI: Any = nAccum.mul(sph.h as Any);
    colorFieldNormal.element(i).assign(vec4(nI, float(0.0)));
  })().compute(fluidParticles.count);
}

export interface BuildSurfaceTensionScatterKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  readonly coh: CohesionUniforms;
  readonly restDensity: UniformNode<'float', number>;
  readonly mass: UniformNode<'float', number>;
  readonly density: StorageBufferNode<'float'>;
  readonly colorFieldNormal: StorageBufferNode<'vec4'>;
  readonly dt: UniformNode<'float', number>;
  readonly velocityAccumulator: VelocityAccumulator;
  readonly fluidParticles: ParticleRange;
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Pass 2 — per-pair scatter of the combined surface-tension force
 * (Akinci 2013 eqs. 1 + 3 + 4 + 5; Macklin & Müller 2013 Algorithm 1
 * line 6 amortized neighbour iteration).
 *
 * Paper references (Verified 2026-04-23, M-2 checkpoint):
 *   eq. 1: `F^{cohesion}_{i←j} = −γ · m_i · m_j · C(|x_i − x_j|) · r̂`
 *   eq. 3: `F^{curvature}_{i←j} = −γ · m_i · (n_i − n_j)`
 *   eq. 4: `K_{ij} = 2·ρ_0 / (ρ_i + ρ_j)`
 *   eq. 5: `F^{st}_{i←j} = K_{ij} · (F^{cohesion}_{i←j} + F^{curvature}_{i←j})`
 *
 * Per-pair scatter loop:
 *   for each fluid particle i:
 *     for each neighbour j in the pair list with `j > i` (canonical ordering):
 *       compute F_{i←j} (above)
 *       Δv = F_{i←j} · Δt / m_i   (equal-mass fluid; see note below)
 *       atomically: velAcc[i] += Δv, velAcc[j] -= Δv
 *
 * Canonical ordering `j > i` ensures each pair is visited exactly once
 * across all threads; without it, a pair `(i, j)` with `j > i` would be
 * processed by thread i AND by thread j, doubling every force. The pair
 * list's symmetry contract (`j ∈ pair-list(i) ⟺ i ∈ pair-list(j)`)
 * guarantees the smaller-index thread processes each pair and the
 * larger-index thread skips it.
 *
 * Reads positions from `particles.predictedPositions` to match the
 * pair list's build-time read source. Within-h cutoff is applied at
 * pair-list build time, so the redundant `rSq ≥ hSq` early-exit is
 * dropped here. The `r < 1e-20` self-skip is preserved — it guards
 * `1/r` in `rHat = diff/r` and is independent of the within-h check.
 *
 * Equal-mass note: for v1 MVP all fluid particles have identical mass
 * (`m = ρ_0 · spacing³`). The scatter uses `m = mass` for both the
 * `+Δv` side (dividing F_{i←j} by m_i) AND the `−Δv` side (dividing by
 * m_j). Unequal-mass fluid is out of scope; the paper's eq. 3 itself
 * breaks Newton-3 for unequal masses (see U-33 resolution notes) so it
 * would require a reformulation anyway.
 *
 * Boundary exclusion: cohesion + curvature are fluid-fluid only;
 * adhesion handles fluid-boundary (separate kernel). Skip neighbour j
 * when `boundaryVolume[j] > 0`.
 */
export function buildSurfaceTensionScatterKernel(
  args: BuildSurfaceTensionScatterKernelArgs,
): ComputeNode {
  const {
    particles,
    sph,
    coh,
    restDensity,
    mass,
    density,
    colorFieldNormal,
    dt,
    velocityAccumulator,
    fluidParticles,
    pairList,
    pairCount,
  } = args;
  validateRange('buildSurfaceTensionScatterKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const ni: Any = colorFieldNormal.element(i).xyz.toVar();
    const rhoI: Any = density.element(i).toVar();
    const twoRho0: Any = (restDensity as Any).mul(float(2.0));
    const invM: Any = float(1.0).div(mass as Any);

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        // Canonical pair ordering: only process pairs where j > i, so
        // each pair is visited exactly once across all threads. Without
        // this, both thread i and thread j would process the (i, j)
        // pair and the total force would be doubled.
        If(j.lessThanEqual(i), () => {
          Continue();
        });
        // Skip boundary — adhesion handles fluid-boundary.
        const Vjb: Any = particles.boundaryVolume.element(j);
        If(Vjb.greaterThan(float(0.0)), () => {
          Continue();
        });

        const xj: Any = particles.predictedPositions.element(j).xyz;
        const diff: Any = xi.sub(xj).toVar();
        const r: Any = diff.dot(diff).sqrt().toVar();
        If(r.lessThan(float(1e-20)), () => {
          Continue();
        });

        // K_ij (eq. 4). Density seeded to ρ_0 at FluidSystem
        // construction; on subsequent substeps it carries last-iter
        // values from perIterKernels' density kernel. `max(1)` guards
        // against literal zero if the seed is ever skipped.
        const rhoJ: Any = density.element(j);
        const rhoSum: Any = rhoI.add(rhoJ).max(float(1.0));
        const Kij: Any = twoRho0.div(rhoSum);

        // Cohesion (eq. 1). rHat = (x_i − x_j)/r points from j→i; the
        // leading minus sign in eq. 1 flips it to i→j for attractive
        // C > 0 — i is pulled toward j.
        const Cr: Any = emitCohesionSpline(r, sph, coh);
        const rHat: Any = diff.div(r);
        const gamma: Any = coh.gamma as Any;
        const mI: Any = mass as Any;
        const mJ: Any = mass as Any;
        const cohMag: Any = gamma.mul(mI).mul(mJ).mul(Cr).negate();
        const fCoh: Any = rHat.mul(cohMag);

        // Curvature (eq. 3).
        const nj: Any = colorFieldNormal.element(j).xyz;
        const fCurv: Any = ni.sub(nj).mul(gamma.mul(mI)).negate();

        // Combined per-pair force (eq. 5). Convert to velocity impulse
        // `Δv = F · Δt / m` so accumulator ticks represent m/s.
        const fSt: Any = fCoh.add(fCurv).mul(Kij);
        const dv: Any = fSt
          .mul(invM)
          .mul(dt as Any)
          .toVar();

        // Scatter +Δv to i's accumulator, −Δv to j's. Atomic i32
        // adds enforce Newton-3 by construction.
        emitAccumulateVelocityDelta(velocityAccumulator, i, dv);
        emitAccumulateVelocityDelta(velocityAccumulator, j, dv.negate());
      },
    });
  })().compute(fluidParticles.count);
}

export interface BuildApplyVelocityImpulseKernelArgs {
  readonly particles: ParticleSystem;
  readonly velocityAccumulator: VelocityAccumulator;
  readonly dt: UniformNode<'float', number>;
  readonly fluidParticles: ParticleRange;
}

/**
 * Pass 3 — drain the scattered Δv into `velocities` AND
 * `predictedPositions`, then reset the accumulator.
 *
 * For each fluid particle:
 *   Δv = accumulated ticks · invScale       (reconstruct float Δv)
 *   velocities[i]          += Δv
 *   predictedPositions[i]  += Δv · Δt       (paper Alg. 1 line 3 contribution)
 *   accumulator[i]         := 0             (prep for next substep)
 *
 * The `predictedPositions += Δv·Δt` update is the algebraic equivalent
 * of integrating F_st into predict — see the file-level docstring's
 * pipeline description. Without the x\* update the Δv is overwritten by
 * advect (`v = (x* − x)/Δt`) before anything reads it, and the force
 * has no physical effect.
 */
export function buildApplyVelocityImpulseKernel(
  args: BuildApplyVelocityImpulseKernelArgs,
): ComputeNode {
  const { particles, velocityAccumulator, dt, fluidParticles } = args;
  validateRange('buildApplyVelocityImpulseKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const base: Any = i.mul(uint(3));
    const invScale: Any = velocityAccumulator.invScale as Any;

    // atomicLoad + atomicStore required: the buffer is declared atomic,
    // so non-atomic reads/writes are invalid TSL.
    const dxTicks: Any = atomicLoad(velocityAccumulator.delta.element(base));
    const dyTicks: Any = atomicLoad(velocityAccumulator.delta.element(base.add(uint(1))));
    const dzTicks: Any = atomicLoad(velocityAccumulator.delta.element(base.add(uint(2))));
    const dvx: Any = dxTicks.toFloat().mul(invScale);
    const dvy: Any = dyTicks.toFloat().mul(invScale);
    const dvz: Any = dzTicks.toFloat().mul(invScale);

    const v: Any = particles.velocities.element(i).toVar();
    (particles.velocities.element(i) as Any).assign(
      vec4(v.x.add(dvx), v.y.add(dvy), v.z.add(dvz), v.w),
    );

    const xStar: Any = particles.predictedPositions.element(i).toVar();
    const dtV: Any = dt as Any;
    (particles.predictedPositions.element(i) as Any).assign(
      vec4(
        xStar.x.add(dvx.mul(dtV)),
        xStar.y.add(dvy.mul(dtV)),
        xStar.z.add(dvz.mul(dtV)),
        xStar.w,
      ),
    );

    // Reset this particle's accumulator slots for the next substep. We
    // own reset here (not via a dedicated reset kernel) — the scatter +
    // apply pair owns the accumulator lifecycle for surface tension.
    atomicStore(velocityAccumulator.delta.element(base), int(0));
    atomicStore(velocityAccumulator.delta.element(base.add(uint(1))), int(0));
    atomicStore(velocityAccumulator.delta.element(base.add(uint(2))), int(0));
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
