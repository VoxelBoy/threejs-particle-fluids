import { Continue, Fn, If, float, instanceIndex, uint, uniform } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ParticleRange, ParticleSystem, VelocityAccumulator } from '../../core/index.js';
import { emitAccumulateVelocityDelta, emitForEachPair } from '../../core/index.js';

import type { SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/*
 * Akinci, Akinci, Teschner 2013 §3 — fluid ↔ boundary adhesion.
 *
 *
 * In MVP, boundary particles are kinematic (not integrated by predict
 * /advect); they are prescribed by the scene or driven by an external
 * animation. The momentum that would scatter to the boundary side is
 * therefore *absorbed* by whatever drives the boundary — it does not
 * need to appear in our fluid velocity accumulator. Phase 15 rigid-
 * body coupling revisits this: rigid particles are particles too, and
 * the adhesion reaction will then scatter into their impulse buffer.
 *
 * Practical implication: this scatter kernel dispatches ONE thread per
 * FLUID particle, walks its neighbours, filters to boundary (`V_b > 0`),
 * and scatters `+Δv = F_{i←k}·Δt/m_i` into the velocity accumulator
 * slot for i. The `−Δv` that would go to boundary k is elided.
 *
 * Reuses the Akinci 2012 `boundaryVolume` attribute already populated
 * by `FluidSystem.registerBoundaryParticles` (Phase 08). No new
 * per-particle data.
 */

export interface AdhesionUniforms {
  readonly adhesionCoef: UniformNode<'float', number>;
  readonly beta: UniformNode<'float', number>;
  readonly setAdhesionH: (newH: number) => void;
  readonly setBeta: (newBeta: number) => void;
}

function adhesionCoefFor(h: number): number {
  return 0.007 / Math.pow(h, 3.25);
}

export function createAdhesionUniforms(h: number, beta: number): AdhesionUniforms {
  if (!Number.isFinite(h) || h <= 0) {
    throw new Error(`createAdhesionUniforms: h must be a positive finite number, got ${h}`);
  }
  if (!Number.isFinite(beta)) {
    throw new Error(`createAdhesionUniforms: beta must be a finite number, got ${beta}`);
  }
  const coefU = uniform(adhesionCoefFor(h), 'float');
  const betaU = uniform(beta, 'float');
  return {
    adhesionCoef: coefU,
    beta: betaU,
    setAdhesionH(newH: number): void {
      if (!Number.isFinite(newH) || newH <= 0) {
        throw new Error(`setAdhesionH: h must be a positive finite number, got ${newH}`);
      }
      (coefU as Any).value = adhesionCoefFor(newH);
    },
    setBeta(newBeta: number): void {
      if (!Number.isFinite(newBeta)) {
        throw new Error(`setBeta: beta must be a finite number, got ${newBeta}`);
      }
      (betaU as Any).value = newBeta;
    },
  };
}

/**
 * Emit Akinci 2013 eq. 7 adhesion spline `A(r)` as a scalar TSL node.
 *
 * `A(r) = (0.007 / h^{3.25}) · { ⁴√(−4r²/h + 6r − 2h)   for 2r > h ∧ r ≤ h
 *                                0                       otherwise }`
 *
 * Support `[h/2, h]` only (paper Figure 4) — zero below h/2 is
 * intentional: Akinci 2012 boundary-volume pressure already prevents
 * clustering there, so adhesion would double-act if enabled.
 */
function emitAdhesionSpline(r: Any, sph: SphKernelUniforms, adh: AdhesionUniforms): Any {
  const h: Any = sph.h as Any;
  // −4r²/h + 6r − 2h = −(4/h)(r − h/2)(r − h); non-negative on [h/2, h],
  // zero at endpoints. `.max(0)` guards FP underflow past the endpoint.
  const arg: Any = r
    .mul(r)
    .mul(float(-4.0))
    .div(h)
    .add(r.mul(float(6.0)))
    .sub(h.mul(float(2.0)))
    .max(float(0.0))
    .toVar();
  const quartRoot: Any = arg.sqrt().sqrt();
  const twoR: Any = r.mul(float(2.0));
  const inRange: Any = twoR.greaterThan(h).and(r.lessThanEqual(h));
  return inRange.select(quartRoot.mul(adh.adhesionCoef as Any), float(0.0));
}

export interface BuildAdhesionScatterKernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  readonly adh: AdhesionUniforms;
  /** `ρ_0` — rest density, for `Ψ_{b_k} = ρ_0 · V_{b_k}`. */
  readonly restDensity: UniformNode<'float', number>;
  /** Per-fluid-particle mass `m_i = ρ_0 · spacing³` (scene-constant). */
  readonly mass: UniformNode<'float', number>;
  readonly dt: UniformNode<'float', number>;
  readonly velocityAccumulator: VelocityAccumulator;
  readonly fluidParticles: ParticleRange;
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Build the fluid ↔ boundary adhesion scatter kernel.
 *
 * Paper eq. 6: `F^{adhesion}_{i←k} = −β · m_i · Ψ_{b_k} · A(r) · r̂`
 * with `Ψ_{b_k} = ρ_0 · V_{b_k}`. Macklin & Müller 2013 Algorithm 1
 * line 6 amortized neighbour iteration: walks the per-substep pair list
 * instead of re-walking the 27-cell hash grid every dispatch.
 *
 * One thread per fluid particle i; iterates the pair list, filters to
 * boundary neighbours (`V_b > 0`), and scatters the per-pair Δv into
 * the velocity accumulator for i. No `j > i` filter — the boundary
 * side is kinematic and its −Δv scatter is elided (see file-level
 * docstring).
 *
 * Reads positions from `particles.predictedPositions` to match the
 * pair list's build-time read source. Within-h cutoff is applied at
 * pair-list build time, so the redundant `rSq ≥ hSq` early-exit is
 * dropped here. The `r < 1e-20` self-skip is preserved — it guards
 * `1/r` in `rHat = diff/r` and is independent of the within-h check.
 */
export function buildAdhesionScatterKernel(args: BuildAdhesionScatterKernelArgs): ComputeNode {
  const {
    particles,
    sph,
    adh,
    restDensity,
    mass,
    dt,
    velocityAccumulator,
    fluidParticles,
    pairList,
    pairCount,
  } = args;
  validateRange('buildAdhesionScatterKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const invM: Any = float(1.0).div(mass as Any);

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (k: Any) => {
        // Boundary-only filter. Fluid-fluid pairs are cohesion's job.
        const Vk: Any = particles.boundaryVolume.element(k).toVar();
        If(Vk.lessThanEqual(float(0.0)), () => {
          Continue();
        });

        const xk: Any = particles.predictedPositions.element(k).xyz;
        const diff: Any = xi.sub(xk).toVar();
        const r: Any = diff.dot(diff).sqrt().toVar();
        If(r.lessThan(float(1e-20)), () => {
          Continue();
        });

        // Ψ_k (paper §3 just before eq. 6).
        const psiK: Any = (restDensity as Any).mul(Vk);
        const Ar: Any = emitAdhesionSpline(r, sph, adh);
        const rHat: Any = diff.div(r);
        // F_{i←k} = −β · m_i · Ψ_k · A(r) · r̂
        const magAcc: Any = (adh.beta as Any)
          .mul(mass as Any)
          .mul(psiK)
          .mul(Ar)
          .negate();
        const fAdh: Any = rHat.mul(magAcc);
        const dv: Any = fAdh.mul(invM).mul(dt as Any);

        // Scatter +Δv into fluid particle i's accumulator. Boundary
        // side absorbed by whatever drives the boundary's kinematic
        // motion — no scatter to k.
        emitAccumulateVelocityDelta(velocityAccumulator, i, dv);
      },
    });
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
