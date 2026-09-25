import { Continue, Fn, If, Return, float, instanceIndex, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ParticleRange, ParticleSystem } from '../../core/index.js';
import { emitForEachPair } from '../../core/index.js';

import { emitSpikyGrad, type SphKernelUniforms } from './kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/*
 * Macklin 2013 §5 vorticity confinement — three passes dispatched once
 * per substep in `FluidSystem.postAdvectKernels`, after advect, before
 * the shared velocity-friction block.
 *
 *
 * The sign is observable: if Pass 1 and Pass 2 are kerneled with
 * inconsistent conventions, the force direction in Pass 3 flips and the
 * kernel anti-confines vorticity (the dam-break golden would render
 * with a visibly *flatter* splash crown, not crash). The inline
 * comments below anchor the chosen convention at each pass.
 */

export interface BuildVorticityPass1KernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  /** Per-fluid-particle `V = m/ρ_0`. See `FluidSystem.particleVolumeUniform`. */
  readonly particleVolume: UniformNode<'float', number>;
  readonly omega: StorageBufferNode<'vec4'>;
  readonly omegaMag: StorageBufferNode<'float'>;
  readonly fluidParticles: ParticleRange;

  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Pass 1 — per-particle vorticity vector and its magnitude.
 *
 * Paper (unit-mass, §5 eq. 15, neighbor-walk amortized via Algorithm 1
 * line 6 / §6 paragraph 2 per-substep pair list):
 * `ω_i = Σ_j v_{ij} × ∇_{p_j} W`. Our dimensional Monaghan-canonical
 * form restores the `m_j/ρ_j` weight that paper absorbs:
 *
 *   `ω_i = Σ_j (m_j/ρ_j) · (v_i − v_j) × ∇_{p_i} W_ij`
 *
 * For fluid neighbors `m_j/ρ_j ≈ particleVolume`. For Akinci boundary
 * neighbors we use the stored `boundaryVolume[j]`. Dropping this factor
 * gives `ω` that is `1/V ≈ 64 000×` too large for our spacing / ρ_0
 * scene — the NaN blowup I saw 2026-04-23.
 *
 * Sign convention: `(v_i − v_j) × ∇_{p_i} W` is equivalent to paper's
 * `(v_j − v_i) × ∇_{p_j} W` by the two-flip identity (see file-level
 * docstring above).
 *
 * Pair-list iteration: pairs were filtered to `|r|² < h²` at build time,
 * so no within-h check is needed here.
 */
export function buildVorticityPass1Kernel(args: BuildVorticityPass1KernelArgs): ComputeNode {
  const { particles, sph, particleVolume, omega, omegaMag, fluidParticles, pairList, pairCount } =
    args;
  validateRange('buildVorticityPass1Kernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.positions.element(i).xyz.toVar();
    const vi: Any = particles.velocities.element(i).xyz.toVar();
    const omegaSum: Any = vec3(0, 0, 0).toVar();

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        const xj: Any = particles.positions.element(j).xyz;
        const diff: Any = xi.sub(xj).toVar();

        const Vj_boundary: Any = particles.boundaryVolume.element(j);
        const isBoundary: Any = Vj_boundary.greaterThan(float(0.0));
        const Vj: Any = isBoundary.select(Vj_boundary, particleVolume as Any);

        const vj: Any = particles.velocities.element(j).xyz;
        const vij: Any = vi.sub(vj);
        const g: Any = emitSpikyGrad(diff, sph);
        omegaSum.addAssign(vij.cross(g).mul(Vj));
      },
    });

    omega.element(i).assign(vec4(omegaSum, float(0.0)));
    omegaMag.element(i).assign(omegaSum.length());
  })().compute(fluidParticles.count);
}

export interface BuildVorticityPass2KernelArgs {
  readonly particles: ParticleSystem;
  readonly sph: SphKernelUniforms;
  /** Per-fluid-particle `V = m/ρ_0`. Same uniform Pass 1 receives. */
  readonly particleVolume: UniformNode<'float', number>;
  readonly omegaMag: StorageBufferNode<'float'>;
  readonly eta: StorageBufferNode<'vec4'>;
  readonly fluidParticles: ParticleRange;
  /** Same per-substep pair list Pass 1 reads. See Pass 1's docstring. */
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
}

/**
 * Pass 2 — per-particle `η_i = ∇|ω|_i` via the symmetric Monaghan 1992
 * SPH gradient estimator applied to the scalar field `|ω|` (neighbor-
 * walk amortized via Macklin 2013 Algorithm 1 line 6 / §6 paragraph 2
 * per-substep pair list):
 *
 *   `η_i = Σ_j (m_j/ρ_j) · (|ω_j| − |ω_i|) · ∇_{p_i} W_ij`
 *
 * Boundary neighbors are gated out: `|ω_j|` at a boundary is undefined
 * under our convention (Pass 1 doesn't compute omega for kinematic
 * slots), and the symmetric-gradient estimator cancels constant fields
 * so there's nothing physical to gain by including them. The boundary
 * gate is independent of the within-h check and stays inside the
 * pair-list callback — pairs in the cached list can still include
 * boundary neighbors, only the within-h filter is dropped.
 *
 * The `m_j/ρ_j = V_j = particleVolume` weight restores the Monaghan
 * dimensional factor paper drops under its unit-mass convention —
 * without it, `η` is `1/m ≈ 64×` too small and combined with Pass 1's
 * matching fix this makes the N = η/|η| unit vector come out correct.
 * (N is magnitude-normalized, so it would have been OK on its own, but
 * using consistent V weighting keeps the implementation honest.)
 */
export function buildVorticityPass2Kernel(args: BuildVorticityPass2KernelArgs): ComputeNode {
  const { particles, sph, particleVolume, omegaMag, eta, fluidParticles, pairList, pairCount } =
    args;
  validateRange('buildVorticityPass2Kernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const xi: Any = particles.positions.element(i).xyz.toVar();
    const omegaMagI: Any = omegaMag.element(i).toVar();
    const etaSum: Any = vec3(0, 0, 0).toVar();

    emitForEachPair({
      pairList,
      pairCount,
      queryIdx: i,
      fluidStart: startIdx,
      fluidCount: fluidParticles.count,
      onCandidate: (j: Any) => {
        // Gate to fluid neighbors — boundary particles do not participate
        // in the |ω| field.
        const Vjb: Any = particles.boundaryVolume.element(j);
        If(Vjb.greaterThan(float(0.0)), () => {
          Continue();
        });

        const xj: Any = particles.positions.element(j).xyz;
        const diff: Any = xi.sub(xj).toVar();

        const omegaMagJ: Any = omegaMag.element(j);
        const g: Any = emitSpikyGrad(diff, sph);
        const coef: Any = omegaMagJ.sub(omegaMagI).mul(particleVolume as Any);
        etaSum.addAssign(g.mul(coef));
      },
    });

    eta.element(i).assign(vec4(etaSum, float(0.0)));
  })().compute(fluidParticles.count);
}

export interface BuildVorticityPass3KernelArgs {
  readonly particles: ParticleSystem;
  readonly omega: StorageBufferNode<'vec4'>;
  readonly eta: StorageBufferNode<'vec4'>;
  readonly strength: UniformNode<'float', number>;
  readonly dt: UniformNode<'float', number>;
  readonly fluidParticles: ParticleRange;
}

/**
 * Pass 3 — apply `v_i += dt · ε · (N × ω_i)`.
 *
 * Paper: Macklin 2013 §5 eq. 16. Uses *unnormalized* `ω` in the cross
 * product on purpose — paper: "we do not use normalized ω as this would
 * increase vorticity indiscriminately; instead we use the unnormalized
 * value, which only adds vorticity where it already exists."
 *
 * Applied as an acceleration (no explicit `/m` division). In the MVP's
 * equal-mass fluid regime this is indistinguishable from a force-with-
 * divided-mass form; if a future multi-mass fluid phase changes this,
 * file an UNKNOWN.
 */
export function buildVorticityPass3Kernel(args: BuildVorticityPass3KernelArgs): ComputeNode {
  const { particles, omega, eta, strength, dt, fluidParticles } = args;
  validateRange('buildVorticityPass3Kernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const etaI: Any = eta.element(i).xyz.toVar();
    const etaLen: Any = etaI.length().toVar();
    // `N = η/|η|`. Skip the confinement entirely if |η| is effectively
    // zero (homogeneous-vorticity region — no gradient direction to
    // apply force along). Without this early-out, `etaI.div(etaLen)`
    // below produces `0/0 = NaN`, and `0 * NaN = NaN`, so the guard is
    // required even when `strength = 0` (pure-settle scenes).
    //
    // `Return()` is the TSL kernel-exit primitive; a bare JS `return`
    // inside the arrow body emits no TSL and the guard silently becomes
    // a no-op — cause of the 2026-04-23 both-demos-diverge regression
    // (contact/generate.ts carries the canonical comment about this
    // Fn-level-exit semantics).
    If(etaLen.lessThan(float(1e-12)), () => {
      Return();
    });
    const N: Any = etaI.div(etaLen);
    const omegaI: Any = omega.element(i).xyz;
    const force: Any = N.cross(omegaI).mul(strength as Any);
    const existing: Any = particles.velocities.element(i).toVar();
    const newV: Any = existing.xyz.add(force.mul(dt as Any));
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
