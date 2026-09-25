import { Fn, float, instanceIndex, max, min, uint, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { Vector3 } from 'three';
import type { ParticleRange, ParticleSystem } from '../../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildDragKernelArgs {
  readonly particles: ParticleSystem;

  readonly density: StorageBufferNode<'float'>;
  /** Scene rest density `ρ_0` (kg/m³). */
  readonly restDensity: UniformNode<'float', number>;
  /** Drag coefficient `k` (kg/s). Paper §7.2.2: `k ≈ 5..50` qualitatively. */
  readonly k: UniformNode<'float', number>;

  readonly vEnv: UniformNode<'vec3', Vector3>;
  /** Substep `Δt` from the shared XPBD uniforms. */
  readonly dt: UniformNode<'float', number>;
  readonly fluidParticles: ParticleRange;
}

/**
 * Free-surface drag kernel — Macklin 2014 §7.2.2 eq. 29:
 *
 *   `f_drag_i = −k · (v_i − v_env) · (1 − ρ_i / ρ_0)`
 *
 * Applied as an external force in the standard Algorithm-1-line-2 shape
 * (`v += Δt · f / m`):
 *
 *   `v_i ← v_i − Δt · k · invMass_i · (v_i − v_env) · clamp01(1 − ρ_i / ρ_0)`
 *
 * The `(1 − ρ_i / ρ_0)` attenuation activates near the free surface
 * (where `ρ_i < ρ_0`) and vanishes in the bulk (where `ρ_i ≈ ρ_0`). Paper
 * §7 eq. 26 makes the density constraint unilateral so `ρ_i ≤ ρ_0` in
 * equilibrium, but transient over-density between iters can flip the
 * sign — the `clamp01` guards against that.
 *
 * **Substep slot — postAdvect, not predict.** Paper eq. 29 is presented as
 * an external force entering Alg. 1 line 2 (predict); core's `SimLoop`
 * has no pre-predict hook today (`src/core/loop.ts` pushes
 * `predict → predictRotation → preIterKernels`). Drag therefore runs in
 * `Material.postAdvectKernels` alongside the other Macklin 2013 §5
 * velocity modifiers (vorticity confinement, XSPH viscosity), which all
 * operate on the realised post-advect velocity `v = (x* − x) / Δt`. The
 * functional difference vs a pre-predict slot is one substep of phase
 * shift in the discretisation: drag damps the substep that just
 * completed instead of the one about to start. **Assumed** acceptable
 * for MVP — at S=4 substeps and `α·dt = k·invMass/(60·S) ≈ O(0.01)` the
 * shift sits at sub-percent of one substep's velocity change. If a
 * future scene needs paper-faithful predict-time application, the
 * remediation is a new `Material.prePredictKernels` slot in core.
 *
 * No new buffers — reads `density[i]` (FluidSystem-owned), `velocities[i]`
 * + `invMass[i]` (`ParticleSystem`-owned), and the `k` / `vEnv` / `dt`
 * uniforms; writes `velocities[i]`.
 */
export function buildDragKernel(args: BuildDragKernelArgs): ComputeNode {
  const { particles, density, restDensity, k, vEnv, dt, fluidParticles } = args;
  validateRange('buildDragKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const mInv: Any = particles.invMass.element(i);
    const v: Any = particles.velocities.element(i).toVar();
    const rho: Any = density.element(i);

    // (1 − ρ_i / ρ_0), clamped to [0, 1]. Eq. 26 makes ρ ≤ ρ_0 in
    // equilibrium so the upper clamp at 1 only matters when the buffer
    // is zero-initialised at startup; the lower clamp at 0 prevents
    // transient over-density from inverting the drag sign.
    const surfaceFactor: Any = min(
      max(float(1.0).sub(rho.div(restDensity as Any)), float(0.0)),
      float(1.0),
    );

    // α = k · invMass · surfaceFactor — the per-particle damping rate
    // (1/s). For kinematic particles (invMass = 0) this collapses to
    // 0 → no drag → no write, by virtue of the multiplicative form below.
    const alpha: Any = (k as Any).mul(mInv).mul(surfaceFactor);
    const damping: Any = alpha.mul(dt as Any);

    const relV: Any = v.xyz.sub(vEnv as Any);
    const newV3: Any = v.xyz.sub(relV.mul(damping));
    particles.velocities.element(i).assign(vec4(newV3, v.w));
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
