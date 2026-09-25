import { Fn, If, Return, float, instanceIndex, uint, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ParticleRange, ParticleSystem } from '../../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildFusedVorticityXsphApplyKernelArgs {
  readonly particles: ParticleSystem;
  readonly omega: StorageBufferNode<'vec4'>;
  readonly eta: StorageBufferNode<'vec4'>;
  readonly xsphDeltaV: StorageBufferNode<'vec4'>;
  readonly strength: UniformNode<'float', number>;
  readonly dt: UniformNode<'float', number>;
  readonly fluidParticles: ParticleRange;
}

/**
 * Fused vorticity-Pass-3 + XSPH-apply tail — one per-particle dispatch
 * that reads `ω`, `η`, and `Δv` and writes `velocities[i]` once.
 *
 * Replaces the separate `buildVorticityPass3Kernel` + `buildXsphApply
 * Kernel` pair when both vorticity and XSPH are enabled.
 *
 * Vorticity-confinement force (Macklin 2013 §5 eq. 16):
 *   `v_i ← v_i + dt · ε · (N_i × ω_i)`,  `N_i = η_i / |η_i|`.
 * Uses the *unnormalized* ω in the cross product per paper note: "we do
 * not use normalized ω as this would increase vorticity indiscriminately;
 * instead we use the unnormalized value, which only adds vorticity where
 * it already exists."
 *
 * XSPH viscosity apply (Macklin 2013 §5 eq. 17 commit step):
 *   `v_i ← v_i + Δv_i`.
 *
 * The two corrections are summed together and added to the existing
 * velocity in one read-modify-write. `etaLen < 1e-12` guards the
 * vorticity contribution against division-by-zero in homogeneous-
 * vorticity regions (no gradient direction); under that branch we keep
 * the XSPH contribution by jumping past Pass-3 only — `Return()` would
 * skip XSPH too, so we use a select rather than a kernel-exit.
 *
 * Gather mode — each thread writes only `velocities[i]`. No atomics.
 */
export function buildFusedVorticityXsphApplyKernel(
  args: BuildFusedVorticityXsphApplyKernelArgs,
): ComputeNode {
  const { particles, omega, eta, xsphDeltaV, strength, dt, fluidParticles } = args;
  validateRange('buildFusedVorticityXsphApplyKernel', particles, fluidParticles);
  const startIdx = fluidParticles.start;

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(startIdx)).toVar();
    const etaI: Any = eta.element(i).xyz.toVar();
    const etaLen: Any = etaI.length().toVar();
    const xsphDV: Any = xsphDeltaV.element(i).xyz.toVar();
    const existing: Any = particles.velocities.element(i).toVar();

    // Vorticity Pass-3 contribution. The `etaLen < 1e-12` branch in the
    // un-fused Pass 3 kernel issues `Return()` (kernel exit) to skip the
    // confinement entirely. Here we keep XSPH alive in the same regime,
    // so we early-return only after committing the XSPH-only update.
    If(etaLen.lessThan(float(1e-12)), () => {
      const newV: Any = existing.xyz.add(xsphDV);
      particles.velocities.element(i).assign(vec4(newV, existing.w));
      Return();
    });

    const N: Any = etaI.div(etaLen);
    const omegaI: Any = omega.element(i).xyz;
    const vortDV: Any = N.cross(omegaI)
      .mul(strength as Any)
      .mul(dt as Any);
    const newV: Any = existing.xyz.add(vortDV).add(xsphDV);
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
