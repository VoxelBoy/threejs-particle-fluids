import { Vector3 } from 'three';
import { instancedArray, uniform } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';
import { buildDragKernel } from '../../../src/fluids/index.js';

// Phase 16 G1 — Macklin 2014 §7.2.2 drag attenuation.
//
// Single fluid particle in vacuum, given initial velocity v_0 = (1, 0, 0).
// Drag k = 10, v_env = 0, density = 0 → surfaceFactor = (1 − ρ/ρ_0) = 1.
//
// Continuous ODE: dv/dt = −α·(v − v_env) with α = k·invMass·surfaceFactor
//                                                = 10·1·1 = 10 1/s.
// Solution: v(t) = v_0·exp(−α·t). Half-time τ = ln(2)/α = 0.0693 s.
//
// Plan tolerance: measured half-time must match analytical within 2%.
//
// This is a focused unit test on the drag kernel — not the full
// FluidSystem. We bypass SimLoop and dispatch the kernel directly so we
// can pin density = 0 (the SimLoop iter loop's lambda kernel would
// otherwise write a non-zero self-density term `ψ_self · W(0)` for an
// isolated particle, partially attenuating drag and breaking the closed-
// form match). The kernel under test is the exact one FluidSystem
// builds — same TSL nodes, same uniforms, same buffers.

describe('Phase 16 — drag attenuation (Macklin 2014 §7.2.2 eq. 29)', () => {
  it('exponential velocity decay matches analytical τ = ln(2)/(k·invMass) within 2%', async () => {
    const renderer = await createParticleRenderer();
    try {
      const restDensity = 1000;
      const k = 10;
      const dtValue = 0.001; // 1 ms — keeps explicit-Euler truncation < 0.5%.
      const T = 0.1; // total integration time, s. Past one half-time.
      const numSteps = Math.round(T / dtValue);

      const initial: ParticleInit[] = [
        {
          position: [0, 0, 0],
          velocity: [1, 0, 0],
          invMass: 1, // m = 1 kg → α = k·invMass = 10 1/s.
          phase: 0,
        },
      ];

      const particles = new ParticleSystem(renderer, 1, 0.01);
      particles.uploadParticles(initial);

      // Density buffer pinned at 0 (zero-init from instancedArray) so
      // surfaceFactor = clamp01(1 − 0/ρ_0) = 1 throughout the run.
      const density = instancedArray(1, 'float');

      const restDensityU = uniform(restDensity, 'float');
      const kU = uniform(k, 'float');
      const vEnvU = uniform(new Vector3(0, 0, 0));
      const dtU = uniform(dtValue, 'float');

      const dragKernel = buildDragKernel({
        particles,
        density,
        restDensity: restDensityU,
        k: kU,
        vEnv: vEnvU,
        dt: dtU,
        fluidParticles: { start: 0, count: 1 },
      });

      // Dispatch the drag kernel `numSteps` times. Each dispatch applies
      // one explicit-Euler step `v ← v · (1 − α·dt)`.
      for (let step = 0; step < numSteps; step++) {
        await renderer.computeAsync(dragKernel);
      }

      const velArr = new Float32Array(
        await renderer.getArrayBufferAsync(particles.velocities.value),
      );
      const vxFinal = velArr[0]; // x component of particle 0.

      // Analytical comparison via half-time.
      //   v(T) = exp(−α·T)  ⇒  α_measured = −ln(v(T)) / T
      //   τ_measured = ln(2) / α_measured
      const alphaAnalytical = k * 1; // invMass = 1, surfaceFactor = 1
      const tauAnalytical = Math.log(2) / alphaAnalytical;
      expect(vxFinal).toBeGreaterThan(0); // sanity — exponential, never crosses 0
      const alphaMeasured = -Math.log(vxFinal!) / T;
      const tauMeasured = Math.log(2) / alphaMeasured;
      const relErr = Math.abs(tauMeasured - tauAnalytical) / tauAnalytical;

      // eslint-disable-next-line no-console
      console.log(
        `[drag-attenuation] T=${T.toFixed(3)}s vx=${vxFinal!.toFixed(6)} ` +
          `τ_measured=${tauMeasured.toFixed(6)}s τ_analytical=${tauAnalytical.toFixed(6)}s ` +
          `relErr=${(relErr * 100).toFixed(3)}%`,
      );
      expect(relErr).toBeLessThan(0.02);
    } finally {
      renderer.dispose();
    }
  });
});
