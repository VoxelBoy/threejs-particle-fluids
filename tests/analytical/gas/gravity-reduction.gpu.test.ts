import { describe, expect, it } from 'vitest';
import { ParticleSystem, SimLoop, createParticleRenderer } from '../../../src/index.js';

// Reduced gravity reaches the integrator.
//
// Gas scenes set scene gravity to `g · α` (Macklin et al. 2014 §7.2 —
// fluid particles fill the domain with reduced gravity to model buoyant
// gas). This smoke test confirms a live change to `SimLoop.gravity` is
// honoured by the predict kernel — there is no gas-specific code path
// for it.
//
// One particle, free fall under g_eff = 0.1·g for T = 1 s.
//   v(T) = g_eff · T = -0.981 m/s
// Tolerance: 1e-4 m/s (integration error well below the physical signal).

describe('reduced gravity reaches the integrator', () => {
  it('single particle under g·0.1 falls at 0.1·g·t within 1e-4 m/s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const G = 9.81;
      const ALPHA = 0.1;
      // dt = 1/128 = 0.0078125 is exactly representable in f32, so 128
      // sequential `v += dt·g` adds in the predict kernel accumulate
      // only ULP-level rounding (bounded ~1e-5 m/s at this scale) —
      // well under the 1e-4 m/s tolerance.
      const dt = 1 / 128;
      const numSteps = 128;
      const T = numSteps * dt; // = 1.0 exactly

      const particles = new ParticleSystem(renderer, 1, 0.01);
      particles.uploadParticles([{ position: [0, 0, 0] }]);
      const loop = new SimLoop(particles, { substeps: 1, iterations: 1 });
      loop.gravity.set(0, -G * ALPHA, 0);

      for (let step = 0; step < numSteps; step++) {
        await loop.step(dt);
      }

      const vel = new Float32Array(await renderer.getArrayBufferAsync(particles.velocities.value));
      const vyMeasured = vel[1]!;
      const vyExpected = -G * ALPHA * T;
      const err = Math.abs(vyMeasured - vyExpected);
      console.log(
        `[gravity-reduction] α=${ALPHA} T=${T}s vy_measured=${vyMeasured.toFixed(6)} vy_expected=${vyExpected.toFixed(6)} err=${err.toExponential(2)} m/s`,
      );
      expect(err).toBeLessThan(1e-4);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });
});
