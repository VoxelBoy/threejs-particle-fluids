import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 16 G1 — α-scaled gravity reaches the predict kernel.
//
// Gas scenes set scene gravity to `g · α` (Macklin 2014 §7.2 — fluid
// particles fill the domain with reduced gravity to model buoyant gas).
// This is a trivial smoke test confirming `SimLoop.gravity` is honoured
// by `core/integrate.ts::predict` — there is no gas-specific code path
// for it (gas reuses the existing predict + advect untouched).
//
// One particle, free fall under g_eff = 0.1·g for T = 1 s.
//   v(T) = g_eff · T = -0.981 m/s
// Plan tolerance: 1e-4 m/s (i.e. integration error well below the
// physical signal).

describe('Phase 16 — α-scaled gravity reaches the predict kernel', () => {
  it('single particle under g·0.1 falls at 0.1·g·t within 1e-4 m/s²', async () => {
    const renderer = await createParticleRenderer();
    try {
      const G = 9.81;
      const ALPHA = 0.1;
      // dt = 1/128 = 0.0078125 is exactly representable in f32, so 128
      // sequential `v += dt·g` adds in the predict kernel accumulate
      // only ULP-level rounding (bounded ~1e-5 m/s at this scale) —
      // well under the plan's 1e-4 m/s² tolerance.
      const dt = 1 / 128;
      const numSteps = 128;
      const T = numSteps * dt; // = 1.0 exactly

      const initial: ParticleInit[] = [
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ];
      const particles = new ParticleSystem(renderer, 1, 0.01);
      particles.uploadParticles(initial);
      const hashGrid = new HashGrid(particles, { cellSize: 0.05 });
      const xpbd = createXpbdUniforms(dt);
      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        xpbd,
        hashGrid,
        materials: [],
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -G * ALPHA, 0);

      for (let step = 0; step < numSteps; step++) {
        await loop.step(dt);
      }

      const vel = new Float32Array(await renderer.getArrayBufferAsync(particles.velocities.value));
      const vyMeasured = vel[1];
      const vyExpected = -G * ALPHA * T;
      const err = Math.abs(vyMeasured! - vyExpected);
      // eslint-disable-next-line no-console
      console.log(
        `[gravity-reduction] α=${ALPHA} T=${T}s vy_measured=${vyMeasured!.toFixed(6)} vy_expected=${vyExpected.toFixed(6)} err=${err.toExponential(2)} m/s`,
      );
      expect(err).toBeLessThan(1e-4);
    } finally {
      renderer.dispose();
    }
  });
});
