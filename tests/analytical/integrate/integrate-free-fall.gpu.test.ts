import { describe, expect, it } from 'vitest';
import { ParticleSystem, SimLoop, createParticleRenderer } from '../../../src/core/index.js';

describe('Phase 02 — integrate: free-fall', () => {
  it('reproduces semi-implicit Euler after 60 steps under gravity', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 1, 0.05);
      particles.uploadParticles([
        {
          position: [0, 0, 0],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 0,
        },
      ]);
      // Phase 02 semantics: one predict+advect per `step(dt)`. Phase 04's
      // default of `substeps: 4` multi-steps the integrator, so the CPU
      // reference below (which does exactly 60 semi-implicit Euler steps at
      // dt = 1/60) would otherwise disagree with the GPU output by O(1/S).
      const loop = new SimLoop(particles, { substeps: 1 });
      // Plan spec starts at origin; the Phase 02 floor clamp (default
      // floorY = 0) would otherwise pin y ≥ 0 and mask integration.
      loop.kernels.floorY.value = -1e9;

      const dt = 1 / 60;
      const g = 9.81;
      const steps = 60;

      let refV = 0;
      let refY = 0;
      for (let n = 0; n < steps; n++) {
        refV -= g * dt;
        refY += refV * dt;
      }

      for (let n = 0; n < steps; n++) {
        await loop.step(dt);
      }
      const snap = await particles.readback();
      const gpuY = snap.positions[1]!;

      // eslint-disable-next-line no-console
      console.info(
        `[free-fall] refY=${refY.toFixed(6)} gpuY=${gpuY.toFixed(6)} |err|=${Math.abs(gpuY - refY).toExponential(3)}`,
      );
      expect(gpuY).toBeLessThan(0);
      expect(Math.abs(gpuY - refY)).toBeLessThan(1e-4);

      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
