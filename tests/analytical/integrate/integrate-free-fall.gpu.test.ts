import { describe, expect, it } from 'vitest';
import { ParticleSystem, SimLoop, createParticleRenderer } from '../../../src/index.js';

describe('integrate: free fall', () => {
  it('reproduces semi-implicit Euler after 60 steps under gravity', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 1, 0.05);
      particles.uploadParticles([
        {
          position: [0, 0, 0],
          velocity: [0, 0, 0],
          invMass: 1,
        },
      ]);
      // One predict+advect per `step(dt)`. The default `substeps: 4` would
      // multi-step the integrator, so the CPU reference below (which does
      // exactly 60 semi-implicit Euler steps at dt = 1/60) would otherwise
      // disagree with the GPU output by O(1/S). No collider, so the
      // particle falls freely from the origin.
      const loop = new SimLoop(particles, { substeps: 1 });

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

      console.info(
        `[free-fall] refY=${refY.toFixed(6)} gpuY=${gpuY.toFixed(6)} |err|=${Math.abs(gpuY - refY).toExponential(3)}`,
      );
      expect(gpuY).toBeLessThan(0);
      expect(Math.abs(gpuY - refY)).toBeLessThan(1e-4);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
