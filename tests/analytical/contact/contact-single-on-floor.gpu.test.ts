import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Single dynamic particle dropped on an analytic plane, with particle
// contacts enabled.
//
// On a flat plane the particle must come to rest exactly one radius above
// it. With exactly ONE particle–collider contact, a regression here is easy
// to diagnose: no collision-group noise, no multi-contact λ accumulation, no
// neighboring particles.

describe('contact: single particle on an analytic floor', () => {
  it('ball dropped from h=6r lands near y=r and comes to rest', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;

      const initial: ParticleInit[] = [
        { position: [0, 6 * r, 0], velocity: [0, 0, 0], invMass: 1 },
      ];
      const dynamicIdx = 0;

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles);
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.6,
        muK: 0.5,
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        contact: { maxContacts: 64, muS: 0.6, muK: 0.5 },
        colliders: [colliders],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const totalFrames = 60;

      const trace: Array<{ y: number; vy: number }> = [];
      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
        const snap = await particles.readback();
        trace.push({
          y: snap.positions[dynamicIdx * 4 + 1]!,
          vy: snap.velocities[dynamicIdx * 4 + 1]!,
        });
      }

      const finalY = trace[totalFrames - 1]!.y;
      const finalVy = trace[totalFrames - 1]!.vy;
      const maxY = Math.max(...trace.map((t) => t.y));

      console.info(
        `[contact-single] finalY=${finalY.toFixed(4)} finalVy=${finalVy.toExponential(3)} ` +
          `maxY=${maxY.toFixed(4)} (expected ≈ r=${r})`,
      );

      // The ball must rest at y ≈ r (centre one radius above the plane at
      // y = 0). A flat plane leaves no surface features to absorb error into,
      // so the tolerance is tight.
      expect(finalY).toBeGreaterThan(r - 1e-3);
      expect(finalY).toBeLessThan(r + 5e-3);
      expect(Math.abs(finalVy)).toBeLessThan(0.1);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
