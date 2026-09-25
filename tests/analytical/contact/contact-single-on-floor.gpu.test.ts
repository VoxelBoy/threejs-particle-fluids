import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 05 diagnostic — single dynamic particle on an analytic floor.
//
// Originally authored against a 3×3 kinematic sphere floor (Phase 05
// particle-floor surrogate). Migrated to an analytic plane in Phase 06
// per the plan's §"Migration from Phase 5's particle-floor surrogate"
// migration list. The kinematic-sphere-floor valley-trap pathology
// (r · (1 − √3/2) ≈ 0.007 m deep valleys between adjacent floor
// particles) no longer applies — particles rest on a flat plane, and
// the settle-to-y=r invariant is exact rather than being perturbed by
// valley geometry.
//
// Single-ball-on-floor has exactly ONE particle-collider contact, which
// keeps diagnostics trivial when a regression surfaces: no phase-mask
// noise, no multi-contact λ accumulation, no neighbor coloring.

describe('Phase 05 — contact: single particle on analytic floor (diagnostic)', () => {
  it('ball dropped from h=6r lands near y=r and comes to rest', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;

      const initial: ParticleInit[] = [
        {
          position: [0, 6 * r, 0],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 0,
        },
      ];
      const dynamicIdx = 0;

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      // Hash grid kept around for any future multi-particle variant; with
      // N=1 it's a no-op except for the one-time initialization cost.
      const hashGrid = new HashGrid(particles, {
        cellSize: 2 * r * 1.1,
      });

      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.6,
        muK: 0.5,
      });
      colliders.upload();

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        contact: {
          hashGrid,
          maxContacts: 64,
          friction: { muS: 0.6, muK: 0.5 },
        },
        colliders: { colliders },
      });
      loop.kernels.floorY.value = -1e9;
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

      // eslint-disable-next-line no-console
      console.info(
        `[contact-single] finalY=${finalY.toFixed(4)} finalVy=${finalVy.toExponential(3)} ` +
          `maxY=${maxY.toFixed(4)} (expected ≈ r=${r})`,
      );

      // Ball must rest at y ≈ r (particle centre one radius above plane
      // at y=0). Post-plane migration the tolerance is tight — no valley
      // perturbation to absorb into.
      expect(finalY).toBeGreaterThan(r - 1e-3);
      expect(finalY).toBeLessThan(r + 5e-3);
      expect(Math.abs(finalVy)).toBeLessThan(0.1);

      particles.destroy();
      colliders.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
