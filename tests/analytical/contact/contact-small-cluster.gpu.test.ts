import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Small cluster settling on an analytic floor.
//
// Stress-tests vertical stacking in the per-iteration solve: middle
// particles receive corrections from contact neighbors above and below, and
// the bottom layer also from the plane beneath it. The pile must come to
// rest with the bottom layer near y = r.

describe('contact: 3³ cluster on an analytic floor', () => {
  it('cluster settles on plane with bottom layer at y ≈ r', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const twoR = 2 * r;

      const initial: ParticleInit[] = [];
      // 3×3×3 stacked cluster. Grid spacing = 3r so pairs start NOT
      // touching and the pile has to do real work to settle.
      const clusterSide = 3;
      const clusterSpacing = 3 * r;
      const startY = 2 * twoR;
      for (let ix = 0; ix < clusterSide; ix++) {
        for (let iy = 0; iy < clusterSide; iy++) {
          for (let iz = 0; iz < clusterSide; iz++) {
            initial.push({
              position: [
                (ix - (clusterSide - 1) / 2) * clusterSpacing,
                startY + iy * clusterSpacing,
                (iz - (clusterSide - 1) / 2) * clusterSpacing,
              ],
              velocity: [0, 0, 0],
              invMass: 1,
            });
          }
        }
      }
      const capacity = initial.length;

      const particles = new ParticleSystem(renderer, capacity, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles);
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.5,
        muK: 0.4,
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        contact: { maxContacts: 512, muS: 0.5, muK: 0.4 },
        colliders: [colliders],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      for (let n = 0; n < 90; n++) {
        await loop.step(frameDt);
      }

      const snap = await particles.readback();
      let minY = Infinity;
      let maxSpeed = 0;
      for (let i = 0; i < capacity; i++) {
        const base = i * 4;
        const y = snap.positions[base + 1]!;
        if (y < minY) minY = y;
        const vx = snap.velocities[base]!;
        const vy = snap.velocities[base + 1]!;
        const vz = snap.velocities[base + 2]!;
        const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (speed > maxSpeed) maxSpeed = speed;
      }
      console.info(
        `[small-cluster] minY=${minY.toFixed(4)} (expected ≈ r=${r}) maxSpeed=${maxSpeed.toFixed(4)} m/s`,
      );

      // Bottom-layer particles must rest at y ≥ r (no penetration).
      expect(minY).toBeGreaterThan(r - 2e-3);
      // The pile must be settled — max speed well under the free-fall peak.
      expect(maxSpeed).toBeLessThan(1.0);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
