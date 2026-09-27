import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Moving-floor friction: a horizontal plane slides at constant velocity
// under a particle with μ = 0.6; friction must bring the particle up to the
// plane's velocity within the expected settling time.
//
// Analytic estimate: under kinetic friction (μ_k = 0.5) the velocity change
// per substep is clamped to `μ_k · g · h`, where h is the substep length. At
// v_plane = 1 m/s and g = 9.81 m/s², the particle needs at most
//   t_settle ≈ v_plane / (μ_k · g) = 1 / (0.5 · 9.81) ≈ 0.204 s
// to match the plane's velocity. With static friction (μ_s = 0.6 > μ_k),
// once the particle moves with the plane the cone holds and no further slip
// occurs.

describe('collider: moving floor friction', () => {
  it('particle on horizontal plane sliding at 1 m/s acquires plane velocity', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const vPlane = 1.0; // m/s, along +x
      const initial: ParticleInit[] = [
        // Already resting on the plane surface.
        { position: [0, r, 0], velocity: [0, 0, 0], invMass: 1 },
      ];

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles);
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.6,
        muK: 0.5,
        velocity: new Vector3(vPlane, 0, 0),
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        colliders: [colliders],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      // 0.5 s >> the analytic 0.204 s settling time.
      const totalFrames = 30;

      const velTrace: Array<{ t: number; vx: number; vy: number }> = [];
      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
        const snap = await particles.readback();
        velTrace.push({
          t: (n + 1) * frameDt,
          vx: snap.velocities[0]!,
          vy: snap.velocities[1]!,
        });
      }

      const final = velTrace[velTrace.length - 1]!;
      console.info(
        `[collider-moving-floor] final vx=${final.vx.toFixed(4)} (expected ≈ ${vPlane}) vy=${final.vy.toExponential(
          3,
        )}`,
      );
      for (let i = 0; i < Math.min(8, velTrace.length); i++) {
        const t = velTrace[i]!;
        console.info(
          `[collider-moving-floor] t=${t.t.toFixed(3)} vx=${t.vx.toFixed(4)} vy=${t.vy.toFixed(4)}`,
        );
      }

      // Settling: final vx within 10% of the plane velocity.
      expect(final.vx).toBeGreaterThan(0.9 * vPlane);
      expect(final.vx).toBeLessThan(1.1 * vPlane);
      // Vertical velocity should be near zero (the particle stays on the
      // plane; the normal correction balances gravity every substep).
      expect(Math.abs(final.vy)).toBeLessThan(0.1);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
