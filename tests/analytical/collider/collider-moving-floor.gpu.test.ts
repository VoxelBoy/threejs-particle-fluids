import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 06 G1 — "Moving-floor friction" (plan §Validation > Automatic (G1)):
// "Horizontal plane sliding at constant velocity; particle with μ = 0.6
// placed on top; particle acquires plane velocity within expected settling
// time given friction (analytic estimate documented)."
//
// Analytic estimate: under kinetic friction (μ_k = 0.5), the per-substep
// Δv clamp is `μ_k · g · h` where h is the substep dt. At v_plane = 1 m/s
// and g = 9.81 m/s², the particle needs at most
//   t_settle ≈ v_plane / (μ_k · g) = 1 / (0.5 · 9.81) ≈ 0.204 s
// to match the plane's velocity. With static friction (μ_s = 0.6 > μ_k),
// once the particle is moving with the plane, the cone holds and no
// further slip occurs.

describe('Phase 06 — collider: moving floor friction', () => {
  it('particle on horizontal plane sliding at 1 m/s acquires plane velocity', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const vPlane = 1.0; // m/s, along +x
      const initial: ParticleInit[] = [
        {
          position: [0, r, 0], // already resting on the plane surface
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 0,
        },
      ];

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.6,
        muK: 0.5,
        linearVelocity: new Vector3(vPlane, 0, 0),
      });
      colliders.upload();

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        colliders: { colliders },
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      // 0.5 s >> analytic 0.204 s settling time.
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
      // eslint-disable-next-line no-console
      console.info(
        `[collider-moving-floor] final vx=${final.vx.toFixed(4)} (expected ≈ ${vPlane}) vy=${final.vy.toExponential(
          3,
        )}`,
      );
      for (let i = 0; i < Math.min(8, velTrace.length); i++) {
        const t = velTrace[i]!;
        // eslint-disable-next-line no-console
        console.info(
          `[collider-moving-floor] t=${t.t.toFixed(3)} vx=${t.vx.toFixed(4)} vy=${t.vy.toFixed(4)}`,
        );
      }

      // Settling assertion: final vx within 10% of plane velocity.
      expect(final.vx).toBeGreaterThan(0.9 * vPlane);
      expect(final.vx).toBeLessThan(1.1 * vPlane);
      // Vertical velocity should be near zero (particle stays on the
      // plane; gravity balanced by normal force each substep).
      expect(Math.abs(final.vy)).toBeLessThan(0.1);

      particles.destroy();
      colliders.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
