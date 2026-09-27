import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Single particle dropped on an analytic plane.
//
// The smallest test that exercises every moving part of the collider path:
//   (1) `PrimitiveSet.addPlane` authoring.
//   (2) A `SimLoop` with colliders but no particle contacts still allocates
//       and applies its position/velocity accumulators.
//   (3) The per-particle collider solve resolves normal penetration against
//       a plane at y = 0.
//
// A fast canary that fails loudly if the collider wiring regresses, before
// the more demanding bowl / box / capsule tests get a chance to run.

describe('collider: particle resting on a plane', () => {
  it('particle dropped on an analytic plane settles at y = r', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const initial: ParticleInit[] = [{ position: [0, 0.5, 0], velocity: [0, 0, 0], invMass: 1 }];

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles);
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.5,
        muK: 0.4,
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        colliders: [colliders],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      // 60 frames (1 s) is more than enough — a drop from 0.5 m lands in
      // ~0.32 s under 9.81 m/s².
      const totalFrames = 60;

      const trace: Array<{ y: number; vy: number }> = [];
      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
        const snap = await particles.readback();
        trace.push({
          y: snap.positions[1]!,
          vy: snap.velocities[1]!,
        });
      }

      const finalY = trace[totalFrames - 1]!.y;
      const finalVy = trace[totalFrames - 1]!.vy;
      const minY = Math.min(...trace.map((t) => t.y));

      console.info(
        `[collider-plane-rest] finalY=${finalY.toFixed(5)} finalVy=${finalVy.toExponential(
          3,
        )} minY=${minY.toFixed(5)} (expected ≈ r=${r})`,
      );
      for (let i = 0; i < Math.min(5, trace.length); i++) {
        console.info(
          `[collider-plane-rest] frame=${i} y=${trace[i]!.y.toFixed(5)} vy=${trace[i]!.vy.toFixed(5)}`,
        );
      }

      // Particle must rest above the plane (no penetration): y ≥ r - ε.
      // Small ε for the f32 round-trip plus quantization through the i32
      // position accumulator (~1e8 ticks/m).
      const noPenetrationTol = 1e-4;
      expect(finalY).toBeGreaterThan(r - noPenetrationTol);
      // Particle must rest near the plane (contact). Looser tolerance to
      // allow one substep of velocity residual converting to position.
      expect(finalY).toBeLessThan(r + 5e-3);
      // Velocity at rest should be close to zero.
      expect(Math.abs(finalVy)).toBeLessThan(0.1);
      // Frame minimum must not go below r - ε (no penetration transient).
      expect(minY).toBeGreaterThan(r - noPenetrationTol);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
