import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 06 pilot — single particle on an analytic plane.
//
// Smallest test that exercises every moving part of the Phase 06 pipeline:
//   (1) `PrimitiveSet.addPlane` authoring + `upload`.
//   (2) `SimLoop` collider-only path allocates its own accumulators.
//   (3) The per-particle collider solve kernel runs and resolves normal
//       penetration against a plane at y = 0.
//
// Not in the plan's exit-criteria list; intended as a fast regression
// canary that fails loudly if the pipeline wiring regresses before the
// more demanding bowl / box / capsule tests get a chance to run.

describe('Phase 06 — collider plane rest (pilot)', () => {
  it('particle dropped on an analytic plane settles at y = r', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const initial: ParticleInit[] = [
        {
          position: [0, 0.5, 0],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 0,
        },
      ];

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.5,
        muK: 0.4,
      });
      colliders.upload();

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        colliders: { colliders },
      });
      // eslint-disable-next-line no-console
      console.info(
        `[collider-plane-rest] accumulator=${loop.accumulator ? 'yes' : 'no'} velocityAccumulator=${loop.velocityAccumulator ? 'yes' : 'no'}`,
      );
      // Floor kill-plane at a large negative value so the integrator's
      // cheap floor clamp doesn't mask the collider solve.
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      // 60 frames (1 s) is more than enough — drop from 0.5 m lands in
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

      // eslint-disable-next-line no-console
      console.info(
        `[collider-plane-rest] finalY=${finalY.toFixed(5)} finalVy=${finalVy.toExponential(
          3,
        )} minY=${minY.toFixed(5)} (expected ≈ r=${r})`,
      );
      for (let i = 0; i < Math.min(5, trace.length); i++) {
        // eslint-disable-next-line no-console
        console.info(
          `[collider-plane-rest] frame=${i} y=${trace[i]!.y.toFixed(5)} vy=${trace[i]!.vy.toFixed(5)}`,
        );
      }

      // Particle must rest above the plane (no penetration): y ≥ r - ε.
      // Small ε for f32 round-trip + quantization through the i32
      // accumulator at 1e8 ticks/m.
      const noPenetrationTol = 1e-4;
      expect(finalY).toBeGreaterThan(r - noPenetrationTol);
      // Particle must rest near the plane (contact): y ≤ r + 2ε.
      // Looser tolerance to allow one substep of velocity residual
      // converting to position over 1/60 s.
      expect(finalY).toBeLessThan(r + 5e-3);
      // Velocity at rest should be close to zero.
      expect(Math.abs(finalVy)).toBeLessThan(0.1);
      // Frame minimum must not go below r - ε (no penetration transient).
      expect(minY).toBeGreaterThan(r - noPenetrationTol);

      particles.destroy();
      colliders.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
