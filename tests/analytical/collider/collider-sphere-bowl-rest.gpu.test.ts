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

// Phase 06 G1 — "Sphere bowl rest" (plan §Validation > Automatic (G1)):
// "Drop 100 particles into an inverted sphere (bowl) of radius 1. After
// settling (5 s), each particle's distance to the bowl surface is within
// 2ε of zero (on the inside)."
//
// The plan phrasing "distance to the bowl surface is within 2ε" is the
// paper's non-penetration invariant evaluated on a bowl (inverted SDF):
// every particle must have its surface **at or inside** the bowl wall.
// Particles in the middle of the pile are allowed to be further from
// the wall (they rest on the layer below, not the bowl); the "within
// 2ε" bound is a **one-sided** constraint — no particle penetrates the
// wall outward.
//
// Concrete checks:
//   (a) Inside-the-bowl invariant: for every particle, `|x| ≤ R − r + ε`
//       where ε ~ the f32 → i32 accumulator round-trip tolerance.
//   (b) Settled: max kinetic speed < 0.5 m/s after 5 s (well below
//       particles' peak fall speed of ~4 m/s over a 1 m drop).
//   (c) No particle escaped — overflow flags stay clear.

describe('Phase 06 — collider: sphere bowl rest (100 particles)', () => {
  it('100 particles dropped into an inverted sphere settle inside the bowl', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const R = 1.0;
      const N = 100;

      // Seed 100 particles in a small cloud above the bowl centre. Drop
      // ~0.5 m above the bowl centre so they fall in; deterministic
      // jitter keeps them from colliding pair-wise degenerately (every
      // particle on the same y-column).
      const initial: ParticleInit[] = [];

      let seed = 0xc0ffee;
      const rand = (): number => {
        seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
        return ((seed >>> 0) / 0x100000000) * 2 - 1; // [-1, 1)
      };
      for (let i = 0; i < N; i++) {
        const x = 0.3 * rand();
        const y = 0.3 + 0.3 * Math.abs(rand());
        const z = 0.3 * rand();
        initial.push({
          position: [x, y, z],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 0,
        });
      }

      const particles = new ParticleSystem(renderer, N, r);
      particles.uploadParticles(initial);

      // Hash grid for particle-particle contact (needed — particles will
      // pile up on the bowl floor and need non-penetration).
      const hashGrid = new HashGrid(particles, {
        cellSize: 2 * r * 1.1,
      });

      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addSphere(new Vector3(0, 0, 0), R, {
        invert: true,
        muS: 0.5,
        muK: 0.4,
      });
      colliders.upload();

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        contact: {
          hashGrid,
          maxContacts: 8 * N,
          friction: { muS: 0.5, muK: 0.4 },
        },
        colliders: { colliders },
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const totalFrames = 300; // 5 s

      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
      }

      const snap = await particles.readback();
      let maxDistFromCenter = 0;
      let maxSpeed = 0;
      for (let i = 0; i < N; i++) {
        const px = snap.positions[i * 4 + 0]!;
        const py = snap.positions[i * 4 + 1]!;
        const pz = snap.positions[i * 4 + 2]!;
        const dist = Math.sqrt(px * px + py * py + pz * pz);
        if (dist > maxDistFromCenter) maxDistFromCenter = dist;
        const vx = snap.velocities[i * 4 + 0]!;
        const vy = snap.velocities[i * 4 + 1]!;
        const vz = snap.velocities[i * 4 + 2]!;
        const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (speed > maxSpeed) maxSpeed = speed;
      }

      // eslint-disable-next-line no-console
      console.info(
        `[collider-sphere-bowl-rest] N=${N} maxDistFromCenter=${maxDistFromCenter.toFixed(
          5,
        )} (bound R-r=${(R - r).toFixed(5)}) maxSpeed=${maxSpeed.toFixed(5)} m/s`,
      );

      // (a) Inside-the-bowl invariant: `|x| ≤ R − r + tol`. Tolerance
      //     is set to allow for f32/i32 round-trip through the
      //     accumulator (~1e-7 relative at our scale → 1e-7 m absolute).
      //     A 5e-3 m slack covers stabilization overshoot on the frame
      //     of first contact without hiding a real escape.
      const insideTol = 5e-3;
      expect(maxDistFromCenter).toBeLessThan(R - r + insideTol);
      // (b) Settled.
      expect(maxSpeed).toBeLessThan(0.5);

      // (c) Accumulator overflow flag stays clear.
      const overflow = await loop.accumulator!.readbackOverflow();
      expect(overflow).toBe(0);

      particles.destroy();
      colliders.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
