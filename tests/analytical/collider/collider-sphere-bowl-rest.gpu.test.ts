import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type Accumulator,
  type Collider,
  type ParticleInit,
} from '../../../src/index.js';

// Sphere bowl rest: 100 particles dropped into an inverted sphere (a bowl)
// of radius 1 must, after settling for 5 s, all be on the inside of the
// bowl wall.
//
// This is the non-penetration invariant evaluated on an inverted SDF: every
// particle's surface must be at or inside the bowl wall. Particles in the
// middle of the pile may be far from the wall (they rest on the layer
// below, not on the bowl), so the bound is one-sided — no particle
// penetrates the wall outward.
//
// Concrete checks:
//   (a) Inside the bowl: for every particle, `|x| ≤ R − r + ε`, where ε
//       covers the f32 → i32 accumulator round-trip.
//   (b) Settled: max speed < 0.5 m/s after 5 s (well below the peak fall
//       speed of ~4 m/s over a 1 m drop).
//   (c) Nothing blew up — the position accumulator's overflow flag is clear.

describe('collider: sphere bowl rest (100 particles)', () => {
  it('100 particles dropped into an inverted sphere settle inside the bowl', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const R = 1.0;
      const N = 100;

      // Seed 100 particles in a small cloud above the bowl centre, so they
      // fall in; deterministic jitter keeps them from stacking degenerately
      // on the same vertical column.
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
        initial.push({ position: [x, y, z], velocity: [0, 0, 0], invMass: 1 });
      }

      const particles = new ParticleSystem(renderer, N, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles);
      colliders.addSphere(new Vector3(0, 0, 0), R, {
        invert: true,
        muS: 0.5,
        muK: 0.4,
      });

      // The loop's position accumulator is internal; capture it from the
      // context the loop passes to its colliders so its overflow flag can
      // be checked at the end.
      let positionAccumulator: Accumulator | undefined;
      const observed: Collider = {
        particles,
        update: (dt) => colliders.update(dt),
        buildKernels: (context) => {
          positionAccumulator = context.positions;
          return colliders.buildKernels(context);
        },
        dispose: () => colliders.dispose(),
      };

      // Particle contacts are needed: the particles pile up on the bowl
      // floor and must not overlap.
      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        contact: { maxContacts: 8 * N, muS: 0.5, muK: 0.4 },
        colliders: [observed],
      });
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

      console.info(
        `[collider-sphere-bowl-rest] N=${N} maxDistFromCenter=${maxDistFromCenter.toFixed(
          5,
        )} (bound R-r=${(R - r).toFixed(5)}) maxSpeed=${maxSpeed.toFixed(5)} m/s`,
      );

      // (a) Inside the bowl: `|x| ≤ R − r + tol`. The accumulator round-trip
      //     alone is ~1e-7 m at this scale; 5e-3 m of slack covers the
      //     stabilization overshoot on the frame of first contact without
      //     hiding a real escape.
      const insideTol = 5e-3;
      expect(maxDistFromCenter).toBeLessThan(R - r + insideTol);
      // (b) Settled.
      expect(maxSpeed).toBeLessThan(0.5);

      // (c) The position accumulator's overflow flag is clear.
      const overflow = await positionAccumulator!.readbackOverflow();
      expect(overflow).toBe(false);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
