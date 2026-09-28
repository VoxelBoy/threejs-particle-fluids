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

// Box container: particles poured into an inverted box of size (1, 1, 1)
// must, after settling, all be strictly inside it (the inverted SDF keeps
// them in), with no wall penetration.
//
// 1k particles keep the runtime short; the invariant is per particle and
// holds at any count.

describe('collider: box container (1000 particles)', () => {
  it('particles poured into an inverted box settle inside with no wall penetration', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.02;
      const N = 1000;
      const halfExtents = new Vector3(0.5, 0.5, 0.5);

      // Deterministic pseudo-random seeding.
      let seed = 0xdecafbad;
      const rand = (): number => {
        seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
        return (seed >>> 0) / 0x100000000;
      };
      const initial: ParticleInit[] = [];
      for (let i = 0; i < N; i++) {
        const x = (rand() * 2 - 1) * 0.4;
        const y = 0.2 + rand() * 0.25;
        const z = (rand() * 2 - 1) * 0.4;
        initial.push({ position: [x, y, z], velocity: [0, 0, 0], invMass: 1 });
      }

      const particles = new ParticleSystem(renderer, N, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles);
      colliders.addBox(new Vector3(0, 0, 0), halfExtents, {
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

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        contact: { maxContacts: 8 * N },
        colliders: [observed],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      // 2 s is plenty to settle 1k small particles falling ~0.5 m.
      const totalFrames = 120;
      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
      }

      const snap = await particles.readback();
      let maxOverrun = 0;
      let maxSpeed = 0;
      let escaped = 0;
      for (let i = 0; i < N; i++) {
        const px = snap.positions[i * 4 + 0]!;
        const py = snap.positions[i * 4 + 1]!;
        const pz = snap.positions[i * 4 + 2]!;
        // Outward penetration: particle centre outside the half-extents by
        // more than `r`. Zero = strictly inside.
        const overrun = Math.max(
          Math.abs(px) - (halfExtents.x - r),
          Math.abs(py) - (halfExtents.y - r),
          Math.abs(pz) - (halfExtents.z - r),
          0,
        );
        if (overrun > maxOverrun) maxOverrun = overrun;
        if (overrun > 0) escaped++;
        const vx = snap.velocities[i * 4 + 0]!;
        const vy = snap.velocities[i * 4 + 1]!;
        const vz = snap.velocities[i * 4 + 2]!;
        const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (speed > maxSpeed) maxSpeed = speed;
      }

      console.info(
        `[collider-box-container] N=${N} maxOverrun=${maxOverrun.toFixed(
          5,
        )} escaped=${escaped} maxSpeed=${maxSpeed.toFixed(5)} m/s`,
      );

      // Tolerance: f32/i32 accumulator round-trip plus the one-pass
      // stabilization overshoot. 2e-3 m is well below the particle radius.
      const tol = 2e-3;
      expect(maxOverrun).toBeLessThan(tol);
      // A reasonable settled threshold for small particles with friction.
      expect(maxSpeed).toBeLessThan(0.5);

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
