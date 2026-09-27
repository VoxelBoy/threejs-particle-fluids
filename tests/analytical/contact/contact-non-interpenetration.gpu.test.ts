import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Non-interpenetration invariant: after every step, every particle pair
// satisfies `|x_i − x_j| ≥ r_i + r_j − ε` (ε = a small floating-point
// tolerance).
//
// Scene: a small pile of dynamic particles dropped onto a floor plane. After
// each frame every particle is read back and no pair may have
// `|x_i − x_j| < 2r − ε`. This covers the invariant for the ENTIRE pair
// graph, not just emitted contacts — a correct contact pipeline keeps even
// pairs it didn't emit well separated (pairs are emitted within
// `2r · CONTACT_RADIUS_EXPANSION`, which is > 2r, so every pair closer than
// 2r is a candidate).
//
// This is a structural invariant, not a determinism test: every frame of the
// run must satisfy it.

describe('contact: non-interpenetration invariant', () => {
  it('dropped pile never has overlapping pairs after a full step', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const twoR = 2 * r;
      // ε: the solver's residual overlap, at the r = 0.05 scale. The worst
      // case is the frame the ~3 m/s column hits the floor. The floor plane
      // is a constraint in the same iterations as the contacts, so after 4
      // Jacobi iterations the stack keeps up to ~1.7 mm of pair overlap on
      // that frame (a floor that only clamps predicted positions, and gives
      // way during the iterations, leaves ~0.9 mm). Every other frame stays
      // below 0.7 mm.
      const epsilon = 2e-3;

      // Compact 3×3×3 cluster above the floor, dropped into a pile. Spacing
      // slightly larger than 2r so the cluster starts strictly non-overlapping.
      const spacing = twoR * 1.05;
      const initial: ParticleInit[] = [];
      for (let ix = 0; ix < 3; ix++) {
        for (let iy = 0; iy < 3; iy++) {
          for (let iz = 0; iz < 3; iz++) {
            initial.push({
              position: [(ix - 1) * spacing, 0.5 + iy * spacing, (iz - 1) * spacing],
              velocity: [0, 0, 0],
              invMass: 1,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      // Floor at y = 0.
      const floor = new PrimitiveSet(particles);
      floor.addPlane(new Vector3(0, 1, 0), new Vector3());

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        contact: {
          maxContacts: 256, // plenty for 27 particles in a tight cluster
          muS: 0.4,
          muK: 0.3,
        },
        colliders: [floor],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const totalFrames = 90; // 1.5 s — enough to bottom out and settle

      let worstOverlap = 0; // twoR − min(dist) across all frames and pairs
      let worstPair: [number, number] = [-1, -1];
      let worstFrame = -1;

      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
        const snap = await particles.readback();
        for (let a = 0; a < initial.length; a++) {
          for (let b = a + 1; b < initial.length; b++) {
            const dx = snap.positions[a * 4]! - snap.positions[b * 4]!;
            const dy = snap.positions[a * 4 + 1]! - snap.positions[b * 4 + 1]!;
            const dz = snap.positions[a * 4 + 2]! - snap.positions[b * 4 + 2]!;
            const dist = Math.hypot(dx, dy, dz);
            const overlap = twoR - dist;
            if (overlap > worstOverlap) {
              worstOverlap = overlap;
              worstPair = [a, b];
              worstFrame = n;
            }
          }
        }
      }

      console.info(
        `[contact-non-interpenetration] worst overlap=${worstOverlap.toExponential(3)} ` +
          `at frame ${worstFrame} pair=(${worstPair[0]},${worstPair[1]}) ` +
          `(tolerance ε=${epsilon.toExponential(3)}, 2r=${twoR})`,
      );

      expect(worstOverlap).toBeLessThan(epsilon);

      loop.dispose();
      particles.dispose();
      floor.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
