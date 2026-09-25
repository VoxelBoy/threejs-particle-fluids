import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 05 G3 — non-interpenetration invariant (BLOCKING, plan §Exit criteria).
//
// Plan §Validation: "After a full step, all contact pairs satisfy |x_i − x_j|
// ≥ r_i + r_j − ε (ε = small tolerance for floating-point)."
//
// Scene: small pile of dynamic particles dropped on the Phase-02 floor clamp.
// After each frame we readback every particle and verify no pair has
// `|x_i − x_j| < 2r − ε`. This covers the invariant for the ENTIRE pair
// graph, not just emitted contacts — a correctly functioning contact pipeline
// keeps even pairs the emitter doesn't claim well-separated (the emission
// filter is `dist ≤ 2r·radiusExpansion`; non-emitted pairs are above that
// threshold by definition, which is > 2r).
//
// Tier choice (G4): not a determinism test; this is a G3 structural
// invariant. Every frame of the run must satisfy it.

describe('Phase 05 — contact: non-interpenetration invariant', () => {
  it('dropped pile never has overlapping pairs after a full step', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const twoR = 2 * r;
      const epsilon = 1e-3;

      // Compact cluster of 27 particles above a floor at y=0, dropped into
      // a pile. 3×3×3 grid with spacing slightly larger than 2r to start
      // strictly non-overlapping.
      const spacing = twoR * 1.05;
      const initial: ParticleInit[] = [];
      for (let ix = 0; ix < 3; ix++) {
        for (let iy = 0; iy < 3; iy++) {
          for (let iz = 0; iz < 3; iz++) {
            initial.push({
              position: [(ix - 1) * spacing, 0.5 + iy * spacing, (iz - 1) * spacing],
              velocity: [0, 0, 0],
              invMass: 1,
              phase: 0,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const hashGrid = new HashGrid(particles, {
        cellSize: twoR * 1.1,
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        contact: {
          hashGrid,
          maxContacts: 256, // plenty for 27 particles in a tight cluster
          friction: { muS: 0.4, muK: 0.3 },
        },
      });
      // Floor at y = 0 (default).
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const totalFrames = 90; // 1.5 s — enough to bottom out + settle

      let worstOverlap = 0; // twoR − min(dist) across all frames + pairs
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

      // eslint-disable-next-line no-console
      console.info(
        `[contact-non-interpenetration] worst overlap=${worstOverlap.toExponential(3)} ` +
          `at frame ${worstFrame} pair=(${worstPair[0]},${worstPair[1]}) ` +
          `(tolerance ε=${epsilon.toExponential(3)}, 2r=${twoR})`,
      );

      // Plan's ε: "small tolerance for floating-point." 1e-3 is ~1 ULP of
      // the r=0.05 scale on the XPBD's iterative residual.
      expect(worstOverlap).toBeLessThan(epsilon);

      particles.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
