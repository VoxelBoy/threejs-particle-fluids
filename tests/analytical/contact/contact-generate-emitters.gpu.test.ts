import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type Material,
  type ParticleInit,
  type ParticleRange,
} from '../../../src/index.js';

// Contact generation from a subset of emitting particles.
//
// A material that keeps its own particles apart (a fluid's pressure solve)
// reports them as `noSelfContacts`. The loop then lets only the other
// particles ("emitters") search for contacts: a pair with one emitter is
// found from the emitter's side, and a pair of two non-emitters is never
// generated. When no pair of two non-emitters could produce a contact
// anyway, this must find exactly the pairs the all-particles search finds.
//
// Scene: a column of 200 particles resting on a pinned 60-particle slab.
// Column–column pairs share a collision group and slab–slab pairs share
// another (and are both pinned), so every contact is a (slab, column) pair.
// Run one substep with every particle emitting, and one with the column
// marked `noSelfContacts` (only the slab emits), then compare the unordered
// pair sets.
//
// The all-particles search stores `(i, j)` with `i < j`; the emitter search
// stores `(emitter, other)`. Pairs are normalized before comparing, and the
// raw order confirms which search produced them.

describe('contact generation: emitter subset', () => {
  it('finds the same pair set as the all-particles search', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.025;
      const COLUMN_GROUP = 1;
      const SLAB_GROUP = 2;

      const columnCount = 200;
      const slabCount = 60;
      const total = columnCount + slabCount;

      const initial: ParticleInit[] = [];
      // Column. The bottom layer at y = 0.5·spacing sits directly on the
      // slab, within the contact candidate radius (2r · 1.1 = 0.055 m).
      const spacing = 2 * r;
      const columnSide = Math.ceil(Math.cbrt(columnCount));
      for (let i = 0; i < columnCount; i++) {
        const x = i % columnSide;
        const y = Math.floor(i / columnSide) % columnSide;
        const z = Math.floor(i / (columnSide * columnSide));
        initial.push({
          position: [
            -columnSide * spacing * 0.5 + (x + 0.5) * spacing,
            (y + 0.5) * spacing, // y starts at 0.025
            -columnSide * spacing * 0.5 + (z + 0.5) * spacing,
          ],
          velocity: [0, 0, 0],
          invMass: 1,
          collisionGroup: COLUMN_GROUP,
        });
      }
      // Slab at y = −0.025, directly under the column's bottom layer at
      // y = 0.025: centre-to-centre distance 0.05 = 2r, inside the 0.055
      // candidate radius.
      const slabSide = Math.ceil(Math.sqrt(slabCount));
      for (let i = 0; i < slabCount; i++) {
        const x = i % slabSide;
        const z = Math.floor(i / slabSide);
        initial.push({
          position: [
            -slabSide * spacing * 0.5 + (x + 0.5) * spacing,
            -spacing * 0.5,
            -slabSide * spacing * 0.5 + (z + 0.5) * spacing,
          ],
          velocity: [0, 0, 0],
          invMass: 0,
          collisionGroup: SLAB_GROUP,
        });
      }

      const columnRange: ParticleRange = { start: 0, count: columnCount };

      // Build a fresh ParticleSystem + SimLoop, run one substep (which
      // generates contacts once) and read back the stored pairs. With
      // `quiet`, a material marks that range as never contacting itself.
      const runOnce = async (quiet?: ParticleRange): Promise<Uint32Array> => {
        const particles = new ParticleSystem(renderer, total, r);
        particles.uploadParticles(initial);

        const materials: Material[] = quiet ? [{ build: () => ({ noSelfContacts: quiet }) }] : [];
        const loop = new SimLoop(particles, {
          substeps: 1,
          iterations: 1,
          contact: { maxContacts: total * 8 },
          materials,
        });

        await loop.step(1 / 60);
        const { pairs } = await loop.contacts!.readbackPairs();

        loop.dispose();
        particles.dispose();
        return pairs;
      };
      const toSet = (pairs: Uint32Array): Set<string> => {
        const set = new Set<string>();
        for (let p = 0; p < pairs.length / 2; p++) {
          const a = pairs[2 * p]!;
          const b = pairs[2 * p + 1]!;
          set.add(`${Math.min(a, b)}-${Math.max(a, b)}`);
        }
        return set;
      };

      const allPairs = await runOnce();
      const emitterPairs = await runOnce(columnRange);
      const allSet = toSet(allPairs);
      const emitterSet = toSet(emitterPairs);

      console.info(`[contact-generate-emitters] all=${allSet.size} emitters=${emitterSet.size}`);

      expect(emitterSet.size).toBeGreaterThan(0);
      expect(emitterSet.size).toBe(allSet.size);

      // Set equality.
      for (const pair of allSet) {
        expect(emitterSet.has(pair)).toBe(true);
      }
      for (const pair of emitterSet) {
        expect(allSet.has(pair)).toBe(true);
      }

      // The emitter search stored every pair from the slab's side, so it
      // really ran; the all-particles search stored the lower index first.
      for (let p = 0; p < emitterPairs.length / 2; p++) {
        expect(emitterPairs[2 * p]!).toBeGreaterThanOrEqual(columnCount);
      }
      for (let p = 0; p < allPairs.length / 2; p++) {
        expect(allPairs[2 * p]!).toBeLessThan(allPairs[2 * p + 1]!);
      }
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
