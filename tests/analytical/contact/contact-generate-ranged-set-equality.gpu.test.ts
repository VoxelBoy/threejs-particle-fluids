import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
  type ParticleRange,
} from '../../../src/core/index.js';

// Phase Perf-14 G1 — pair-set equality between legacy and ranged
// contact.generate kernels.
//
// Build a small mixed scene (fluid + boundary), run one substep through
// each kernel variant, read back the contact pairs, and assert the two
// kernels emit the SAME unordered set of pairs.
//

//
// The legacy kernel emits pairs with `i < j` always; the ranged kernel
// may emit `(i, j)` with `j < i` when `j ∉ E` (the emit set), so the
// normalize-then-sort step is mandatory.

describe('Phase Perf-14 — contact.generate ranged-vs-legacy set equality', () => {
  it('emits identical (unordered) pair sets', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.025;
      const phaseFor = (id: number): number => ((id & 0xffff) << 16) >>> 0;
      const PHASE_FLUID = 1;
      const PHASE_BOUNDARY = 2;

      // Small scene: 200 fluid in a column, 60 boundary in a slab below.
      // Pair-set will be entirely (boundary, fluid) cross-pairs because
      // fluid-fluid pairs are phase-suppressed (PHASE_FLUID == PHASE_FLUID
      // and group != 0) and boundary-boundary pairs share group too.
      const fluidCount = 200;
      const boundaryCount = 60;
      const total = fluidCount + boundaryCount;

      const initial: ParticleInit[] = [];
      // Fluid column. Bottom layer at y = 0.5*SPACING so it directly
      // sits on the boundary slab — neighbors within the candidate
      // radius (2*r*1.1 = 0.055 m).
      const SPACING = 2 * r;
      const fluidSide = Math.ceil(Math.cbrt(fluidCount));
      for (let i = 0; i < fluidCount; i++) {
        const x = i % fluidSide;
        const y = Math.floor(i / fluidSide) % fluidSide;
        const z = Math.floor(i / (fluidSide * fluidSide));
        initial.push({
          position: [
            -fluidSide * SPACING * 0.5 + (x + 0.5) * SPACING,
            (y + 0.5) * SPACING, // y starts at 0.025
            -fluidSide * SPACING * 0.5 + (z + 0.5) * SPACING,
          ],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: phaseFor(PHASE_FLUID),
        });
      }
      // Boundary slab — at y = -0.025, directly under the fluid bottom
      // layer at y = 0.025. Center-to-center distance = 0.05 = 2*r,
      // inside the 0.055 candidate radius.
      const bSide = Math.ceil(Math.sqrt(boundaryCount));
      for (let i = 0; i < boundaryCount; i++) {
        const x = i % bSide;
        const z = Math.floor(i / bSide);
        initial.push({
          position: [
            -bSide * SPACING * 0.5 + (x + 0.5) * SPACING,
            -SPACING * 0.5,
            -bSide * SPACING * 0.5 + (z + 0.5) * SPACING,
          ],
          velocity: [0, 0, 0],
          invMass: 0,
          phase: phaseFor(PHASE_BOUNDARY),
        });
      }

      const boundaryRange: ParticleRange = {
        start: fluidCount,
        count: boundaryCount,
      };

      // Helper: build a fresh ParticleSystem + SimLoop, run one substep
      // (which dispatches contact.generate once), read back the pairs.
      const runOnce = async (emittingRanges?: readonly ParticleRange[]): Promise<Set<string>> => {
        const particles = new ParticleSystem(renderer, total, r);
        particles.uploadParticles(initial);

        const hashGrid = new HashGrid(particles, { cellSize: 4 * r });
        const colliders = new PrimitiveSet(particles, { capacity: 1 });
        colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
        colliders.upload();

        const xpbd = createXpbdUniforms(1 / 60);
        const loop = new SimLoop(particles, {
          substeps: 1,
          iterations: 1,
          xpbd,
          hashGrid,
          contact: {
            hashGrid,
            maxContacts: total * 8,
            ...(emittingRanges ? { emittingRanges } : {}),
          },
          colliders: { colliders },
        });

        await loop.step(1 / 60);

        const { pairs, nContacts } = await loop.contacts!.readbackPairs();
        const set = new Set<string>();
        for (let p = 0; p < nContacts; p++) {
          const a = pairs[2 * p]!;
          const b = pairs[2 * p + 1]!;
          const lo = Math.min(a, b);
          const hi = Math.max(a, b);
          set.add(`${lo}-${hi}`);
        }

        return set;
      };

      const legacySet = await runOnce(undefined);
      const rangedSet = await runOnce([boundaryRange]);

      // eslint-disable-next-line no-console
      console.log(
        `[contact-generate-ranged-set-equality] legacy=${legacySet.size} ranged=${rangedSet.size}`,
      );

      expect(rangedSet.size).toBeGreaterThan(0);
      expect(rangedSet.size).toBe(legacySet.size);

      // Set equality.
      for (const pair of legacySet) {
        expect(rangedSet.has(pair)).toBe(true);
      }
      for (const pair of rangedSet) {
        expect(legacySet.has(pair)).toBe(true);
      }
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
