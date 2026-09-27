// Neighbor-list overflow flag.
//
// When a particle has more than MAX_NEIGHBORS neighbors within the
// radius, the build kernel must (a) set the overflow flag and (b)
// truncate that particle's list to MAX_NEIGHBORS entries without
// corrupting other particles' lists.
//

import { describe, expect, it } from 'vitest';
import { uniform } from 'three/tsl';
import {
  HashGrid,
  MAX_NEIGHBORS,
  NeighborList,
  ParticleSystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

describe('neighbor list overflow flag', () => {
  it('fires when a particle exceeds MAX_NEIGHBORS within h', async () => {
    const renderer = await createParticleRenderer();
    try {
      // Cluster of N = MAX_NEIGHBORS + 32 particles all within h of
      // each other — guarantees every particle has > MAX_NEIGHBORS
      // candidates in its list. We use a tiny lattice spacing so
      // every pair is within h.
      const N = MAX_NEIGHBORS + 32;
      const H = 1.0;
      const H_SQ = H * H;
      // Pack into a small box ≪ h on each axis so every pair is
      // within range; verify by brute force below.
      const SIDE = Math.ceil(Math.cbrt(N));
      const SPACING = 0.05; // SIDE * SPACING ≪ H = 1.0

      const initial: ParticleInit[] = [];
      for (let z = 0; z < SIDE && initial.length < N; z++) {
        for (let y = 0; y < SIDE && initial.length < N; y++) {
          for (let x = 0; x < SIDE && initial.length < N; x++) {
            initial.push({ position: [x * SPACING, y * SPACING, z * SPACING] });
          }
        }
      }
      expect(initial.length).toBe(N);

      // Sanity check: every pair should be within h.
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const xi = initial[i]!.position;
          const xj = initial[j]!.position;
          const dx = xi[0] - xj[0];
          const dy = xi[1] - xj[1];
          const dz = xi[2] - xj[2];
          expect(dx * dx + dy * dy + dz * dz).toBeLessThan(H_SQ);
        }
      }

      const particles = new ParticleSystem(renderer, N, SPACING * 0.5);
      particles.uploadParticles(initial);

      const grid = new HashGrid(particles, { cellSize: H });
      const neighbors = new NeighborList(particles, { start: 0, count: N });
      const buildKernels = neighbors.buildKernels(grid, uniform(H_SQ, 'float'));

      await renderer.computeAsync([...grid.rebuildPipeline, ...buildKernels]);

      const counts = new Uint32Array(await renderer.getArrayBufferAsync(neighbors.counts.value));

      expect(await neighbors.readbackOverflow()).toBe(true);
      // Every particle's count must be clamped to MAX_NEIGHBORS — they
      // all have N = MAX_NEIGHBORS + 32 actual neighbors within h
      // (self included).
      for (let i = 0; i < N; i++) {
        expect(counts[i]).toBe(MAX_NEIGHBORS);
      }

      grid.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
