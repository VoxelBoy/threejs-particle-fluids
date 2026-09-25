// Phase Perf — pair-list overflow flag G1.
//
// When a particle has more than MAX_NEIGHBORS within-h candidates,
// the build kernel must (a) set `pairOverflowFlag` to 1 and (b)
// truncate that particle's pair list to MAX_NEIGHBORS entries
// without corrupting other particles' lists.
//

import { describe, expect, it } from 'vitest';
import { uniform } from 'three/tsl';
import {
  HashGrid,
  MAX_NEIGHBORS,
  ParticleSystem,
  allocatePairListStorage,
  buildPairListKernel,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

describe('Phase Perf — pair-list overflow flag', () => {
  it('fires when a particle exceeds MAX_NEIGHBORS within h', async () => {
    const renderer = await createParticleRenderer();
    try {
      // Cluster of N = MAX_NEIGHBORS + 32 particles all within h of
      // each other — guarantees every particle has > MAX_NEIGHBORS
      // candidates in its pair list. We use a tiny lattice spacing so
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
            initial.push({
              position: [x * SPACING, y * SPACING, z * SPACING],
              velocity: [0, 0, 0],
              invMass: 1,
              phase: 0,
            });
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
      const storage = allocatePairListStorage(N);
      const hSqU = uniform(H_SQ, 'float');
      const buildKernel = buildPairListKernel({
        particles,
        hashGrid: grid,
        hSq: hSqU,
        fluidParticles: { start: 0, count: N },
        ...storage,
      });

      await renderer.computeAsync([...grid.rebuildPipeline, buildKernel]);

      const [pairCountBuf, overflowBuf] = await Promise.all([
        renderer.getArrayBufferAsync(storage.pairCount.value),
        renderer.getArrayBufferAsync(storage.pairOverflowFlag.value),
      ]);
      const pairCount = new Uint32Array(pairCountBuf);
      const overflow = new Uint32Array(overflowBuf)[0]!;

      expect(overflow).toBe(1);
      // Every particle's count must be clamped to MAX_NEIGHBORS — they
      // all have N - 1 = MAX_NEIGHBORS + 31 actual candidates within h.
      for (let i = 0; i < N; i++) {
        expect(pairCount[i]).toBe(MAX_NEIGHBORS);
      }

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
