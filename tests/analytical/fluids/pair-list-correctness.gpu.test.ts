// Phase Perf — pair-list correctness G1.
//
// For a settled-cube scene at known h and spacing, the pair-list
// build kernel must populate the per-particle slot range with exactly
// the set of within-h neighbors that a CPU brute-force search would
// find. The slot order depends on the hash-grid 27-cell walk + Morton
// bucket sequence + sortedIndices order, so we compare as sets rather
// than sequences. `pairCount[i]` must equal the cardinality of that
// set per particle. Layout is column-major (Phase Perf-18): slot k of
// particle i lives at `pairList[k * fluidCount + i]`.
//

import { describe, expect, it } from 'vitest';
import { uniform } from 'three/tsl';
import {
  HashGrid,
  ParticleSystem,
  allocatePairListStorage,
  buildPairListKernel,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

describe('Phase Perf — pair-list build correctness', () => {
  it('matches CPU brute-force neighbor sets on a 5×5×5 lattice', async () => {
    const renderer = await createParticleRenderer();
    try {
      // 5³ = 125 particles. Spacing 0.025, h = 0.04. h² = 0.0016 sits
      // between the 2-axis pair distance² (0.00125) and the 3-axis
      // pair distance² (0.001875), so f32/f64 precision differences
      // cannot flip a pair's neighbor status. Self is included in the
      // pair list per paper density formula.
      const LATTICE = 5;
      const SPACING = 0.025;
      const H = 0.04;
      const H_SQ = H * H;
      const N = LATTICE ** 3;

      const initial: ParticleInit[] = [];
      for (let z = 0; z < LATTICE; z++) {
        for (let y = 0; y < LATTICE; y++) {
          for (let x = 0; x < LATTICE; x++) {
            initial.push({
              position: [x * SPACING, y * SPACING, z * SPACING],
              velocity: [0, 0, 0],
              invMass: 1,
              phase: 0,
            });
          }
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

      const [pairListBuf, pairCountBuf, overflowBuf] = await Promise.all([
        renderer.getArrayBufferAsync(storage.pairList.value),
        renderer.getArrayBufferAsync(storage.pairCount.value),
        renderer.getArrayBufferAsync(storage.pairOverflowFlag.value),
      ]);
      const pairList = new Uint32Array(pairListBuf);
      const pairCount = new Uint32Array(pairCountBuf);
      const overflow = new Uint32Array(overflowBuf)[0]!;

      expect(overflow).toBe(0);

      // CPU brute-force ground truth.
      const expected: Set<number>[] = [];
      for (let i = 0; i < N; i++) {
        const set = new Set<number>();
        const xi = initial[i]!.position;
        for (let j = 0; j < N; j++) {
          const xj = initial[j]!.position;
          const dx = xi[0] - xj[0];
          const dy = xi[1] - xj[1];
          const dz = xi[2] - xj[2];
          if (dx * dx + dy * dy + dz * dz < H_SQ) set.add(j);
        }
        expected.push(set);
      }

      // Column-major layout (Phase Perf-18): slot k for particle i lives
      // at `pairList[k * N + i]`, where N = fluidCount. The buffer is
      // sized to `N * MAX_NEIGHBORS` so the last valid k is
      // `MAX_NEIGHBORS - 1`.
      for (let i = 0; i < N; i++) {
        const count = pairCount[i]!;
        expect(count).toBe(expected[i]!.size);

        const observed = new Set<number>();
        for (let k = 0; k < count; k++) {
          observed.add(pairList[k * N + i]!);
        }
        expect(observed).toEqual(expected[i]);
      }

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
