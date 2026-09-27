// Neighbor-list correctness.
//
// For a lattice at known spacing and radius, the neighbor-list build
// kernel must store, for every particle, exactly the set of neighbors a
// CPU brute-force search finds within the radius. The slot order depends
// on the hash grid's 27-cell walk, Morton bucket order and sort order, so
// the lists are compared as sets. `counts[i]` must equal the size of that
// set. Storage is column-major: neighbor k of particle i lives at
// `indices[k * count + i]`.
//

import { describe, expect, it } from 'vitest';
import { uniform } from 'three/tsl';
import {
  HashGrid,
  NeighborList,
  ParticleSystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

describe('neighbor list build', () => {
  it('matches CPU brute-force neighbor sets on a 5×5×5 lattice', async () => {
    const renderer = await createParticleRenderer();
    try {
      // 5³ = 125 particles. Spacing 0.025, h = 0.04. h² = 0.0016 sits
      // between the 2-axis pair distance² (0.00125) and the 3-axis
      // pair distance² (0.001875), so f32/f64 precision differences
      // cannot flip a pair's neighbor status. Each particle is its own
      // neighbor, as the SPH density sum requires.
      const LATTICE = 5;
      const SPACING = 0.025;
      const H = 0.04;
      const H_SQ = H * H;
      const N = LATTICE ** 3;

      const initial: ParticleInit[] = [];
      for (let z = 0; z < LATTICE; z++) {
        for (let y = 0; y < LATTICE; y++) {
          for (let x = 0; x < LATTICE; x++) {
            initial.push({ position: [x * SPACING, y * SPACING, z * SPACING] });
          }
        }
      }

      const particles = new ParticleSystem(renderer, N, SPACING * 0.5);
      particles.uploadParticles(initial);

      const grid = new HashGrid(particles, { cellSize: H });
      const neighbors = new NeighborList(particles, { start: 0, count: N });
      const buildKernels = neighbors.buildKernels(grid, uniform(H_SQ, 'float'));

      await renderer.computeAsync([...grid.rebuildPipeline, ...buildKernels]);

      const [indicesBuf, countsBuf] = await Promise.all([
        renderer.getArrayBufferAsync(neighbors.indices.value),
        renderer.getArrayBufferAsync(neighbors.counts.value),
      ]);
      const indices = new Uint32Array(indicesBuf);
      const counts = new Uint32Array(countsBuf);

      expect(await neighbors.readbackOverflow()).toBe(false);

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

      // Column-major layout: neighbor k of particle i lives at
      // `indices[k * N + i]`. The buffer holds `N * MAX_NEIGHBORS`
      // entries, so the last valid k is `MAX_NEIGHBORS - 1`.
      for (let i = 0; i < N; i++) {
        const count = counts[i]!;
        expect(count).toBe(expected[i]!.size);

        const observed = new Set<number>();
        for (let k = 0; k < count; k++) {
          observed.add(indices[k * N + i]!);
        }
        expect(observed).toEqual(expected[i]);
      }

      grid.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
