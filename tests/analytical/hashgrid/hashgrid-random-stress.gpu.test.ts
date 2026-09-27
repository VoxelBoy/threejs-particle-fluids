import { Fn, If, instanceIndex, instancedArray, uint } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  emitForEachNeighbor,
  type ParticleInit,
} from '../../../src/index.js';

// Random stress. 100k particles uniformly distributed in the unit cube,
// h=0.05 (ε=0). Sample 100 query particles; CPU brute force gives the ground
// truth. Uses a per-query neighbor count — comparing the count (not the set)
// is sufficient because the mapping from sortedIndices to candidate
// neighbors is deterministic up to intra-cell order, and we filter by
// distance² < h² anyway.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

describe('HashGrid: 100k random stress', () => {
  it('per-particle neighbor count matches CPU brute force on 100 samples', async () => {
    const renderer = await createParticleRenderer();
    try {
      const N = 100_000;
      const H = 0.05;
      const H2 = H * H;
      const rand = lcg(0xc0ffee);
      const data: ParticleInit[] = [];
      // Keep particles slightly inside the unit cube so the per-pair distance
      // check is independent of boundary clamp behavior — bounds handling has
      // its own dedicated test.
      const MARGIN = 0.01;
      for (let i = 0; i < N; i++) {
        data.push({
          position: [
            MARGIN + rand() * (1 - 2 * MARGIN),
            MARGIN + rand() * (1 - 2 * MARGIN),
            MARGIN + rand() * (1 - 2 * MARGIN),
          ],
          velocity: [0, 0, 0],
          invMass: 1,
        });
      }

      const particles = new ParticleSystem(renderer, N, 0.005);
      particles.uploadParticles(data);

      const grid = new HashGrid(particles, {
        cellSize: H,
      });
      await grid.rebuild();

      const countsOut = instancedArray(N, 'uint');
      const countKernel = Fn(() => {
        const p: Any = instanceIndex;
        const pos: Any = particles.positions.element(p).xyz;
        const count: Any = uint(0).toVar();
        emitForEachNeighbor(grid, pos, (n) => {
          If(n.notEqual(p), () => {
            const npos: Any = particles.positions.element(n).xyz;
            const diff: Any = pos.sub(npos);
            const d2: Any = diff.dot(diff);
            If(d2.lessThan(H2), () => {
              count.addAssign(uint(1));
            });
          });
        });
        countsOut.element(p).assign(count);
      })().compute(N);

      await renderer.computeAsync(countKernel);
      const gpuCounts = new Uint32Array(await renderer.getArrayBufferAsync(countsOut.value));

      // Sample 100 particles deterministically and brute-force each against
      // the full 100k set. That's 10M pair comparisons — ~100 ms on a modern
      // laptop JS runtime, well within the test budget.
      const sampleRand = lcg(0xbeef_cafe);
      const SAMPLES = 100;
      for (let s = 0; s < SAMPLES; s++) {
        const p = Math.floor(sampleRand() * N);
        const dp = data[p]!.position;
        let cpuCount = 0;
        for (let q = 0; q < N; q++) {
          if (q === p) continue;
          const dq = data[q]!.position;
          const dx = dp[0] - dq[0];
          const dy = dp[1] - dq[1];
          const dz = dp[2] - dq[2];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < H2) cpuCount++;
        }
        expect(gpuCounts[p]).toBe(cpuCount);
      }

      grid.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
