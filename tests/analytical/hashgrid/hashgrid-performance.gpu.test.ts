import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 03 — informational performance floor. Plan §Validation —
// "100k particles, build + query in < 2 ms on an RTX 3060 / M1 Pro.
//  This is a ballpark sanity check; missing it is a signal to optimize, not
//  to block the phase."
//
// We measure `rebuild()` only. Query-side performance depends on the caller
// kernel and is the responsibility of later phases. Warmup + 20 timed runs,
// log the median so TSL/WebGPU first-dispatch compilation doesn't skew the
// number. No assertion on the timing itself — the floor is informational.

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

describe('Phase 03 — HashGrid: performance (informational)', () => {
  it('logs median build time over 100k particles', async () => {
    const renderer = await createParticleRenderer();
    try {
      const N = 100_000;
      const rand = lcg(0xf00dbabe);
      const data: ParticleInit[] = [];
      for (let i = 0; i < N; i++) {
        data.push({
          position: [rand(), rand(), rand()],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 0,
        });
      }

      const particles = new ParticleSystem(renderer, N, 0.005);
      particles.uploadParticles(data);
      const grid = new HashGrid(particles, {
        cellSize: 0.05,
      });

      // Warmup: forces kernel compilation and first-dispatch allocation.
      await grid.rebuild();
      await renderer.getArrayBufferAsync(grid.cellStart.value);

      const runs: number[] = [];
      for (let i = 0; i < 20; i++) {
        const t0 = performance.now();
        await grid.rebuild();
        // Force synchronization so wall-clock reflects GPU completion, not
        // just command submission. `getArrayBufferAsync` is the same barrier
        // used elsewhere in the test suite.
        await renderer.getArrayBufferAsync(grid.cellStart.value);
        runs.push(performance.now() - t0);
      }
      runs.sort((a, b) => a - b);
      const median = runs[Math.floor(runs.length / 2)]!;
      const best = runs[0]!;
      const worst = runs[runs.length - 1]!;

      // eslint-disable-next-line no-console
      console.info(
        `[hashgrid-performance] N=${N} hashTableSize=${grid.hashTableSize} padded=${grid.hashTableSizePadded} ` +
          `build median=${median.toFixed(2)}ms best=${best.toFixed(2)}ms worst=${worst.toFixed(2)}ms`,
      );

      // Liveness-only assertion — the timing itself is not gated.
      expect(median).toBeGreaterThan(0);

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
