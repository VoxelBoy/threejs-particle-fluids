import { Fn, If, instanceIndex, instancedArray, uint } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  emitForEachNeighbor,
  type ParticleInit,
} from '../../../src/core/index.js';

// HashGrid out-of-origin correctness (G1) — motivating-bug regression.
//

//
// Under Morton bucketing, the hash grid still has no declared domain, but
// the Morton encoding has a finite supported cell-coordinate range
// (`±MORTON_BIAS` per axis, currently 512 cells = ±51.2 m at cellSize=0.1).
// Scenes far from the world origin must declare a `hashOrigin` offset to
// recenter cells into that range. This test pins both:
//   - The hashOrigin offset shifts cells into Morton range (overflow=0).
//   - Pairs at sub-cellSize separation are correctly returned as neighbors.

import { Vector3 } from 'three';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

describe('HashGrid: out-of-origin correctness (G1)', () => {
  it('two particles at (10000, 0, 0) and (10000 + h/2, 0, 0) see each other as neighbors with hashOrigin offset', async () => {
    const renderer = await createParticleRenderer();
    try {
      const H = 0.1;
      const OFFSET = 10_000;
      const data: ParticleInit[] = [
        { position: [OFFSET, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [OFFSET + H / 2, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ];
      const particles = new ParticleSystem(renderer, 2, 0.02);
      particles.uploadParticles(data);

      // Scenes far from world origin should declare hashOrigin so cells
      // land inside the Morton encoding's supported range. Without this
      // offset, both particles would hash to wrap-aliased buckets — the
      // simulation would still work (per-pair distance filter would still
      // accept them) but the overflow flag would fire.
      const grid = new HashGrid(particles, {
        cellSize: H,
        hashOrigin: new Vector3(OFFSET, 0, 0),
      });
      await grid.rebuild();

      const countsOut = instancedArray(2, 'uint');
      const countKernel = Fn(() => {
        const p: Any = instanceIndex;
        const pos: Any = particles.positions.element(p).xyz;
        const count: Any = uint(0).toVar();
        emitForEachNeighbor({
          queryPosXyz: pos,
          hashOrigin: grid.hashOriginUniform,
          cellSize: grid.cellSizeUniform,
          hashTableSize: grid.hashTableSize,
          cellStart: grid.cellStart,
          cellEnd: grid.cellEnd,
          sortedIndices: grid.sortedIndices,
          onCandidate: (n) => {
            If((n as Any).notEqual(p), () => {
              count.addAssign(uint(1));
            });
          },
        });
        countsOut.element(p).assign(count);
      })().compute(2);

      await renderer.computeAsync(countKernel);
      const gpu = new Uint32Array(await renderer.getArrayBufferAsync(countsOut.value));

      // Each particle should see exactly the other — no more, no fewer.
      expect(gpu[0]).toBe(1);
      expect(gpu[1]).toBe(1);

      // Overflow flag must stay 0 — the hashOrigin offset places the
      // particles at cell ~0 (well inside the Morton range) and ~10⁴ m
      // is far inside the f32→i32 saturation range at cellSize = 0.1.
      expect(await grid.readbackOverflow()).toBe(0);

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
