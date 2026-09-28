import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';
import { MORTON_BIAS } from '../../../src/core/hashGrid/mortonHash.js';

// Morton encoding correctness.
//
// Hand-verified Morton encoding for a small set of (cx, cy, cz) cell-coord
// triples covering origin, axis-aligned offsets, and edge cases at the
// supported [-MORTON_BIAS, +MORTON_BIAS-1] range. The reference encoding
// is computed CPU-side using the same bit-spreading algorithm as
// `mortonHash.ts`; we then check that particles placed at those cells
// land in the bucket the encoding predicts.
//
// This pins the GPU↔CPU bit-exactness of the Morton encoding so any
// future refactor that breaks the encoding fails loudly.

function part1by2CPU(n: number): number {
  let m = n & 0x000003ff;
  m = (m | (m << 16)) & 0xff0000ff;
  m = (m | (m << 8)) & 0x0300f00f;
  m = (m | (m << 4)) & 0x030c30c3;
  m = (m | (m << 2)) & 0x09249249;
  return m >>> 0;
}

function mortonCPU(cx: number, cy: number, cz: number): number {
  const ux = (cx + MORTON_BIAS) >>> 0;
  const uy = (cy + MORTON_BIAS) >>> 0;
  const uz = (cz + MORTON_BIAS) >>> 0;
  return (part1by2CPU(ux) | (part1by2CPU(uy) << 1) | (part1by2CPU(uz) << 2)) >>> 0;
}

describe('HashGrid: Morton bucket encoding', () => {
  it('GPU bucket index matches CPU Morton encoding for hand-verified cells', async () => {
    const renderer = await createParticleRenderer();
    try {
      const CELL = 0.1;
      // Test set of (cx, cy, cz) cell coords covering origin, single-axis
      // unit offsets, mixed offsets, and edge-of-range values.
      const testCells: readonly [number, number, number][] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [-1, -1, -1],
        [10, 20, 30],
        [-10, -20, -30],
        [MORTON_BIAS - 1, 0, 0], // upper edge
        [-MORTON_BIAS, 0, 0], // lower edge
        [MORTON_BIAS - 1, MORTON_BIAS - 1, MORTON_BIAS - 1], // upper-corner
      ];

      // Place particles at the centers of those cells and verify each
      // particle lands in the bucket the CPU Morton encoding predicts.
      const data: ParticleInit[] = testCells.map(([cx, cy, cz]) => ({
        position: [(cx + 0.5) * CELL, (cy + 0.5) * CELL, (cz + 0.5) * CELL],
        velocity: [0, 0, 0],
        invMass: 1,
      }));

      const particles = new ParticleSystem(renderer, data.length, 0.02);
      particles.uploadParticles(data);

      // hashTableSize chosen large enough that lower-bit Morton codes do
      // not mask-collide for our test set (max code ~ 30 bits).
      const grid = new HashGrid(particles, {
        cellSize: CELL,
        hashTableSize: 1 << 18,
        hashOrigin: new Vector3(0, 0, 0),
      });
      await grid.rebuild();

      const snapshot = await grid.readback();
      for (let i = 0; i < testCells.length; i++) {
        const [cx, cy, cz] = testCells[i]!;
        const expected = mortonCPU(cx, cy, cz) & (grid.hashTableSize - 1);
        const observed = snapshot.cellIndex[i]!;
        expect(observed).toBe(expected);
      }

      // Overflow flag should stay 0 for cells inside the Morton range.
      expect(await grid.readbackOverflow()).toBe(false);

      grid.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('overflow flag fires for a particle one cell past the Morton range', async () => {
    const renderer = await createParticleRenderer();
    try {
      const CELL = 0.1;
      // Cell coord MORTON_BIAS is one past the encodable upper bound.
      const data: ParticleInit[] = [
        {
          position: [(MORTON_BIAS + 0.5) * CELL, 0, 0],
          velocity: [0, 0, 0],
          invMass: 1,
        },
      ];
      const particles = new ParticleSystem(renderer, data.length, 0.02);
      particles.uploadParticles(data);

      const grid = new HashGrid(particles, { cellSize: CELL });
      await grid.rebuild();

      expect(await grid.readbackOverflow()).toBe(true);

      grid.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
