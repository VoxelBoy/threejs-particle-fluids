import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// HashGrid overflow-flag invariant (G3).
//
// The hash grid carries TWO overflow checks, both surfaced through a single
// `overflowFlag`:
//   1. f32→i32 saturation clamp at `±2^30`. Protects the cast itself for
//      simulation-collapse positions (`|x| ≳ 10⁸ m`).
//   2. Morton-range check at `±MORTON_BIAS` (currently 512 cells per axis).
//      Cells outside this range still simulate but lose Morton's spatial-
//      locality guarantee — they wrap-alias into the bucket space.
//
// At `cellSize = 0.1 m`, the Morton range corresponds to physical positions
// `|x| < 51.2 m` (using default `hashOrigin = (0,0,0)`). Scenes farther
// from the world origin should declare a `hashOrigin` offset to recenter
// — see `hashgrid-out-of-origin.gpu.test.ts` for that pattern.
//
// This file pins:
//   - Within-Morton scenes (positions within ±50 m at `cellSize = 0.1 m`):
//     flag stays 0. Every non-overflow test in the suite depends on this.
//   - Beyond-Morton positions (`|x| ≳ 60 m` at `cellSize = 0.1 m`): flag
//     surfaces 1, but the simulation continues (correctness preserved by
//     the per-pair distance filter in every neighbor-walk caller).
//   - Reset behavior: the rebuild pipeline's `resetOverflowFlag` runs
//     first, so a previous frame's set flag does not stay sticky.

describe('HashGrid: overflow flag (G3)', () => {
  it('stays 0 when all particles are within the Morton cell-coordinate range', async () => {
    const renderer = await createParticleRenderer();
    try {
      // At cellSize = 0.1 the Morton range is ±51.1 m. These positions sit
      // comfortably within that window.
      const data: ParticleInit[] = [
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [40, -25, 30], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [-30, 15, -45], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ];
      const particles = new ParticleSystem(renderer, data.length, 0.02);
      particles.uploadParticles(data);

      const grid = new HashGrid(particles, { cellSize: 0.1 });
      await grid.rebuild();

      const flag = await grid.readbackOverflow();
      expect(flag).toBe(0);

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('surfaces 1 when any particle drifts past the Morton cell-coordinate range', async () => {
    const renderer = await createParticleRenderer();
    try {
      // At cellSize = 0.1 the Morton range is ±51.1 m. 1e10 m is well past
      // that AND past the f32→i32 saturation clamp at ~10⁸ m, so both
      // overflow conditions fire — either alone would suffice.
      const data: ParticleInit[] = [
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [1e10, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ];
      const particles = new ParticleSystem(renderer, data.length, 0.02);
      particles.uploadParticles(data);

      const grid = new HashGrid(particles, { cellSize: 0.1 });
      await grid.rebuild();

      const flag = await grid.readbackOverflow();
      expect(flag).toBe(1);

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('resets to 0 on subsequent rebuilds once particles move back into range', async () => {
    const renderer = await createParticleRenderer();
    try {
      const data: ParticleInit[] = [
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [1e10, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ];
      const particles = new ParticleSystem(renderer, data.length, 0.02);
      particles.uploadParticles(data);

      const grid = new HashGrid(particles, { cellSize: 0.1 });
      await grid.rebuild();
      expect(await grid.readbackOverflow()).toBe(1);

      // Pull the escapee back. Rebuild should clear the flag — the
      // pipeline's resetOverflowFlag kernel runs before the histogram
      // kernel that would otherwise leave the previous frame's flag sticky.
      particles.uploadParticles([
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [0.5, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ]);
      await grid.rebuild();
      expect(await grid.readbackOverflow()).toBe(0);

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
