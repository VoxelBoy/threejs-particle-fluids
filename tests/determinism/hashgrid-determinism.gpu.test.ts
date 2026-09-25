import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../src/core/index.js';

// Phase 03 G4 — first concrete example of the two-tier G4 policy.
//

//
//   Tier 1 — bit-exact.        Asserted on `cellStart`, `cellEnd`, and the
//                               per-cell **set** of `sortedIndices`.
//                               These come from histogram + prefix-scan
//                               (order-independent reductions) and the
//                               sort's cell-membership (a multiset defined
//                               purely by `cellIndex`, itself bit-exact).
//   Tier 2 — bounded max error. Would be used if this test were about an
//                               f32 reduction over neighbor order. It is
//                               not — every readback here is a uint
//                               permutation or partition.
//
// The intra-cell sequence of `sortedIndices` is NOT asserted. The atomic-
// scatter cursor orders intra-cell particles in warp-schedule order, which
// varies ULP-equivalently across rebuilds on the same GPU. Per the U-18
// physical-accuracy analysis, the downstream effect is ≤1 ULP per f32
// reduction, dominated by symplectic-Euler truncation and by XPBD's
// contractive constraint projection — a test-harness cost, not a physics
// cost.

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function buildScene(): ParticleInit[] {
  const N = 4096;
  const rand = lcg(0xc0ffee_42);
  const data: ParticleInit[] = [];
  for (let i = 0; i < N; i++) {
    data.push({
      position: [rand(), rand(), rand()],
      velocity: [0, 0, 0],
      invMass: 1,
      phase: 0,
    });
  }
  return data;
}

function firstMismatch(a: Uint32Array, b: Uint32Array): number {
  if (a.length !== b.length) return -2;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return i;
  return -1;
}

function sortedCellSlice(
  sortedIndices: Uint32Array,
  cellStart: Uint32Array,
  cellEnd: Uint32Array,
  c: number,
): Uint32Array {
  const slice = sortedIndices.slice(cellStart[c]!, cellEnd[c]!);
  slice.sort();
  return slice;
}

describe('Phase 03 — HashGrid: determinism (tier-1, 100 rebuilds)', () => {
  it('cellStart and cellEnd are bit-identical; per-cell sortedIndices set is stable', async () => {
    const renderer = await createParticleRenderer();
    try {
      const data = buildScene();
      const particles = new ParticleSystem(renderer, data.length, 0.005);
      particles.uploadParticles(data);
      const grid = new HashGrid(particles, {
        cellSize: 0.05,
      });

      await grid.rebuild();
      const reference = await grid.readback();

      // Cell-sorted slices for the reference run — one Uint32Array per
      // non-empty cell, each sorted ascending. Comparison runs re-sort
      // their own slice and compare element-by-element.
      const refSlices = new Map<number, Uint32Array>();
      for (let c = 0; c < grid.hashTableSize; c++) {
        if (reference.cellEnd[c]! > reference.cellStart[c]!) {
          refSlices.set(
            c,
            sortedCellSlice(reference.sortedIndices, reference.cellStart, reference.cellEnd, c),
          );
        }
      }

      const REBUILDS = 100;
      let startMismatchRun = -1;
      let startMismatchIdx = -1;
      let endMismatchRun = -1;
      let endMismatchIdx = -1;
      let sliceMismatchRun = -1;
      let sliceMismatchCell = -1;
      for (let r = 0; r < REBUILDS; r++) {
        await grid.rebuild();
        const snap = await grid.readback();

        const start = firstMismatch(reference.cellStart, snap.cellStart);
        if (start !== -1 && startMismatchRun === -1) {
          startMismatchRun = r;
          startMismatchIdx = start;
        }
        const end = firstMismatch(reference.cellEnd, snap.cellEnd);
        if (end !== -1 && endMismatchRun === -1) {
          endMismatchRun = r;
          endMismatchIdx = end;
        }

        for (const [c, refSlice] of refSlices) {
          const cur = sortedCellSlice(snap.sortedIndices, snap.cellStart, snap.cellEnd, c);
          if (firstMismatch(refSlice, cur) !== -1 && sliceMismatchRun === -1) {
            sliceMismatchRun = r;
            sliceMismatchCell = c;
            break;
          }
        }
      }

      if (startMismatchRun !== -1 || endMismatchRun !== -1 || sliceMismatchRun !== -1) {
        // eslint-disable-next-line no-console
        console.error(
          `[hashgrid-determinism] start@run=${startMismatchRun} idx=${startMismatchIdx}; ` +
            `end@run=${endMismatchRun} idx=${endMismatchIdx}; ` +
            `cell-slice@run=${sliceMismatchRun} cell=${sliceMismatchCell}`,
        );
      }

      expect(startMismatchRun).toBe(-1);
      expect(endMismatchRun).toBe(-1);
      expect(sliceMismatchRun).toBe(-1);

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 180_000);
});
