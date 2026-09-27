import { describe, expect, it } from 'vitest';
import { colorConstraints } from '../../../src/index.js';

// Graph coloring correctness. CPU-only; no GPU required.
//
// Two properties:
//   1. No particle appears twice in any group.
//   2. The number of groups stays bounded by the max constraint degree of
//      any particle.

function maxDegree(
  arity: number,
  participants: readonly number[] | Uint32Array,
  nConstraints: number,
): number {
  const counts = new Map<number, number>();
  for (let c = 0; c < nConstraints; c++) {
    for (let k = 0; k < arity; k++) {
      const p = participants[c * arity + k]!;
      counts.set(p, (counts.get(p) ?? 0) + 1);
    }
  }
  let m = 0;
  for (const v of counts.values()) if (v > m) m = v;
  return m;
}

function assertNoIntraGroupConflict(
  arity: number,
  participants: readonly number[] | Uint32Array,
  nConstraints: number,
  groupOf: Uint32Array,
  numGroups: number,
): void {
  for (let g = 0; g < numGroups; g++) {
    const seen = new Set<number>();
    for (let c = 0; c < nConstraints; c++) {
      if (groupOf[c] !== g) continue;
      for (let k = 0; k < arity; k++) {
        const p = participants[c * arity + k]!;
        expect(seen.has(p), `group ${g}: particle ${p} appears twice`).toBe(false);
        seen.add(p);
      }
    }
  }
}

describe('colorConstraints: graph coloring', () => {
  it('handles the empty graph', () => {
    const r = colorConstraints({
      arity: 2,
      nConstraints: 0,
      participantsPerConstraint: new Uint32Array(0),
    });
    expect(r.numGroups).toBe(0);
    expect(r.groupOf.length).toBe(0);
  });

  it('a linear chain of N particles / N-1 edges colors with 2 groups', () => {
    // Chain 0-1-2-3-4-5-6-7-8-9 has maxDegree=2 (interior particles) → 2
    // groups suffice (alternating pattern).
    const N = 10;
    const pairs: number[] = [];
    for (let i = 0; i < N - 1; i++) pairs.push(i, i + 1);
    const participants = Uint32Array.from(pairs);
    const r = colorConstraints({
      arity: 2,
      nConstraints: N - 1,
      participantsPerConstraint: participants,
    });
    expect(r.numGroups).toBeLessThanOrEqual(2);
    assertNoIntraGroupConflict(2, participants, N - 1, r.groupOf, r.numGroups);
  });

  it('a star graph (hub + N leaves) requires N groups', () => {
    // Star: particle 0 connected to each of 1..N. Hub has degree N, so
    // ≥ N groups required; the greedy algorithm achieves exactly N.
    const N = 8;
    const pairs: number[] = [];
    for (let leaf = 1; leaf <= N; leaf++) pairs.push(0, leaf);
    const participants = Uint32Array.from(pairs);
    const r = colorConstraints({
      arity: 2,
      nConstraints: N,
      participantsPerConstraint: participants,
    });
    expect(r.numGroups).toBe(N);
    assertNoIntraGroupConflict(2, participants, N, r.groupOf, r.numGroups);
  });

  it('bounds numGroups by maxDegree on 10 random graphs', () => {
    // For graphs of bounded max-degree Δ, greedy coloring uses at most
    // Δ + 1 colors (each constraint forbids at most Δ·(arity-1) others at
    // a given participant). For distance (arity=2), the bound is exactly
    // maxDegree — a constraint at particle p with degree d touches d-1
    // other constraints through p, plus up to d-1 through its other
    // endpoint; still O(maxDegree) groups total.
    //
    // This test checks the looser bound numGroups ≤ 1 + 2·(maxDegree-1)
    // ≈ 2·maxDegree, which is sufficient to catch quadratic-blowup bugs
    // in colorConstraints.

    let rng = 0xdeadc0de;
    const next = (): number => {
      rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
      return rng / 0x100000000;
    };

    for (let trial = 0; trial < 10; trial++) {
      const nParticles = 20 + Math.floor(next() * 30);
      const nEdges = 30 + Math.floor(next() * 50);
      const seen = new Set<string>();
      const pairs: number[] = [];
      let attempts = 0;
      while (pairs.length / 2 < nEdges && attempts < nEdges * 10) {
        attempts++;
        const i = Math.floor(next() * nParticles);
        const j = Math.floor(next() * nParticles);
        if (i === j) continue;
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (seen.has(key)) continue;
        seen.add(key);
        pairs.push(i, j);
      }
      const n = pairs.length / 2;
      const participants = Uint32Array.from(pairs);
      const deg = maxDegree(2, participants, n);
      const r = colorConstraints({
        arity: 2,
        nConstraints: n,
        participantsPerConstraint: participants,
      });
      assertNoIntraGroupConflict(2, participants, n, r.groupOf, r.numGroups);
      expect(
        r.numGroups,
        `trial ${trial}: numGroups=${r.numGroups} exceeds 2·maxDegree=${2 * deg} for ${n} edges`,
      ).toBeLessThanOrEqual(Math.max(1, 2 * deg));
    }
  });

  it('high-arity constraint (K=4) colors correctly', () => {
    // Four tetrahedral constraints sharing a center particle (0):
    //   c0: (0,1,2,3)   c1: (0,4,5,6)   c2: (0,7,8,9)   c3: (0,10,11,12)
    // Particle 0 has degree 4 → needs 4 groups.
    const arity = 4;
    const pairs = [
      [0, 1, 2, 3],
      [0, 4, 5, 6],
      [0, 7, 8, 9],
      [0, 10, 11, 12],
    ];
    const participants = new Uint32Array(pairs.length * arity);
    pairs.forEach((tet, c) => {
      tet.forEach((p, k) => {
        participants[c * arity + k] = p;
      });
    });
    const r = colorConstraints({
      arity,
      nConstraints: pairs.length,
      participantsPerConstraint: participants,
    });
    expect(r.numGroups).toBe(4);
    assertNoIntraGroupConflict(arity, participants, pairs.length, r.groupOf, r.numGroups);
  });
});
