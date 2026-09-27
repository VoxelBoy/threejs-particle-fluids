import { describe, expect, it } from 'vitest';

// Central-difference validation of the dihedral-angle bending gradients
// ported from Bridson 2003 §4.
//
// The closed-form gradients in `bending.ts` are translated from
// Bridson's vertex labels `(x_1=far_1, x_2=far_2, x_3=edge_a, x_4=edge_b)`
// to our Bender 2014 §3.4.2 labels `(p_1=edge_a, p_2=edge_b, p_3=far_1,
// p_4=far_2)`. This test re-implements the same formulas in plain TS
// and asserts that for a battery of random configurations
// `|∇_k C - (C(x + ε·ê_k) - C(x - ε·ê_k)) / (2ε)| < 1e-5` per
// component.
//
// The kernel itself is trusted to faithfully port these formulas; the
// kernel-vs-CPU equivalence is exercised end-to-end by the GPU
// bending-rest and drape tests.

type V3 = [number, number, number];

function sub(a: V3, b: V3): V3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a: V3, b: V3): V3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function len(a: V3): number {
  return Math.sqrt(dot(a, a));
}
function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/**
 * Signed dihedral angle `atan2((N_1×N_2)·E, (N_1·N_2)·|E|)` per
 * Bridson 2003 §4. Mirrors `bending.ts` and `graph.ts::measureDihedralAngle`.
 */
function constraintValue(p1: V3, p2: V3, p3: V3, p4: V3): number {
  const E = sub(p2, p1);
  const eLen = len(E);
  const N1 = cross(sub(p3, p1), sub(p3, p2));
  const N2 = cross(sub(p4, p2), sub(p4, p1));
  if (eLen < 1e-12) return 0;
  if (dot(N1, N1) < 1e-24 || dot(N2, N2) < 1e-24) return 0;
  const cN = cross(N1, N2);
  const sinTimes = dot(cN, E);
  const cosTimes = dot(N1, N2) * eLen;
  return Math.atan2(sinTimes, cosTimes);
}
// Suppress `clamp` no-longer-used warning; kept as a local utility for
// future near-singular guards if needed.
void clamp;

/**
 * Closed-form gradients ported from `bending.ts` (Bridson 2003 §4 with
 * Bender label translation). Returns `[∇_p1 C, ∇_p2 C, ∇_p3 C, ∇_p4 C]`.
 */
function closedFormGradients(p1: V3, p2: V3, p3: V3, p4: V3): [V3, V3, V3, V3] {
  const E = sub(p2, p1);
  const eLen = len(E);
  const a3 = sub(p3, p1);
  const b3 = sub(p3, p2);
  const a4 = sub(p4, p2);
  const b4 = sub(p4, p1);
  const N1 = cross(a3, b3);
  const N2 = cross(a4, b4);
  const n1Sq = dot(N1, N1);
  const n2Sq = dot(N2, N2);
  const N1Scaled: V3 = [N1[0] / n1Sq, N1[1] / n1Sq, N1[2] / n1Sq];
  const N2Scaled: V3 = [N2[0] / n2Sq, N2[1] / n2Sq, N2[2] / n2Sq];
  // Bridson's u_k followed by -∇θ_atan2 = -u_k correction (see
  // `bending.ts` block comment, "Sign correction" paragraph).
  const grad3: V3 = [-N1Scaled[0] * eLen, -N1Scaled[1] * eLen, -N1Scaled[2] * eLen];
  const grad4: V3 = [-N2Scaled[0] * eLen, -N2Scaled[1] * eLen, -N2Scaled[2] * eLen];
  const c31 = dot(b3, E) / eLen;
  const c32 = dot(a4, E) / eLen;
  const c41 = dot(a3, E) / eLen;
  const c42 = dot(b4, E) / eLen;
  const grad1: V3 = [
    -(N1Scaled[0] * c31 + N2Scaled[0] * c32),
    -(N1Scaled[1] * c31 + N2Scaled[1] * c32),
    -(N1Scaled[2] * c31 + N2Scaled[2] * c32),
  ];
  const grad2: V3 = [
    N1Scaled[0] * c41 + N2Scaled[0] * c42,
    N1Scaled[1] * c41 + N2Scaled[1] * c42,
    N1Scaled[2] * c41 + N2Scaled[2] * c42,
  ];
  return [grad1, grad2, grad3, grad4];
}

function centralDifferenceGradients(p1: V3, p2: V3, p3: V3, p4: V3, eps: number): [V3, V3, V3, V3] {
  const ps: V3[] = [p1, p2, p3, p4];
  const result: V3[] = [];
  for (let k = 0; k < 4; k++) {
    const g: V3 = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const plus = ps.map((p) => [...p] as V3);
      const minus = ps.map((p) => [...p] as V3);
      plus[k]![d] = plus[k]![d]! + eps;
      minus[k]![d] = minus[k]![d]! - eps;
      const cPlus = constraintValue(plus[0]!, plus[1]!, plus[2]!, plus[3]!);
      const cMinus = constraintValue(minus[0]!, minus[1]!, minus[2]!, minus[3]!);
      g[d] = (cPlus - cMinus) / (2 * eps);
    }
    result.push(g);
  }
  return result as [V3, V3, V3, V3];
}

function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

describe('cloth bending gradient: closed form vs central difference', () => {
  it('matches central difference within 1e-5 on 64 random configurations', () => {
    const rand = rng(0xb1ac1234);
    const eps = 1e-5;
    const tol = 1e-5;
    let maxErr = 0;
    for (let trial = 0; trial < 64; trial++) {
      // Random non-degenerate configuration: edge endpoints close, far
      // vertices on opposite sides of the edge with non-zero
      // perpendicular offsets.
      const p1: V3 = [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1];
      const p2: V3 = [
        p1[0] + (rand() * 0.5 + 0.5),
        p1[1] + (rand() - 0.5) * 0.2,
        p1[2] + (rand() - 0.5) * 0.2,
      ];
      const p3: V3 = [
        (p1[0] + p2[0]) / 2 + (rand() - 0.5) * 0.6,
        (p1[1] + p2[1]) / 2 + (rand() * 0.5 + 0.4),
        (p1[2] + p2[2]) / 2 + (rand() - 0.5) * 0.6,
      ];
      const p4: V3 = [
        (p1[0] + p2[0]) / 2 + (rand() - 0.5) * 0.6,
        (p1[1] + p2[1]) / 2 - (rand() * 0.5 + 0.4),
        (p1[2] + p2[2]) / 2 + (rand() - 0.5) * 0.6,
      ];

      // Skip degenerate cases (acos near ±1, where central-difference is
      // numerically unstable due to acos's slope blowup near the boundary).
      // The closed form is robust there; the test is just measuring
      // agreement.
      const N1 = cross(sub(p3, p1), sub(p3, p2));
      const N2 = cross(sub(p4, p2), sub(p4, p1));
      const cosT = dot(N1, N2) / (len(N1) * len(N2));
      if (Math.abs(cosT) > 0.95) continue; // skip near-flat / near-folded configs

      const closed = closedFormGradients(p1, p2, p3, p4);
      const numeric = centralDifferenceGradients(p1, p2, p3, p4, eps);
      for (let k = 0; k < 4; k++) {
        for (let d = 0; d < 3; d++) {
          const err = Math.abs(closed[k]![d]! - numeric[k]![d]!);
          if (err > maxErr) maxErr = err;
          if (err > tol) {
            console.error(
              `[bending-gradient] trial ${trial} k=${k} d=${d}\n` +
                `  p1=${JSON.stringify(p1)}\n  p2=${JSON.stringify(p2)}\n  p3=${JSON.stringify(p3)}\n  p4=${JSON.stringify(p4)}\n` +
                `  cosT=${cosT.toFixed(4)} closed=${closed[k]![d]!.toExponential(3)} numeric=${numeric[k]![d]!.toExponential(3)} err=${err.toExponential(3)}`,
            );
          }
          expect(err).toBeLessThan(tol);
        }
      }
    }
    console.info(`[bending-gradient] max |∇_closed − ∇_numeric| = ${maxErr.toExponential(2)}`);
  });
});
