import { describe, expect, it } from 'vitest';

type Vec3 = readonly [number, number, number];
type Quat = readonly [number, number, number, number]; // (x, y, z, w)

const v3 = (x: number, y: number, z: number): Vec3 => [x, y, z];
const q4 = (x: number, y: number, z: number, w: number): Quat => [x, y, z, w];

function quatMul(a: Quat, b: Quat): Quat {
  const ax = a[0],
    ay = a[1],
    az = a[2],
    aw = a[3];
  const bx = b[0],
    by = b[1],
    bz = b[2],
    bw = b[3];
  return [
    aw * bx + bw * ax + ay * bz - az * by,
    aw * by + bw * ay + az * bx - ax * bz,
    aw * bz + bw * az + ax * by - ay * bx,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function quatRotate(q: Quat, v: Vec3): Vec3 {
  // Standard Lemma 4: v + 2 q.xyz × (q.xyz × v + q.w · v).
  const qx = q[0],
    qy = q[1],
    qz = q[2],
    qw = q[3];
  const ax = qy * v[2] - qz * v[1] + qw * v[0];
  const ay = qz * v[0] - qx * v[2] + qw * v[1];
  const az = qx * v[1] - qy * v[0] + qw * v[2];
  return [
    v[0] + 2 * (qy * az - qz * ay),
    v[1] + 2 * (qz * ax - qx * az),
    v[2] + 2 * (qx * ay - qy * ax),
  ];
}

function quatNeg(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], -q[3]];
}

function dot4(a: Quat, b: Quat): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

function add4(a: Quat, b: Quat): Quat {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
}

function scale4(a: Quat, s: number): Quat {
  return [a[0] * s, a[1] * s, a[2] * s, a[3] * s];
}

function len4(a: Quat): number {
  return Math.hypot(a[0], a[1], a[2], a[3]);
}

/**
 * Reference DLB skinning function. Given K influences each described
 * by (rotation quaternion `q_k`, world-space position `x_star_k`,
 * body-rest-local position `r_k`) and weights `w_k`, returns the
 * world-space skinned position of body-rest-local point `vRest`.
 *
 * This is the exact algorithm the TSL shader runs:
 *   1. t_k = x*_k − R(q_k) · r_k   (per-particle rigid translation)
 *   2. q̂_k = q_k + ε · ½ · (0, t_k) ⊗ q_k
 *   3. Antipodality: flip lanes whose real part has negative dot with
 *      lane 0. (Algorithm 2.)
 *   4. Σ w_k · q̂_k → real_sum, dual_sum  (Eq. 11.)
 *   5. Normalize by ‖real_sum‖.
 *   6. v' = R(realN) · vRest + 2(a₀·d_ε − a_ε·d₀ + d₀ × d_ε).
 */
function skinDLB(
  influences: { q: Quat; xStar: Vec3; r: Vec3 }[],
  weights: number[],
  vRest: Vec3,
): Vec3 {
  // 1, 2: per-influence dual quaternions.
  const reals: Quat[] = [];
  const duals: Quat[] = [];
  for (let k = 0; k < influences.length; k++) {
    const { q, xStar, r } = influences[k]!;
    const Rr = quatRotate(q, r);
    const t: Vec3 = [xStar[0] - Rr[0], xStar[1] - Rr[1], xStar[2] - Rr[2]];
    // (0, t) ⊗ q
    const tQuat: Quat = [t[0], t[1], t[2], 0];
    const dual = scale4(quatMul(tQuat, q), 0.5);
    reals.push(q);
    duals.push(dual);
  }

  // 3: antipodality flip vs lane 0.
  const pivot = reals[0]!;
  for (let k = 1; k < reals.length; k++) {
    if (dot4(pivot, reals[k]!) < 0) {
      reals[k] = quatNeg(reals[k]!);
      duals[k] = quatNeg(duals[k]!);
    }
  }

  // 4: weighted sum.
  let realSum: Quat = [0, 0, 0, 0];
  let dualSum: Quat = [0, 0, 0, 0];
  for (let k = 0; k < reals.length; k++) {
    realSum = add4(realSum, scale4(reals[k]!, weights[k]!));
    dualSum = add4(dualSum, scale4(duals[k]!, weights[k]!));
  }

  // 5: normalize.
  const norm = Math.max(len4(realSum), 1e-20);
  const realN = scale4(realSum, 1 / norm);
  const dualN = scale4(dualSum, 1 / norm);

  // 6: apply.
  const a0 = realN[3];
  const d0: Vec3 = [realN[0], realN[1], realN[2]];
  const ae = dualN[3];
  const de: Vec3 = [dualN[0], dualN[1], dualN[2]];

  const rotated = quatRotate(realN, vRest);
  // 2(a₀·d_ε − a_ε·d₀ + d₀ × d_ε)
  const cross: Vec3 = [
    d0[1] * de[2] - d0[2] * de[1],
    d0[2] * de[0] - d0[0] * de[2],
    d0[0] * de[1] - d0[1] * de[0],
  ];
  return [
    rotated[0] + 2 * (a0 * de[0] - ae * d0[0] + cross[0]),
    rotated[1] + 2 * (a0 * de[1] - ae * d0[1] + cross[1]),
    rotated[2] + 2 * (a0 * de[2] - ae * d0[2] + cross[2]),
  ];
}

/** Reference LBS — Σ_k w_k · (R_k · vRest + (x*_k − R_k · r_k)). */
function skinLBS(
  influences: { q: Quat; xStar: Vec3; r: Vec3 }[],
  weights: number[],
  vRest: Vec3,
): Vec3 {
  let out: Vec3 = [0, 0, 0];
  for (let k = 0; k < influences.length; k++) {
    const { q, xStar, r } = influences[k]!;
    const Rv = quatRotate(q, vRest);
    const Rr = quatRotate(q, r);
    const t: Vec3 = [xStar[0] - Rr[0], xStar[1] - Rr[1], xStar[2] - Rr[2]];
    const w = weights[k]!;
    out = [out[0] + w * (Rv[0] + t[0]), out[1] + w * (Rv[1] + t[1]), out[2] + w * (Rv[2] + t[2])];
  }
  return out;
}

/** Axis-angle to (x, y, z, w) quaternion. Axis must be unit. */
function axisAngleQuat(axis: Vec3, angleRad: number): Quat {
  const s = Math.sin(angleRad / 2);
  const c = Math.cos(angleRad / 2);
  return [axis[0] * s, axis[1] * s, axis[2] * s, c];
}

const I_QUAT: Quat = [0, 0, 0, 1];

describe('Phase 13 G1 — DLB algebra reference', () => {
  it('reduces to a rigid transform when every influence shares the same q and t', () => {
    // Body-rest particles forming a 2x2x2 grid; rotate all by 30° about
    // y, translate by (0.7, -0.2, 1.1). Every particle and the rendered
    // vertex should map to the same world-space position the rigid
    // transform produces directly.
    const angle = Math.PI / 6;
    const axis: Vec3 = [0, 1, 0];
    const q = axisAngleQuat(axis, angle);
    const trans: Vec3 = [0.7, -0.2, 1.1];

    const restR: Vec3[] = [
      [-0.05, -0.05, -0.05],
      [0.05, -0.05, -0.05],
      [-0.05, 0.05, -0.05],
      [0.05, 0.05, -0.05],
    ];
    const influences = restR.map((r) => {
      const Rr = quatRotate(q, r);
      const xStar: Vec3 = [Rr[0] + trans[0], Rr[1] + trans[1], Rr[2] + trans[2]];
      return { q, xStar, r };
    });
    const weights = [0.25, 0.25, 0.25, 0.25];

    const vRest: Vec3 = [0.02, -0.03, 0.04];
    const dlb = skinDLB(influences, weights, vRest);
    const Rv = quatRotate(q, vRest);
    const expected: Vec3 = [Rv[0] + trans[0], Rv[1] + trans[1], Rv[2] + trans[2]];
    for (let i = 0; i < 3; i++) expect(dlb[i]).toBeCloseTo(expected[i]!, 6);
  });

  it('§5.3 LBS-equivalence: DLB matches LBS exactly when all influence quaternions are identical', () => {
    // Same shared rotation, but particles displaced from their rigid
    // positions (simulates a §5.3 body where shape matching has not
    // converged). DLB and LBS must agree analytically.
    // Axis normalized — a non-unit axis would produce a non-unit q,
    // which breaks DLB's unit-DQ-on-output invariant.
    const axisRaw: Vec3 = [0.3, 0.7, 0.6];
    const axisNorm = Math.hypot(...axisRaw);
    const q = axisAngleQuat(
      [axisRaw[0] / axisNorm, axisRaw[1] / axisNorm, axisRaw[2] / axisNorm],
      0.4,
    );
    const restR: Vec3[] = [
      [-0.04, -0.03, -0.02],
      [0.05, -0.02, 0.03],
      [-0.03, 0.05, 0.04],
      [0.04, 0.03, -0.05],
    ];
    const influences = restR.map((r, k) => {
      // Off-rigid x* — sprinkle small noise.
      const xStar: Vec3 = [0.5 + 0.01 * (k - 2), 0.6 - 0.02 * k, 0.1 + 0.03 * Math.sin(k)];
      return { q, xStar, r };
    });
    const weights = [0.4, 0.3, 0.2, 0.1];

    const vRest: Vec3 = [0.01, 0.02, -0.01];
    const dlb = skinDLB(influences, weights, vRest);
    const lbs = skinLBS(influences, weights, vRest);
    for (let i = 0; i < 3; i++) {
      // Plan §Validation gates DLB ≈ LBS at 1e-5 m on §5.3 bodies; the
      // analytic identity is exact (DLB(q,t) reduces to R·v + t when
      // every q_k is shared and q is a unit quaternion).
      expect(Math.abs(dlb[i]! - lbs[i]!)).toBeLessThan(1e-7);
    }
  });

  it('candy-wrapper-absence: DQB returns a unit rotation when influences are 180° apart', () => {
    // Two influences rotated 180° apart around the same axis. LBS would
    // average to a near-zero (degenerate) rotation matrix; DQB picks a
    // valid rigid transform whose rotation quaternion stays unit.
    const axis: Vec3 = [1, 0, 0];
    const q1 = axisAngleQuat(axis, 0); // identity
    const q2 = axisAngleQuat(axis, Math.PI); // 180° about +x
    // Antipodality flip should align them so the blend is meaningful.
    // After flip, q2 becomes (-1, 0, 0, 0) → its negative is (1, 0, 0, 0)?
    // Actually q1 = (0,0,0,1) and q2 = (1,0,0,0). dot = 0, no flip.
    // The DLB still produces a unit-norm result rather than degenerate.

    // Place particles at the origin so translations vanish; we only
    // care about the rotation part.
    const r: Vec3 = [0, 0, 0];
    const xStar: Vec3 = [0, 0, 0];
    const influences = [
      { q: q1, xStar, r },
      { q: q2, xStar, r },
    ];
    const weights = [0.5, 0.5];

    // Reproduce internal blend — after antipodality + sum + normalize,
    // the real part should be a unit quaternion (rotation 90° around
    // the +x axis).
    const reals: Quat[] = [q1, q2];
    if (dot4(q1, q2) < 0) reals[1] = quatNeg(q2);
    let realSum: Quat = [0, 0, 0, 0];
    for (let k = 0; k < reals.length; k++) {
      realSum = add4(realSum, scale4(reals[k]!, weights[k]!));
    }
    const norm = len4(realSum);
    expect(norm).toBeGreaterThan(0.5); // not collapsed
    const realN = scale4(realSum, 1 / norm);
    expect(len4(realN)).toBeCloseTo(1, 6);

    // Driving the full skin path on a non-zero v: the blended rotation
    // applied to (0, 1, 0) should give a finite, unit-length rotated
    // vector — LBS would shrink it.
    const out = skinDLB(influences, weights, [0, 1, 0]);
    expect(Math.hypot(out[0], out[1], out[2])).toBeCloseTo(1, 5);
  });

  it('identity influence with zero translation is a no-op on vRest', () => {
    const influences = [
      { q: I_QUAT, xStar: v3(0, 0, 0), r: v3(0, 0, 0) },
      { q: I_QUAT, xStar: v3(0, 0, 0), r: v3(0, 0, 0) },
      { q: I_QUAT, xStar: v3(0, 0, 0), r: v3(0, 0, 0) },
      { q: I_QUAT, xStar: v3(0, 0, 0), r: v3(0, 0, 0) },
    ];
    const weights = [0.25, 0.25, 0.25, 0.25];
    const vRest: Vec3 = [0.42, -0.11, 0.07];
    const out = skinDLB(influences, weights, vRest);
    for (let i = 0; i < 3; i++) expect(out[i]).toBeCloseTo(vRest[i]!, 6);
  });

  it('pure translation (q = identity, x*_k = r_k + t shared) translates vRest by t', () => {
    const t: Vec3 = [1.5, -0.2, 0.3];
    const restR: Vec3[] = [
      [-0.05, -0.05, -0.05],
      [0.05, -0.05, -0.05],
      [-0.05, 0.05, -0.05],
      [0.05, 0.05, -0.05],
    ];
    const influences = restR.map((r) => ({
      q: I_QUAT,
      xStar: v3(r[0] + t[0], r[1] + t[1], r[2] + t[2]) as Vec3,
      r,
    }));
    const weights = [0.4, 0.3, 0.2, 0.1];
    const vRest: Vec3 = [0.02, 0.03, -0.04];
    const out = skinDLB(influences, weights, vRest);
    expect(out[0]).toBeCloseTo(vRest[0]! + t[0], 6);
    expect(out[1]).toBeCloseTo(vRest[1]! + t[1], 6);
    expect(out[2]).toBeCloseTo(vRest[2]! + t[2], 6);
  });
});

// `q4` is exported as a helper but the suite intentionally constructs
// quaternions via `axisAngleQuat`. Keep the import-or-die contract by
// referencing the symbol once.
void q4;
