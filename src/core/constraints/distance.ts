import { Fn, If, Return, float, instanceIndex, instancedArray, uint, vec4 } from 'three/tsl';

import type { ParticleSystem } from '../particles.js';
import type { XpbdUniforms } from './xpbd.js';
import { xpbdDeltaLambda } from './xpbd.js';
import {
  NO_CONSTRAINT,
  colorConstraints,
  type ConstraintGroup,
  type ConstraintType,
} from './types.js';

// TSL's @types surface many nodes as bare `Node`, dropping proxy methods. See
// the same pattern in `integrate.ts`, `hashGrid/*.ts`, and `_probe/*`.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * XPBD distance constraints, shared by cloth and direct constraint users.
 *
 * Constraint function (paper Macklin 2016 §6.1 "Spring"):
 *   `C(x_i, x_j) = |x_i − x_j| − L_0`
 *
 * Gradients:
 *   `∇_i C = n`,  `∇_j C = −n`,   where `n = (x_i − x_j) / |x_i − x_j|`.
 * Hence `|∇_i C|² = |∇_j C|² = 1` and
 *   `Σ_k w_k · |∇_k C|² = w_i + w_j`.
 *
 * Solve kernel design — single-mode particle-centric gather per plan §Solve
 * mode. Dispatched once per group over `capacity` particles. Thread `p`:
 *   1. Reads `c = particleToConstraint[p]`. If `NO_CONSTRAINT`, returns.
 *   2. Loads participants `(i, j)` and their predicted positions + invMass.
 *   3. Computes `Δλ` via Macklin 2016 eq. (18) — see `xpbdDeltaLambda`.
 *   4. Applies its **own** `Δx_p = w_p · (±n) · Δλ` to `predictedPositions[p]`.
 *      Sign is `+` if `p == i`, `−` if `p == j`.
 *   5. **Leader write**: if `p == i`, additionally writes
 *      `lambda[c] += Δλ` per Macklin 2016 eq. (13). Only the `i`-side thread
 *      writes λ; graph coloring guarantees `(i, j)` appear in at most one
 *      constraint of this group, so no atomics are needed.
 *
 */
export function createDistanceConstraints(args: {
  readonly particles: ParticleSystem;
  readonly pairs: readonly [number, number][];
  /** Compliance α (s²/kg). Scalar broadcasts to every pair. */
  readonly compliance: number | readonly number[];
  /** Rest length per pair (metres). Required — must match `pairs.length`. */
  readonly restLength: readonly number[];
  readonly xpbd: XpbdUniforms;
}): ConstraintType {
  const { particles, pairs, compliance, restLength, xpbd } = args;
  const nConstraints = pairs.length;
  if (restLength.length !== nConstraints) {
    throw new Error(
      `createDistanceConstraints: restLength length ${restLength.length} ≠ pairs.length ${nConstraints}`,
    );
  }
  const complianceArray = Array.isArray(compliance)
    ? compliance
    : Array.from({ length: nConstraints }, () => compliance as number);
  if (complianceArray.length !== nConstraints) {
    throw new Error(
      `createDistanceConstraints: compliance length ${complianceArray.length} ≠ pairs.length ${nConstraints}`,
    );
  }
  for (const [i, j] of pairs) {
    if (
      !Number.isInteger(i) ||
      !Number.isInteger(j) ||
      i < 0 ||
      j < 0 ||
      i >= particles.capacity ||
      j >= particles.capacity
    ) {
      throw new Error(
        `createDistanceConstraints: pair (${i}, ${j}) has out-of-range index for capacity ${particles.capacity}`,
      );
    }
    if (i === j) {
      throw new Error(`createDistanceConstraints: pair (${i}, ${j}) references the same particle`);
    }
  }

  const arity = 2;

  // ---- Per-constraint SoA storage ----
  const particleIndices = instancedArray(nConstraints * arity, 'uint');
  const complianceBuf = instancedArray(nConstraints, 'float');
  const restBuf = instancedArray(nConstraints, 'float');
  const lambda = instancedArray(nConstraints, 'float');

  // Upload CPU → GPU. Trailing slots (if any) are zero-initialized which is
  // fine: `compliance = 0` reads back as a rigid constraint and `rest = 0`
  // is a degenerate case, but trailing slots are not referenced because we
  // dispatch only over `capacity` via the group's inverted index.
  {
    const idxArr = particleIndices.value.array as Uint32Array;
    for (let c = 0; c < nConstraints; c++) {
      const [i, j] = pairs[c]!;
      idxArr[c * arity + 0] = i;
      idxArr[c * arity + 1] = j;
    }
    particleIndices.value.needsUpdate = true;

    const compArr = complianceBuf.value.array as Float32Array;
    const restArr = restBuf.value.array as Float32Array;
    for (let c = 0; c < nConstraints; c++) {
      compArr[c] = complianceArray[c]!;
      restArr[c] = restLength[c]!;
    }
    complianceBuf.value.needsUpdate = true;
    restBuf.value.needsUpdate = true;
  }

  // ---- Graph coloring ----
  const flatParticipants = new Uint32Array(nConstraints * arity);
  for (let c = 0; c < nConstraints; c++) {
    const [i, j] = pairs[c]!;
    flatParticipants[c * arity + 0] = i;
    flatParticipants[c * arity + 1] = j;
  }
  const { groupOf, numGroups } = colorConstraints({
    arity,
    nConstraints,
    participantsPerConstraint: flatParticipants,
  });

  // ---- Per-group inverted index + solve kernel ----
  const groups: ConstraintGroup[] = [];
  for (let g = 0; g < numGroups; g++) {
    const particleToConstraint = instancedArray(particles.capacity, 'uint');
    const invArr = particleToConstraint.value.array as Uint32Array;
    invArr.fill(NO_CONSTRAINT);
    for (let c = 0; c < nConstraints; c++) {
      if (groupOf[c] !== g) continue;
      const [i, j] = pairs[c]!;
      // Coloring guarantees i and j are not yet assigned to this group.
      invArr[i] = c;
      invArr[j] = c;
    }
    particleToConstraint.value.needsUpdate = true;

    const solveKernel = Fn(() => {
      const p: Any = instanceIndex;
      const c: Any = particleToConstraint.element(p).toVar();
      If(c.equal(uint(NO_CONSTRAINT)), () => {
        Return();
      });

      const iIdx: Any = particleIndices.element(c.mul(uint(arity))).toVar();
      const jIdx: Any = particleIndices.element(c.mul(uint(arity)).add(uint(1))).toVar();

      const xi: Any = particles.predictedPositions.element(iIdx).xyz.toVar();
      const xj: Any = particles.predictedPositions.element(jIdx).xyz.toVar();
      const wi: Any = particles.invMass.element(iIdx).toVar();
      const wj: Any = particles.invMass.element(jIdx).toVar();
      const alpha: Any = complianceBuf.element(c).toVar();
      const rest: Any = restBuf.element(c).toVar();
      const lamCurrent: Any = lambda.element(c).toVar();

      // w_i + w_j = 0 → both pinned → no correction. Also skip when len < eps
      // (degenerate — coincident particles have no defined gradient).
      const wSum: Any = wi.add(wj).toVar();
      If(wSum.lessThanEqual(float(0.0)), () => {
        Return();
      });

      const diff: Any = xi.sub(xj).toVar();
      const len: Any = diff.length().toVar();
      If(len.lessThan(float(1e-12)), () => {
        Return();
      });

      const n: Any = diff.div(len).toVar();
      const C: Any = len.sub(rest);

      // α̃ = α / dt²  — Macklin 2016, text between eq. (7) and (8).
      const dtVal: Any = xpbd.dt;
      const alphaTilde: Any = alpha.div(dtVal.mul(dtVal));

      // Σ w_k |∇_k C|² = w_i + w_j   (|∇_i C|² = |∇_j C|² = 1).
      const dLambda: Any = xpbdDeltaLambda({
        C,
        sumGradSqInvMass: wSum,
        alphaTilde,
        lambdaCurrent: lamCurrent,
      }).toVar();

      // Apply Δx to the thread's own particle slot only. Leader (p == i)
      // additionally commits the λ accumulator.
      If(p.equal(iIdx), () => {
        // ∇_i C = n  ⇒  Δx_i = w_i · Δλ · n
        const dx: Any = n.mul(wi.mul(dLambda));
        const newXi: Any = xi.add(dx);
        particles.predictedPositions.element(iIdx).assign(vec4(newXi, float(0.0)));
        lambda.element(c).assign(lamCurrent.add(dLambda));
      });
      If(p.equal(jIdx), () => {
        // ∇_j C = -n  ⇒  Δx_j = w_j · Δλ · (-n) = -w_j · Δλ · n
        const dx: Any = n.mul(wj.mul(dLambda)).negate();
        const newXj: Any = xj.add(dx);
        particles.predictedPositions.element(jIdx).assign(vec4(newXj, float(0.0)));
      });
    })().compute(particles.capacity);

    groups.push({ particleToConstraint, solveKernel });
  }

  // ---- Lambda reset kernel (Macklin 2016 Algorithm 1 line 4) ----
  const resetLambdaKernel = Fn(() => {
    const c: Any = instanceIndex;
    lambda.element(c).assign(float(0.0));
  })().compute(nConstraints);

  return {
    arity,
    nConstraints,
    particleIndices,
    compliance: complianceBuf,
    restValue: restBuf,
    lambda,
    groups,
    resetLambdaKernel,
  };
}
