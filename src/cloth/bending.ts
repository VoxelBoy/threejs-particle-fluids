import {
  Fn,
  If,
  Return,
  atan,
  cross,
  float,
  instanceIndex,
  instancedArray,
  uint,
  vec4,
} from 'three/tsl';

import {
  NO_CONSTRAINT,
  colorConstraints,
  xpbdDeltaLambda,
  type ConstraintGroup,
  type ConstraintType,
  type ParticleSystem,
  type XpbdUniforms,
} from '../core/index.js';

// TSL's @types surface many nodes as bare `Node`, dropping proxy methods
// (`.add`/`.mul`/`.div`/`.dot`/`.cross`/...). Same loose-alias pattern as
// `distance.ts` and the softbody kernels.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Cloth bending constraint factory.
 *
 * **Constraint form (Bender 2014 §3.4.2 / Bridson 2003 §4 normal
 * convention).** For a tuple `(p1, p2, p3, p4)` where `(p1, p2)` is
 * the shared edge and `p3, p4` are the far vertices of the two
 * incident triangles:
 *
 *   `N_1 = (p3 - p1) × (p3 - p2)`            (Bridson §4 area-weighted normal, far_1)
 *   `N_2 = (p4 - p2) × (p4 - p1)`            (area-weighted normal, far_2)
 *   `θ   = atan2((N_1×N_2)·E, (N_1·N_2)·|E|)`  (Bridson §4 SIGNED dihedral angle)
 *   `C(p1..p4) = θ - φ_0`                    (Bender 2014 §3.4.2 with Bridson signed-angle convention)
 *
 * **Why signed θ (atan2), not unsigned (acos).** Bridson 2003 §4
 * defines `sin(θ/2) = ±√((1−n̂_1·n̂_2)/2)` with the sign taken from
 * `(n̂_1 × n̂_2) · ê` (top of page 5, "Accurate Model for Bending"). The
 * closed-form gradients `u_k` below are therefore the gradient of the
 * SIGNED θ ∈ (−π, π]. A naive `C = acos(n̂_1·n̂_2)` constraint folds
 * negative- and positive-side bends onto the same non-negative C, so
 * `u_k` ends up pointing the wrong way on one half of the
 * configuration space — and any small perturbation that drove the
 * cloth to the negative-sin half would be amplified rather than
 * damped. Caught at Phase 18 entry by the central-difference test
 * (`tests/analytical/cloth/bending-gradient.test.ts`) and again by
 * the cloth-flag demo before this fix landed (the sheet exploded
 * into a self-intersecting spike-mass within the first few frames).
 * `atan2` keeps θ a smooth scalar in (−π, π] with consistent gradient
 * sign on both halves, matching `u_k` byte-for-byte.
 *
 * Flat planar config gives `θ = 0`; valley/mountain folds give signed
 * θ in (−π, π]. Rest angle `φ_0` is measured from the input geometry
 * by `fromBufferGeometry` (using the same atan2 form) so the resting
 * state of the cloth is the unstressed state.
 *
 * **Closed-form gradients (Bridson 2003 §4, with sign correction).**
 * The paper derives `u_k` via a 12-D modal decomposition of the four-
 * velocity space — orthogonal to the eleven non-bending modes.
 * Bridson labels `(x_1, x_2)` = far vertices, `(x_3, x_4)` = edge
 * endpoints, with `E = x_4 - x_3`. Translated to our Bender labels
 * `(x_1, x_2, x_3, x_4) ↔ (p_3, p_4, p_1, p_2)` and `E = p_2 - p_1`:
 *
 *   `u_1 = |E| · N_1 / |N_1|²`
 *   `u_2 = |E| · N_2 / |N_2|²`
 *   `u_3 = ((p3-p2)·E)/|E| · N_1/|N_1|² + ((p4-p2)·E)/|E| · N_2/|N_2|²`
 *   `u_4 = -((p3-p1)·E)/|E| · N_1/|N_1|² - ((p4-p1)·E)/|E| · N_2/|N_2|²`
 *
 * **Sign correction (Phase 18 entry, 2026-05-04).** Bridson 2003
 * page 4 introduces `u` parenthetically as "the gradient (the
 * steepest ascent direction) of the dihedral angle" — and Figure 1
 * captions the dihedral angle as `π − θ` (where his θ is the angle
 * between the unit normals). So `u_k = ∇(π − θ) = −∇θ`. The
 * companion formula `dθ/dt = u_1·v_1 + ... + u_4·v_4` on the same
 * page is internally inconsistent with that — it would require
 * `u_k = +∇θ`. Numerical perturbation testing
 * (`tests/analytical/cloth/bending-gradient.test.ts`) confirms
 * `u_k = −∇θ_atan2` for every component at flat and non-flat
 * configurations. Since our constraint is `C = θ_atan2 - φ_0`, the
 * gradient we need is `∇C = ∇θ = −u_k` — so the closed forms are
 * applied with an overall negation. This sign correction was the
 * root cause of the cloth-flag demo exploding in the first run-up
 * to the visual sign-off.
 *
 * Gradients of θ stay finite at the singular angles `θ ∈ {0, π}`
 * where `1/sin(θ)` would blow up — that is the elegant property
 * Bridson §4 calls out. The XPBD update applies eq. 17 with these
 * gradients and Macklin 2016 eq. 18 for `Δλ`:
 *
 *   `Δλ = (-C - α̃·λ) / (Σ_k w_k · |∇_k C|² + α̃)`,   α̃ = α/dt²
 *   `Δx_k = w_k · ∇_k C · Δλ`
 *
 * **Solve mode.** Gather + leader-write λ scatter, identical
 * structure to `distance.ts` extended to arity 4. Graph coloring
 * (Phase 04 `colorConstraints`) partitions tuples so no two in the
 * same color share any of their 4 participants — within a color the
 * dispatch can be data-parallel over `particles.capacity` with no
 * atomics. The "leader" thread for the λ accumulator write is the
 * `p1`-side thread (first participant); the same coloring guarantee
 * makes that single-writer.
 *
 * **Pinned vertices.** Slots with `invMass = 0` contribute 0 to both
 * `Σ_k w_k · |∇_k C|²` and to their own `Δx_k`, so the kernel handles
 * pinning naturally without a special branch.
 */
export function createClothBendingConstraints(args: {
  readonly particles: ParticleSystem;
  /** Absolute slot offset of this cloth's first particle. */
  readonly particleOffset: number;
  /**
   * Cloth-local bending tuples `[p1, p2, p3, p4]` with
   * `0 ≤ p_k < nClothParticles`. Typically the per-tuple entry from
   * {@link "./graph.js".ClothGraph.bendingTuples}.
   */
  readonly tuples: readonly (readonly [number, number, number, number])[];
  /**
   * Rest dihedral angle per tuple, in radians ∈ [0, π]. Typically the
   * per-tuple entry from {@link "./graph.js".ClothGraph.bendingRestAngles}.
   */
  readonly restAngles: readonly number[];
  /**
   * XPBD compliance `α` (s²/rad²·kg). Scalar broadcast to every tuple.
   * Phase 18 default `1e-5` — produces visible bend resistance at MVP
   * S/I (8/1) without locking to a flat reference under gravity.
   */
  readonly compliance: number;
  readonly xpbd: XpbdUniforms;
}): ConstraintType {
  const { particles, particleOffset, tuples, restAngles, compliance, xpbd } = args;

  if (!Number.isInteger(particleOffset) || particleOffset < 0) {
    throw new Error(
      `createClothBendingConstraints: particleOffset must be a non-negative integer, got ${particleOffset}`,
    );
  }
  const nConstraints = tuples.length;
  if (restAngles.length !== nConstraints) {
    throw new Error(
      `createClothBendingConstraints: restAngles.length ${restAngles.length} ≠ tuples.length ${nConstraints}`,
    );
  }
  for (const [a, b, c, d] of tuples) {
    if (
      !Number.isInteger(a) ||
      !Number.isInteger(b) ||
      !Number.isInteger(c) ||
      !Number.isInteger(d) ||
      a < 0 ||
      b < 0 ||
      c < 0 ||
      d < 0
    ) {
      throw new Error(
        `createClothBendingConstraints: tuple (${a},${b},${c},${d}) has a non-integer or negative index`,
      );
    }
    if (a === b || a === c || a === d || b === c || b === d || c === d) {
      throw new Error(
        `createClothBendingConstraints: tuple (${a},${b},${c},${d}) has duplicate indices`,
      );
    }
    const aAbs = particleOffset + a;
    const bAbs = particleOffset + b;
    const cAbs = particleOffset + c;
    const dAbs = particleOffset + d;
    if (
      aAbs >= particles.capacity ||
      bAbs >= particles.capacity ||
      cAbs >= particles.capacity ||
      dAbs >= particles.capacity
    ) {
      throw new Error(
        `createClothBendingConstraints: tuple absolute indices (${aAbs},${bAbs},${cAbs},${dAbs}) exceed capacity ${particles.capacity}`,
      );
    }
  }

  const arity = 4;

  const particleIndices = instancedArray(nConstraints * arity, 'uint');
  const complianceBuf = instancedArray(nConstraints, 'float');
  const restBuf = instancedArray(nConstraints, 'float');
  const lambda = instancedArray(nConstraints, 'float');
  {
    const idxArr = particleIndices.value.array as Uint32Array;
    const compArr = complianceBuf.value.array as Float32Array;
    const restArr = restBuf.value.array as Float32Array;
    for (let cIdx = 0; cIdx < nConstraints; cIdx++) {
      const [p1, p2, p3, p4] = tuples[cIdx]!;
      idxArr[cIdx * arity + 0] = particleOffset + p1;
      idxArr[cIdx * arity + 1] = particleOffset + p2;
      idxArr[cIdx * arity + 2] = particleOffset + p3;
      idxArr[cIdx * arity + 3] = particleOffset + p4;
      compArr[cIdx] = compliance;
      restArr[cIdx] = restAngles[cIdx]!;
    }
    particleIndices.value.needsUpdate = true;
    complianceBuf.value.needsUpdate = true;
    restBuf.value.needsUpdate = true;
  }

  // Graph coloring over flattened participants.
  const flat = new Uint32Array(nConstraints * arity);
  for (let cIdx = 0; cIdx < nConstraints; cIdx++) {
    const [p1, p2, p3, p4] = tuples[cIdx]!;
    flat[cIdx * arity + 0] = particleOffset + p1;
    flat[cIdx * arity + 1] = particleOffset + p2;
    flat[cIdx * arity + 2] = particleOffset + p3;
    flat[cIdx * arity + 3] = particleOffset + p4;
  }
  const { groupOf, numGroups } = colorConstraints({
    arity,
    nConstraints,
    participantsPerConstraint: flat,
  });

  const groups: ConstraintGroup[] = [];
  for (let g = 0; g < numGroups; g++) {
    const particleToConstraint = instancedArray(particles.capacity, 'uint');
    const invArr = particleToConstraint.value.array as Uint32Array;
    invArr.fill(NO_CONSTRAINT);
    for (let cIdx = 0; cIdx < nConstraints; cIdx++) {
      if (groupOf[cIdx] !== g) continue;
      const [p1, p2, p3, p4] = tuples[cIdx]!;
      invArr[particleOffset + p1] = cIdx;
      invArr[particleOffset + p2] = cIdx;
      invArr[particleOffset + p3] = cIdx;
      invArr[particleOffset + p4] = cIdx;
    }
    particleToConstraint.value.needsUpdate = true;

    const solveKernel = Fn(() => {
      const p: Any = instanceIndex;
      const c: Any = particleToConstraint.element(p).toVar();
      If(c.equal(uint(NO_CONSTRAINT)), () => {
        Return();
      });

      // Load 4 participant indices.
      const i1: Any = particleIndices.element(c.mul(uint(arity))).toVar();
      const i2: Any = particleIndices.element(c.mul(uint(arity)).add(uint(1))).toVar();
      const i3: Any = particleIndices.element(c.mul(uint(arity)).add(uint(2))).toVar();
      const i4: Any = particleIndices.element(c.mul(uint(arity)).add(uint(3))).toVar();

      // Predicted positions (vec3).
      const x1: Any = particles.predictedPositions.element(i1).xyz.toVar();
      const x2: Any = particles.predictedPositions.element(i2).xyz.toVar();
      const x3: Any = particles.predictedPositions.element(i3).xyz.toVar();
      const x4: Any = particles.predictedPositions.element(i4).xyz.toVar();
      const w1: Any = particles.invMass.element(i1).toVar();
      const w2: Any = particles.invMass.element(i2).toVar();
      const w3: Any = particles.invMass.element(i3).toVar();
      const w4: Any = particles.invMass.element(i4).toVar();
      const alpha: Any = complianceBuf.element(c).toVar();
      const phi0: Any = restBuf.element(c).toVar();
      const lamCurrent: Any = lambda.element(c).toVar();

      // E = p2 - p1.
      const E: Any = x2.sub(x1).toVar();
      const eLen2: Any = E.dot(E).toVar();
      // Degenerate edge — skip this thread.
      If(eLen2.lessThan(float(1e-24)), () => {
        Return();
      });
      const eLen: Any = eLen2.sqrt().toVar();

      // N_1 = (p3 - p1) × (p3 - p2)  — Bridson §4 area-weighted normal of triangle 1.
      const a3: Any = x3.sub(x1).toVar();
      const b3: Any = x3.sub(x2).toVar();
      const N1: Any = cross(a3, b3).toVar();
      // N_2 = (p4 - p2) × (p4 - p1)  — Bridson §4 area-weighted normal of triangle 2.
      const a4: Any = x4.sub(x2).toVar();
      const b4: Any = x4.sub(x1).toVar();
      const N2: Any = cross(a4, b4).toVar();
      const n1Sq: Any = N1.dot(N1).toVar();
      const n2Sq: Any = N2.dot(N2).toVar();
      // Degenerate triangle (collapsed) — skip.
      If(n1Sq.lessThan(float(1e-24)), () => {
        Return();
      });
      If(n2Sq.lessThan(float(1e-24)), () => {
        Return();
      });

      // Constraint value C = atan2((N_1×N_2)·E, (N_1·N_2)·|E|) - φ_0.
      // Both atan2 arguments share the factor |N_1|·|N_2|·|E| (positive),
      // so atan2 returns the signed dihedral angle θ ∈ (−π, π] directly
      // without per-pair normalisation. This matches the signed-θ
      // convention Bridson §4 uses for the closed-form gradients below.
      // No clamp needed — atan2 is well-defined for all inputs.
      const cross12: Any = cross(N1, N2);
      const sinTimes: Any = cross12.dot(E);
      const cosTimes: Any = N1.dot(N2).mul(eLen);
      // TSL exports a 2-arg `atan(y, x)` that compiles to WGSL `atan2(y, x)`.
      const theta: Any = atan(sinTimes, cosTimes).toVar();
      const C: Any = theta.sub(phi0).toVar();

      // Bridson §4 closed-form `u_k`, then NEGATE to get ∇C = ∇θ_atan2.
      // See block comment above ("Sign correction") for why u_k = -∇θ.
      //   u_1 = |E| · N_1/|N_1|²
      //   u_2 = |E| · N_2/|N_2|²
      //   u_3 = c31·N_1/|N_1|² + c32·N_2/|N_2|²
      //   u_4 = -(c41·N_1/|N_1|² + c42·N_2/|N_2|²)
      // with cKL = (relevant_diff · E) / |E|.
      const N1Scaled: Any = N1.div(n1Sq).toVar(); // N_1 / |N_1|²
      const N2Scaled: Any = N2.div(n2Sq).toVar(); // N_2 / |N_2|²
      // ∇_{p_3} C = -u_1
      const grad3: Any = N1Scaled.mul(eLen).negate().toVar();
      // ∇_{p_4} C = -u_2
      const grad4: Any = N2Scaled.mul(eLen).negate().toVar();
      // Coefficients for u_3 / u_4 — `(... · E) / |E|`.
      const c31: Any = b3.dot(E).div(eLen).toVar(); // (p3-p2)·E / |E|
      const c32: Any = a4.dot(E).div(eLen).toVar(); // (p4-p2)·E / |E|
      const c41: Any = a3.dot(E).div(eLen).toVar(); // (p3-p1)·E / |E|
      const c42: Any = b4.dot(E).div(eLen).toVar(); // (p4-p1)·E / |E|
      // ∇_{p_1} C = -u_3
      const grad1: Any = N1Scaled.mul(c31).add(N2Scaled.mul(c32)).negate().toVar();
      // ∇_{p_2} C = -u_4 = +(c41·N1Scaled + c42·N2Scaled)  (Bridson's u_4 already carries a leading minus)
      const grad2: Any = N1Scaled.mul(c41).add(N2Scaled.mul(c42)).toVar();

      // Σ w_k |∇_k C|².
      const sumGradSq: Any = grad1
        .dot(grad1)
        .mul(w1)
        .add(grad2.dot(grad2).mul(w2))
        .add(grad3.dot(grad3).mul(w3))
        .add(grad4.dot(grad4).mul(w4))
        .toVar();
      // All four pinned (sum = 0) → no correction.
      If(sumGradSq.lessThanEqual(float(1e-30)), () => {
        Return();
      });

      const dtVal: Any = xpbd.dt;
      const alphaTilde: Any = alpha.div(dtVal.mul(dtVal));
      const dLambda: Any = xpbdDeltaLambda({
        C,
        sumGradSqInvMass: sumGradSq,
        alphaTilde,
        lambdaCurrent: lamCurrent,
      }).toVar();

      // Apply Δx for whichever participant THIS thread is, and have the
      // p1-leader commit λ.
      If(p.equal(i1), () => {
        const dx: Any = grad1.mul(w1.mul(dLambda));
        const newX: Any = x1.add(dx);
        particles.predictedPositions.element(i1).assign(vec4(newX, float(0.0)));
        lambda.element(c).assign(lamCurrent.add(dLambda));
      });
      If(p.equal(i2), () => {
        const dx: Any = grad2.mul(w2.mul(dLambda));
        const newX: Any = x2.add(dx);
        particles.predictedPositions.element(i2).assign(vec4(newX, float(0.0)));
      });
      If(p.equal(i3), () => {
        const dx: Any = grad3.mul(w3.mul(dLambda));
        const newX: Any = x3.add(dx);
        particles.predictedPositions.element(i3).assign(vec4(newX, float(0.0)));
      });
      If(p.equal(i4), () => {
        const dx: Any = grad4.mul(w4.mul(dLambda));
        const newX: Any = x4.add(dx);
        particles.predictedPositions.element(i4).assign(vec4(newX, float(0.0)));
      });
    })().compute(particles.capacity);

    groups.push({ particleToConstraint, solveKernel });
  }

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
