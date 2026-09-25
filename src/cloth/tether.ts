import { Fn, If, Return, float, instanceIndex, instancedArray, uint, vec4 } from 'three/tsl';

import {
  NO_CONSTRAINT,
  colorConstraints,
  xpbdDeltaLambda,
  type ConstraintGroup,
  type ConstraintType,
  type ParticleSystem,
  type XpbdUniforms,
} from '../core/index.js';

import type { TetherConstraint } from './tetherBuild.js';

// TSL's @types surface many nodes as bare `Node`, dropping proxy methods.
// Same loose-alias pattern as `distance.ts` and `bending.ts`.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Cloth tether (Long-Range-Attachment) constraint factory — Kim,
 * Chentanez, Müller-Fischer 2012 §3.1.
 *
 * **Constraint form (paper §3.1).** Per LRA between free particle
 * `p_i` and a fixed attachment point `a`:
 *
 *   `C(x_i) = |x_i - a| - r_i`     (active when `C > 0`, else inactive)
 *   `∇_i C = n = (x_i - a) / |x_i - a|`,   `|∇_i C|² = 1`
 *
 * Unilateral semantics: the constraint forbids the particle from
 * leaving the sphere of radius `r_i` centered at `a`, but allows
 * free movement inside. The kernel returns early when `C ≤ 0` so
 * the constraint imposes no force during compression / buckling
 * — local distance + bending constraints (Phase 18) handle wrinkle
 * formation. Paper §3.1: "LRA constraints are unilateral - they
 * get activated only when cloth is stretched, and do not influence
 * the buckling behavior".
 *
 * **XPBD update (Macklin 2016).** When `C > 0`:
 *
 *   `α̃ = α / Δt²`,  `Δλ = (-C - α̃·λ) / (w + α̃)`,  `Δx_i = w · n · Δλ`
 *
 * `tetherCompliance = α = 0` (the default) reduces this to plain
 * PBD: `Δx_i = -C · n` projects `x_i` onto the constraint sphere
 * exactly — matching Kim 2012's implicit "infinite stiffness"
 * behaviour described in their abstract.
 *
 * **λ behaviour during inactive periods.** When `C ≤ 0` the kernel
 * resets `λ ← 0` before returning. For soft tethers (α > 0) this
 * matches the unilateral semantics: a satisfied tether must not
 * pull the particle inward via the historical-λ memory term. For
 * the strict default (α = 0) the reset is a no-op since `α̃·λ`
 * vanishes from `Δλ` anyway.
 *
 * **Multi-island averaging deviation from paper §3.4 (Assumed).**
 * Kim 2012 §3.4 specifies Jacobi-style averaging across the up-to-N
 * LRAs assigned to one free particle. We instead use graph-coloring
 * + Gauss-Seidel-across-colors via the standard
 * `ConstraintType` scaffolding — when a particle has K LRAs to K
 * different islands, the K constraints land in K distinct color
 * groups and are projected sequentially within an iter. For the
 * single-island scenes Phase 19's validation gates exercise (32×32
 * sheet pinned along one edge — N degenerates to 1 per particle),
 * the two are identical. For multi-island scenes (paper Fig. 7
 * dress pinned at shoulder + waist) the GS-across-colors path is
 * biased toward the last-processed island. Documented as a Phase 19
 * post-MVP item; would require a Jacobi-via-i32-scatter pipeline
 * (the per-(particle, LRA) average can't be expressed in the
 * graph-colored gather pattern without atomics).
 *
 * **Solve mode.** Gather + leader-write λ scatter, arity = 1. The
 * inverted index `particleToConstraint` maps the free particle's
 * slot directly to `c` and every other slot to `NO_CONSTRAINT`, so
 * the kernel dispatches over `particles.capacity` and returns
 * immediately on every non-participating thread. Same conflict-
 * free-write pattern as the Phase 18 distance / bending kernels;
 * graph coloring guarantees no two constraints in a color share
 * the free particle.
 *
 * **Pinned vertices** — never reach this kernel. `buildTethers`
 * filters them out at the build stage (no LRA is emitted for a
 * pinned particle), and the scatter-side per-particle inverted
 * index leaves their slot as `NO_CONSTRAINT`.
 */
export function createClothTetherConstraints(args: {
  readonly particles: ParticleSystem;
  /** Absolute slot offset of this cloth's first particle. */
  readonly particleOffset: number;
  /**
   * Cloth-local LRA constraint list — typically the output of
   * {@link "./tetherBuild.js".buildTethers}. Each entry is one
   * (free-particle, anchor, restRadius) tuple.
   */
  readonly tethers: readonly TetherConstraint[];
  /**
   * XPBD compliance `α` (s²/kg). Default `0` matches Kim 2012's
   * implicit infinite-stiffness behaviour; non-zero produces a
   * "soft" tether that allows some over-stretch under load.
   */
  readonly compliance: number;
  readonly xpbd: XpbdUniforms;
}): ConstraintType {
  const { particles, particleOffset, tethers, compliance, xpbd } = args;

  if (!Number.isInteger(particleOffset) || particleOffset < 0) {
    throw new Error(
      `createClothTetherConstraints: particleOffset must be a non-negative integer, got ${particleOffset}`,
    );
  }
  if (!Number.isFinite(compliance) || compliance < 0) {
    throw new Error(
      `createClothTetherConstraints: compliance must be a non-negative finite number, got ${compliance}`,
    );
  }

  const nConstraints = tethers.length;
  const arity = 1;

  for (const t of tethers) {
    const abs = particleOffset + t.particle;
    if (!Number.isInteger(t.particle) || t.particle < 0 || abs >= particles.capacity) {
      throw new Error(
        `createClothTetherConstraints: tether particle ${t.particle} (absolute ${abs}) out of range for capacity ${particles.capacity}`,
      );
    }
    if (!Number.isFinite(t.restRadius) || t.restRadius < 0) {
      throw new Error(
        `createClothTetherConstraints: tether for particle ${t.particle} has invalid restRadius ${t.restRadius}`,
      );
    }
    for (const v of t.anchor) {
      if (!Number.isFinite(v)) {
        throw new Error(
          `createClothTetherConstraints: tether for particle ${t.particle} has non-finite anchor coord ${v}`,
        );
      }
    }
  }

  // ---- Per-constraint SoA storage ----
  const particleIndices = instancedArray(Math.max(1, nConstraints * arity), 'uint');
  const anchorBuf = instancedArray(Math.max(1, nConstraints), 'vec4');
  const complianceBuf = instancedArray(Math.max(1, nConstraints), 'float');
  const restBuf = instancedArray(Math.max(1, nConstraints), 'float');
  const lambda = instancedArray(Math.max(1, nConstraints), 'float');

  if (nConstraints > 0) {
    const idxArr = particleIndices.value.array as Uint32Array;
    const anchorArr = anchorBuf.value.array as Float32Array;
    const compArr = complianceBuf.value.array as Float32Array;
    const restArr = restBuf.value.array as Float32Array;
    for (let c = 0; c < nConstraints; c++) {
      const t = tethers[c]!;
      idxArr[c] = particleOffset + t.particle;
      anchorArr[c * 4 + 0] = t.anchor[0];
      anchorArr[c * 4 + 1] = t.anchor[1];
      anchorArr[c * 4 + 2] = t.anchor[2];
      anchorArr[c * 4 + 3] = 0;
      compArr[c] = compliance;
      restArr[c] = t.restRadius;
    }
    particleIndices.value.needsUpdate = true;
    anchorBuf.value.needsUpdate = true;
    complianceBuf.value.needsUpdate = true;
    restBuf.value.needsUpdate = true;
  }

  // ---- Graph coloring ----
  // For arity-1 LRAs, two constraints conflict iff they share the
  // same free particle (i.e. the particle has multiple LRAs to
  // different islands). The greedy pass produces one color per
  // multiplicity level: a particle with K LRAs gets K colors.
  const flat = new Uint32Array(nConstraints * arity);
  for (let c = 0; c < nConstraints; c++) {
    flat[c] = particleOffset + tethers[c]!.particle;
  }
  const { groupOf, numGroups } = colorConstraints({
    arity,
    nConstraints,
    participantsPerConstraint: flat,
  });

  // ---- Per-group inverted index + solve kernel ----
  const groups: ConstraintGroup[] = [];
  for (let g = 0; g < numGroups; g++) {
    const particleToConstraint = instancedArray(particles.capacity, 'uint');
    const invArr = particleToConstraint.value.array as Uint32Array;
    invArr.fill(NO_CONSTRAINT);
    for (let c = 0; c < nConstraints; c++) {
      if (groupOf[c] !== g) continue;
      invArr[particleOffset + tethers[c]!.particle] = c;
    }
    particleToConstraint.value.needsUpdate = true;

    const solveKernel = Fn(() => {
      const p: Any = instanceIndex;
      const c: Any = particleToConstraint.element(p).toVar();
      If(c.equal(uint(NO_CONSTRAINT)), () => {
        Return();
      });

      const idx: Any = particleIndices.element(c).toVar();
      const x: Any = particles.predictedPositions.element(idx).xyz.toVar();
      const w: Any = particles.invMass.element(idx).toVar();
      If(w.lessThanEqual(float(0.0)), () => {
        Return();
      });

      const a: Any = anchorBuf.element(c).xyz.toVar();
      const alpha: Any = complianceBuf.element(c).toVar();
      const rest: Any = restBuf.element(c).toVar();
      const lamCurrent: Any = lambda.element(c).toVar();

      const diff: Any = x.sub(a).toVar();
      const len: Any = diff.length().toVar();

      // Unilateral check (Kim 2012 §3.1): no projection when |x − a| ≤ r.
      // Reset λ ← 0 so a soft tether (α > 0) doesn't carry stale
      // historical multiplier into the next iter where the constraint
      // might re-activate; for the α = 0 default the reset is a no-op
      // since α̃·λ vanishes from Δλ regardless.
      If(len.lessThanEqual(rest), () => {
        lambda.element(c).assign(float(0.0));
        Return();
      });
      // Degenerate (`x ≡ a`) — gradient undefined. Leave λ as-is.
      If(len.lessThan(float(1e-12)), () => {
        Return();
      });

      const C: Any = len.sub(rest);
      const n: Any = diff.div(len).toVar();

      const dtVal: Any = xpbd.dt;
      const alphaTilde: Any = alpha.div(dtVal.mul(dtVal));
      const dLambda: Any = xpbdDeltaLambda({
        C,
        sumGradSqInvMass: w,
        alphaTilde,
        lambdaCurrent: lamCurrent,
      }).toVar();

      // ∇_i C = n  ⇒  Δx_i = w · n · Δλ
      const dx: Any = n.mul(w.mul(dLambda));
      const newX: Any = x.add(dx);
      particles.predictedPositions.element(idx).assign(vec4(newX, float(0.0)));
      lambda.element(c).assign(lamCurrent.add(dLambda));
    })().compute(particles.capacity);

    groups.push({ particleToConstraint, solveKernel });
  }

  // ---- λ reset (Macklin 2016 Algorithm 1 line 4) ----
  // Dispatch over at least one slot — `instancedArray` would refuse
  // a 0-length compute. When `nConstraints === 0` the kernel runs
  // over the placeholder slot, which is harmless (writes 0 to
  // unused slot 0 of `lambda`).
  const resetLambdaKernel = Fn(() => {
    const c: Any = instanceIndex;
    lambda.element(c).assign(float(0.0));
  })().compute(Math.max(1, nConstraints));

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
