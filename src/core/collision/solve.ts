import { Fn, If, Loop, Return, float, instanceIndex, uint, vec3 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleSystem } from '../particles.js';
import { type ContactAccumulator, emitAccumulateDelta } from '../contact/accumulator.js';
import type { PrimitiveSet } from './PrimitiveSet.js';
import { emitColliderSdf } from './primitives.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Build the per-particle collider solve kernel — the Phase 6 analogue of
 * the Phase 05 particle-particle contact solve, specialized for the case
 * where one side of the contact is a kinematic analytic primitive.
 *
 * Dispatch shape: one thread per particle, iterating the `numColliders`
 * active collider slots inline. Each thread reads its own `(p, 0..N_c)`
 * slice of the λ_n / λ_t buffers — no cross-thread contention, so the λ
 * buffers are non-atomic f32 (see {@link PrimitiveSet}). The per-particle
 * Δx correction is scattered into the shared {@link ContactAccumulator}
 * via `atomicAdd`, so the collider solve co-exists with the Phase 05
 * particle-particle scatter: both kernels accumulate into the same buffer
 * per iter, and the shared `buildApplyAccumulatorToPredictedKernel` then
 * commits the sum to `predictedPositions`. This is Jacobi-style under the
 * XPBD small-step form — intentional, for parity with particle-particle
 * contact.
 *
 * Per-collider math (paper references in-line):
 *
 *   1. SDF evaluation. `emitColliderSdf` returns `phi` (signed distance
 *      from particle centre to collider surface) and `∇phi` (outward
 *      unit normal). Penetration is `phi < r` (the particle radius), i.e.
 *      the particle's surface has crossed the collider's surface.
 *
 *   2. Normal projection. Macklin 2014 §6.1 eq. (22) non-penetration
 *      constraint `C = phi - r ≥ 0`. With one kinematic side, w_collider
 *      = 0 so `w_sum = w_i` and the Lagrange multiplier collapses to
 *      `Δλ_n = d / w_i`, correction `Δx_i = w_i · Δλ_n · ∇phi = d · ∇phi`.
 *
 *
 *      The proactive-form gate (vs. the paper's reactive `λ_t < μ_s · λ_n`
 *      test) mirrors the Phase 05a treatment in `contact/solve.ts` —
 *      under our parallel-scatter position solve the reactive form is
 *      trivially satisfied at iter 1 and over-applies static friction.
 *
 *   Kinetic friction is NOT handled here; the §3.6 velocity-level pass in
 *   `collision/frictionVelocity.ts` runs after advect and handles the
 *   slip branch.
 *
 * dt: shared with the XPBD / velocity-friction dt uniform. The collider's
 * kinematic displacement over the substep is `linVel · dt`, subtracted
 * from the particle's Δp so `Δp_t` measures relative tangential slip,
 * not absolute tangential motion.
 *
 * Determinism: every write is gather (per-particle kernel writing only to
 * its own slot in λ_n / λ_t) or through the shared accumulator's i32
 * atomicAdd. Both paths are G4 tier-1 bit-exact — see the particle-
 * particle scatter's determinism note in `contact/solve.ts`.
 */
export function buildColliderSolveKernel(args: {
  readonly particles: ParticleSystem;
  readonly colliders: PrimitiveSet;
  readonly accumulator: ContactAccumulator;
  readonly dt: UniformNode<'float', number>;
}): ComputeNode {
  const { particles, colliders, accumulator, dt } = args;
  const r = particles.particleRadius;
  const capacity = colliders.capacity;

  return Fn(() => {
    const p: Any = instanceIndex;
    const w: Any = particles.invMass.element(p).toVar();

    // Kinematic particles don't receive collider corrections. Paper:
    // Macklin 2014 §3 "kinematic particles (w = 0) are handled as
    // constraint anchors — other particles solve against them but they
    // themselves don't move."
    If(w.lessThanEqual(float(0.0)), () => {
      Return();
    });

    const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
    const x0: Any = particles.positions.element(p).xyz.toVar();
    const dp: Any = xStar.sub(x0).toVar(); // particle Δp over substep

    Loop(
      { start: uint(0), end: colliders.numColliders as Any, type: 'uint', condition: '<' },
      ({ i }: { i: Any }) => {
        const phi: Any = float(0.0).toVar();
        const grad: Any = vec3(float(0.0), float(0.0), float(0.0)).toVar();
        emitColliderSdf(colliders, i, xStar, phi, grad);

        // Penetration test — paper Macklin 2014 §6.1 eq. (22) with one
        // kinematic side: `phi - r < 0` means the particle's surface has
        // crossed the collider's surface.
        If(phi.lessThan(float(r)), () => {
          const d: Any = float(r).sub(phi).toVar(); // penetration depth (> 0)

          // ---- Normal projection (Macklin 2014 §6.1 eq. (22)) ----
          // w_collider = 0 → w_sum = w_i → Δx_i = d · grad.
          const dxN: Any = grad.mul(d).toVar();

          // Accumulate λ_n. Units: kg·m. Paper convention: `Δλ_n = d / w_sum`
          // with w_sum = w_i, so `Δλ_n = d / w_i = d · m_i`.
          // Interleaved layout: `lambdaNT[2·(p·capacity + c) + 0]` = λ_n,
          // `lambdaNT[2·(p·capacity + c) + 1]` = λ_t — see PrimitiveSet.
          const pairBase: Any = p.mul(uint(capacity)).add(i).mul(uint(2));
          const dLambdaN: Any = d.div(w).toVar();
          const prevLambdaN: Any = colliders.lambdaNT.element(pairBase).toVar();
          const lambdaNNow: Any = prevLambdaN.add(dLambdaN).toVar();
          colliders.lambdaNT.element(pairBase).assign(lambdaNNow);

          // ---- Static friction (Macklin 2020 §3.5 eqs. 26-28) ----
          // Δp_rel = particle Δp over substep - collider Δp over substep.
          // Collider is rigid-kinematic with linear velocity only (U-21
          // deferral): collider Δp = linVel · dt.
          const linVelPacked: Any = colliders.linVel.element(i);
          const linVel: Any = linVelPacked.xyz;
          // μ_s is stashed in `data1.w` (layout note on PrimitiveSet).
          const data1Slot: Any = colliders.data1.element(i);
          const muS: Any = data1Slot.w;
          const dpCollider: Any = linVel.mul(dt).toVar();
          const dpRel: Any = dp.sub(dpCollider).toVar();
          const tangential: Any = dpRel.sub(grad.mul(dpRel.dot(grad))).toVar();
          const tanLen: Any = tangential.length().toVar();

          // Proactive cone gate (Phase 05a adaptation for the scatter /
          // per-iter accumulation case — see `contact/solve.ts` for the
          // full derivation). Checks whether applying the full
          // tangential correction at this iter keeps λ_t inside
          // `μ_s · λ_n`; if not, static friction is skipped and the
          // §3.6 velocity-level pass handles the slip after advect.
          const lambdaTIdx: Any = pairBase.add(uint(1));
          const prevLambdaT: Any = colliders.lambdaNT.element(lambdaTIdx).toVar();
          const correctionLambdaT: Any = tanLen.div(w).toVar();
          const coneCapacity: Any = muS.mul(lambdaNNow).sub(prevLambdaT).toVar();
          const tanValid: Any = tanLen.greaterThan(float(1e-10));
          const withinStatic: Any = correctionLambdaT.lessThanEqual(coneCapacity);
          const applyStatic: Any = tanValid.and(withinStatic);

          // Δx_i = -Δp_t (Bender 2014 §3.5 split with w_j = 0).
          const dxF: Any = tangential.negate();
          const totalDx: Any = applyStatic.select(dxN.add(dxF), dxN).toVar();
          emitAccumulateDelta(accumulator, p, totalDx);

          // Accumulate Δλ_t on static hit.
          const dLambdaT: Any = applyStatic.select(tanLen.div(w), float(0.0)).toVar();
          colliders.lambdaNT.element(lambdaTIdx).assign(prevLambdaT.add(dLambdaT));
        });
      },
    );
  })().compute(particles.capacity);
}
