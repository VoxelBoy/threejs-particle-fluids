import { Fn, If, Loop, Return, float, instanceIndex, uint, vec3 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleSystem } from '../particles.js';
import {
  type VelocityAccumulator,
  emitAccumulateVelocityDelta,
} from '../contact/velocityAccumulator.js';
import type { PrimitiveSet } from './PrimitiveSet.js';
import { emitColliderSdf } from './primitives.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Build the per-particle velocity-level dynamic-friction kernel for
 * particle-vs-analytic-collider contacts.
 *
 *
 *   Eq. 29: relative contact-point velocity decomposition into normal and
 *           tangential components. For particle-collider the collider's
 *           rotation is zero (Phase 6 — see U-21), so the collider's
 *           contact-point velocity is just its linear velocity.
 *   Eq. 30: Δv ← −(v_t/|v_t|) · min(h · μ_d · |f_n|, |v_t|) with
 *           f_n = λ_n / h². The min-clamp makes the friction correction
 *           unconditionally stable; the tangential velocity is never
 *           over-corrected past zero.
 *   Eq. 33: inverse-mass split, particle-simplified. With w_collider = 0
 *           the full Δv is absorbed by the particle side.
 *
 * Unit note on eq. 30 (same as `contact/frictionVelocity.ts`): the paper
 * writes `h · μ_d · |f_n|` with f_n = λ_n / h². For particle-collider,
 * dimensional consistency requires the reduced-mass factor (w_i + w_j) =
 * w_i (w_collider = 0). The implementation below writes the threshold as
 *   threshold = μ_d · λ_n · w_i / h
 * which has units `(dimless) · kg·m · (1/kg) / s = m/s`, matching |v_t|.
 *
 * Dispatch shape: one thread per particle, iterating active collider
 * slots inline. Each thread reads its own `(p, 0..N_c)` slice of
 * `colliders.lambdaN` — non-atomic. The per-particle Δv correction is
 * scattered into {@link VelocityAccumulator} via i32 atomicAdd.
 *
 * Determinism: per-particle gather on λ_n plus the accumulator's i32
 * atomicAdd (associative + commutative + exact) gives G4 tier-1 bit-exact
 * across repeat runs.
 *
 * Guard cases (mirror `contact/frictionVelocity.ts`):
 *   - Kinematic particle (w ≤ 0): return.
 *   - λ_n ≤ 0 for this (p, collider): no accumulated normal impulse, no
 *     friction to apply — Continue to next collider.
 *   - Gradient-evaluation fallback: SDF evaluated at post-advect position.
 *     If gradient is near-zero (degenerate geometry touching the particle
 *     at a singular locus), skip.
 *   - |v_t| near zero: no tangential motion to correct.
 */
export function buildColliderFrictionVelocityKernel(args: {
  readonly particles: ParticleSystem;
  readonly colliders: PrimitiveSet;
  readonly velocityAccumulator: VelocityAccumulator;
  /**
   * Substep timestep h (seconds). Shares the same uniform handle as the
   * XPBD dt uniform; updates via `SimLoop.step` propagate.
   */
  readonly dt: UniformNode<'float', number>;
}): ComputeNode {
  const { particles, colliders, velocityAccumulator, dt } = args;
  const capacity = colliders.capacity;

  return Fn(() => {
    const p: Any = instanceIndex;
    const w: Any = particles.invMass.element(p).toVar();
    If(w.lessThanEqual(float(0.0)), () => {
      Return();
    });

    // Post-advect: `positions` == `predictedPositions` == committed x*.
    // Reading `predictedPositions` matches the particle-particle pass in
    // `contact/frictionVelocity.ts` for consistency.
    const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
    const v: Any = particles.velocities.element(p).xyz.toVar();

    Loop(
      { start: uint(0), end: colliders.numColliders as Any, type: 'uint', condition: '<' },
      ({ i }: { i: Any }) => {
        // Interleaved λ layout: λ_n lives at even slot, λ_t at odd.
        const pairBase: Any = p.mul(uint(capacity)).add(i).mul(uint(2));
        const lambdaN: Any = colliders.lambdaNT.element(pairBase).toVar();

        // No accumulated normal impulse → no friction to clamp.
        If(lambdaN.lessThanEqual(float(0.0)), () => {
          // TSL has no `Continue` inside an `If` nested in `Loop` that
          // returns from the If callback while remaining inside the Loop
          // iteration. Wrap the rest of the loop body in the inverse If
          // so this early-skip keeps the kernel within its iter.
        });
        If(lambdaN.greaterThan(float(0.0)), () => {
          // Re-evaluate SDF at post-solve position to recover the contact
          // normal. The position was projected onto the collider surface
          // during the last iter of the position solve, so `phi ≈ r` and
          // `grad` is well-defined (except at singular loci — guarded).
          const phi: Any = float(0.0).toVar();
          const grad: Any = vec3(float(0.0), float(0.0), float(0.0)).toVar();
          emitColliderSdf(colliders, i, xStar, phi, grad);
          const gradLen: Any = grad.length().toVar();
          If(gradLen.greaterThan(float(1e-6)), () => {
            // Normalize defensively; `emitColliderSdf` returns unit gradient
            // in the common case, but sphere / capsule SDFs guard against
            // center-singularity with `max(len, 1e-8)` which can leave
            // magnitude slightly off when the particle lands at the centre.
            const n: Any = grad.div(gradLen).toVar();

            // Relative contact-point velocity. Kinematic collider has no
            // angular velocity (U-21) → collider contact-point velocity is
            // just its linear velocity.
            const linVelPacked: Any = colliders.linVel.element(i);
            const linVel: Any = linVelPacked.xyz;
            // μ_k is stashed in `linVel.w`.
            const muK: Any = linVelPacked.w;
            const vRel: Any = v.sub(linVel).toVar();
            const vN: Any = n.dot(vRel);
            const vT: Any = vRel.sub(n.mul(vN)).toVar();
            const vTLen: Any = vT.length().toVar();

            If(vTLen.greaterThan(float(1e-6)), () => {
              // Eq. 30 threshold with explicit reduced-mass factor.
              const threshold: Any = muK
                .mul(lambdaN)
                .mul(w)
                .div(dt as Any)
                .toVar();
              const deltaVMag: Any = threshold.min(vTLen).toVar();
              // w_collider = 0 → full correction lands on particle.
              const deltaV: Any = vT.div(vTLen).mul(deltaVMag).negate().toVar();
              emitAccumulateVelocityDelta(velocityAccumulator, p, deltaV);
            });
          });
        });
      },
    );
  })().compute(particles.capacity);
}
