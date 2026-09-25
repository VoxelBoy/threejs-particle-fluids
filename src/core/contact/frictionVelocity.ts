import { Fn, If, Return, atomicLoad, float, instanceIndex, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleSystem } from '../particles.js';
import type { ContactBuffer } from './ContactBuffer.js';
import { type VelocityAccumulator, emitAccumulateVelocityDelta } from './velocityAccumulator.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Build the velocity-level dynamic friction scatter kernel.
 *
 *
 *   Eq. 29: relative contact-point velocity decomposed into normal and
 *           tangential components.
 *   Eq. 30: Δv ← −(v_t/|v_t|) · min(h · μ_d · |f_n|, |v_t|) with
 *           f_n = λ_n / h². The min-clamp ensures the velocity correction
 *           never exceeds the velocity itself, making the explicit friction
 *           integration unconditionally stable.
 *   Eq. 33: inverse-mass split, particle-simplified (rotational ω updates
 *           dropped for particle contacts).
 *
 * Coefficient note: the kernel reads `friction.muK`, which is the kinetic
 * Coulomb coefficient from the solve.ts FrictionUniforms type. In Coulomb
 * friction, kinetic and dynamic are synonymous, so μ_k = μ_d; the Phase 5
 * API name is preserved for backward compatibility and the paper's μ_d
 * notation is used only in the equation comments below.
 *
 * Unit note on eq. 30: the paper writes the clamp threshold as
 * `h · μ_d · |f_n|` with f_n = λ_n / h². Taken literally with λ_n in XPBD's
 * kg·m units, this product has units kg·m/s — momentum, which does not
 * match |v_t|'s units of m/s. Dimensional consistency requires dividing by
 * a reduced mass; the particle-pair reduced-mass is 1 / (w_i + w_j). The
 * implementation below writes the threshold as
 * `μ_d · λ_n · (w_i + w_j) / h` — equivalent to the paper's expression with
 * the implicit reduced-mass factor made explicit. Dimensionally:
 * `(dimless) · kg·m · (1/kg) / s = m/s`, matching |v_t|.
 *
 * Dispatch shape: one thread per contact, over maxContacts threads. Each
 * thread reads the final accumulated normal Lagrange multiplier λ_n (summed
 * across all position-solve iterations of the current substep), computes
 * its pair's Δv correction, and scatters via atomicAdd into the velocity
 * accumulator. Called once per substep after the XPBD iter loop and after
 * the position accumulator has been applied.
 *
 * Determinism: i32 atomicAdd is associative, commutative, and exact. G4
 * tier-1 bit-exact across repeat runs, matching the position scatter
 * contact solve.
 *
 * Guard cases:
 *   - Early return if the contact index is past the emitted counter.
 *   - Early return if |x*_i − x*_j| is near zero (degenerate pair; normal
 *     is undefined).
 *   - Early return if both particles are kinematic (w_i + w_j ≤ 0).
 *   - Early return if |v_t| is near zero (no tangential motion to correct).
 *   - Early return if λ_n ≤ 0 (no accumulated normal impulse at this
 *     contact; without a normal force there is no friction force to apply).
 */
export function buildContactFrictionVelocityKernel(args: {
  readonly particles: ParticleSystem;
  readonly contacts: ContactBuffer;
  readonly velocityAccumulator: VelocityAccumulator;
  /**
   * Substep timestep h (seconds). Share the same uniform handle as the
   * XPBD solver's dt uniform; updates via {@link SimLoop.step} propagate.
   */
  readonly dt: UniformNode<'float', number>;
}): ComputeNode {
  const { particles, contacts, velocityAccumulator, dt } = args;
  const maxContacts = contacts.maxContacts;
  const invLambdaScale = 1 / contacts.lambdaScale;

  return Fn(() => {
    const c: Any = instanceIndex;
    const nRaw: Any = atomicLoad(contacts.counter.element(uint(0)));
    If(c.greaterThanEqual(nRaw).or(c.greaterThanEqual(uint(maxContacts))), () => {
      Return();
    });

    // Phase 21a — single struct-buffer ContactRecord access. All per-pair
    // fields (i, j, lambdaN, normal) are members of `records[c]`.
    const rec: Any = contacts.records.element(c);
    const i: Any = rec.get('i').toVar();
    const j: Any = rec.get('j').toVar();

    // Macklin 2020 eq. 29: contact-point relative velocity decomposition.
    // For particle contacts (no rotation) the formula collapses to
    // v = v_i − v_j. Phase 15: read `n` from the record's `normal` field
    // (written by whichever position-solve handled this pair — spherical
    // or SDF) instead of recomputing from `xij/|xij|`. Recomputing here
    // injects spurious tangential velocity on rigid-rigid pairs whose SDF
    // normal diverges from the spherical inter-particle direction.
    const wi: Any = particles.invMass.element(i).toVar();
    const wj: Any = particles.invMass.element(j).toVar();

    const wSum: Any = wi.add(wj).toVar();
    If(wSum.lessThanEqual(float(0.0)), () => {
      Return();
    });

    const n: Any = rec.get('normal').toVar();
    // The position-solve writes `n` in lockstep with `lambdaN[c]`. The
    // `lambdaN > 0` gate below doubles as a "did any solve fire on this
    // pair" gate (zero-init lambdaN means no solve fired, so a stale
    // `n` from a previous substep would be ignored anyway).

    const vi: Any = particles.velocities.element(i).xyz.toVar();
    const vj: Any = particles.velocities.element(j).xyz.toVar();
    const v: Any = vi.sub(vj).toVar();
    const vN: Any = n.dot(v).toVar();
    const vT: Any = v.sub(n.mul(vN)).toVar();
    const vTLen: Any = vT.length().toVar();
    If(vTLen.lessThan(float(1e-6)), () => {
      Return();
    });

    // Read the final accumulated λ_n for this contact. Descale from
    // fixed-point ticks to kg·m. Phase 21a — `lambdaN` is an atomic struct
    // field on `ContactRecord`.
    const lambdaNLoaded: Any = atomicLoad(rec.get('lambdaN'));
    const lambdaN: Any = lambdaNLoaded.toFloat().mul(float(invLambdaScale)).toVar();
    If(lambdaN.lessThanEqual(float(0.0)), () => {
      Return();
    });

    // Macklin 2020 eq. 30 with the implicit (w_i + w_j) factor made explicit
    // for dimensional consistency — see the top-of-file unit note. Phase 21
    // — μ_k is per-pair, populated by `contact/generate.ts` from the
    // `FrictionTable` LUT; combine rule applied at emit time, this read
    // is a scalar.
    //
    //   threshold = μ_d · λ_n · (w_i + w_j) / h    (units: m/s)
    //   Δv       = −(v_t / |v_t|) · min(threshold, |v_t|)
    const muKPair: Any = rec.get('muK').toVar();
    const threshold: Any = muKPair
      .mul(lambdaN)
      .mul(wSum)
      .div(dt as Any)
      .toVar();
    const deltaVMag: Any = threshold.min(vTLen).toVar();
    const deltaV: Any = vT.div(vTLen).mul(deltaVMag).negate().toVar();

    // Macklin 2020 eq. 33 particle-simplified: inverse-mass split.
    //   p = Δv / (w_i + w_j)         (momentum impulse vector)
    //   Δv_i = +p · w_i              (velocity change on particle i)
    //   Δv_j = −p · w_j              (velocity change on particle j)
    // Momentum-conserving: m_i · Δv_i + m_j · Δv_j = p − p = 0.
    const p: Any = deltaV.div(wSum).toVar();
    const deltaVi: Any = p.mul(wi).toVar();
    const deltaVj: Any = p.mul(wj).negate().toVar();

    emitAccumulateVelocityDelta(velocityAccumulator, i, deltaVi);
    emitAccumulateVelocityDelta(velocityAccumulator, j, deltaVj);
  })().compute(maxContacts);
}
