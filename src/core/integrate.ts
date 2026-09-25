import { Vector3 } from 'three';
import { Fn, If, atan, cos, float, instanceIndex, length, sin, uniform, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { ParticleSystem } from './particles.js';

// TSL's @types surface many nodes as bare `Node`, which drops the
// proxy-provided `.element()/.add()/.assign()/.xyz/...` methods. We loosen
// here rather than scaffold module-augmentation that outlives Phase 02.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Integration kernels and their mutable uniforms.
 *
 * Mutate `dt.value` / `gravity.value` between dispatches. TSL forwards the
 * updated values to the next `renderer.computeAsync(kernel)` call; there is
 * no per-step rebuild.
 */
export interface IntegrationKernels {
  readonly dt: UniformNode<'float', number>;
  readonly gravity: UniformNode<'vec3', Vector3>;
  /**
   * Opt-in implicit floor height enforced inside `predict` —
   * `x*.y = max(x*.y, floorY)` after integration. Default `0` is a
   * convenience for simple particle demos and
   * for the simplest softbody tests (`tests/analytical/softbody/
   * softbody-implicit.gpu.test.ts`) that want a "ground at y = 0"
   * without an analytic plane primitive.
   *
   * **Scenes with particles below `y = floorY` MUST set this to a large
   * negative value** (every fluid + softbody + rigid demo does
   * `loop.kernels.floorY.value = -1e9`). Otherwise predict teleports
   * those particles to `y = floorY` every substep, producing an
   * instantaneous over-density at that plane that the unilateral PBF
   * constraint relieves outward — the "blasts to the walls" failure
   * mode (Phase 16 gas demos exit, 2026-05-04).
   *
   * Originally tagged "Removed in Phase 06 when the analytic plane
   * primitive lands"; the analytic plane (Phase 06) coexists with the
   * clamp rather than replacing it. The clamp stays opt-in for tests
   * that don't want a `PrimitiveSet` allocation and for scenes that
   * find a single y-clamp simpler than authoring a plane.
   */
  readonly floorY: UniformNode<'float', number>;
  readonly predict: ComputeNode;
  readonly advect: ComputeNode;
  /**
   * Phase 12 — substep prologue, Mueller 2011 §4.1 Eq. 11. Per particle:
   * `qp ← rotateBy(ω, Δt) · q` with the `|ω| < ε` short-circuit `qp ← q`.
   * Always-on; on particles whose `ω = 0` (every particle except those a
   * §5.1 implicit-shape-matching kernel is updating) the kernel collapses
   * to a per-particle identity copy. Dispatched immediately after `predict`
   * so `predictedRotation` is in place before any iter-loop kernel reads it.
   */
  readonly predictRotation: ComputeNode;
  /**
   * Phase 12 — substep epilogue, Mueller 2011 §4.1 Eqs. 12–15. Per particle:
   * finite-difference `ω_i ← axis(r) · angle(r) / Δt` where `r = qp · q⁻¹`
   * with the shorter-rotation rule (`r.w < 0 → r ← −r`) and the
   * `|angle(r)| < ε` zero-clamp; then `q ← qp`. Dispatched alongside
   * `advect` so `q` and `ω` step together with `x` and `v`.
   */
  readonly advectRotation: ComputeNode;
}

/**
 * Build the predict + advect compute kernels for a `ParticleSystem`.
 *
 *
 *   Algorithm 1 line 2 — `v_i ← v_i + Δt · f_ext(x_i)`
 *     Here `f_ext` is an **acceleration** (gravity); no mass term. Non-
 *     gravitational forces contributed by later-phase modules will arrive
 *     via a separate per-particle force accumulator with a mass-weighted
 *     `v += Δt · F · invMass` branch.
 *   Algorithm 1 line 3 — `x*_i ← x_i + Δt · v_i`
 *     Uses the updated `v` from line 2 — semi-implicit (symplectic) Euler.
 *   Algorithm 1 lines 23–24 — velocity reconstruction / advect:
 *     `v_i ← (x*_i - x_i) / Δt ; x_i ← x*_i`
 *
 * The `x*.y = max(x*.y, floorY)` clamp in `predict` is opt-in via the
 * `floorY` uniform — see {@link IntegrationKernels.floorY} for the
 * disable-by-default contract scenes with particles below `floorY`
 * must follow.
 */
export function buildIntegrationKernels(particles: ParticleSystem): IntegrationKernels {
  const dt = uniform(1 / 60, 'float');
  const gravity = uniform(new Vector3(0, -9.81, 0));
  const floorY = uniform(0, 'float');

  const predict = Fn(() => {
    const i: Any = instanceIndex;
    const mInv: Any = particles.invMass.element(i);
    If(mInv.greaterThan(0.0), () => {
      const v: Any = particles.velocities.element(i);
      const x: Any = particles.positions.element(i);
      const xStar: Any = particles.predictedPositions.element(i);
      const newV3: Any = v.xyz.add((gravity as Any).mul(dt));
      v.assign(vec4(newV3, 0.0));
      const predicted3: Any = x.xyz.add(newV3.mul(dt));
      const clampedY: Any = predicted3.y.max(floorY);
      xStar.assign(vec4(predicted3.x, clampedY, predicted3.z, 0.0));
    });
  })().compute(particles.capacity);

  const advect = Fn(() => {
    const i: Any = instanceIndex;
    const mInv: Any = particles.invMass.element(i);
    If(mInv.greaterThan(0.0), () => {
      const x: Any = particles.positions.element(i);
      const xStar: Any = particles.predictedPositions.element(i);
      const v: Any = particles.velocities.element(i);
      const newV3: Any = xStar.xyz.sub(x.xyz).div(dt);
      v.assign(vec4(newV3, 0.0));
      x.assign(xStar);
    });
  })().compute(particles.capacity);

  // Phase 12 prologue + epilogue — Mueller 2011 §4.1.
  //
  // Eq. 11 (prologue): qp ← rotateBy(ω, Δt) · q.
  //   spinQuat = [(ω/|ω|)·sin(|ω|·Δt/2), cos(|ω|·Δt/2)]; qp = spinQuat · q.
  //   Stability: |ω| < ε ⇒ qp = q (avoids division by zero; below this
  //   threshold the rotation in one substep is sub-f32-quaternion precision).
  //
  // Eqs. 12–15 (epilogue): r = qp · q⁻¹ (q⁻¹ = conj(q) for unit q).
  //   Shorter-rotation rule: r.w < 0 ⇒ r ← −r (pick the ≤π arc).
  //   ω = axis(r) · angle(r) / Δt; |angle| < ε ⇒ ω = 0.
  //   angle = 2·atan2(|r.xyz|, r.w) (numerically stable near 0 and π).
  //   q ← qp.
  //
  // Both passes are unconditional per-particle (no invMass gate). Particles
  // with ω = 0 — the default for everything except implicit-mode softbody
  // particles whose §5.1 Pass 4 has overwritten `predictedRotation` — flow
  // through both passes as identities. The cost is one read + one write per
  // particle per pass, ~3 µs at 50k particles on Apple M-series.
  const OMEGA_EPS = 1e-8;
  const ANGLE_HALF_EPS = 1e-8;
  // Maximum |ω| (rad/s) committed by the §4.1 epilogue. A noisy R_i from
  // an ill-conditioned polar decomp (Phase 12 §5.1) yields a huge ω via
  // the Eq. 14 finite-difference, which the next substep's prologue
  // applies as an over-rotation, feeding bad R_prev into Pass 2's A_j
  // sum and cascading through the body. 100 rad/s is ~16 rev/s — well
  // above any plausible physical rotation rate for MVP scenes; only
  // pathological cases trigger the cap.
  // Non-softbody particles never generate non-zero ω so this cap is a
  // no-op for fluids / kinematic / explicit-mode softbody.
  const OMEGA_MAX = 100;

  const predictRotation = Fn(() => {
    const i: Any = instanceIndex;
    const q: Any = particles.rotation.element(i).toVar();
    const omega: Any = particles.angularVelocity.element(i).xyz.toVar();
    const omegaMag: Any = length(omega);
    If(omegaMag.lessThan(float(OMEGA_EPS)), () => {
      particles.predictedRotation.element(i).assign(q);
    }).Else(() => {
      const halfAngle: Any = omegaMag.mul(dt).mul(0.5);
      // (sin(halfAngle) / |ω|) is the ω-axis-magnitude scale that turns
      // ω into the imaginary part of the spin quaternion.
      const sScale: Any = sin(halfAngle).div(omegaMag);
      const spinXYZ: Any = omega.mul(sScale);
      const spinW: Any = cos(halfAngle);

      // Quaternion product spin · q.
      //   (spin · q).xyz = spin.w·q.xyz + q.w·spin.xyz + cross(spin.xyz, q.xyz)
      //   (spin · q).w   = spin.w·q.w   − dot(spin.xyz, q.xyz)
      const qXYZ: Any = q.xyz;
      const qW: Any = q.w;
      const newXYZ: Any = qXYZ.mul(spinW).add(spinXYZ.mul(qW)).add(spinXYZ.cross(qXYZ));
      const newW: Any = spinW.mul(qW).sub(spinXYZ.dot(qXYZ));
      particles.predictedRotation.element(i).assign(vec4(newXYZ.x, newXYZ.y, newXYZ.z, newW));
    });
  })().compute(particles.capacity);

  const advectRotation = Fn(() => {
    const i: Any = instanceIndex;
    const q: Any = particles.rotation.element(i).toVar();
    const qpRaw: Any = particles.predictedRotation.element(i).toVar();

    // Normalize qp at read. §5.1 Pass 4 writes qp from quatFromMat3(R_i)
    // directly; a near-singular polar decomp can produce R with det ≠ 1
    // and the resulting qp is non-unit. Subsequent quaternion math (the
    // r = qp · q⁻¹ → ω finite difference, and the q ← qp commit) needs a
    // unit qp to interpret the rotation angle correctly. The same divide-
    // by-zero guard `max(1e-12)` as on the commit side.
    const qpMagSq0: Any = qpRaw.x
      .mul(qpRaw.x)
      .add(qpRaw.y.mul(qpRaw.y))
      .add(qpRaw.z.mul(qpRaw.z))
      .add(qpRaw.w.mul(qpRaw.w));
    const qpInvMag0: Any = float(1.0).div(qpMagSq0.max(float(1e-12)).sqrt());
    const qp: Any = vec4(
      qpRaw.x.mul(qpInvMag0),
      qpRaw.y.mul(qpInvMag0),
      qpRaw.z.mul(qpInvMag0),
      qpRaw.w.mul(qpInvMag0),
    ).toVar();

    // r = qp · q⁻¹  where q⁻¹ = (−q.xyz, q.w) for unit q.
    const qInvXYZ: Any = q.xyz.negate();
    const qInvW: Any = q.w;
    const rXYZpre: Any = qp.xyz.mul(qInvW).add(qInvXYZ.mul(qp.w)).add(qp.xyz.cross(qInvXYZ));
    const rWpre: Any = qp.w.mul(qInvW).sub(qp.xyz.dot(qInvXYZ));

    // Shorter-rotation rule: if r.w < 0, flip r so angle(r) ∈ [0, π].
    const flipSign: Any = rWpre.lessThan(0.0).select(float(-1.0), float(1.0));
    const rXYZ: Any = rXYZpre.mul(flipSign).toVar();
    const rW: Any = rWpre.mul(flipSign).toVar();

    const halfSinMag: Any = length(rXYZ);
    If(halfSinMag.lessThan(float(ANGLE_HALF_EPS)), () => {
      particles.angularVelocity.element(i).assign(vec4(0.0, 0.0, 0.0, 0.0));
    }).Else(() => {
      // angle = 2·atan2(|r.xyz|, r.w); ω = axis·angle/Δt = (r.xyz/|r.xyz|)·angle/Δt.
      const angle: Any = atan(halfSinMag, rW).mul(2.0);
      const axisScale: Any = angle.div(halfSinMag).div(dt);
      const newOmegaRaw: Any = rXYZ.mul(axisScale);
      // Clamp |ω| to OMEGA_MAX. Matches the per-pair Δx clamp in
      // shapeMatchImplicit.ts §"Pass 3 scatter" — a noisy R_i would
      // otherwise feed an arbitrarily large ω into next substep's
      // Eq. 11 prologue and over-rotate qp. Cap formula:
      //   scale = sqrt(MAX² / max(|ω|², MAX²)) = min(1, MAX/|ω|).
      const omegaMagSq: Any = newOmegaRaw.x
        .mul(newOmegaRaw.x)
        .add(newOmegaRaw.y.mul(newOmegaRaw.y))
        .add(newOmegaRaw.z.mul(newOmegaRaw.z));
      const omegaCap: Any = float(OMEGA_MAX);
      const omegaScale: Any = omegaCap
        .mul(omegaCap)
        .div(omegaMagSq.max(omegaCap.mul(omegaCap)))
        .sqrt();
      const newOmega: Any = newOmegaRaw.mul(omegaScale);
      particles.angularVelocity
        .element(i)
        .assign(vec4(newOmega.x, newOmega.y, newOmega.z, float(0.0)));
    });

    // qp was already normalized at read; commit it directly. The chain
    // reaction we're guarding against: §5.1 Pass 4 writes a non-unit qp
    // when A_pq is ill-conditioned → mat3FromQuat next substep yields a
    // non-rotation matrix → A_j sum gets bad regularization → cascade.
    // Normalizing on the boundary (qp at read, q at commit) breaks it.
    particles.rotation.element(i).assign(qp);
  })().compute(particles.capacity);

  return { dt, gravity, floorY, predict, advect, predictRotation, advectRotation };
}
