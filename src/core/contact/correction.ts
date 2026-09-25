import { atomicAdd, atomicLoad, float } from 'three/tsl';

import type { ParticleSystem } from '../particles.js';
import type { ContactBuffer } from './ContactBuffer.js';
import { type ContactAccumulator, emitAccumulateDelta } from './accumulator.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Per-pair contact-solve body, shared across every geometry mode.
 *
 * Inputs:
 *   - `(n, d)`: unit contact normal pointing from j toward i, and the
 *     positive penetration depth in metres. Computed by either the
 *     spherical default (Macklin 2014 §6.1 eq. 22) or a registered
 *     {@link ContactGeometryExtension} (Phase 15a — softbody's SDF mode
 *     for rigid-rigid pairs, paper §5.1 eqs. 17–20).
 *   - `(i, j, c)`: the pair indices and the contact slot index, all TSL
 *     nodes evaluated inside the kernel.
 *
 * The helper performs four operations per call:
 *
 *   1. Δλ_n = d / (w_i + w_j) and the Bender 2014 §3.5 momentum-conserving
 *      normal scatter (normalDxI/J).
 *   2. Δλ_n accumulation into `contacts.lambdaN[c]` and the per-contact
 *      normal write into `contacts.normal[c]` so the §3.6 velocity-friction
 *      pass reads the same `n` this iter wrote (otherwise a recomputed
 *      spherical normal would diverge from an SDF-mode write and inject
 *      tangential energy on rigid-rigid stacks).
 *   3. Static-friction cone gate (Macklin 2020 §3.5 eqs. 26–28). The
 *      proactive form `correctionLambdaT ≤ μ_s · λ_n − λ_t` is used
 *      because parallel scatter cannot see same-iter λ_t updates from
 *      other threads — see solve.ts §"Sign convention" for the rationale.
 *   4. Δλ_t accumulation into `contacts.lambdaT[c]` when the gate fires.
 *
 * Reads `particles.contactInvMass`, not `particles.invMass`. The per-
 * substep copy kernel (`buildCopyContactInvMassKernel`) seeds
 * `contactInvMass[i] = invMass[i]` at the head of the contact preIter
 * block; material modules whose contact response needs a mass-scaling
 * override (Macklin 2014 §5.2 eq. 21 stiff stacks) overwrite from their
 * own preIter, which sequences after the copy. The decoupling keeps the
 * physical mass channel (`invMass`) free of contact-pipeline-specific
 * scaling that would otherwise leak into predict / advect / the
 * constraint scheduler.
 *
 * Determinism: every write is an `atomicAdd` (G4 Tier 1 bit-exact) or an
 * `assign` to a per-c slot the caller has uniquely indexed.
 *
 * **Precondition.** Caller must have early-returned on:
 *   - `c >= contacts.counter` (past the emitted contact count),
 *   - `c >= contacts.maxContacts` (past the buffer size),
 *   - degenerate normal direction (`|x_ij| < ε` or geometry-mode-specific
 *     equivalent),
 *   - non-penetration (`d ≤ 0`),
 *   - `w_i + w_j ≤ 0` (both kinematic).
 * The helper assumes a penetrating, non-degenerate pair with at least one
 * dynamic participant.
 */
export interface EmitContactSolveCorrectionArgs {
  readonly i: Any;
  readonly j: Any;
  readonly c: Any;
  readonly n: Any;
  readonly d: Any;
  readonly particles: ParticleSystem;
  readonly contacts: ContactBuffer;
  readonly accumulator: ContactAccumulator;
}

export function emitContactSolveCorrection(args: EmitContactSolveCorrectionArgs): void {
  const { i, j, c, n, d, particles, contacts, accumulator } = args;
  const lambdaScale = contacts.lambdaScale;
  const invLambdaScale = 1 / lambdaScale;

  // Phase 15a — contact-time inverse mass channel. Reads of
  // `particles.invMass` here would silently bypass material overrides
  // (softbody §5.2 stiff stacks).
  const wi: Any = particles.contactInvMass.element(i).toVar();
  const wj: Any = particles.contactInvMass.element(j).toVar();
  const wSum: Any = wi.add(wj).toVar();

  // ---- Normal projection (Macklin 2014 §6.1 eq. 22, α = 0) ----
  const dLambda: Any = d.div(wSum).toVar();
  const normalDxI: Any = n.mul(wi.mul(dLambda)).toVar();
  const normalDxJ: Any = n.mul(wj.mul(dLambda)).negate().toVar();

  // Phase 5a: accumulate Δλ_n into the per-contact record so §3.5 below
  // and the §3.6 velocity-friction pass downstream can read the final
  // accumulated multiplier. Phase 21a — `lambdaN`/`lambdaT` are atomic
  // struct fields on `ContactRecord`.
  const rec: Any = contacts.records.element(c);
  const dLambdaTicks: Any = dLambda.mul(float(lambdaScale)).toInt();
  atomicAdd(rec.get('lambdaN'), dLambdaTicks);

  // Phase 15: store the contact normal so friction-velocity reads the
  // same `n` we used here. The per-iter write is idempotent across
  // iterations of the same substep (n is constant for the pair under a
  // given geometry mode); cross-substep, it is overwritten before
  // friction-velocity reads. Phase 21a — `normal` is a vec3 struct field;
  // assign vec3 directly (no vec4-padding needed — std430 alignment in
  // the struct adds the trailing 4-byte pad implicitly).
  rec.get('normal').assign(n);

  // ---- Static friction (Macklin 2020 §3.5 eqs. 26–28) ----
  // Δp = (p_1 − p̃_1) − (p_2 − p̃_2). For particle contacts p_i = x*_i
  // and p̃_i = x_i (substep-start committed position).
  const xiStar: Any = particles.predictedPositions.element(i).xyz.toVar();
  const xjStar: Any = particles.predictedPositions.element(j).xyz.toVar();
  const xi: Any = particles.positions.element(i).xyz.toVar();
  const xj: Any = particles.positions.element(j).xyz.toVar();
  const deltaI: Any = xiStar.sub(xi).toVar();
  const deltaJ: Any = xjStar.sub(xj).toVar();
  const deltaRel: Any = deltaI.sub(deltaJ).toVar();
  const tangential: Any = deltaRel.sub(n.mul(deltaRel.dot(n))).toVar(); // Δp_t
  const tanLen: Any = tangential.length().toVar();

  // Proactive cone gate. The paper's reactive test `λ_t < μ_s · λ_n`
  // assumes a Gauss-Seidel solve where λ_t updates are visible to
  // subsequent contacts in the same iteration. Under our parallel
  // scatter (U-16 Path B), `λ_t` is unchanged across a whole dispatch,
  // so the reactive test is trivially true at iter 1 (λ_t = 0) and
  // over-applies static friction — the cone's impulse budget is
  // effectively ignored. The proactive form below checks whether
  // applying the full tangential correction at this iter would keep
  // `λ_t` inside `μ_s · λ_n`; if not, static friction is skipped
  // entirely at this contact and the velocity-level kinetic friction
  // pass (§3.6) handles the resulting slip.
  //
  // Note that `λ_n` here includes this iter's contribution (same-
  // thread atomic ordering guarantees the load sees the add).
  const lambdaNLoaded: Any = atomicLoad(rec.get('lambdaN'));
  const lambdaTLoaded: Any = atomicLoad(rec.get('lambdaT'));
  const lambdaNNow: Any = lambdaNLoaded.toFloat().mul(float(invLambdaScale)).toVar();
  const lambdaTNow: Any = lambdaTLoaded.toFloat().mul(float(invLambdaScale)).toVar();
  const correctionLambdaT: Any = tanLen.div(wSum).toVar();
  // Phase 21 — μ_s is per-pair, populated by `contact/generate.ts` from
  // a `FrictionTable` LUT keyed on the high 16 bits of `phase`. Combine
  // rule (`min`) was applied at emit time; this read is a scalar.
  const muSPair: Any = rec.get('muS').toVar();
  const coneCapacity: Any = muSPair.mul(lambdaNNow).sub(lambdaTNow).toVar();
  const tanValid: Any = tanLen.greaterThan(float(1e-10));
  const withinStatic: Any = correctionLambdaT.lessThanEqual(coneCapacity);
  const applyStatic: Any = tanValid.and(withinStatic);

  // Bender 2014 §3.5 momentum-conserving split applied to Δp_t.
  const frictionDxI: Any = tangential.mul(wi.div(wSum)).negate().toVar();
  const frictionDxJ: Any = tangential.mul(wj.div(wSum)).toVar();

  // Fold the normal and static-friction corrections into a single
  // scatter per particle. `select` collapses the "friction applied" vs
  // "friction skipped" branches without a runtime If — cheaper under TSL
  // and matches the pattern the Phase 5 solve kernel already used.
  const totalI: Any = applyStatic.select(normalDxI.add(frictionDxI), normalDxI);
  const totalJ: Any = applyStatic.select(normalDxJ.add(frictionDxJ), normalDxJ);
  emitAccumulateDelta(accumulator, i, totalI);
  emitAccumulateDelta(accumulator, j, totalJ);

  // Accumulate Δλ_t when static friction fires. XPBD form:
  // Δλ_t = |Δp_t| / (w_i + w_j), matching the kg·m units of Δλ_n and
  // keeping the gate comparison well-scaled across iterations.
  const dLambdaT: Any = applyStatic.select(tanLen.div(wSum), float(0.0)).toVar();
  const dLambdaTTicks: Any = dLambdaT.mul(float(lambdaScale)).toInt();
  atomicAdd(rec.get('lambdaT'), dLambdaTTicks);
}
