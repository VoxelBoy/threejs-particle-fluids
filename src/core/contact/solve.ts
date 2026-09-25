import { Fn, If, Return, atomicLoad, bool, float, instanceIndex, uint, vec3 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';

import type { ParticleSystem } from '../particles.js';
import type { ContactBuffer } from './ContactBuffer.js';
import type { ContactAccumulator } from './accumulator.js';
import { emitContactSolveCorrection } from './correction.js';
import { emitGeometrySelection, type ContactGeometryExtension } from './extension.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Build the scatter-mode contact solve kernel.
 *
 * The kernel performs two Macklin-paper operations per contact per iter:
 *
 *   1. Normal projection (Macklin 2014 §6.1 eq. 22). Unchanged from Phase 5.
 *      Computes Δλ_n = d / (w_i + w_j) for the non-penetration constraint
 *      and scatters the positional correction. Additionally, Phase 5a
 *      accumulates Δλ_n into the per-contact λ_n buffer on ContactBuffer
 *      so that the Macklin 2020 §3.5 static gate below and the §3.6
 *      velocity-friction pass can read the final accumulated multiplier.
 *
 *   2. Static friction (Macklin 2020 §3.5 eqs. 26–28, replacing the
 *      Macklin 2014 §6.1 eqs. 23–25 that Phase 5 shipped). Computes the
 *      tangential component of the contact-point displacement since
 *      substep start and, gated by the accumulated-λ static cone
 *      `λ_t < μ_s · λ_n`, scatters a position correction that removes the
 *      tangential slip. If the gate fails, no static correction is applied
 *      at this iter; the velocity-level dynamic friction pass (§3.6, in
 *      `frictionVelocity.ts`) handles the slip case after the position
 *      solve completes.
 *
 *

 *
 * Dispatch shape: one thread per contact, over maxContacts threads. Each
 * thread processes its `(i, j)` pair and scatters via atomicAdd into the
 * per-particle fixed-point accumulator (see `accumulator.ts`) and into
 * the per-contact λ_n / λ_t buffers (see `ContactBuffer.ts`).
 *
 * Determinism: i32 atomicAdd is associative, commutative, and exact. The
 * scatter is G4 tier-1 bit-exact across repeat runs — same guarantee as
 * hash-grid `cellStart` / `cellEnd` per the U-18 archive.
 *
 * Invariance note: the tangential Δp_t is invariant under the normal
 * correction because the normal correction is parallel to n, so both are
 * computed in a single pass using pre-correction x*, no intermediate
 * dispatch needed.
 */
export function buildContactSolveKernel(args: {
  readonly particles: ParticleSystem;
  readonly contacts: ContactBuffer;
  readonly accumulator: ContactAccumulator;
  /**
   * Phase 15a — geometry-mode extensions (e.g. softbody's rigid-rigid
   * SDF mode, paper §5.1 eqs. 17–20) tried per pair before the spherical
   * default. Empty list keeps Phase 5/15 spherical behavior unchanged.
   *
   * See `core/contact/extension.ts` for the protocol.
   */
  readonly geometryExtensions?: readonly ContactGeometryExtension[];
}): ComputeNode {
  const { particles, contacts, accumulator } = args;
  const geometryExtensions = args.geometryExtensions ?? [];
  const twoR = 2 * particles.particleRadius;
  const maxContacts = contacts.maxContacts;

  return Fn(() => {
    const c: Any = instanceIndex;
    // Early-exit threads past the emitted-contact count.
    const nRaw: Any = atomicLoad(contacts.counter.element(uint(0)));
    If(c.greaterThanEqual(nRaw).or(c.greaterThanEqual(uint(maxContacts))), () => {
      Return();
    });

    const rec: Any = contacts.records.element(c);
    const i: Any = rec.get('i').toVar();
    const j: Any = rec.get('j').toVar();

    // Geometry selection: per-pair `(n, d)` chosen by the first claiming
    // extension; spherical default fires if none claim. The spherical
    // default leaves `outHandled = false` when its guards reject the pair
    // (degenerate, non-penetrating, kinematic) so the post-composer gate
    // skips. Extensions follow the same convention.
    const outN: Any = vec3(0, 0, 0).toVar();
    const outD: Any = float(0).toVar();
    const outHandled: Any = bool(false).toVar();

    emitGeometrySelection(
      {
        i,
        j,
        c,
        particles,
        outN,
        outD,
        outHandled,
      },
      geometryExtensions,
      () => {
        // Spherical default — Macklin 2014 §6.1 eq. 22 unilateral.
        const xiStar: Any = particles.predictedPositions.element(i).xyz.toVar();
        const xjStar: Any = particles.predictedPositions.element(j).xyz.toVar();
        const diff: Any = xiStar.sub(xjStar).toVar();
        const len: Any = diff.length().toVar();
        const C: Any = len.sub(float(twoR)).toVar();
        const wSum: Any = particles.contactInvMass
          .element(i)
          .add(particles.contactInvMass.element(j))
          .toVar();
        const valid: Any = len
          .greaterThan(float(1e-8))
          .and(C.lessThan(float(0.0)))
          .and(wSum.greaterThan(float(0.0)));
        If(valid, () => {
          outN.assign(diff.div(len));
          outD.assign(C.negate());
          outHandled.assign(bool(true));
        });
      },
    );

    If(outHandled.not(), () => {
      Return();
    });

    emitContactSolveCorrection({
      i,
      j,
      c,
      n: outN,
      d: outD,
      particles,
      contacts,
      accumulator,
    });
  })().compute(maxContacts);
}
