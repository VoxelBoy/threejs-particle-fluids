import { Fn, instanceIndex } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';

import type { ParticleSystem } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Per-substep `contactInvMass[i] = invMass[i]` copy kernel (Phase 15a).
 *
 * Dispatched once per substep at the HEAD of the contact preIter block,
 * before any material's `preIterKernels` run. Material modules whose
 * physics requires a contact-time mass override (e.g. softbody's stiff-
 * stack mass scaling, paper §5.2 eq. 21) write into
 * `particles.contactInvMass` from their own preIter kernel — those
 * writes survive because materials' preIter is sequenced after this
 * copy.
 *
 * Decouples the contact-solve helper (`emitContactSolveCorrection`,
 * which reads `contactInvMass`) from physical mass changes that would
 * otherwise leak into predict / advect / the constraint scheduler.
 *
 * One thread per particle; trivially per-particle so determinism is
 * Tier 1 bit-exact (G4) — same guarantee as `integrate.ts`.
 */
export function buildCopyContactInvMassKernel(particles: ParticleSystem): ComputeNode {
  return Fn(() => {
    const i: Any = instanceIndex;
    particles.contactInvMass.element(i).assign(particles.invMass.element(i));
  })().compute(particles.capacity);
}
