import { Fn, exp, float, instanceIndex, max, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { ParticleSystem } from '../core/index.js';

import { FLAG_RIGID } from './flags.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Macklin 2014 §5.2 stiff-stack mass scaling — eq. 21.
 *
 *   s_i(x*_i) = exp(-k · h(x*_i))         (paper eq. 21)
 *   m*_i      = s_i · m_i
 *   ⇒ invMass*_i = invMass_i / s_i = invMass_i · exp(+k · h(x*_i))
 *
 * Writes `particles.contactInvMass[i]` (Phase 15a — replaces the Phase 15
 * softbody-private `scaledInvMass` buffer). SimLoop's per-substep
 * `buildCopyContactInvMassKernel` seeds every slot to `invMass[i]` at
 * the head of the contact preIter block; this kernel runs in the
 * material's preIter and overrides the rigid slots with the §5.2 scaled
 * value. Non-rigid slots are left at the copy value (= `invMass`), so
 * the contact-solve helper's `contactInvMass` read is correct for every
 * pair flavour.
 *
 *
 * Height function h is the simple ground-plane heuristic from paper §5.2:
 * `h(x) = max(0, x.y - groundY)`. Non-planar stacking surfaces are filed
 * post-MVP under U-29 — the `groundY` uniform is the only knob the artist
 * gets in this phase.
 *
 * Scope (paper §5.2 ¶ "scaled particle mass"): the scaled mass is used
 * ONLY during contact processing. Shape-matching reads the original
 * `particles.invMass`, not `contactInvMass` — `contactInvMass` is the
 * named contact-time channel.
 *
 * Kinematic guard: when `invMass[i] == 0` (pinned / kinematic) the eq. 21
 * scaling preserves zero — `0 · anything = 0`. Computed without a branch.
 *
 * Non-rigid guard: the kernel dispatches over `particles.capacity` and
 * gates the write on `flags[i] & FLAG_RIGID`. Non-rigid slots are left
 * at the copy-kernel-seeded `invMass[i]`.
 *
 * Cadence: once per substep, in `RigidBodySystem.lastIterPreContactKernels`
 * (when stack stabilization is enabled). SimLoop dispatches the slot only
 * on the FINAL iter, after `materialsPerIterKernels` and before the contact-
 * solve scatter, per Macklin 2014 §5.2 ¶ "perform mass modification only in
 * the final solver iteration" (page 6, right column). Iters 1..N-1 see the
 * physical `contactInvMass` seeded by SimLoop's per-substep copy of
 * `invMass`; only iter N sees this kernel's §5.2-scaled override. The
 * paper presents the final-iter-only placement as the mitigation for the
 * "k too high" failure mode where lower particles stop responding to
 * upper-particle interactions; in the height regime our scenes typically
 * run (8-cube × 0.4 m stacks at k=3, mass ratio ~4500×) the mitigation is
 * mandatory, not optional.
 */
export interface BuildStiffStacksKernelArgs {
  readonly particles: ParticleSystem;
  /**
   * Mass-scaling exponent `k`. Paper §5.2 examples use `k ∈ [1, 5]`. MVP
   * default `k = 3` is set by `RigidBodySystem`; the uniform here is the
   * runtime-mutable knob.
   */
  readonly kUniform: UniformNode<'float', number>;
  /**
   * Ground-plane height in world space. `h(x) = max(0, x.y - groundY)`.
   * Only valid for scenes with a horizontal ground plane (paper §5.2
   * acknowledges this limitation; non-planar surfaces tracked under U-29).
   */
  readonly groundYUniform: UniformNode<'float', number>;
}

export function buildStiffStacksKernel(args: BuildStiffStacksKernelArgs): ComputeNode {
  const { particles, kUniform, groundYUniform } = args;

  return Fn(() => {
    const i: Any = instanceIndex;
    const flagsI: Any = particles.flags.element(i).toVar();
    const isRigid: Any = flagsI.bitAnd(uint(FLAG_RIGID)).notEqual(uint(0));

    const wOriginal: Any = particles.invMass.element(i).toVar();
    const xiPredicted: Any = particles.predictedPositions.element(i).xyz.toVar();
    const yPredicted: Any = xiPredicted.y;
    const heightAboveGround: Any = max(yPredicted.sub(groundYUniform), float(0.0)).toVar();
    // invMass*_i = invMass_i · exp(+k · h)  — see header doc for sign-flip
    // derivation (paper m*_i = s · m_i with s = exp(-k·h)).
    const k: Any = kUniform;
    const scale: Any = exp(k.mul(heightAboveGround)).toVar();
    const wScaled: Any = wOriginal.mul(scale);

    // Non-rigid slots are left at the per-substep-copied `invMass`
    // value (SimLoop's `buildCopyContactInvMassKernel`). Only rigid
    // slots get the §5.2 override.
    const out: Any = isRigid.select(wScaled, particles.contactInvMass.element(i));
    particles.contactInvMass.element(i).assign(out);
  })().compute(particles.capacity);
}
