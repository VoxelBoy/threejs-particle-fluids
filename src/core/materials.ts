import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';

import type { ContactGeometryExtension } from './contact/extension.js';

/**
 *
 *
 *
 * `SimLoop` iterates `options.materials` once at construction and appends
 * each material's kernel arrays to the corresponding pipeline stage.
 * Registration order is stable: materials registered earlier have their
 * `perIterKernels` dispatched earlier within each iter.
 *
 */
export interface Material {
  /**
   * Runs once per substep before the iter loop. Typical use: `resetLambda`
   * for constraints that accumulate Lagrange multipliers across iterations
   * (Macklin 2016 XPBD Algorithm 1 line 4).
   */
  readonly preIterKernels?: readonly ComputeNode[];

  /**
   * Runs once per solver iteration. All of one material's `perIterKernels`
   * dispatch before the next material's — registration order in
   * `SimLoopOptions.materials` determines inter-material ordering; within a
   * material, kernels dispatch in array order.
   *
   * Gather-mode kernels only. Materials that need scatter must route
   * through the shared contact/collider accumulator (Phase 08+ has no such
   * material; add an UNKNOWN if a future phase does).
   */
  readonly perIterKernels?: readonly ComputeNode[];

  /**
   * Runs once per substep after `advect`, before the cross-cutting
   * velocity-friction passes (contact §3.6, collider §3.6, SDF §3.6).
   * Typical use: velocity post-processing that replaces energy lost to
   * constraint projection — Macklin 2013 §5 vorticity confinement and
   * XSPH viscosity for fluids are the canonical case.
   */
  readonly postAdvectKernels?: readonly ComputeNode[];

  /**
   * Runs once per substep, at the head of the FINAL iter only, before
   * the contact-solve scatter. Used for kernels that need to fire just
   * once per substep with the iter loop's intermediate state already in
   * `predictedPositions`.
   *
   * Phase 15b / Macklin 2014 §5.2 — paper's mitigation for the "k too
   * high" failure mode of stiff stacks: *"perform mass modification
   * only in the final solver iteration."* RigidBodySystem's
   * `stiffStacks` kernel sits here. Iters 1..N-1 see physical
   * `contactInvMass` (seeded by SimLoop's per-substep copy of
   * `invMass`), letting lower particles respond to upper-particle
   * corrections; iter N sees the §5.2-scaled override that delivers the
   * stack-acceleration push without the per-iter asymmetric drift the
   * paper warned about.
   */
  readonly lastIterPreContactKernels?: readonly ComputeNode[];

  /**
   * Phase 15a — optional contact geometry contribution. Materials whose
   * contact pairs need a non-spherical `(n, d)` (currently only
   * softbody's `RigidBodySystem` for rigid-rigid pairs, paper §5.1
   * eqs. 17–20) return one extension; everything else leaves this
   * undefined and falls through to the spherical default in
   * `core/contact/extension.ts`.
   *
   * `SimLoop` collects extensions from registered materials in
   * registration order before building the contact-solve and
   * stabilize kernels — see `core/contact/extension.ts` §"Cross-
   * extension composition" for ordering and conflict semantics.
   *
   * Replaces the Phase 15 `bindContactPipeline` lifecycle hook: the
   * extension closure carries any state the geometry mode needs
   * (e.g. softbody's `restSDF` storage), so materials no longer need
   * direct access to `ContactBuffer` or the accumulator.
   */
  contactGeometryExtension?(): ContactGeometryExtension | undefined;
}
