import type { Vector3 } from 'three';
import { instancedArray } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';

import { buildIntegrationKernels, type IntegrationKernels } from './integrate.js';
import type { ParticleSystem, ParticleRange } from './particles.js';
import { ConstraintScheduler } from './constraints/scheduler.js';
import { createXpbdUniforms, type XpbdUniforms } from './constraints/xpbd.js';
import type { ConstraintType } from './constraints/types.js';
import type { HashGrid } from './hashGrid/HashGrid.js';
import type { Material } from './materials.js';
import type { ContactGeometryExtension } from './contact/extension.js';
import {
  ContactAccumulator,
  ContactBuffer,
  DEFAULT_MAX_VELOCITY,
  FrictionTable,
  VelocityAccumulator,
  buildApplyAccumulatorToBothKernel,
  buildApplyAccumulatorToPredictedKernel,
  buildApplyVelocityAccumulatorKernel,
  buildContactFrictionVelocityKernel,
  buildContactGenerateKernel,
  buildContactGenerateRangedKernel,
  buildContactSolveKernel,
  buildContactStabilizeKernel,
  buildCopyContactInvMassKernel,
  buildResetAccumulatorKernel,
  buildResetOverflowFlagKernel,
  buildResetVelocityAccumulatorKernel,
  buildResetVelocityOverflowFlagKernel,
  deriveAccumulatorScale,
  deriveVelocityAccumulatorScale,
} from './contact/index.js';
import {
  buildColliderFrictionVelocityKernel,
  buildColliderSolveKernel,
  buildSdfFrictionVelocityKernel,
  buildSdfSolveKernel,
  type PrimitiveSet,
  type SDFCollider,
} from './collision/index.js';

export interface ContactOptions {
  /**
   * Hash grid used for neighbor lookup during contact emission. Must be
   * constructed against the same {@link ParticleSystem} as this loop — and
   * with `cellSize ≥ 2·particleRadius · radiusExpansion` so the 27-cell
   * neighborhood covers every potential contact pair (paper §9 expanded
   * radius; cross-validated in Phase 03's
   * `tests/analytical/hashgrid-expanded-radius.gpu.test.ts`).
   */
  readonly hashGrid: HashGrid;
  /**
   * Hard cap on simultaneous contact pairs. Overflow is silently dropped —
   * the emission kernel still counts past the cap so the CPU-side overflow
   * check sees the overflow magnitude. Plan §Contact generation. A typical
   * default for granular scenes is `~8·capacity` (each particle can be in
   * at most a handful of tight pair contacts; each undirected pair counts
   * once with i<j).
   */
  readonly maxContacts: number;
  /**
   * Coulomb friction. Phase 21 — accepts either a scalar `{muS, muK}` for
   * scenes with a single uniform friction across all phase groups, or a
   * pre-built {@link FrictionTable} for scenes that need per-phase-group
   * friction (e.g. fluid+rigid where the fluid surface is "slick" and the
   * rigid is "sticky"). The kernel-side per-pair μ is always populated by
   * `contact/generate.ts` from the LUT — when the scalar form is supplied,
   * SimLoop constructs an internal `FrictionTable` seeded with that scalar
   * across every group, preserving Phase 5 / Phase 17 semantics.
   *
   * Default if omitted: `{muS: 0.5, muK: 0.4}` (dry friction).
   */
  readonly friction?: { readonly muS: number; readonly muK: number } | FrictionTable;
  /**
   * Number of pre-stabilization iterations per substep (paper §4.4). Default
   * `1` — paper: "1-2 iterations is usually sufficient". Higher values may
   * help when scenes open with interpenetrations (e.g. seeded sand piles).
   */
  readonly stabIters?: number;
  /** Candidate radius expansion for contact generate; see `generate.ts`. */
  readonly radiusExpansion?: number;
  /**
   * Override the position accumulator's fixed-point scale (ticks per metre).
   * If omitted, derived from {@link maxCorrectionMeters} via
   * {@link deriveAccumulatorScale}. Callers generally should not set this —
   * the auto-derivation is correct for any scene whose per-iteration
   * `|Δx_k|` stays below the correction bound.
   */
  readonly accumulatorScale?: number;
  /**
   * Upper bound on `|Δx_k|` per solver iteration, in metres. Sizes the
   * position accumulator's fixed-point quantum. Default `10 m` — three
   * orders of magnitude above U-16's 50k-particle measurement of
   * `|Δx| < 0.01 m`. Scenes with unusually compliant XPBD constraints or
   * bootstrap corrections can raise this; the accumulator's overflow flag
   * (readable via {@link ContactAccumulator.readbackOverflow}) trips on
   * saturation, so misconfiguration is detectable rather than silent.
   */
  readonly maxCorrectionMeters?: number;
  /**
   * Upper bound on per-particle velocity magnitude (m/s) used to derive the
   * velocity accumulator's fixed-point scale for the Macklin 2020 §3.6
   * dynamic friction pass. Defaults to {@link DEFAULT_MAX_VELOCITY} (50 m/s).
   */
  readonly maxVelocity?: number;
  /**
   * Phase Perf-14 — opt-in dispatch-shape reduction for `contact.generate`.
   * When provided, the contact pipeline dispatches `contact.generate` only
   * over the union of these ranges instead of the full
   * `particles.capacity`. Suitable for scenes where most particles are
   * fluid (or otherwise self-suppressed by phase mask) and only a small
   * subset can originate cross-pair contacts — typical of MVP boundary-
   * coupled scenes (fluid + body / cloth / rigid). Pair-set output is
   * set-identical to the legacy `particles.capacity` dispatch by
   * construction; pair order within `pairs[]` differs because atomic-
   * counter increments interleave differently across the smaller dispatch
   * (downstream solve scatter is order-independent — G4 tier-1).
   *
   * When omitted (default), `SimLoop` constructs the legacy
   * `buildContactGenerateKernel`. Backwards compatible.
   */
  readonly emittingRanges?: readonly ParticleRange[];
}

export interface ColliderOptions {
  /**
   * Phase 06 — analytic primitive colliders (plane, sphere, box, capsule).
   * The per-substep pipeline runs a per-particle collider solve kernel
   * (position-level §3.5 static friction) within the iter loop and a
   * collider velocity-friction kernel (§3.6) in the post-advect pass. The
   * collider λ buffers are reset once per substep.
   *
   * The provided {@link PrimitiveSet} must be constructed against the same
   * {@link ParticleSystem} as this loop; this is not enforced at construction
   * but mismatches will cause undefined behavior.
   *
   * Optional as of Phase 07 — a scene may use only {@link sdfColliders} with
   * no analytic primitives.
   */
  readonly colliders?: PrimitiveSet;
  /**
   * Phase 07 — baked static SDF colliders. Each collider is its own
   * kernel (a 3D-texture binding cannot be slot-indexed at runtime) so
   * SimLoop fans out reset/solve/friction kernels per entry. Constructed
   * by the caller — see `src/core`'s `SDFCollider` — before the
   * SimLoop because the accumulators are shared with the rest of the
   * pipeline and allocated inside SimLoop's constructor.
   */
  readonly sdfColliders?: readonly SDFCollider[];
  /**
   * Fixed-point scale (ticks per metre) for the position accumulator used
   * by the collider solve's scatter. If the loop also has `contact` enabled
   * the position accumulator is shared across both kernels — this value is
   * only consulted if the loop owns its own accumulator (collider-only mode).
   * If omitted in collider-only mode, derived from {@link maxCorrectionMeters}
   * via {@link deriveAccumulatorScale}.
   */
  readonly accumulatorScale?: number;
  /**
   * Upper bound on `|Δx_k|` per solver iteration, in metres. Sizes the
   * position accumulator's fixed-point quantum in collider-only mode.
   * Default `10 m`. Ignored if `contact` is enabled (the accumulator is
   * shared and sized by {@link ContactOptions.maxCorrectionMeters}).
   */
  readonly maxCorrectionMeters?: number;
  /**
   * Upper bound on velocity magnitude (m/s) used to size the velocity
   * accumulator in collider-only mode. Defaults to {@link DEFAULT_MAX_VELOCITY}
   * (50 m/s). Ignored if `contact` is enabled (the accumulator is shared).
   */
  readonly maxVelocity?: number;
}

export interface SimLoopOptions {
  readonly substeps?: number;
  /**
   * Number of constraint-solve iterations per substep. Default `2`. Plan
   * §"Outer loop" — Macklin 2019 favors more substeps over more iterations.
   */
  readonly iterations?: number;
  /**
   * XPBD uniforms (at minimum, the substep timestep) shared with every
   * constraint kernel. If omitted, {@link SimLoop} creates its own and
   * exposes them as {@link SimLoop.xpbd} so downstream modules can read the
   * same uniform handles.
   *
   * When constraints are passed in {@link SimLoopOptions.constraints}, they
   * MUST have been built against the same {@link XpbdUniforms} instance —
   * the `dt` uniform is shared by identity, not by value.
   */
  readonly xpbd?: XpbdUniforms;
  /**
   * Already-constructed constraint types to register. Type order is
   * preserved: earlier types are solved first within each iteration. Default
   * empty — Phase 02 gravity-only scenes work without any.
   */
  readonly constraints?: readonly ConstraintType[];
  /**
   * Phase 05 contact pipeline. If provided, the per-substep pipeline adds
   * hash-grid rebuild → contact emit → coloring → stabilization → (solve
   * inside each iteration). Omit entirely for Phase 02–04 scenes with no
   * particle-particle contacts.
   */
  readonly contact?: ContactOptions;
  /**
   * Phase 06 analytic-collider pipeline. If provided, the per-substep
   * pipeline adds a per-particle collider position solve (per iter) and a
   * post-advect velocity-friction pass. Can be combined with `contact` (in
   * which case both kernels share the position / velocity accumulators) or
   * used standalone (the loop allocates its own accumulators sized via
   * {@link ColliderOptions.accumulatorScale} / {@link ColliderOptions.maxVelocity}).
   */
  readonly colliders?: ColliderOptions;
  /**
   * Phase 07b — shared hash grid for neighbor queries. The canonical
   * location now that Phase 08+ material modules (fluid first) need the
   * grid even in scenes without a contact pipeline. If `contact.hashGrid`
   * is also set it MUST reference the same `HashGrid` instance — SimLoop
   * asserts this at construction. `ContactOptions.hashGrid` stays required
   * for Phase 05/06 back-compat.
   *
   * When present from any source, the rebuild pipeline is dispatched once
   * at the top of the substep's pre-iter stage.
   */
  readonly hashGrid?: HashGrid;
  /**
   * Phase 07b — physics modules registered as `Material` instances. SimLoop
   * iterates this list once at construction and appends each material's
   * `preIterKernels` / `perIterKernels` / `postAdvectKernels` to the
   * corresponding pipeline stages. Registration order is stable:
   * materials earlier in the array run earlier within each iter.
   *
   */
  readonly materials?: readonly Material[];
}

/**
 * Simulation loop — owns the outer XPBD substep loop, the integration
 * kernels (predict + advect), the optional contact pipeline, and the
 * constraint scheduler.
 *
 * Per-substep pipeline (paper Macklin 2014 Algorithm 1 + Phase 05 plan,
 * post-U-16 Path B resolution 2026-04-21, extended with Phase 05a Macklin
 * 2020 §3.6 velocity-friction pass 2026-04-21):
 *
 *   predict                                    // integrate.ts
 *   if contact:
 *     hashGrid.rebuild                         // 8 kernels from Phase 03
 *     contacts.resetKernel                     // zero the atomic counter
 *     contactGenerate                          // emit {i,j} pairs
 *     resetAccumulator + resetOverflowFlag
 *     contacts.resetLambdaKernel               // zero λ_n, λ_t (Phase 05a)
 *     for stabIter in 1..stabIters:            // paper §4.4 pre-stabilization
 *       contactStabilize (scatter, 1 dispatch)
 *       applyAccumulator(x AND x*)
 *     resetLambda[*]                           // per constraint type
 *     for iter in 1..I:
 *       for type in registered types:
 *         for group in type.groups:
 *           solveKernel                         // gather-mode per plan
 *       contactSolve (scatter, 1 dispatch — normal + §3.5 static friction)
 *       applyAccumulator(x*)
 *   else:
 *     resetLambda[*]
 *     for iter in 1..I: (types × groups)
 *   advect                                     // integrate.ts — v = (x* − x)/h
 *   if contact:
 *     // Phase 05a velocity-friction pass (Macklin 2020 §3.6) runs AFTER
 *     // advect per the paper's Algorithm 2; otherwise the advect step
 *     // would overwrite any Δv the friction pass applied.
 *     resetVelocityAccumulator + resetVelocityOverflow
 *     frictionVelocity (scatter, 1 dispatch)
 *     applyVelocityAccumulator
 *
 * The contact solve uses **fixed-point i32 scatter** (U-16 Path B,
 * 2026-04-21): one thread per contact, `atomicAdd` into per-particle
 * Δx accumulator, separate apply kernel commits the sum to
 * `predictedPositions`. Replaces the 32-color gather fan-out that
 * exceeded the Path A 6.67 ms budget on the reference platform.
 *
 *
 * Phase 04 scope preserved: when `contact` is omitted, the per-substep
 * pipeline collapses back to `predict → resetLambda → iters × groups →
 * advect`, identical to Phase 04.
 */

/**
 * Phase Perf-12 — contact-pipeline kernel inventory exposed for
 * bench-harness attribution. See `SimLoop.contactKernels`.
 */
export interface SimLoopContactKernels {
  readonly copyInvMass: ComputeNode;
  readonly resetCounter: ComputeNode;
  readonly generate: ComputeNode;
  readonly resetLambda: ComputeNode;
  readonly stabilize?: ComputeNode;
  readonly applyAccumulatorToBoth: ComputeNode;
  readonly solve: ComputeNode;
  readonly frictionVelocity: ComputeNode;
}

/**
 * Phase Perf-16 — collider-kernel inventory exposed for bench-harness
 * attribution. See `SimLoop.colliderKernels`. Same observability-only
 * shape as `SimLoopContactKernels`; `loop.step()` continues to dispatch
 * via the pre-flattened `pipeline` array. Production cost: zero (just
 * extra references to existing kernel objects).
 */
export interface SimLoopColliderKernelTriple {
  /** preIter, once per substep — zeroes per-collider λ state. */
  readonly resetLambda: ComputeNode;
  /** perIter, per particle — projects out interpenetration with this collider. */
  readonly solve: ComputeNode;
  /** postIter, once per substep — Coulomb-friction velocity correction. */
  readonly frictionVelocity: ComputeNode;
}
export interface SimLoopColliderKernels {
  /**
   * Primitive (analytic plane / box / sphere / capsule) collider
   * kernels. Present iff `options.colliders.colliders` (a `PrimitiveSet`)
   * was provided.
   */
  readonly primitive?: SimLoopColliderKernelTriple;
  /**
   * Per-SDF-collider kernel triple. One entry per
   * `options.colliders.sdfColliders[i]` (Phase 07 fans out one triple
   * per SDF because 3D-texture bindings are compile-time per
   * `SDFCollider`).
   */
  readonly sdf: readonly SimLoopColliderKernelTriple[];
}

export class SimLoop {
  readonly particles: ParticleSystem;
  readonly kernels: IntegrationKernels;
  readonly xpbd: XpbdUniforms;
  readonly scheduler: ConstraintScheduler;
  readonly substeps: number;
  readonly iterations: number;

  /** Phase 05 — present if and only if `contact` was passed to the ctor. */
  readonly contacts?: ContactBuffer;
  readonly hashGrid?: HashGrid;
  readonly accumulator?: ContactAccumulator;
  /**
   * Phase 21 — per-phase-group Coulomb friction LUT. Always present when
   * `contact` is enabled, even for scenes that supplied the legacy scalar
   * `{muS, muK}` form (the scalar gets seeded into every group). Demos
   * that previously read `loop.friction.muS.value = v` for live tuning
   * should call `loop.friction.setMuS(v)` / `setMuK(v)` instead.
   */
  readonly friction?: FrictionTable;
  readonly stabIters: number;
  /** Phase 05a — velocity-level friction accumulator (Macklin 2020 §3.6). */
  readonly velocityAccumulator?: VelocityAccumulator;

  /**
   * Phase Perf-12 — kernels constructed inside the contact block, exposed
   * for bench-harness attribution. Present if and only if `contact` was
   * passed to the ctor. `stabilize` is `undefined` when `stabIters: 0`.
   * The bench harness reads these to populate `PerfKernelSpec[]`; no
   * production code dispatches via this object — `loop.step()` still owns
   * the dispatch sequence.
   */
  readonly contactKernels?: SimLoopContactKernels;

  /**
   * Phase Perf-16 — kernels constructed inside the colliders block,
   * exposed for bench-harness attribution. Present if and only if
   * `colliders` was passed to the ctor. `primitive` is undefined when
   * only SDF colliders were supplied; `sdf` is empty when only
   * primitives were supplied. As with `contactKernels`, the production
   * `loop.step()` dispatch sequence is unchanged — these references
   * are observability-only.
   */
  readonly colliderKernels?: SimLoopColliderKernels;

  private readonly pipeline: ComputeNode[];

  constructor(particles: ParticleSystem, options: SimLoopOptions = {}) {
    const substeps = options.substeps ?? 4;
    const iterations = options.iterations ?? 2;
    if (!Number.isInteger(substeps) || substeps <= 0) {
      throw new Error(`SimLoop: substeps must be a positive integer, got ${substeps}`);
    }
    if (!Number.isInteger(iterations) || iterations < 0) {
      throw new Error(`SimLoop: iterations must be a non-negative integer, got ${iterations}`);
    }

    this.particles = particles;
    this.kernels = buildIntegrationKernels(particles);
    this.xpbd = options.xpbd ?? createXpbdUniforms(1 / 60);
    this.substeps = substeps;
    this.iterations = iterations;
    this.scheduler = new ConstraintScheduler();

    for (const c of options.constraints ?? []) this.scheduler.register(c);

    // ---- Pipeline slots shared by contact (Phase 05/05a) and colliders (Phase 06) ----
    const preIterKernels: ComputeNode[] = [];
    const perIterKernels: ComputeNode[] = [];
    const postIterKernels: ComputeNode[] = [];
    const stabIters = options.contact?.stabIters ?? 1;
    this.stabIters = stabIters;
    if (!Number.isInteger(stabIters) || stabIters < 0) {
      throw new Error(
        `SimLoop: contact.stabIters must be a non-negative integer, got ${stabIters}`,
      );
    }

    // Accumulators — shared across the contact and collider kernels when
    // both are active. Created lazily so a scene that uses neither allocates
    // nothing, and a scene that uses only one allocates only what it needs.
    let accumulator: ContactAccumulator | undefined;
    let velocityAccumulator: VelocityAccumulator | undefined;
    let resetAccumulatorKernel: ComputeNode | undefined;
    let resetOverflowKernel: ComputeNode | undefined;
    let applyToPredictedKernel: ComputeNode | undefined;
    let resetVelocityAccumulatorKernel: ComputeNode | undefined;
    let resetVelocityOverflowKernel: ComputeNode | undefined;
    let applyVelocityAccumulatorKernel: ComputeNode | undefined;

    const ensureAccumulator = (
      scaleTicks: number,
      maxVelocity: number,
    ): { accum: ContactAccumulator; velAccum: VelocityAccumulator } => {
      if (!accumulator) {
        accumulator = new ContactAccumulator(particles, scaleTicks);
        resetAccumulatorKernel = buildResetAccumulatorKernel(accumulator);
        resetOverflowKernel = buildResetOverflowFlagKernel(accumulator);
        applyToPredictedKernel = buildApplyAccumulatorToPredictedKernel(accumulator);
      }
      if (!velocityAccumulator) {
        velocityAccumulator = new VelocityAccumulator(
          particles,
          deriveVelocityAccumulatorScale(maxVelocity),
        );
        resetVelocityAccumulatorKernel = buildResetVelocityAccumulatorKernel(velocityAccumulator);
        resetVelocityOverflowKernel = buildResetVelocityOverflowFlagKernel(velocityAccumulator);
        applyVelocityAccumulatorKernel = buildApplyVelocityAccumulatorKernel(velocityAccumulator);
      }
      return { accum: accumulator, velAccum: velocityAccumulator };
    };

    // ---- Resolve shared HashGrid (Phase 07b) ----
    // SimLoopOptions.hashGrid is the canonical source; ContactOptions.hashGrid
    // stays required for Phase 05/06 back-compat. If both are set, they MUST
    // reference the same HashGrid instance. The rebuild pipeline is pushed
    // once at the top of preIterKernels so fluid-only scenes (no contact)
    // have a hook too.
    const hashGridFromSimLoop = options.hashGrid;
    const hashGridFromContact = options.contact?.hashGrid;
    let resolvedHashGrid: HashGrid | undefined;
    if (hashGridFromSimLoop && hashGridFromContact) {
      if (hashGridFromSimLoop !== hashGridFromContact) {
        throw new Error(
          'SimLoop: options.hashGrid and options.contact.hashGrid must reference the same HashGrid instance when both are provided',
        );
      }
      resolvedHashGrid = hashGridFromSimLoop;
    } else {
      resolvedHashGrid = hashGridFromSimLoop ?? hashGridFromContact;
    }
    if (resolvedHashGrid && resolvedHashGrid.particles !== particles) {
      throw new Error('SimLoop: hashGrid must be constructed against the same ParticleSystem');
    }
    if (resolvedHashGrid) {
      preIterKernels.push(...resolvedHashGrid.rebuildPipeline);
    }

    // ---- Material contact-geometry extensions (Phase 15a) ----
    // Collected BEFORE the contact block so the solve and stabilize
    // kernels can be parameterised by the registered extensions.
    // Materials whose contact pairs need a non-spherical (n, d)
    // (currently only softbody's RigidBodySystem for rigid-rigid pairs,
    // paper §5.1 eqs. 17–20) return one extension here; everything else
    // leaves it undefined and falls through to the spherical default.
    // Registration order = `options.materials` order.
    const geometryExtensions: ContactGeometryExtension[] = [];
    if (options.materials) {
      for (const material of options.materials) {
        const ext = material.contactGeometryExtension?.();
        if (ext) geometryExtensions.push(ext);
      }
    }

    if (options.contact) {
      const contactOpts = options.contact;
      const hashGrid = contactOpts.hashGrid;
      // HashGrid/ParticleSystem consistency asserted in the shared resolver
      // above (Phase 07b); no duplicate assert here.
      const contacts = new ContactBuffer(particles.renderer, {
        maxContacts: contactOpts.maxContacts,
      });
      // Phase 21 — resolve the friction option to a `FrictionTable`. The
      // `FrictionTable` form is used as-is; the legacy scalar `{muS, muK}`
      // form constructs an internal `FrictionTable` seeded with the scalar
      // at every group (preserving Phase 5 / Phase 17 semantics for scenes
      // that don't differentiate by group). If omitted entirely, defaults
      // to dry-friction `{muS: 0.5, muK: 0.4}` matching the pre-21 default.
      let friction: FrictionTable;
      const fOpt = contactOpts.friction;
      if (fOpt instanceof FrictionTable) {
        friction = fOpt;
      } else {
        friction = new FrictionTable({
          defaultMuS: fOpt?.muS ?? 0.5,
          defaultMuK: fOpt?.muK ?? 0.4,
        });
      }
      const scaleTicks =
        contactOpts.accumulatorScale ??
        deriveAccumulatorScale(contactOpts.maxCorrectionMeters ?? 10);
      const maxVelocity = contactOpts.maxVelocity ?? DEFAULT_MAX_VELOCITY;
      const { accum, velAccum } = ensureAccumulator(scaleTicks, maxVelocity);

      // Phase Perf-14 — when `emittingRanges` is supplied, build the
      // dispatch-shape-reduced kernel; otherwise build the legacy kernel.
      let generateKernel: ComputeNode;
      if (contactOpts.emittingRanges && contactOpts.emittingRanges.length > 0) {
        // Build the emit-set indirection table + flag buffer.
        let emittingCount = 0;
        for (const range of contactOpts.emittingRanges) {
          if (
            !Number.isInteger(range.start) ||
            !Number.isInteger(range.count) ||
            range.start < 0 ||
            range.count <= 0 ||
            range.start + range.count > particles.capacity
          ) {
            throw new Error(
              `SimLoop: invalid contact.emittingRanges entry start=${range.start} count=${range.count} capacity=${particles.capacity}`,
            );
          }
          emittingCount += range.count;
        }
        const emittingIndices = instancedArray(emittingCount, 'uint');
        const isEmittingFlag = instancedArray(particles.capacity, 'uint');
        // Populate the indirection table + flag buffer once at construction.
        // Both buffers are static — no per-frame writes.
        const idxArr = emittingIndices.value.array as Uint32Array;
        const flagArr = isEmittingFlag.value.array as Uint32Array;
        let cursor = 0;
        for (const range of contactOpts.emittingRanges) {
          for (let k = 0; k < range.count; k++) {
            const globalIdx = range.start + k;
            idxArr[cursor++] = globalIdx;
            flagArr[globalIdx] = 1;
          }
        }
        emittingIndices.value.needsUpdate = true;
        isEmittingFlag.value.needsUpdate = true;
        generateKernel = buildContactGenerateRangedKernel({
          particles,
          hashGrid,
          contacts,
          frictionTable: friction,
          emittingIndices,
          emittingCount,
          isEmittingFlag,
          ...(contactOpts.radiusExpansion !== undefined
            ? { radiusExpansion: contactOpts.radiusExpansion }
            : {}),
        });
      } else {
        generateKernel = buildContactGenerateKernel(
          contactOpts.radiusExpansion !== undefined
            ? {
                particles,
                hashGrid,
                contacts,
                frictionTable: friction,
                radiusExpansion: contactOpts.radiusExpansion,
              }
            : { particles, hashGrid, contacts, frictionTable: friction },
        );
      }
      const stabilizeKernel = buildContactStabilizeKernel({
        particles,
        contacts,
        accumulator: accum,
        geometryExtensions,
      });
      const solveKernel = buildContactSolveKernel({
        particles,
        contacts,
        accumulator: accum,
        geometryExtensions,
      });
      const applyToBoth = buildApplyAccumulatorToBothKernel(accum);
      const frictionVelocityKernel = buildContactFrictionVelocityKernel({
        particles,
        contacts,
        velocityAccumulator: velAccum,
        dt: this.xpbd.dt,
      });

      this.contacts = contacts;
      this.hashGrid = hashGrid;
      this.friction = friction;

      // Pre-iter: copy invMass → contactInvMass once per substep so the
      // solve-side mass override channel (Phase 15a, paper §5.2 stiff-stack
      // scaling) starts each substep at the physical mass. Materials whose
      // preIter writes `contactInvMass` (currently only softbody's stiff
      // stacks) sequence after this copy, so their override survives into
      // the iter loop.
      const copyInvMassKernel = buildCopyContactInvMassKernel(particles);
      preIterKernels.push(copyInvMassKernel);
      // Pre-iter: contact generate → reset accumulators. HashGrid rebuild is
      // pushed once by the shared resolver above (Phase 07b). Keep the
      // "reset accumulator(s)" block ordered before the first scatter.
      preIterKernels.push(contacts.resetKernel, generateKernel, contacts.resetLambdaKernel);
      // Stabilization (paper §4.4) — scatters into the accumulator then
      // applies to both x and x*. The applyToBoth kernel zeroes the
      // accumulator on the way out, so no explicit reset is needed before
      // the main iter loop's reset below.
      for (let s = 0; s < stabIters; s++) {
        preIterKernels.push(stabilizeKernel, applyToBoth);
      }
      // Per-iter: particle-particle scatter solve. The shared apply kernel
      // is appended below (after colliders are considered) so both scatter
      // kernels accumulate before the single apply commits their sum.
      perIterKernels.push(solveKernel);
      // Post-iter: velocity-friction pass.
      postIterKernels.push(frictionVelocityKernel);

      // Phase Perf-12 — expose for bench-harness attribution. Read-only
      // pointers to the kernels we just constructed; the dispatch sequence
      // remains owned by the pipeline above + step() below.
      this.contactKernels = {
        copyInvMass: copyInvMassKernel,
        resetCounter: contacts.resetKernel,
        generate: generateKernel,
        resetLambda: contacts.resetLambdaKernel,
        ...(stabIters > 0 ? { stabilize: stabilizeKernel } : {}),
        applyAccumulatorToBoth: applyToBoth,
        solve: solveKernel,
        frictionVelocity: frictionVelocityKernel,
      };
    }

    if (options.colliders) {
      const colliderOpts = options.colliders;
      const primitives = colliderOpts.colliders;
      const sdfColliders = colliderOpts.sdfColliders ?? [];
      if (!primitives && sdfColliders.length === 0) {
        throw new Error(
          'SimLoop: colliders option must provide at least one of `colliders` (PrimitiveSet) or `sdfColliders` (SDFCollider[])',
        );
      }
      const scaleTicks =
        colliderOpts.accumulatorScale ??
        deriveAccumulatorScale(colliderOpts.maxCorrectionMeters ?? 10);
      const maxVelocity = colliderOpts.maxVelocity ?? DEFAULT_MAX_VELOCITY;
      const { accum, velAccum } = ensureAccumulator(scaleTicks, maxVelocity);

      // Phase Perf-16 — collect kernel references for `colliderKernels`
      // exposure as we build them. Observability-only: the kernels are
      // pushed into preIter/perIter/postIter as before; this block just
      // additionally retains references.
      let primitiveTriple: SimLoopColliderKernelTriple | undefined;
      const sdfTriples: SimLoopColliderKernelTriple[] = [];

      if (primitives) {
        const colliderSolve = buildColliderSolveKernel({
          particles,
          colliders: primitives,
          accumulator: accum,
          dt: this.xpbd.dt,
        });
        const colliderVelocityFriction = buildColliderFrictionVelocityKernel({
          particles,
          colliders: primitives,
          velocityAccumulator: velAccum,
          dt: this.xpbd.dt,
        });
        // λ reset runs once per substep, before the iter loop.
        preIterKernels.push(primitives.resetLambdaKernel);
        perIterKernels.push(colliderSolve);
        postIterKernels.push(colliderVelocityFriction);
        primitiveTriple = {
          resetLambda: primitives.resetLambdaKernel,
          solve: colliderSolve,
          frictionVelocity: colliderVelocityFriction,
        };
      }

      // Phase 07 — fan out one reset/solve/friction per SDF collider.
      // Each SDF is its own kernel because 3D-texture bindings are
      // compile-time (see `SDFCollider` class doc).
      for (const sdf of sdfColliders) {
        if (sdf.particles !== particles) {
          throw new Error(
            'SimLoop: every sdfColliders entry must be constructed against the same ParticleSystem as the loop',
          );
        }
        const solve = buildSdfSolveKernel({
          sdf,
          accumulator: accum,
          dt: this.xpbd.dt,
        });
        const friction = buildSdfFrictionVelocityKernel({
          sdf,
          velocityAccumulator: velAccum,
          dt: this.xpbd.dt,
        });
        preIterKernels.push(sdf.resetLambdaKernel);
        perIterKernels.push(solve);
        postIterKernels.push(friction);
        sdfTriples.push({
          resetLambda: sdf.resetLambdaKernel,
          solve,
          frictionVelocity: friction,
        });
      }

      // Expose for bench-harness attribution (Phase Perf-16). Only
      // assigned when at least one of (primitive, sdfTriples) is non-empty
      // — guarded by the outer block's `primitives || sdfColliders.length`
      // check above.
      this.colliderKernels = {
        ...(primitiveTriple ? { primitive: primitiveTriple } : {}),
        sdf: sdfTriples,
      };
    }

    // ---- Materials iteration (Phase 07b) ----
    // Each material contributes up to three kernel arrays, appended to
    // the corresponding pipeline stages in registration order. Geometry
    // extensions were already collected above and folded into the solve
    // and stabilize kernels (Phase 15a — replaces the Phase 15 lifecycle
    // hook `bindContactPipeline`).
    const materialsPerIterKernels: ComputeNode[] = [];
    const materialsPostAdvectKernels: ComputeNode[] = [];
    const materialsLastIterPreContactKernels: ComputeNode[] = [];
    if (options.materials) {
      for (const material of options.materials) {
        if (material.preIterKernels) {
          preIterKernels.push(...material.preIterKernels);
        }
        if (material.perIterKernels) {
          materialsPerIterKernels.push(...material.perIterKernels);
        }
        if (material.postAdvectKernels) {
          materialsPostAdvectKernels.push(...material.postAdvectKernels);
        }
        if (material.lastIterPreContactKernels) {
          materialsLastIterPreContactKernels.push(...material.lastIterPreContactKernels);
        }
      }
      // Sanity guard: a material that contributes a geometry extension
      // requires the contact pipeline to be enabled — its (n, d) writes
      // are consumed only by the solve/stabilize kernels SimLoop builds
      // when `options.contact` is set. Without contact, the extension is
      // dead code that the user almost certainly didn't intend.
      if (geometryExtensions.length > 0 && !this.contacts) {
        throw new Error(
          'SimLoop: a registered material returns a contactGeometryExtension but `options.contact` was not provided — extensions only fire when the contact pipeline is enabled',
        );
      }
    }

    // Expose accumulators if either pipeline allocated them. Guarded
    // assignment per `exactOptionalPropertyTypes` — the declared field
    // type `ContactAccumulator | undefined` permits undefined reads but
    // not undefined writes.
    if (accumulator) this.accumulator = accumulator;
    if (velocityAccumulator) this.velocityAccumulator = velocityAccumulator;

    // Accumulator reset must run BEFORE any scatter source in the substep.
    // Prepending rather than appending (regardless of which pipeline
    // populated preIterKernels) maintains that invariant — both the
    // Phase 05 stabilization and the Phase 06 collider solve scatter into
    // the accumulator, and neither should see stale values from the
    // previous substep's overflow-flag state.
    if (accumulator && resetAccumulatorKernel && resetOverflowKernel) {
      preIterKernels.unshift(resetAccumulatorKernel, resetOverflowKernel);
    }

    // Per-iter apply: one apply per iter, sums the particle-particle scatter
    // + the collider scatter + zeroes the accumulator for the next iter.
    const sharedApplyToPredicted = applyToPredictedKernel;
    // Post-iter (velocity-friction) reset + apply.
    const sharedVelApply = applyVelocityAccumulatorKernel;
    const sharedVelReset = resetVelocityAccumulatorKernel;
    const sharedVelOverflowReset = resetVelocityOverflowKernel;

    // ---- Per-substep pipeline assembly ----
    const substepChain: ComputeNode[] = [];
    substepChain.push(this.kernels.predict);
    // Phase 12 — Mueller 2011 §4.1 Eq. 11. Per-particle quaternion prediction
    // runs immediately after position predict so any iter-loop kernel that
    // reads `predictedRotation` (Phase 12 §5.1 Pass 2 Aᵢ term) sees the
    // substep-start prediction. Always-on per-particle no-op when ω = 0.
    substepChain.push(this.kernels.predictRotation);
    substepChain.push(...preIterKernels);

    const registeredTypes = this.scheduler.registeredTypes;
    for (const t of registeredTypes) substepChain.push(t.resetLambdaKernel);
    for (let i = 0; i < iterations; i++) {
      for (const t of registeredTypes) {
        for (const g of t.groups) substepChain.push(g.solveKernel);
      }
      // Material per-iter gather kernels (Phase 07b). Run after scheduler
      // types, before scatter — so materials see the scheduler's projections
      // and contact/colliders see the materials' projections.
      substepChain.push(...materialsPerIterKernels);
      // Phase 15b — Macklin 2014 §5.2 ¶ "perform mass modification only
      // in the final solver iteration." Materials whose mass overrides
      // would destabilize at intermediate iters (currently rigid stiff
      // stacks) populate `lastIterPreContactKernels`; SimLoop dispatches
      // them only in iter N-1, after `materialsPerIterKernels` (so
      // shape-match Pass 3 has just run) and BEFORE the contact-solve
      // scatter.
      if (i === iterations - 1) {
        substepChain.push(...materialsLastIterPreContactKernels);
      }
      // Scatter: particle-particle + collider kernels, any combination.
      substepChain.push(...perIterKernels);
      // Apply: single commit per iter, feeds the accumulated Δx into
      // predictedPositions and zeroes the accumulator for the next iter.
      if (sharedApplyToPredicted) substepChain.push(sharedApplyToPredicted);
    }

    substepChain.push(this.kernels.advect);
    // Phase 12 — Mueller 2011 §4.1 Eqs. 12–15. Finite-difference ω from
    // (q, qp) and commit q ← qp. Runs alongside `advect` (which does the
    // analogous v-from-x, x ← x* finite-difference for translation) so the
    // post-substep state on (x, v, q, ω) is internally consistent for the
    // next substep's predict + predictRotation.
    substepChain.push(this.kernels.advectRotation);
    // Material post-advect kernels (Phase 07b) — vorticity, XSPH, etc.
    // Run BEFORE velocity-friction so friction damps what remains after
    // the material's energy-replacement pass.
    substepChain.push(...materialsPostAdvectKernels);
    if (postIterKernels.length > 0) {
      // Velocity-friction reset → scatter kernels → apply. The reset and
      // apply wrappers are allocated by `ensureAccumulator`; the scatter
      // kernels come from `postIterKernels`.
      if (sharedVelReset) substepChain.push(sharedVelReset);
      if (sharedVelOverflowReset) substepChain.push(sharedVelOverflowReset);
      substepChain.push(...postIterKernels);
      if (sharedVelApply) substepChain.push(sharedVelApply);
    }

    // Repeat the substep chain S times. The one-`computeAsync` pattern for
    // the whole frame is preserved.
    const frameChain: ComputeNode[] = [];
    for (let s = 0; s < substeps; s++) frameChain.push(...substepChain);
    this.pipeline = frameChain;
  }

  /** Mutable gravity vector — see {@link IntegrationKernels.gravity}. */
  get gravity(): Vector3 {
    return this.kernels.gravity.value;
  }

  /**
   * Read-only view of the per-frame compute pipeline — the ordered list of
   * `ComputeNode`s that {@link SimLoop.step} dispatches via
   * `renderer.computeAsync()`.
   *
   * Phase Perf — added 2026-04-25 so the perf bench harness can enumerate
   * kernels in execution order for completeness checks (every dispatched
   * kernel registered with the `PerfRunner`). Production callers do not
   * iterate this list; the field name avoids collision with
   * {@link SimLoop.kernels} (which is the integration kernels — predict +
   * advect — not the full pipeline).
   *
   * The returned array is aliased with the internal pipeline buffer; the
   * `readonly` element type prevents mutation through the type system but
   * does not prevent a caller from casting back to a mutable view. Callers
   * MUST treat this as read-only.
   */
  get computeNodes(): readonly ComputeNode[] {
    return this.pipeline;
  }

  async step(dt: number): Promise<void> {
    const subDt = dt / this.substeps;
    this.kernels.dt.value = subDt;
    this.xpbd.dt.value = subDt;
    await this.particles.renderer.computeAsync(this.pipeline);
  }
}
