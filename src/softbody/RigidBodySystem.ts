import { instancedArray, uniform } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type {
  ContactGeometryExtension,
  Material,
  ParticleRange,
  ParticleSystem,
  XpbdUniforms,
} from '../core/index.js';

import { FLAG_RIGID } from './flags.js';
import {
  buildCenterOfMassKernel,
  buildMomentAndPolarDecompKernel,
  buildResetLambdaKernel,
  buildShapeMatchDeltaApplyKernel,
} from './shapeMatch.js';
import { buildStiffStacksKernel } from './stiffStacks.js';
import { buildRigidGeometryExtension } from './rigidGeometryExtension.js';

/**
 * One rigid body's configuration as supplied to {@link RigidBodySystem}.
 * Phase 15: shape-matching cluster + per-particle SDF data for paper
 * §5.1 inter-body contact.
 *
 * The scene must have already uploaded each body's particles to
 * `particleRange` via {@link ParticleSystem.uploadParticles} with
 * `phase = (phaseId & 0xFFFF) << 16` so the high-16 self-collision
 * mask reads back to `phaseId`.
 */
export interface RigidBodyDef {
  /** Particle slots this body owns. */
  readonly particleRange: ParticleRange;
  /**
   * Flat xyz per particle, length `3 · particleRange.count`. World-space
   * rest configuration. RigidBodySystem pre-centers to the rest COM at
   * construction (`r_i = x_i^0 − c̄`).
   */
  readonly restPositions: Float32Array;
  /**
   * Per-particle `(φ, ∇φ_x, ∇φ_y, ∇φ_z)`, length `4 · particleRange.count`.
   * Output of `voxelize(mesh, { ..., bakeSdf: true })`. Sign convention is
   * paper-faithful (see voxelize.ts §"Gradient direction"): `φ ≥ 0`
   * inside, `∇φ` points INWARD (the direction eqs. 17–20 expect to
   * produce correct separation).
   */
  readonly restSDF: Float32Array;
  /**
   * Self-collision phase tag (ARCH §Cross-cutting mechanism row
   * `phase: u32`, Macklin 2014 §3). Encoded into `phase` as
   * `(phaseId & 0xFFFF) << 16` — the rigid-rigid contact filter uses
   * the standard contact-emit phase mask so same-body pairs are
   * suppressed at pair-emission time.
   */
  readonly phaseId: number;
  /**
   * Shape-matching XPBD compliance `α` in s²/kg. Default `0` (perfectly
   * rigid). Soft-rigid is achieved by passing a small positive `α`.
   */
  readonly compliance?: number;
}

export interface RigidBodySystemOptions {
  readonly particles: ParticleSystem;
  /** Shared XPBD uniforms — typically `SimLoop.xpbd` (identity-shared). */
  readonly xpbd: XpbdUniforms;
  readonly bodies: readonly RigidBodyDef[];
  /**
   * Macklin 2014 §5.2 stiff-stack mass scaling. When omitted, every rigid
   * particle's contact-time inverse mass equals its physical
   * `particles.invMass` (no acceleration).
   */
  readonly stackStabilization?: {
    /** Mass-scaling exponent. Paper §5.2 examples use `[1, 5]`; default 3. */
    readonly k?: number;
    /** Ground-plane height. Default 0. */
    readonly groundY?: number;
  };
}

/**
 * Rigid-body material — Macklin 2014 §5 shape matching + §5.1 sparse
 * SDF contact + §5.2 stiff stacks. Implements the core {@link Material}
 * interface; `SimLoop` dispatches the kernels in the unified pipeline.
 *
 *
 * Cadence (per substep / per iter):
 *
 *   preIter (once per substep):
 *     1. shape-match λ reset.
 *     2. shape-match Pass 1 (centre of mass).
 *     3. shape-match Pass 2 (moment matrix + polar decomposition).
 *
 *   perIter (each iter):
 *     1. shape-match Pass 3 (Δx apply, in-place to `predictedPositions`).
 *     2. (No rigid-specific contact kernel; the geometry extension
 *        contributes `(n, d)` to core's solve kernel for rigid-rigid
 *        pairs.)
 *
 *   lastIterPreContact (final iter only, when `stackStabilization`
 *   enabled):
 *     1. stiff-stack mass scaling — writes `particles.contactInvMass`
 *        per paper §5.2 eq. 21. Fires once per substep at the head of
 *        the last iter, before the contact-solve scatter, so iters
 *        1..N-1 see physical `contactInvMass` (seeded by SimLoop's
 *        per-substep copy) and only iter N sees the §5.2-scaled
 *        override. Macklin 2014 §5.2 ¶ "perform mass modification only
 *        in the final solver iteration" — paper-specified mitigation
 *        for the "k too high" failure mode.
 *
 * The shared apply-accumulator and friction-velocity kernels are
 * dispatched by `SimLoop` outside this material's slots.
 */
export class RigidBodySystem implements Material {
  readonly particles: ParticleSystem;
  readonly bodies: readonly RigidBodyDef[];

  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly bodyStart: StorageBufferNode<'uint'>;
  readonly bodyCount: StorageBufferNode<'uint'>;
  readonly bodyCenters: StorageBufferNode<'vec4'>;
  readonly bodyRotations: StorageBufferNode<'vec4'>;
  readonly bodyCompliance: StorageBufferNode<'float'>;
  readonly lambda: StorageBufferNode<'vec4'>;

  /** Per-particle SDF `(φ, ∇φ)` — Macklin 2014 §5.1, Figure 7. */
  readonly rigidSDF: StorageBufferNode<'vec4'>;

  readonly stackKUniform: UniformNode<'float', number>;
  readonly groundYUniform: UniformNode<'float', number>;
  readonly stackStabilizationEnabled: boolean;

  /** Shape-matching kernels — built at construction. */
  readonly preIterKernels: readonly ComputeNode[];
  /** Shape-match Pass 3 (Δx apply). */
  readonly perIterKernels: readonly ComputeNode[];
  readonly postAdvectKernels: readonly ComputeNode[] = [];
  /**
   * Phase 15b — Macklin 2014 §5.2 final-iter-only mass modification.
   * Empty unless `stackStabilization` is enabled.
   */
  readonly lastIterPreContactKernels: readonly ComputeNode[];

  private readonly xpbd: XpbdUniforms;

  constructor(options: RigidBodySystemOptions) {
    const { particles, xpbd, bodies } = options;
    if (bodies.length === 0) {
      throw new Error('RigidBodySystem: at least one body is required');
    }

    this.particles = particles;
    this.bodies = bodies;
    this.xpbd = xpbd;

    // ---- GPU buffers ----
    this.restOffsets = instancedArray(particles.capacity, 'vec4');
    this.bodyStart = instancedArray(bodies.length, 'uint');
    this.bodyCount = instancedArray(bodies.length, 'uint');
    this.bodyCenters = instancedArray(bodies.length, 'vec4');
    this.bodyRotations = instancedArray(bodies.length * 3, 'vec4');
    this.bodyCompliance = instancedArray(bodies.length, 'float');
    this.lambda = instancedArray(particles.capacity, 'vec4');
    this.rigidSDF = instancedArray(particles.capacity, 'vec4');

    // Identity-init bodyRotations so frame-0 reads return a valid R.
    const bodyRotArr = this.bodyRotations.value.array as Float32Array;
    for (let b = 0; b < bodies.length; b++) {
      const base = b * 3 * 4;
      bodyRotArr[base + 0] = 1;
      bodyRotArr[base + 5] = 1;
      bodyRotArr[base + 10] = 1;
    }
    this.bodyRotations.value.needsUpdate = true;

    // ---- Validate + pre-center + upload ----
    const restArr = this.restOffsets.value.array as Float32Array;
    const sdfArr = this.rigidSDF.value.array as Float32Array;
    const startArr = this.bodyStart.value.array as Uint32Array;
    const countArr = this.bodyCount.value.array as Uint32Array;
    const complianceArr = this.bodyCompliance.value.array as Float32Array;
    const flagsArr = particles.flags.value.array as Uint32Array;

    const usedSlots = new Set<number>();
    const seenPhaseIds = new Set<number>();
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b]!;
      const { particleRange, restPositions, restSDF, phaseId } = body;

      if (
        !Number.isInteger(particleRange.start) ||
        !Number.isInteger(particleRange.count) ||
        particleRange.start < 0 ||
        particleRange.count <= 0 ||
        particleRange.start + particleRange.count > particles.capacity
      ) {
        throw new Error(
          `RigidBodySystem: body ${b} particleRange invalid — start=${particleRange.start} count=${particleRange.count} capacity=${particles.capacity}`,
        );
      }
      if (restPositions.length !== 3 * particleRange.count) {
        throw new Error(
          `RigidBodySystem: body ${b} restPositions length ${restPositions.length} does not match 3 · count = ${3 * particleRange.count}`,
        );
      }
      if (restSDF.length !== 4 * particleRange.count) {
        throw new Error(
          `RigidBodySystem: body ${b} restSDF length ${restSDF.length} does not match 4 · count = ${4 * particleRange.count} (use voxelize({ ..., bakeSdf: true }))`,
        );
      }
      if (!Number.isInteger(phaseId) || phaseId <= 0 || phaseId > 0xffff) {
        throw new Error(
          `RigidBodySystem: body ${b} phaseId must be an integer in (0, 0xFFFF]; got ${phaseId}`,
        );
      }
      if (seenPhaseIds.has(phaseId)) {
        throw new Error(
          `RigidBodySystem: body ${b} reuses phaseId ${phaseId} — each rigid body must have a unique phase tag`,
        );
      }
      seenPhaseIds.add(phaseId);

      for (let i = 0; i < particleRange.count; i++) {
        const slot = particleRange.start + i;
        if (usedSlots.has(slot)) {
          throw new Error(
            `RigidBodySystem: body ${b} particle slot ${slot} already used by an earlier body`,
          );
        }
        usedSlots.add(slot);
      }

      // Pre-center rest positions (Mueller 2011 §3 — pre-centred rest
      // invariant `Σ m_i r_i = 0` collapses A_pq to a one-pass reduction).
      let cx = 0,
        cy = 0,
        cz = 0;
      const n = particleRange.count;
      for (let i = 0; i < n; i++) {
        cx += restPositions[3 * i + 0]!;
        cy += restPositions[3 * i + 1]!;
        cz += restPositions[3 * i + 2]!;
      }
      const invN = 1 / n;
      cx *= invN;
      cy *= invN;
      cz *= invN;

      for (let i = 0; i < n; i++) {
        const slot = particleRange.start + i;
        const baseRest = slot * 4;
        restArr[baseRest + 0] = restPositions[3 * i + 0]! - cx;
        restArr[baseRest + 1] = restPositions[3 * i + 1]! - cy;
        restArr[baseRest + 2] = restPositions[3 * i + 2]! - cz;
        restArr[baseRest + 3] = 0;

        const baseSdf = slot * 4;
        const srcSdf = i * 4;
        sdfArr[baseSdf + 0] = restSDF[srcSdf + 0]!;
        sdfArr[baseSdf + 1] = restSDF[srcSdf + 1]!;
        sdfArr[baseSdf + 2] = restSDF[srcSdf + 2]!;
        sdfArr[baseSdf + 3] = restSDF[srcSdf + 3]!;

        // Mark every rigid particle. Core's contact-solve early-exits
        // when both pair participants set this bit; the rigid SDF kernel
        // (bound below) processes those pairs instead.
        flagsArr[slot] = flagsArr[slot]! | FLAG_RIGID;
      }

      startArr[b] = particleRange.start;
      countArr[b] = particleRange.count;

      const compliance = body.compliance ?? 0;
      if (!Number.isFinite(compliance) || compliance < 0) {
        throw new Error(
          `RigidBodySystem: body ${b} compliance must be a non-negative finite number, got ${compliance}`,
        );
      }
      complianceArr[b] = compliance;
    }

    this.restOffsets.value.needsUpdate = true;
    this.rigidSDF.value.needsUpdate = true;
    this.bodyStart.value.needsUpdate = true;
    this.bodyCount.value.needsUpdate = true;
    this.bodyCompliance.value.needsUpdate = true;
    particles.flags.value.needsUpdate = true;

    // ---- Stack stabilization ----
    const stack = options.stackStabilization;
    this.stackStabilizationEnabled = stack !== undefined;
    const stackK = stack?.k ?? 3;
    const stackGroundY = stack?.groundY ?? 0;
    if (!Number.isFinite(stackK) || stackK < 0) {
      throw new Error(
        `RigidBodySystem: stackStabilization.k must be a non-negative finite number, got ${stackK}`,
      );
    }
    if (!Number.isFinite(stackGroundY)) {
      throw new Error(
        `RigidBodySystem: stackStabilization.groundY must be finite, got ${stackGroundY}`,
      );
    }
    this.stackKUniform = uniform(stackK, 'float');
    this.groundYUniform = uniform(stackGroundY, 'float');

    // ---- Shape-matching kernels ----
    const resetLambdaKernel = buildResetLambdaKernel({
      particles,
      lambda: this.lambda,
    });
    const stiffStacksKernel = this.stackStabilizationEnabled
      ? buildStiffStacksKernel({
          particles,
          kUniform: this.stackKUniform,
          groundYUniform: this.groundYUniform,
        })
      : undefined;
    const centerOfMassKernel = buildCenterOfMassKernel({
      particles,
      bodyStart: this.bodyStart,
      bodyCount: this.bodyCount,
      bodyCenters: this.bodyCenters,
      numBodies: bodies.length,
    });
    const momentPolarKernel = buildMomentAndPolarDecompKernel({
      particles,
      restOffsets: this.restOffsets,
      bodyStart: this.bodyStart,
      bodyCount: this.bodyCount,
      bodyRotations: this.bodyRotations,
      numBodies: bodies.length,
    });
    const shapeMatchApplyKernel = buildShapeMatchDeltaApplyKernel({
      particles,
      restOffsets: this.restOffsets,
      bodyStart: this.bodyStart,
      bodyCount: this.bodyCount,
      bodyCenters: this.bodyCenters,
      bodyRotations: this.bodyRotations,
      bodyCompliance: this.bodyCompliance,
      lambda: this.lambda,
      xpbd,
      numBodies: bodies.length,
    });

    const preIter: ComputeNode[] = [resetLambdaKernel, centerOfMassKernel, momentPolarKernel];
    this.preIterKernels = preIter;

    this.perIterKernels = [shapeMatchApplyKernel];

    // Phase 15b — Macklin 2014 §5.2 ¶ "perform mass modification only
    // in the final solver iteration" (page 6, right column). Mass
    // scaling fires once per substep at the head of the LAST iter,
    // AFTER shape-match Pass 3 has just run (so `predictedPositions`
    // reflect the iter-N-2 contact + shape-match projections) and
    // BEFORE the iter-N-1 contact-solve scatter. Iters 1..N-2 use
    // physical `contactInvMass` (seeded by SimLoop's per-substep copy);
    // iter N-1 sees the §5.2-scaled override that delivers the stack-
    // acceleration push without the per-iter asymmetric drift the paper
    // warned about for "k too high" regimes.
    this.lastIterPreContactKernels = stiffStacksKernel ? [stiffStacksKernel] : [];
  }

  /**
   * {@link Material.contactGeometryExtension} — softbody contributes the
   * Macklin 2014 §5.1 SDF mode for rigid-rigid pairs (eqs. 17–20). Mixed
   * pairs (rigid + fluid / softbody / cloth) and pairs without any
   * rigid participant fall through to the spherical default.
   */
  contactGeometryExtension(): ContactGeometryExtension {
    return buildRigidGeometryExtension({ restSDF: this.rigidSDF });
  }

  particleRange(bodyIndex: number): ParticleRange {
    const body = this.bodies[bodyIndex];
    if (!body) {
      throw new Error(
        `RigidBodySystem.particleRange: bodyIndex ${bodyIndex} out of bounds (0..${this.bodies.length - 1})`,
      );
    }
    return body.particleRange;
  }
}
