import { instancedArray, uniform } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import {
  ContactAccumulator,
  buildApplyAccumulatorToPredictedKernel,
  buildResetAccumulatorKernel,
  buildResetOverflowFlagKernel,
  deriveAccumulatorScale,
} from '../core/index.js';
import type { Material, ParticleRange, ParticleSystem, XpbdUniforms } from '../core/index.js';

import {
  buildCenterOfMassKernel,
  buildMomentAndPolarDecompKernel,
  buildResetLambdaKernel,
  buildShapeMatchDeltaApplyKernel,
} from './shapeMatch.js';
import {
  buildImplicitMomentPolarKernel,
  buildImplicitNeighborhoodCenterKernel,
  buildImplicitQpWriteKernel,
  buildImplicitResetPairLambdaKernel,
  buildImplicitShapeMatchScatterKernel,
} from './shapeMatchImplicit.js';

/**
 * One soft body's configuration as supplied to {@link SoftbodySystem}.
 *
 * The scene is expected to have already uploaded this body's particles
 * via {@link ParticleSystem.uploadParticles} with `phase = phaseId` for
 * every particle in `particleRange`. SoftbodySystem does not re-write
 * `particles.phase` — the core attribute is owned by the caller, per the
 * ARCH §"Cross-cutting mechanisms" convention for `phase: u32`.
 */
export interface SoftbodyDef {
  /** Which particle slots this body owns. */
  readonly particleRange: ParticleRange;
  /**
   * Flat xyz per particle, length `3 · particleRange.count`. World-space
   * rest configuration. SoftbodySystem pre-centers to the rest COM on
   * construction (`r_i = x_i^0 − c̄`) before uploading to the GPU.
   */
  readonly restPositions: Float32Array;
  /**
   * 1 = surface particle, 0 = interior. Length = `particleRange.count`.
   * Surface particles must be arranged first within the range (indices
   * `[0, surfaceCount)` where `surfaceCount = Σ surfaceFlag`). Voxelize
   * enforces this ordering so {@link SoftbodySystem.surfaceRange}
   * returns a contiguous sub-range.
   *
   * Not read by Phase 10 kernels (shape matching is volumetric); consumed
   * by Phase 17 via `FluidSystem.registerBoundaryParticles(surfaceRange)`
   * for Akinci 2012 §2.2 single-layer boundary coupling.
   */
  readonly surfaceFlag: Uint8Array;
  /**
   * Scene-level self-collision phase tag (ARCH §Cross-cutting mechanism
   * row 2, Macklin 2014 §3). All particles in this body MUST share this
   * value in `particles.phase[i]` — SoftbodySystem does not enforce this
   * at construction (would require a GPU readback); the caller's
   * `uploadParticles` call is the trust boundary.
   */
  readonly phaseId: number;
  /**
   * Shape-matching compliance `α` in s²/kg (not α̃ — the kernel derives
   * `α̃ = α / dt²` per dispatch). Plan §"XPBD adaptation":
   *   - `1e-9` → near-rigid
   *   - `1e-6` → typical soft
   * Consumed by Pass 3 (Δx apply, lands in a later commit).
   */
  readonly matchCompliance: number;
  /**
   * Phase 12 — required when {@link SoftbodySystemOptions.shapeMatchMode} is
   * `'implicit'`; ignored otherwise. Packed undirected edges between adjacent
   * particles in the body's local index frame: `[i0_local, j0_local, ...]`
   * with `i < j` per pair, sourced from {@link voxelize}'s 6-face occupancy
   * walk. Local indices are converted to global particle slots via
   * `particleRange.start` at construction.
   *
   * Source: Mueller, Chentanez 2011b §5.1 — "A group contains the
   * corresponding particle and all the particles connected to it via a
   * single edge." For volumetric voxel-grid bodies the natural edge set is
   * 6-face occupancy adjacency.
   */
  readonly edges?: Uint32Array;
}

export interface SoftbodySystemOptions {
  readonly particles: ParticleSystem;
  /**
   * Shared XPBD uniforms — typically the ones owned by the {@link
   * SimLoop} this soft body participates in. Must be the *same
   * instance* the loop constructs with; the `dt` uniform is shared by
   * identity, not value. `SimLoop.xpbd` is the canonical source when
   * passing through a `SimLoop`.
   */
  readonly xpbd: XpbdUniforms;
  readonly bodies: readonly SoftbodyDef[];
  /**
   * Phase 12 — Mueller 2011 §5.3 (one rotation per body) vs §5.1 (one
   * rotation per particle, computed over the particle's edge-connected
   * neighborhood). Mode is uniform across all bodies in this
   * `SoftbodySystem`; mixed-mode scenes use two `SoftbodySystem` instances
   * over disjoint particle ranges.
   *
   *
   * `'implicit'` mode requires every body to supply
   * {@link SoftbodyDef.edges}; construction throws if any are missing.
   */
  readonly shapeMatchMode?: 'explicit' | 'implicit';
  /**
   * Phase 12 — only consulted when `shapeMatchMode === 'implicit'`. Upper
   * bound on `|Δx_k|` per solver iteration, in metres; sizes the per-system
   * fixed-point Δx accumulator's tick. Default `10 m` (matches Phase 05a's
   * `ContactAccumulator` default; three orders of magnitude above realistic
   * shape-matching corrections).
   */
  readonly maxCorrectionMeters?: number;
}

/**
 *
 *
 * Owns shape-matching GPU state: pre-centered per-particle rest offsets
 * `r_i`, per-body range mappings, and the per-substep cached `(c, R)`.
 * Exposes the core {@link Material} interface so {@link SimLoop}
 * dispatches its kernels in the unified pipeline without a softbody-
 * specific code path in core.
 *
 * Package-boundary invariant (ARCH §"Package boundary rules"):
 * `src/softbody` imports only from `src/core` and `three/tsl`.
 * Cross-module coupling (e.g. fluid sees softbody as SPH boundary in
 * Phase 17) flows through core attributes.
 */
export class SoftbodySystem implements Material {
  readonly particles: ParticleSystem;
  readonly bodies: readonly SoftbodyDef[];
  /**
   * Phase 12 — `'explicit'` (Mueller 2011 §5.3, default) or `'implicit'`
   * (Mueller 2011 §5.1). Uniform across all bodies in this instance.
   */
  readonly shapeMatchMode: 'explicit' | 'implicit';

  /**
   * Pre-centered per-particle rest offsets `r_i = x_i^0 − c̄_body`. Sized
   * to `particles.capacity` (matches every other per-particle core
   * buffer's indexing); slots outside any softbody range are zero.
   * Paper: Mueller 2005 §3.2 `q_i = x_i^0 − t_0`.
   */
  readonly restOffsets: StorageBufferNode<'vec4'>;

  /** Per-body starting index into `particles.predictedPositions`. */
  readonly bodyStart: StorageBufferNode<'uint'>;
  /** Per-body particle count. */
  readonly bodyCount: StorageBufferNode<'uint'>;
  /**
   * Per-body current-configuration centre of mass. Written by
   * {@link preIterKernels}'s Pass 1 (center-of-mass) once per substep;
   * read by Pass 3 (Δx apply) on every iter.
   */
  readonly bodyCenters: StorageBufferNode<'vec4'>;

  /**
   * Per-body shape-matching rotation `R` from the polar decomposition
   * of the cross-covariance `A_pq = Σ x*_i · r_i^T` (Mueller 2011 eq.
   * 7 / Mueller 2005 §3.5). Row-major, three contiguous `vec4` slots
   * per body — `bodyRotations[3·b + row].xyz` is row `row` of `R_b`,
   * `w` unused. Total length = `3 · bodies.length`.
   *
   * Written by {@link preIterKernels}' Pass 2 once per substep; read by
   * {@link perIterKernels}' Pass 3 (Δx apply) on every iter via
   * `goal_i = R · r_i + c`.
   */
  readonly bodyRotations: StorageBufferNode<'vec4'>;

  /**
   * Per-body shape-matching XPBD compliance `α` in s²/kg, uploaded
   * from each {@link SoftbodyDef.matchCompliance} at construction.
   * Kernel derives `α̃ = α / dt²` per dispatch. Immutable post-upload
   * in this commit; a mutable setter lands if an artist tool needs it.
   */
  readonly bodyCompliance: StorageBufferNode<'float'>;

  /**
   * Per-particle shape-matching Lagrange multiplier `λ_i ∈ ℝ³`
   * (vec3 in xyz, w padding). Reset to zero at the start of each
   * substep by {@link preIterKernels}; accumulates Δλ across iters
   * within a substep. Sized to `particles.capacity`; non-softbody slots
   * remain at zero (their reset is harmless).
   *
   * §5.3 mode only. §5.1 uses {@link pairLambda} (per-CSR-entry) instead
   * because particles in §5.1 belong to multiple groups simultaneously
   * and a single per-particle slot would race.
   */
  readonly lambda: StorageBufferNode<'vec4'>;

  // ---- §5.1 implicit-mode-only buffers (undefined in 'explicit' mode) ----
  /**
   * §5.1 per-particle current-frame neighborhood centroid `c_i`. Pass 1
   * output, Pass 2/3 input. Sized `capacity`. Phase 12.
   */
  readonly particleCenters?: StorageBufferNode<'vec4'>;
  /**
   * §5.1 per-particle rotation `R_i`, row-major (3 vec4 per particle).
   * Pass 2 output, Pass 3/4 input. Sized `3 · capacity`. Phase 12.
   */
  readonly particleRotations?: StorageBufferNode<'vec4'>;
  /**
   * §5.1 per-particle rest-frame neighborhood centroid `c̄_i`, computed
   * once at construction from the static edge graph. Sized `capacity`.
   * Phase 12.
   */
  readonly restNeighborhoodCenters?: StorageBufferNode<'vec4'>;
  /**
   * §5.1 CSR offsets — `neighborOffsets[i+1] − neighborOffsets[i]` is
   * `|N(i)|`. Sized `capacity + 1`. Phase 12.
   */
  readonly neighborOffsets?: StorageBufferNode<'uint'>;
  /**
   * §5.1 CSR neighbor indices (global particle slots, including each
   * particle's own self-entry per Mueller 2011 §5.1). Sized
   * `totalDegree`. Phase 12.
   */
  readonly neighborIndices?: StorageBufferNode<'uint'>;
  // particleCompliance dropped in Phase 12 MVP per the WebGPU 10-storage-
  // buffer-per-compute-stage cap; implicit-mode `α` is a single uniform
  // shared across the whole `SoftbodySystem` instance. See
  // `shapeMatchImplicit.ts` `BuildImplicitShapeMatchScatterKernelArgs`
  // §`compliance` for the deviation rationale.
  /**
   * §5.1 per-CSR-entry Lagrange multiplier `λ_{i,j}` (vec3 in xyz, w
   * padding). Sized `totalDegree`; reset to zero at substep start.
   * Phase 12.
   */
  readonly pairLambda?: StorageBufferNode<'vec4'>;
  /**
   * §5.1 dedicated fixed-point Δx accumulator. Phase 12 follows the
   * Phase 11 FluidSystem precedent (memory `phase_11_exit_2026_04_27`):
   * material-owned accumulators sidestep SimLoop construction-order
   * coupling on shared accumulator allocation. Phase 12.
   */
  readonly implicitAccumulator?: ContactAccumulator;

  /**
   * {@link Material} — once per substep: `λ ← 0`, Pass 1 (centre of
   * mass), Pass 2 (moment matrix + polar decomposition). §5.1 mode adds
   * accumulator/overflow/pair-λ resets at the head.
   */
  preIterKernels!: readonly ComputeNode[];
  /**
   * {@link Material} — once per iter. §5.3: Pass 3 (Δx apply, direct
   * write to predictedPositions). §5.1: Pass 3 scatter → accumulator
   * apply → Pass 4 qp write.
   */
  perIterKernels!: readonly ComputeNode[];
  /**
   * {@link Material} — empty in MVP. Plastic creep would have lived here
   * but was deferred post-MVP at the Phase 10 paper-verification
   * checkpoint; see plan §"Plastic deformation" and UNKNOWNS U-37.
   */
  readonly postAdvectKernels: readonly ComputeNode[] = [];

  constructor(options: SoftbodySystemOptions) {
    const { particles, xpbd, bodies } = options;
    const mode = options.shapeMatchMode ?? 'explicit';
    if (bodies.length === 0) {
      throw new Error('SoftbodySystem: at least one body is required');
    }
    if (mode !== 'explicit' && mode !== 'implicit') {
      throw new Error(
        `SoftbodySystem: shapeMatchMode must be 'explicit' or 'implicit', got ${String(mode)}`,
      );
    }

    this.particles = particles;
    this.bodies = bodies;
    this.shapeMatchMode = mode;

    // ---- Allocate GPU buffers ----
    this.restOffsets = instancedArray(particles.capacity, 'vec4');
    this.bodyStart = instancedArray(bodies.length, 'uint');
    this.bodyCount = instancedArray(bodies.length, 'uint');
    this.bodyCenters = instancedArray(bodies.length, 'vec4');
    this.bodyRotations = instancedArray(bodies.length * 3, 'vec4');
    this.bodyCompliance = instancedArray(bodies.length, 'float');
    this.lambda = instancedArray(particles.capacity, 'vec4');

    const restArr = this.restOffsets.value.array as Float32Array;
    const startArr = this.bodyStart.value.array as Uint32Array;
    const countArr = this.bodyCount.value.array as Uint32Array;
    const complianceArr = this.bodyCompliance.value.array as Float32Array;

    // Identity-init `bodyRotations`. Without this, frame 0 reads a zero
    // matrix from the buffer (instancedArray zero-fills), and any consumer
    // that needs a valid R before the polar-decomp kernel runs sees a
    // degenerate rotation. Phase 10 dodged this because the §5.3 kernel
    // writes R every substep and nothing else reads it; Phase 13's mesh
    // skin reads R at render time and hits frame 0 before the first
    // step — the symptom is a "squished" rest pose that fixes itself
    // after one substep. Each body owns three contiguous vec4 rows.
    const bodyRotArr = this.bodyRotations.value.array as Float32Array;
    for (let b = 0; b < bodies.length; b++) {
      const base = b * 3 * 4;
      bodyRotArr[base + 0] = 1; // row 0: (1, 0, 0, _)
      bodyRotArr[base + 5] = 1; // row 1: (0, 1, 0, _)
      bodyRotArr[base + 10] = 1; // row 2: (0, 0, 1, _)
    }
    this.bodyRotations.value.needsUpdate = true;

    // ---- Validate + pre-center + rank-3 check, per body ----
    const usedSlots = new Set<number>();
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b]!;
      const { particleRange, restPositions, surfaceFlag } = body;

      if (
        !Number.isInteger(particleRange.start) ||
        !Number.isInteger(particleRange.count) ||
        particleRange.start < 0 ||
        particleRange.count <= 0 ||
        particleRange.start + particleRange.count > particles.capacity
      ) {
        throw new Error(
          `SoftbodySystem: body ${b} particleRange invalid — start=${particleRange.start} count=${particleRange.count} capacity=${particles.capacity}`,
        );
      }
      if (restPositions.length !== 3 * particleRange.count) {
        throw new Error(
          `SoftbodySystem: body ${b} restPositions length ${restPositions.length} does not match 3 · count = ${3 * particleRange.count}`,
        );
      }
      if (surfaceFlag.length !== particleRange.count) {
        throw new Error(
          `SoftbodySystem: body ${b} surfaceFlag length ${surfaceFlag.length} does not match count = ${particleRange.count}`,
        );
      }

      // Slot-overlap check — two bodies cannot share a particle slot.
      for (let i = 0; i < particleRange.count; i++) {
        const slot = particleRange.start + i;
        if (usedSlots.has(slot)) {
          throw new Error(
            `SoftbodySystem: body ${b} particle slot ${slot} already used by an earlier body`,
          );
        }
        usedSlots.add(slot);
      }

      // ---- Pre-center rest positions (Mueller 2011 §3 — pre-centered
      //      rest invariant `Σ m_i r_i = 0` collapses eq. 7 to a one-pass
      //      reduction without per-particle recentering of x*_i). ----
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

      // ---- Rank-3 precondition (Mueller 2011 §3 page 2 singular-A
      //      caveat; plan §"Particle representation" non-coplanar
      //      precondition — U-36 tracks the post-MVP oriented-particle
      //      resolution for non-volumetric inputs). ----
      //
      // Compute C = Σ r_i r_i^T (3x3 symmetric PSD).
      let c00 = 0,
        c01 = 0,
        c02 = 0,
        c11 = 0,
        c12 = 0,
        c22 = 0;
      for (let i = 0; i < n; i++) {
        const rx = restPositions[3 * i + 0]! - cx;
        const ry = restPositions[3 * i + 1]! - cy;
        const rz = restPositions[3 * i + 2]! - cz;
        c00 += rx * rx;
        c01 += rx * ry;
        c02 += rx * rz;
        c11 += ry * ry;
        c12 += ry * rz;
        c22 += rz * rz;
      }
      const [, , lambdaMin] = eigenvaluesSymmetric3x3(c00, c01, c02, c11, c12, c22);
      // Singular values of the stacked-r matrix are sqrt of eigenvalues
      // of C. Compare σ_min against the threshold directly.
      const sigmaMin = Math.sqrt(Math.max(lambdaMin, 0));
      const threshold = particles.particleRadius * 1e-3;
      if (sigmaMin < threshold) {
        throw new Error(
          `SoftbodySystem: body ${b} particle cloud is rank-deficient — smallest singular value ${sigmaMin.toExponential(3)} < ${threshold.toExponential(3)} (particleRadius · 1e-3). Non-coplanar input required; see plan §"Particle representation".`,
        );
      }

      // ---- Upload pre-centered rest offsets into the global GPU
      //      buffer at each particle's slot. ----
      for (let i = 0; i < n; i++) {
        const slot = particleRange.start + i;
        const base = slot * 4;
        restArr[base + 0] = restPositions[3 * i + 0]! - cx;
        restArr[base + 1] = restPositions[3 * i + 1]! - cy;
        restArr[base + 2] = restPositions[3 * i + 2]! - cz;
        restArr[base + 3] = 0;
      }

      startArr[b] = particleRange.start;
      countArr[b] = particleRange.count;

      if (!Number.isFinite(body.matchCompliance) || body.matchCompliance < 0) {
        throw new Error(
          `SoftbodySystem: body ${b} matchCompliance must be a non-negative finite number, got ${body.matchCompliance}`,
        );
      }
      complianceArr[b] = body.matchCompliance;
    }

    this.restOffsets.value.needsUpdate = true;
    this.bodyStart.value.needsUpdate = true;
    this.bodyCount.value.needsUpdate = true;
    this.bodyCompliance.value.needsUpdate = true;

    if (mode === 'explicit') {
      // ---- §5.3 (Mueller 2011) — Phase 10 kernel set, unchanged ----
      const resetLambdaKernel = buildResetLambdaKernel({
        particles,
        lambda: this.lambda,
      });
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
      this.preIterKernels = [resetLambdaKernel, centerOfMassKernel, momentPolarKernel];
      this.perIterKernels = [shapeMatchApplyKernel];
    } else {
      // ---- §5.1 (Mueller 2011) — Phase 12 implicit kernel set ----
      this.installImplicitKernels(options);
    }
  }

  /**
   * §5.1 implicit-mode construction path. Builds the global CSR neighbor
   * graph from each body's `edges`, computes per-particle rest-frame
   * neighborhood centroids, allocates the implicit-only GPU buffers, and
   * registers the four-pass kernel pipeline. Called once from the
   * constructor when `shapeMatchMode === 'implicit'`.
   */
  private installImplicitKernels(options: SoftbodySystemOptions): void {
    const { particles, xpbd, bodies } = options;
    const restArr = this.restOffsets.value.array as Float32Array;

    // ---- Validate every body has an edge list ----
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b]!;
      if (!body.edges) {
        throw new Error(
          `SoftbodySystem: body ${b} is missing 'edges' (required for shapeMatchMode='implicit')`,
        );
      }
      if (body.edges.length % 2 !== 0) {
        throw new Error(
          `SoftbodySystem: body ${b} edges length ${body.edges.length} is not even (expected packed [i,j] pairs)`,
        );
      }
      for (let e = 0; e < body.edges.length; e += 2) {
        const li = body.edges[e]!;
        const lj = body.edges[e + 1]!;
        if (li >= body.particleRange.count || lj >= body.particleRange.count || li === lj) {
          throw new Error(
            `SoftbodySystem: body ${b} edge (${li},${lj}) out of range or self-loop (count=${body.particleRange.count})`,
          );
        }
      }
    }

    // ---- Build CSR over the global capacity space ----
    // Per-particle degree counts each particle's "self" entry plus every
    // edge incident on it (counted from either end of the undirected edge).
    // CSR indices store global particle slots so the kernel can index
    // particles.predictedPositions / restOffsets directly.
    const capacity = particles.capacity;
    const degree = new Uint32Array(capacity);
    // Initialize self-membership for slots inside any implicit body. Slots
    // outside any body keep degree 0 so the kernels skip them.
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b]!;
      for (let i = 0; i < body.particleRange.count; i++) {
        degree[body.particleRange.start + i]! += 1;
      }
    }
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b]!;
      const base = body.particleRange.start;
      for (let e = 0; e < body.edges!.length; e += 2) {
        const gi = base + body.edges![e]!;
        const gj = base + body.edges![e + 1]!;
        degree[gi]! += 1;
        degree[gj]! += 1;
      }
    }

    // Prefix-sum into offsets.
    const offsets = new Uint32Array(capacity + 1);
    for (let i = 0; i < capacity; i++) {
      offsets[i + 1] = offsets[i]! + degree[i]!;
    }
    const totalDegree = offsets[capacity]!;

    // Fill CSR indices. Use a per-particle write cursor seeded from offsets.
    const cursor = new Uint32Array(capacity);
    cursor.set(offsets.subarray(0, capacity));
    const indices = new Uint32Array(totalDegree);
    // Self-entries first.
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b]!;
      const base = body.particleRange.start;
      for (let i = 0; i < body.particleRange.count; i++) {
        const slot = base + i;
        indices[cursor[slot]!++] = slot;
      }
    }
    // Edge entries (each undirected edge contributes both endpoints).
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b]!;
      const base = body.particleRange.start;
      for (let e = 0; e < body.edges!.length; e += 2) {
        const gi = base + body.edges![e]!;
        const gj = base + body.edges![e + 1]!;
        indices[cursor[gi]!++] = gj;
        indices[cursor[gj]!++] = gi;
      }
    }

    // ---- Compute per-particle rest neighborhood centroid c̄_i ----
    // c̄_i = (1/|N(i)|) Σ restOffsets[j] over j ∈ N(i). Body-local frame
    // (restOffsets is already pre-centered to body COM at this point).
    const restNeighborhoodCenters = new Float32Array(4 * capacity);
    for (let i = 0; i < capacity; i++) {
      const start = offsets[i]!;
      const end = offsets[i + 1]!;
      if (end === start) continue;
      let sx = 0,
        sy = 0,
        sz = 0;
      for (let k = start; k < end; k++) {
        const j = indices[k]!;
        const base = j * 4;
        sx += restArr[base + 0]!;
        sy += restArr[base + 1]!;
        sz += restArr[base + 2]!;
      }
      const inv = 1 / (end - start);
      const out = i * 4;
      restNeighborhoodCenters[out + 0] = sx * inv;
      restNeighborhoodCenters[out + 1] = sy * inv;
      restNeighborhoodCenters[out + 2] = sz * inv;
    }

    // ---- System-wide compliance ----
    // MVP deviation: use a single uniform sourced from `bodies[0]`. All
    // bodies in one implicit-mode SoftbodySystem must agree on
    // matchCompliance. Mixed-stiffness scenes use multiple instances.
    const baseCompliance = bodies[0]!.matchCompliance;
    for (let b = 1; b < bodies.length; b++) {
      if (bodies[b]!.matchCompliance !== baseCompliance) {
        throw new Error(
          `SoftbodySystem: implicit-mode bodies must share a single matchCompliance — body 0 has ${baseCompliance}, body ${b} has ${bodies[b]!.matchCompliance}. Use multiple SoftbodySystem instances for mixed stiffness.`,
        );
      }
    }
    const complianceUniform = uniform(baseCompliance, 'float');

    // ---- Allocate GPU buffers ----
    const particleCenters = instancedArray(capacity, 'vec4');
    const particleRotations = instancedArray(3 * capacity, 'vec4');
    const restCentersBuf = instancedArray(capacity, 'vec4');
    const neighborOffsets = instancedArray(capacity + 1, 'uint');
    const neighborIndices = instancedArray(Math.max(totalDegree, 1), 'uint');
    const pairLambda = instancedArray(Math.max(totalDegree, 1), 'vec4');

    // Upload CPU-built data.
    (restCentersBuf.value.array as Float32Array).set(restNeighborhoodCenters);
    (neighborOffsets.value.array as Uint32Array).set(offsets);
    if (totalDegree > 0) {
      (neighborIndices.value.array as Uint32Array).set(indices);
    }
    restCentersBuf.value.needsUpdate = true;
    neighborOffsets.value.needsUpdate = true;
    neighborIndices.value.needsUpdate = true;

    // Dedicated accumulator for §5.1 Δx scatter — material-owned per the
    // Phase 11 FluidSystem precedent. SimLoop's contact accumulator may not
    // exist (this scene may have no contact pipeline) and even if it did,
    // sharing introduces construction-order coupling.
    const maxCorrectionMeters = options.maxCorrectionMeters ?? 10;
    const accumulator = new ContactAccumulator(
      particles,
      deriveAccumulatorScale(maxCorrectionMeters),
    );

    // r²/5 — sphere inertia coefficient with mass cancelled (uniform-mass
    // simplification; see shapeMatchImplicit.ts module doc).
    const aiScalar = uniform((particles.particleRadius * particles.particleRadius) / 5.0, 'float');

    // ---- Build kernels ----
    const resetAccum = buildResetAccumulatorKernel(accumulator);
    const resetOverflow = buildResetOverflowFlagKernel(accumulator);
    const resetPairLambdaKernel = buildImplicitResetPairLambdaKernel({
      pairLambda,
      totalDegree,
    });
    const centerKernel = buildImplicitNeighborhoodCenterKernel({
      particles,
      neighborOffsets,
      neighborIndices,
      particleCenters,
    });
    const momentPolarKernel = buildImplicitMomentPolarKernel({
      particles,
      restOffsets: this.restOffsets,
      neighborOffsets,
      neighborIndices,
      particleCenters,
      restNeighborhoodCenters: restCentersBuf,
      particleRotations,
      aiScalar,
    });
    const scatterKernel = buildImplicitShapeMatchScatterKernel({
      particles,
      restOffsets: this.restOffsets,
      neighborOffsets,
      neighborIndices,
      particleCenters,
      restNeighborhoodCenters: restCentersBuf,
      particleRotations,
      compliance: complianceUniform,
      pairLambda,
      accumulator,
      xpbd,
    });
    const applyKernel = buildApplyAccumulatorToPredictedKernel(accumulator);
    const qpWriteKernel = buildImplicitQpWriteKernel({
      particles,
      particleRotations,
      neighborOffsets,
    });

    // Expose buffers + accumulator on the instance.
    (this as { particleCenters?: StorageBufferNode<'vec4'> }).particleCenters = particleCenters;
    (this as { particleRotations?: StorageBufferNode<'vec4'> }).particleRotations =
      particleRotations;
    (this as { restNeighborhoodCenters?: StorageBufferNode<'vec4'> }).restNeighborhoodCenters =
      restCentersBuf;
    (this as { neighborOffsets?: StorageBufferNode<'uint'> }).neighborOffsets = neighborOffsets;
    (this as { neighborIndices?: StorageBufferNode<'uint'> }).neighborIndices = neighborIndices;
    (this as { pairLambda?: StorageBufferNode<'vec4'> }).pairLambda = pairLambda;
    (this as { implicitAccumulator?: ContactAccumulator }).implicitAccumulator = accumulator;

    // preIter: reset accumulator + overflow flag + pair λ; compute c_i,
    // R_i once per substep. Order matches §5.3's "reset → centre → moment".
    this.preIterKernels = [
      resetAccum,
      resetOverflow,
      resetPairLambdaKernel,
      centerKernel,
      momentPolarKernel,
    ];
    // perIter: scatter Δx into accumulator, apply (zeroes accumulator for
    // next iter), write qp_i. The qp_i write runs every iter; the last
    // iter's value wins naturally (R_i is constant across iters within a
    // substep, so qp_i is also stable — see shapeMatchImplicit.ts Pass 4
    // doc for the cadence rationale).
    this.perIterKernels = [scatterKernel, applyKernel, qpWriteKernel];
  }

  /**
   * Full particle range belonging to body `bodyIndex`. Used by the scene
   * for contact-related purposes (Phase 05 phase mask skips intra-body
   * pairs) and by the shape-matching kernels.
   */
  particleRange(bodyIndex: number): ParticleRange {
    const body = this.bodies[bodyIndex];
    if (!body) {
      throw new Error(
        `SoftbodySystem.particleRange: bodyIndex ${bodyIndex} out of bounds (0..${this.bodies.length - 1})`,
      );
    }
    return body.particleRange;
  }

  /**
   * Surface sub-range of body `bodyIndex` — the first
   * `Σ surfaceFlag` particles of the body's range. Used by Phase 17
   * `FluidSystem.registerBoundaryParticles(surfaceRange)` for Akinci
   * 2012 §2.2 single-layer sampling. Voxelize is expected to produce a
   * contiguous [surface..., interior...] particle ordering so the sub-
   * range is representable as a plain {@link ParticleRange}.
   */
  surfaceRange(bodyIndex: number): ParticleRange {
    const body = this.bodies[bodyIndex];
    if (!body) {
      throw new Error(
        `SoftbodySystem.surfaceRange: bodyIndex ${bodyIndex} out of bounds (0..${this.bodies.length - 1})`,
      );
    }
    let surfaceCount = 0;
    for (let i = 0; i < body.surfaceFlag.length; i++) {
      if (body.surfaceFlag[i] !== 0) surfaceCount++;
    }
    // Invariant expected from voxelize: surface particles come first in
    // the range. Guard against a malformed surfaceFlag that interleaves.
    for (let i = 0; i < surfaceCount; i++) {
      if (body.surfaceFlag[i] === 0) {
        throw new Error(
          `SoftbodySystem: body ${bodyIndex} surfaceFlag violates [surface..., interior...] ordering at index ${i}`,
        );
      }
    }
    return { start: body.particleRange.start, count: surfaceCount };
  }
}

/**
 * Eigenvalues of a symmetric 3x3 PSD matrix via Jacobi rotations. Input
 * is the upper triangle. Output is sorted descending; smallest is
 * `[2]`. Used by {@link SoftbodySystem} construction to enforce the
 * rank-3 particle-cloud precondition (Mueller 2011 §3 page 2).
 *
 * Jacobi converges quadratically near the solution. 30 sweeps with the
 * largest-|off-diagonal| pivot strategy is far more than needed (3x3
 * typically converges in 4–8 sweeps) but cheap on CPU at construction
 * time and gives a large safety margin for near-degenerate inputs.
 */
function eigenvaluesSymmetric3x3(
  m00: number,
  m01: number,
  m02: number,
  m11: number,
  m12: number,
  m22: number,
): [number, number, number] {
  let a00 = m00;
  let a01 = m01;
  let a02 = m02;
  let a11 = m11;
  let a12 = m12;
  let a22 = m22;

  for (let sweep = 0; sweep < 30; sweep++) {
    const aa01 = Math.abs(a01);
    const aa02 = Math.abs(a02);
    const aa12 = Math.abs(a12);
    const off = aa01 + aa02 + aa12;
    if (off < 1e-14) break;

    let apq: number, app: number, aqq: number;
    let pivot: 0 | 1 | 2;
    if (aa01 >= aa02 && aa01 >= aa12) {
      pivot = 0;
      apq = a01;
      app = a00;
      aqq = a11;
    } else if (aa02 >= aa12) {
      pivot = 1;
      apq = a02;
      app = a00;
      aqq = a22;
    } else {
      pivot = 2;
      apq = a12;
      app = a11;
      aqq = a22;
    }
    if (Math.abs(apq) < 1e-20) break;

    const theta = (aqq - app) / (2 * apq);
    const t =
      theta >= 0
        ? 1 / (theta + Math.sqrt(1 + theta * theta))
        : 1 / (theta - Math.sqrt(1 + theta * theta));
    const cc = 1 / Math.sqrt(1 + t * t);
    const ss = t * cc;

    const newApp = app - t * apq;
    const newAqq = aqq + t * apq;

    if (pivot === 0) {
      const new02 = cc * a02 - ss * a12;
      const new12 = ss * a02 + cc * a12;
      a00 = newApp;
      a11 = newAqq;
      a01 = 0;
      a02 = new02;
      a12 = new12;
    } else if (pivot === 1) {
      const new01 = cc * a01 - ss * a12;
      const new12 = ss * a01 + cc * a12;
      a00 = newApp;
      a22 = newAqq;
      a02 = 0;
      a01 = new01;
      a12 = new12;
    } else {
      const new01 = cc * a01 - ss * a02;
      const new02 = ss * a01 + cc * a02;
      a11 = newApp;
      a22 = newAqq;
      a12 = 0;
      a01 = new01;
      a02 = new02;
    }
  }

  const vals: [number, number, number] = [a00, a11, a22];
  vals.sort((a, b) => b - a);
  return vals;
}
