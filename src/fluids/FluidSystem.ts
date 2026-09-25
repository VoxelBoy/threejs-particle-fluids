import { Vector3 } from 'three';
import { instancedArray, uniform } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type {
  HashGrid,
  Material,
  ParticleRange,
  ParticleSystem,
  XpbdUniforms,
} from '../core/index.js';
import {
  ContactAccumulator,
  DEFAULT_MAX_VELOCITY,
  VelocityAccumulator,
  allocatePairListStorage,
  buildApplyAccumulatorToPredictedKernel,
  buildPairListKernel,
  buildResetAccumulatorKernel,
  buildResetOverflowFlagKernel,
  deriveAccumulatorScale,
  deriveVelocityAccumulatorScale,
} from '../core/index.js';

import { buildBoundaryVolumeKernel } from './sim/boundaryVolume.js';
import { buildLambdaKernel } from './sim/lambda.js';
import { buildApplyDeltaKernel, buildPositionDeltaKernel } from './sim/positionDelta.js';
import { buildXsphApplyKernel, buildXsphComputeKernel } from './sim/xsph.js';
import {
  buildVorticityPass1Kernel,
  buildVorticityPass2Kernel,
  buildVorticityPass3Kernel,
} from './sim/vorticity.js';
import { buildFusedVorticityXsphWalkKernel } from './sim/fusedVorticityXsphWalk.js';
import { buildFusedVorticityXsphApplyKernel } from './sim/fusedVorticityXsphApply.js';
import {
  buildApplyVelocityImpulseKernel,
  buildColorFieldNormalKernel,
  buildSurfaceTensionScatterKernel,
  createCohesionUniforms,
  type CohesionUniforms,
} from './sim/cohesion.js';
import {
  buildAdhesionScatterKernel,
  createAdhesionUniforms,
  type AdhesionUniforms,
} from './sim/adhesion.js';
// Phase Perf-15: `buildSolidReactionScatterKernel` is no longer used by
// FluidSystem (folded into `buildCoupledDeltaKernel`); it is kept
// re-exported from `index.ts` for tests / external callers that want
// the unfused form.
import { buildCoupledDeltaKernel } from './sim/coupledDelta.js';
import { buildDragKernel } from './sim/drag.js';
import { createSphKernelUniforms, type SphKernelUniforms } from './sim/kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const DEFAULT_VORTICITY_STRENGTH = 0.1;
const DEFAULT_XSPH_C = 0.01;

/**
 * Default upper bound on per-particle |Δp| (m) for the Phase 11 solid-
 * reaction `ContactAccumulator`. Same 10 m default the contact pipeline
 * uses (see `accumulator.ts::deriveAccumulatorScale`'s docstring) — at
 * MVP scales the actual per-iter correction is O(mm), three orders of
 * magnitude below the saturation point.
 */
const DEFAULT_SOLID_REACTION_MAX_CORRECTION_M = 10;

export interface FluidSystemOptions {
  readonly particles: ParticleSystem;
  readonly hashGrid: HashGrid;
  /**
   * Shared XPBD uniforms — typically the ones owned by the `SimLoop`
   * this fluid participates in. Must be the *same instance* the loop
   * constructs with; the `dt` uniform is shared by identity, not value.
   */
  readonly xpbd: XpbdUniforms;
  /** `ρ_0` — rest density, kg/m³ (water ≈ 1000). */
  readonly restDensity: number;
  /** Smoothing length `h`, m. Requires `hashGrid.cellSize ≥ h`. */
  readonly h: number;
  /**
   * Per-particle spacing, m. Used to derive `invMass = 1 / (ρ_0 ·
   * spacing³)` for each fluid particle at construction time.
   */
  readonly particleSpacing: number;
  /** XPBD compliance `α`. Scene-tuned; Assumed ~1e-6 pending hydrostatic test. */
  readonly compliance: number;
  /** Range of `ParticleSystem` slots owned by this fluid. */
  readonly fluidParticles: ParticleRange;
  /**
   * XSPH viscosity (Macklin 2013 §5 eq. 17). Omit for no XSPH. Paper
   * default `c = 0.01` when enabled.
   */
  readonly xsph?: { readonly c: number };
  /**
   * Vorticity confinement (Macklin 2013 §5 eqs. 15–16). Omitted →
   * default-on with `strength = 0.1` (Assumed). Pass `{ strength: 0 }`
   * to effectively disable (dispatches still run but apply zero force).
   */
  readonly vorticity?: { readonly strength: number };
  /**
   * Akinci 2013 surface tension coefficient `γ`. Paper default 1
   * (water-like, §4). Omit or pass `0` to skip cohesion + curvature
   * kernel construction entirely — zero-strength means disabled
   * (Phase 08 Finding 8 pattern). Artists tune in `[0.5, 2]` for
   * visual droplet tension.
   */
  readonly surfaceTension?: number;
  /**
   * Akinci 2013 adhesion coefficient `β` (eq. 6). Paper default 0
   * (free-fluid scenes pay nothing). Paper's "moderate hydrophilic"
   * recipe is `β = γ = 1`. Omit or pass `0` to skip kernel construction.
   * Requires at least one boundary range to have been registered via
   * {@link FluidSystem.registerBoundaryParticles} before the first
   * substep runs — the kernel walks fluid neighbors filtered by
   * `boundaryVolume > 0`, so un-registered scenes get zero force
   * (kernel still dispatches but accumulates nothing). Prefer setting
   * `0` explicitly when no boundary particles exist.
   */
  readonly adhesion?: number;
  /**
   * Macklin 2014 §7.2.2 free-surface drag (eq. 29). Used by gas scenes
   * to attenuate fluid (air) motion near the free surface. Omit or pass
   * `k = 0` to skip kernel construction — zero-strength is disabled.
   *
   *
   * The drag kernel runs in `postAdvectKernels` alongside vorticity and
   * XSPH; see `src/fluids/sim/drag.ts` for the substep-slot
   * rationale and discretisation note.
   */
  readonly drag?: { readonly k: number; readonly vEnv?: Vector3 };

  readonly __forceUnfusedPostAdvect?: boolean;
}

/**
 * `src/fluids` Position-Based Fluids material module.
 *
 *

 *
 * Lifecycle note: construction writes `invMass = 1/(ρ_0 · spacing³)`
 * into the fluid slots of `particles.invMass`. This overwrites any
 * previous values the caller uploaded — per plan, the fluid owns mass
 * configuration for its range. The range must have been populated (via
 * `uploadParticles`) before this constructor runs; the override only
 * touches `invMass`, not `position` / `velocity` / `phase`.
 */
export class FluidSystem implements Material {
  readonly particles: ParticleSystem;
  readonly hashGrid: HashGrid;
  readonly xpbd: XpbdUniforms;
  readonly fluidParticles: ParticleRange;
  readonly h: number;
  readonly restDensity: number;
  readonly particleSpacing: number;
  readonly mass: number;

  /** Shared Poly6 / Spiky uniforms (mutable via `sph.setH`). */
  readonly sph: SphKernelUniforms;
  readonly restDensityUniform: UniformNode<'float', number>;
  readonly complianceUniform: UniformNode<'float', number>;
  readonly massUniform: UniformNode<'float', number>;
  readonly cohesion?: CohesionUniforms;
  readonly adh?: AdhesionUniforms;
  /**
   * Per-fluid-particle rest volume `V = m/ρ_0 = spacing³`. Consumed by
   * vorticity and XSPH kernels as the Monaghan `m_j/ρ_j` SPH weight that
   * paper's unit-mass form absorbs. Without it, velocity-post-process
   * kernels diverge by a factor `1/V ≈ 64 000×` for water-scale fluids.
   */
  readonly particleVolumeUniform: UniformNode<'float', number>;
  readonly vorticityStrengthUniform?: UniformNode<'float', number>;
  readonly xsphCUniform?: UniformNode<'float', number>;
  /** Macklin 2014 §7.2.2 drag uniforms — present when `drag` was enabled. */
  readonly dragKUniform?: UniformNode<'float', number>;
  readonly dragVEnvUniform?: UniformNode<'vec3', Vector3>;

  /** Per-particle density `ρ_i`, sized to `particles.capacity`. */
  readonly density: StorageBufferNode<'float'>;
  /** Per-particle Lagrange multiplier `λ_i`, sized to `particles.capacity`. */
  readonly lambda: StorageBufferNode<'float'>;
  /**
   * Per-particle Δx from Macklin 2013 eq. 12. Written by the position-
   * delta kernel, consumed by the apply kernel — two-pass to avoid the
   * read-modify-write race on `predictedPositions` (Algorithm 1 lines
   * 12–18 split).
   */
  readonly deltaX: StorageBufferNode<'vec4'>;
  readonly omega?: StorageBufferNode<'vec4'>;
  readonly omegaMag?: StorageBufferNode<'float'>;
  readonly eta?: StorageBufferNode<'vec4'>;
  /** XSPH two-phase deltaV buffer (allocated only when XSPH is enabled). */
  readonly xsphDeltaV?: StorageBufferNode<'vec4'>;
  /** Akinci 2013 color-field normal `n_i`. Allocated only when surfaceTension > 0. */
  readonly colorFieldNormal?: StorageBufferNode<'vec4'>;
  /**
   * Shared velocity accumulator for surface-tension + adhesion scatter
   * (Phase 09 U-33 resolution: paper §2.3 + §4 specify per-pair scatter,
   * not gather). Owned by FluidSystem when at least one of `surfaceTension
   * > 0` or `adhesion > 0` is set; allocated with a scale derived from
   * `DEFAULT_MAX_VELOCITY`. A future scene with a shared `SimLoop`
   * accumulator (Phase 05a) can lift this; for MVP it's local to the
   * FluidSystem because Phase 09 may ship in scenes without contact.
   */
  readonly surfaceTensionAccumulator?: VelocityAccumulator;

  /**
   * Per-substep neighbor pair list (Macklin & Müller 2013 Algorithm 1
   * line 6 + §6 paragraph 2). Built once per substep, FIRST in
   * `#baselinePreIterKernels` (after the dynamic boundary-volume
   * kernels in the public getter). Consumed by every per-iter kernel
   * that walks neighbors (fused `lambda` (density+λ, Phase Perf-08) +
   * `positionDelta`) plus the pre-iter cohesion + adhesion kernels
   * (Phase Perf-07).
   *
   * Build order (phase-perf-07): the build moved from LAST to FIRST in
   * `#baselinePreIterKernels` so the new pre-iter consumers see a
   * fresh pair list. Per-iter consumers now see a list built before
   * the surface-tension / adhesion impulse-apply step; the resulting
   * `predictedPositions` drift between build and per-iter read is
   * bounded at `~10 µm` per substep at MVP defaults — three orders of
   * magnitude smaller than the per-iter loop staleness Phase Perf-04
   * already accepted, and well inside the Poly6/Spiky self-clamp
   * envelope outside `h`.
   *
   * Layout: row-major `u32[fluidParticles.count × MAX_NEIGHBORS]` for
   * `pairList`, `u32[fluidParticles.count]` for `pairCount`,
   * `atomic<u32>[1]` for `pairOverflowFlag`. See `src/core/
   * hashGrid/pairList.ts` for full layout and lifetime contract.
   */
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
  readonly pairOverflowFlag: StorageBufferNode<'uint'>;

  /**
   * {@link Material} interface — dynamic boundary-volume recompute +
   * surface tension + adhesion land here. Runs ONCE per substep after
   * predict + hashgrid rebuild + contact gen. Empty when no dynamic
   * boundaries are registered AND both `surfaceTension` and `adhesion`
   * are zero/omitted.
   *
   * Order invariant (2026-04-24): dynamic boundary-volume kernels run
   * FIRST, so adhesion's `boundaryVolume` reads and density's `ψ_j =
   * ρ_0 · V_j` branch see the current-substep values. Backed by two
   * internal arrays concatenated in the getter.
   */
  get preIterKernels(): readonly ComputeNode[] {
    return this.#dynamicBoundaryKernels.length === 0
      ? this.#baselinePreIterKernels
      : [...this.#dynamicBoundaryKernels, ...this.#baselinePreIterKernels];
  }
  /**
   * {@link Material} — fused-density-and-lambda → positionDelta (writes
   * deltaX, does NOT touch predictedPositions) → applyDelta (commits
   * deltaX into predictedPositions) per iter. The lambda kernel emits
   * both `density[i]` (always, before any early-return) and `lambda[i]`
   * after a single pair-walk (Phase Perf-08 fusion, 2026-04-27).
   *
   * Phase 11: when `registerBoundaryParticles` has been called at least
   * once, the fluid → solid Newton-3 reaction scatter is interleaved
   * BEFORE `applyDelta`, and the accumulator drain is interleaved AFTER
   * `applyDelta`:
   *   [..., positionDelta, **solidReactionScatter**, applyDelta,
   *    **applyAccumulator**]
   * The scatter slots between positionDelta and applyDelta so it reads
   * the SAME `predictedPositions` positionDelta read — both compute
   * `∇W(x_i − x_j)` against pre-update positions, ensuring per-pair
   * Newton-3 conservation holds. Placing the scatter after applyDelta
   * would have it read fluid `x*_i` AFTER the fluid Δp was applied, while
   * positionDelta computed against `x*_i` BEFORE — different gradients
   * break per-pair conservation.
   *
   * `applyAccumulator` is the existing core
   * `buildApplyAccumulatorToPredictedKernel`; it dispatches over all
   * particles but only the boundary slots have non-zero accumulator
   * values (the scatter filters fluid neighbours), so fluid x* is
   * untouched.
   *
   * Concatenation lives in the getter so a single lazy allocation in
   * `registerBoundaryParticles` propagates without the constructor
   * having to predict whether boundaries will be registered.
   */
  get perIterKernels(): readonly ComputeNode[] {
    if (this.#solidReactionKernels.length === 0) {
      return this.#baselinePerIterKernels;
    }
    // Phase Perf-15 — after `registerBoundaryParticles` the baseline
    // layout is [lambda(fused with density), coupledDelta, applyDelta]:
    // the fused `coupledDelta` kernel folds the per-iter solid-reaction
    // scatter into the position-delta walk (one pass, one barrier
    // saved). `#solidReactionKernels` holds only the accumulator-drain
    // kernel; we append it AFTER applyDelta.
    const [accumApply] = this.#solidReactionKernels;
    if (!accumApply) {
      // Defensive: registerBoundaryParticles always pushes the apply.
      return this.#baselinePerIterKernels;
    }
    return [...this.#baselinePerIterKernels, accumApply];
  }
  /** {@link Material} — vorticity 3-pass + optional XSPH, once per substep. */
  readonly postAdvectKernels: readonly ComputeNode[];

  /**
   * Baseline preIterKernels — surface-tension + adhesion + impulse apply.
   * Populated in the constructor; never mutated after.
   *
   * Phase 11 addition: when the solid-reaction accumulator is allocated
   * by `registerBoundaryParticles`, its reset kernels (i32-buffer reset
   * + overflow-flag reset) are appended into this array on the *first*
   * registration. The reset must run once per substep before any per-
   * iter scatter — the same convention the SimLoop-owned contact
   * accumulator uses.
   */
  readonly #baselinePreIterKernels: ComputeNode[];
  /**
   * Per-substep boundary-volume recompute kernels, one per dynamic range
   * registered via {@link FluidSystem.registerBoundaryParticles}. Mutable
   * until the owning `SimLoop` is constructed — `SimLoop` spreads the
   * getter's snapshot at its own construction. Registrations after that
   * point will not be dispatched; scenes must register before building
   * the loop.
   */
  readonly #dynamicBoundaryKernels: ComputeNode[] = [];
  /**
   * Phase 11 fluid → solid Newton-3 reaction kernels, lazy-built on the
   * first {@link FluidSystem.registerBoundaryParticles} call. Two
   * dispatches per iter: the per-fluid-i scatter into the
   * {@link FluidSystem.solidReactionAccumulator}, then the apply that
   * drains the accumulator into `predictedPositions[j]` for boundary
   * `j`. Empty until the first registration; the {@link FluidSystem.
   * perIterKernels} getter only appends when this array is non-empty.
   */
  readonly #solidReactionKernels: ComputeNode[] = [];
  /**
   * Baseline per-iter kernels — fused (density+lambda) → positionDelta
   * → applyDelta. Set once in the constructor; the Phase 11 scatter +
   * apply are concatenated in the {@link FluidSystem.perIterKernels}
   * getter rather than here so the kernel cost is paid only when at
   * least one boundary range is registered.
   */
  // NOTE: Phase Perf-15 mutates this array in place during
  // `registerBoundaryParticles` to swap `positionDelta` for the fused
  // `coupledDelta`. Field stays `readonly` (the binding can't be
  // reassigned) but the array contents are mutable. Readers (the
  // `perIterKernels` getter) take a snapshot via spread/slice and don't
  // hold a reference, so the swap is safe.
  readonly #baselinePerIterKernels: ComputeNode[];

  /**
   * Phase 11 position-domain scatter accumulator. Lazy-allocated by
   * the *first* {@link FluidSystem.registerBoundaryParticles} call.
   * Pure-fluid scenes pay nothing.
   *
   * FluidSystem-owned (rather than SimLoop-shared) because SimLoop
   * snapshots `perIterKernels` inside its own constructor — the kernels
   * have to exist (and reference an accumulator) before SimLoop is
   * built. Mirrors the {@link FluidSystem.surfaceTensionAccumulator}
   * pattern: per-FluidSystem accumulator + per-FluidSystem apply
   * dispatch, decoupled from any SimLoop-level scatter pipeline.
   */
  solidReactionAccumulator?: ContactAccumulator;

  /**
   * Phase 14c anisotropy storage. Lazy-allocated by
   * {@link FluidSystem.enableAnisotropyBuffers} (typically called once
   * by `FluidSurfaceRenderer` when its `anisotropy` group is
   * constructable). Defined together because the Yu & Turk 2010 §4
   * pipeline writes all three in one compute pass (eq. 6 smoothed
   * centre + eq. 16 G_i^-1 split into diagonal + off-diagonal halves).
   *
   * Layout — symmetric 3×3 G_i^-1 = h_i · R · Σ̃ · R^T (eq. 16 inverted)
   * stored as two vec3 buffers: `anisotropyDiag[i] = (g00, g11, g22)`,
   * `anisotropyOff[i] = (g01, g02, g12)`. Six unique floats per
   * particle, padded to 32 bytes by WebGPU's vec3 alignment. The depth
   * pass reads both halves and reconstructs the full 3×3 to size the
   * imposter quad and run the per-pixel ray-vs-ellipsoid test.
   *
   * `smoothedPositions[i] = x̄_i` per Yu & Turk eq. 6 (Laplacian
   * smoothing with `λ ∈ [0.9, 1.0]`, default 0.95). Used by the depth
   * pass as the imposter centre instead of the raw simulation position.
   */
  anisotropyDiag?: StorageBufferNode<'vec4'>;
  anisotropyOff?: StorageBufferNode<'vec4'>;
  smoothedPositions?: StorageBufferNode<'vec4'>;
  /**
   * Phase 14c diagnostic buffer for anisotropy debug views (#aniso.*):
   *   .x = N — neighbour count within `r_i = 2·h`, including self
   *   .y = σ_1 — largest raw eigenvalue of `C_i` (pre-eq. 15 clamp)
   *   .z = σ_3 — smallest raw eigenvalue
   *   .w = |x̄_i − x_i| — Laplacian smoothing offset magnitude
   * Lets the artist see whether the interior branch is firing
   * (`N > N_ε`) and how anisotropic the local distribution is.
   */
  anisotropyDiagnostic?: StorageBufferNode<'vec4'>;

  constructor(options: FluidSystemOptions) {
    const {
      particles,
      hashGrid,
      xpbd,
      restDensity,
      h,
      particleSpacing,
      compliance,
      fluidParticles,
      xsph,
      vorticity,
      surfaceTension,
      adhesion,
      drag,
      __forceUnfusedPostAdvect,
    } = options;

    // --- Validation ---
    if (!Number.isFinite(restDensity) || restDensity <= 0) {
      throw new Error(
        `FluidSystem: restDensity must be a positive finite number, got ${restDensity}`,
      );
    }
    if (!Number.isFinite(h) || h <= 0) {
      throw new Error(`FluidSystem: h must be a positive finite number, got ${h}`);
    }
    if (!Number.isFinite(particleSpacing) || particleSpacing <= 0) {
      throw new Error(
        `FluidSystem: particleSpacing must be a positive finite number, got ${particleSpacing}`,
      );
    }
    if (!Number.isFinite(compliance) || compliance < 0) {
      throw new Error(
        `FluidSystem: compliance must be a non-negative finite number, got ${compliance}`,
      );
    }
    if (
      !Number.isInteger(fluidParticles.start) ||
      !Number.isInteger(fluidParticles.count) ||
      fluidParticles.start < 0 ||
      fluidParticles.count <= 0 ||
      fluidParticles.start + fluidParticles.count > particles.capacity
    ) {
      throw new Error(
        `FluidSystem: invalid fluidParticles range start=${fluidParticles.start} count=${fluidParticles.count} capacity=${particles.capacity}`,
      );
    }
    if (surfaceTension !== undefined && !Number.isFinite(surfaceTension)) {
      throw new Error(`FluidSystem: surfaceTension must be a finite number, got ${surfaceTension}`);
    }
    if (adhesion !== undefined && !Number.isFinite(adhesion)) {
      throw new Error(`FluidSystem: adhesion must be a finite number, got ${adhesion}`);
    }
    if (drag !== undefined && (!Number.isFinite(drag.k) || drag.k < 0)) {
      throw new Error(`FluidSystem: drag.k must be a non-negative finite number, got ${drag.k}`);
    }
    if (hashGrid.particles !== particles) {
      throw new Error('FluidSystem: hashGrid must be constructed against the same ParticleSystem');
    }
    if (hashGrid.cellSize < h) {
      throw new Error(
        `FluidSystem: hashGrid.cellSize (${hashGrid.cellSize}) must be ≥ h (${h}) — missing neighbors is silent corruption`,
      );
    }

    this.particles = particles;
    this.hashGrid = hashGrid;
    this.xpbd = xpbd;
    this.fluidParticles = fluidParticles;
    this.h = h;
    this.restDensity = restDensity;
    this.particleSpacing = particleSpacing;
    this.mass = restDensity * particleSpacing * particleSpacing * particleSpacing;

    // --- Write fluid-particle invMass ---
    // Per plan: `FluidSystem` owns the mass configuration for its range.
    // Uses `addUpdateRange` per the ARCHITECTURE.md §Compute path finding
    // — partial writes without a range declaration trigger a full-buffer
    // re-upload that can clobber other material modules' uploads.
    const invMassValue = 1 / this.mass;
    const invMassArr = (particles.invMass.value as Any).array as Float32Array;
    for (let k = 0; k < fluidParticles.count; k++) {
      invMassArr[fluidParticles.start + k] = invMassValue;
    }
    (particles.invMass.value as Any).addUpdateRange(fluidParticles.start, fluidParticles.count);
    (particles.invMass.value as Any).needsUpdate = true;

    // --- Uniforms ---
    this.sph = createSphKernelUniforms(h);
    this.restDensityUniform = uniform(restDensity, 'float');
    this.complianceUniform = uniform(compliance, 'float');
    this.particleVolumeUniform = uniform(
      particleSpacing * particleSpacing * particleSpacing,
      'float',
    );
    this.massUniform = uniform(this.mass, 'float');

    // Vorticity is opt-in: pass `vorticity: { strength: > 0 }` to build
    // the three-pass kernels. Passing `{ strength: 0 }` or omitting it
    // entirely skips the 3 per-substep dispatches — a strength-zero
    // dispatch still traverses the 27-cell neighborhood per particle,
    // and the post-advect cost scales with particle count.
    const vorticityStrength = vorticity?.strength ?? DEFAULT_VORTICITY_STRENGTH;
    const vorticityEnabled = vorticity !== undefined && vorticityStrength > 0;
    // XSPH is opt-in as before: pass `xsph: { c: > 0 }` to enable.
    const xsphEnabled = xsph !== undefined && xsph.c > 0;
    const xsphC = xsph?.c ?? DEFAULT_XSPH_C;
    // Surface tension and adhesion are opt-in per Phase 08 Finding 8:
    // zero-strength skips kernel construction entirely (no wasted
    // dispatches on the 27-cell neighbor walk).
    const surfaceTensionEnabled = surfaceTension !== undefined && surfaceTension > 0;
    const adhesionEnabled = adhesion !== undefined && adhesion > 0;
    // Macklin 2014 §7.2.2 drag — opt-in. `k = 0` (or omission) skips
    // kernel construction; non-zero builds the postAdvect dispatch.
    const dragEnabled = drag !== undefined && drag.k > 0;
    if (dragEnabled) {
      this.dragKUniform = uniform(drag.k, 'float');
      this.dragVEnvUniform = uniform(drag.vEnv ?? new Vector3(0, 0, 0));
    }

    // --- Storage ---
    this.density = instancedArray(particles.capacity, 'float');
    this.lambda = instancedArray(particles.capacity, 'float');
    this.deltaX = instancedArray(particles.capacity, 'vec4');

    if (vorticityEnabled) {
      this.omega = instancedArray(particles.capacity, 'vec4');
      this.omegaMag = instancedArray(particles.capacity, 'float');
      this.eta = instancedArray(particles.capacity, 'vec4');
      this.vorticityStrengthUniform = uniform(vorticityStrength, 'float');
    }
    if (xsphEnabled) {
      this.xsphCUniform = uniform(xsphC, 'float');
      this.xsphDeltaV = instancedArray(particles.capacity, 'vec4');
    }
    if (surfaceTensionEnabled) {
      this.colorFieldNormal = instancedArray(particles.capacity, 'vec4');
      this.cohesion = createCohesionUniforms(h, surfaceTension);
    }
    if (adhesionEnabled) {
      this.adh = createAdhesionUniforms(h, adhesion);
    }
    // Shared velocity accumulator for surface-tension + adhesion scatter.
    if (surfaceTensionEnabled || adhesionEnabled) {
      this.surfaceTensionAccumulator = new VelocityAccumulator(
        particles,
        deriveVelocityAccumulatorScale(DEFAULT_MAX_VELOCITY),
      );
    }

    // Per-substep neighbor pair list — paper Algorithm 1 line 6
    // amortization. Sized to fluidParticles.count × MAX_NEIGHBORS.
    const pairStorage = allocatePairListStorage(fluidParticles.count);
    this.pairList = pairStorage.pairList;
    this.pairCount = pairStorage.pairCount;
    this.pairOverflowFlag = pairStorage.pairOverflowFlag;

    // Density bootstrap for Phase 09 surface-tension scatter. The
    // scatter kernel reads `density[i]` and `density[j]` to compute
    // K_ij = 2ρ_0/(ρ_i + ρ_j). On the very first substep those values
    // are zero-initialised from `instancedArray`; seeding to ρ_0 keeps
    // K_ij = 1 on substep 0 (no Akinci correction), and subsequent
    // substeps read last-iter density from perIterKernels — one advect-
    // step of lag, negligible at Δt ≈ 1/180 s.
    if (surfaceTensionEnabled) {
      const densityArr = (this.density.value as Any).array as Float32Array;
      for (let k = 0; k < fluidParticles.count; k++) {
        densityArr[fluidParticles.start + k] = restDensity;
      }
      (this.density.value as Any).addUpdateRange(fluidParticles.start, fluidParticles.count);
      (this.density.value as Any).needsUpdate = true;
    }

    const lambdaKernel = buildLambdaKernel({
      particles,
      sph: this.sph,
      restDensity: this.restDensityUniform,
      dt: xpbd.dt,
      compliance: this.complianceUniform,
      density: this.density,
      lambda: this.lambda,
      fluidParticles,
      pairList: this.pairList,
      pairCount: this.pairCount,
    });
    const positionDeltaKernel = buildPositionDeltaKernel({
      particles,
      sph: this.sph,
      restDensity: this.restDensityUniform,
      lambda: this.lambda,
      deltaX: this.deltaX,
      fluidParticles,
      pairList: this.pairList,
      pairCount: this.pairCount,
    });
    const applyDeltaKernel = buildApplyDeltaKernel({
      particles,
      deltaX: this.deltaX,
      fluidParticles,
    });
    this.#baselinePerIterKernels = [lambdaKernel, positionDeltaKernel, applyDeltaKernel];

    // --- Pre-iter kernels: pair-list build → surface tension + adhesion ---
    // Runs ONCE per substep after predict + hashgrid rebuild + contact
    // generation, plus any registered dynamic boundary-volume kernels
    // (prepended via the `preIterKernels` getter).
    //
    // Sequencing (phase-perf-07): pair-list build runs FIRST so the
    // pre-iter cohesion + adhesion consumers (after their kernel-level
    // migration in commits 4-6 of phase-perf-07) see a fresh list.
    // Per-iter density / lambda / positionDelta still consume the same
    // list; the surface-tension + adhesion impulse-apply now perturbs
    // predictedPositions AFTER the pair list is built, introducing a
    // bounded ~10 µm drift between build and per-iter read. See the
    // `pairList` field JSDoc above for the staleness analysis.
    //
    // Surface tension + adhesion scatter pipeline (Phase 09 per-pair
    // architecture, U-33 resolution). The kernels scatter Δv into a
    // shared velocity accumulator; a single apply kernel drains the
    // accumulator into `velocities` and `predictedPositions` and resets
    // it for the next substep.
    const preIter: ComputeNode[] = [];

    preIter.push(
      buildPairListKernel({
        particles,
        hashGrid,
        hSq: this.sph.hSq,
        fluidParticles,
        pairList: this.pairList,
        pairCount: this.pairCount,
        pairOverflowFlag: this.pairOverflowFlag,
      }),
    );

    const accumulator = this.surfaceTensionAccumulator;
    if (surfaceTensionEnabled && this.colorFieldNormal && this.cohesion && accumulator) {
      preIter.push(
        buildColorFieldNormalKernel({
          particles,
          sph: this.sph,
          particleVolume: this.particleVolumeUniform,
          colorFieldNormal: this.colorFieldNormal,
          fluidParticles,
          pairList: this.pairList,
          pairCount: this.pairCount,
        }),
        buildSurfaceTensionScatterKernel({
          particles,
          sph: this.sph,
          coh: this.cohesion,
          restDensity: this.restDensityUniform,
          mass: this.massUniform,
          density: this.density,
          colorFieldNormal: this.colorFieldNormal,
          dt: xpbd.dt,
          velocityAccumulator: accumulator,
          fluidParticles,
          pairList: this.pairList,
          pairCount: this.pairCount,
        }),
      );
    }
    if (adhesionEnabled && this.adh && accumulator) {
      preIter.push(
        buildAdhesionScatterKernel({
          particles,
          sph: this.sph,
          adh: this.adh,
          restDensity: this.restDensityUniform,
          mass: this.massUniform,
          dt: xpbd.dt,
          velocityAccumulator: accumulator,
          fluidParticles,
          pairList: this.pairList,
          pairCount: this.pairCount,
        }),
      );
    }
    // Apply runs once, regardless of which of (surfaceTension, adhesion)
    // is enabled — both scatter into the same accumulator and the apply
    // kernel drains-and-resets it in one pass.
    if (accumulator) {
      preIter.push(
        buildApplyVelocityImpulseKernel({
          particles,
          velocityAccumulator: accumulator,
          dt: xpbd.dt,
          fluidParticles,
        }),
      );
    }

    // Phase 11: the solid-reaction reset kernel (when allocated by
    // `registerBoundaryParticles`) must run once per substep before any
    // per-iter scatter. We push directly into this array later — keep
    // the assignment as a mutable spread.
    this.#baselinePreIterKernels = preIter;

    const postAdvect: ComputeNode[] = [];
    // Drag goes FIRST in postAdvect — it operates on the freshly-realised
    // post-advect velocity `v = (x* − x) / Δt`, before vorticity adds
    // energy back and XSPH smooths. Order is a discretisation choice; the
    // paper does not specify an ordering against §5 vorticity / XSPH.
    if (dragEnabled && this.dragKUniform && this.dragVEnvUniform) {
      postAdvect.push(
        buildDragKernel({
          particles,
          density: this.density,
          restDensity: this.restDensityUniform,
          k: this.dragKUniform,
          vEnv: this.dragVEnvUniform,
          dt: xpbd.dt,
          fluidParticles,
        }),
      );
    }
    const fuseWalkAndXsph =
      vorticityEnabled &&
      xsphEnabled &&
      !__forceUnfusedPostAdvect &&
      this.omega !== undefined &&
      this.omegaMag !== undefined &&
      this.eta !== undefined &&
      this.vorticityStrengthUniform !== undefined &&
      this.xsphCUniform !== undefined &&
      this.xsphDeltaV !== undefined;
    if (fuseWalkAndXsph) {
      postAdvect.push(
        buildFusedVorticityXsphWalkKernel({
          particles,
          sph: this.sph,
          particleVolume: this.particleVolumeUniform,
          c: this.xsphCUniform!,
          omega: this.omega!,
          omegaMag: this.omegaMag!,
          xsphDeltaV: this.xsphDeltaV!,
          fluidParticles,
          pairList: this.pairList,
          pairCount: this.pairCount,
        }),
        buildVorticityPass2Kernel({
          particles,
          sph: this.sph,
          particleVolume: this.particleVolumeUniform,
          omegaMag: this.omegaMag!,
          eta: this.eta!,
          fluidParticles,
          pairList: this.pairList,
          pairCount: this.pairCount,
        }),
        buildFusedVorticityXsphApplyKernel({
          particles,
          omega: this.omega!,
          eta: this.eta!,
          xsphDeltaV: this.xsphDeltaV!,
          strength: this.vorticityStrengthUniform!,
          dt: xpbd.dt,
          fluidParticles,
        }),
      );
    } else {
      if (
        vorticityEnabled &&
        this.omega &&
        this.omegaMag &&
        this.eta &&
        this.vorticityStrengthUniform
      ) {
        postAdvect.push(
          buildVorticityPass1Kernel({
            particles,
            sph: this.sph,
            particleVolume: this.particleVolumeUniform,
            omega: this.omega,
            omegaMag: this.omegaMag,
            fluidParticles,
            pairList: this.pairList,
            pairCount: this.pairCount,
          }),
          buildVorticityPass2Kernel({
            particles,
            sph: this.sph,
            particleVolume: this.particleVolumeUniform,
            omegaMag: this.omegaMag,
            eta: this.eta,
            fluidParticles,
            pairList: this.pairList,
            pairCount: this.pairCount,
          }),
          buildVorticityPass3Kernel({
            particles,
            omega: this.omega,
            eta: this.eta,
            strength: this.vorticityStrengthUniform,
            dt: xpbd.dt,
            fluidParticles,
          }),
        );
      }
      if (xsphEnabled && this.xsphCUniform && this.xsphDeltaV) {
        postAdvect.push(
          buildXsphComputeKernel({
            particles,
            sph: this.sph,
            c: this.xsphCUniform,
            particleVolume: this.particleVolumeUniform,
            deltaV: this.xsphDeltaV,
            fluidParticles,
            pairList: this.pairList,
            pairCount: this.pairCount,
          }),
          buildXsphApplyKernel({
            particles,
            deltaV: this.xsphDeltaV,
            fluidParticles,
          }),
        );
      }
    }
    this.postAdvectKernels = postAdvect;
  }

  /**
   * Register a boundary-particle range so its density contribution `Ψ_i
   * = ρ_0 · V_i` (Akinci 2012 eq. 5) flows into fluid density and
   * adhesion. Writes `V_i = 1 / Σ_k W(|x_i − x_k|, h)` (eq. 4) into
   * `particles.boundaryVolume[range]`. Neighbors are restricted to the
   * same `range` — boundary particles couple to each other, not to
   * fluid. See {@link buildBoundaryVolumeKernel} for paper provenance.
   *
   * Always runs a one-shot seed computation now (on committed
   * positions) so `boundaryVolume` is valid before the first substep.
   * Rebuilds the hash grid inline for the seed pass. Call after
   * uploading the boundary particles into their slots.
   *
   * `options.dynamic` (default `true`): when `true`, additionally wires
   * a per-substep kernel into {@link FluidSystem.preIterKernels} that
   * recomputes `V_i` from `predictedPositions` at the start of each
   * substep — the paper's prescription for moving / deforming boundaries
   * (Akinci 2012 §2.2 last paragraph). Required for soft-body, rigid-in-
   * proximity, and cloth boundary ranges.
   *
   * Set `{dynamic: false}` ONLY for confirmed-static ranges: the pool
   * container walls in Phase 17, an immovable rigid prop, a scripted
   * scene where the boundary never moves. Static-for-movers was the
   * previous (incorrect) MVP default; see U-17 reopening notes.
   *
   * Call-order requirement for dynamic ranges: `registerBoundaryParticles
   * ({dynamic: true})` must run BEFORE the owning `SimLoop` is
   * constructed. `SimLoop` snapshots `preIterKernels` AND `perIterKernels`
   * at its own construction (see `src/core/loop.ts`); late
   * registrations produce a valid seed but their per-substep dynamic
   * kernel and the Phase 11 fluid → solid Newton-3 reaction kernels
   * are not dispatched.
   *
   * Phase 11 (2026-04-27): the *first* call additionally lazy-allocates
   * {@link FluidSystem.solidReactionAccumulator} and wires three
   * kernels into the per-substep pipeline:
   *   - reset (pre-iter, once per substep) into `#baselinePreIterKernels`
   *   - scatter (per-iter, after `applyDelta`) into `#solidReactionKernels`
   *   - apply (per-iter, after scatter) into `#solidReactionKernels`
   * Subsequent calls reuse the same accumulator and kernels — only the
   * dynamic boundary-volume kernel list grows. Pure-fluid scenes never
   * call this method and pay nothing.
   */
  async registerBoundaryParticles(
    range: ParticleRange,
    options?: { readonly dynamic?: boolean },
  ): Promise<void> {
    if (
      !Number.isInteger(range.start) ||
      !Number.isInteger(range.count) ||
      range.start < 0 ||
      range.count <= 0 ||
      range.start + range.count > this.particles.capacity
    ) {
      throw new Error(
        `FluidSystem.registerBoundaryParticles: invalid range start=${range.start} count=${range.count} capacity=${this.particles.capacity}`,
      );
    }

    const dynamic = options?.dynamic ?? true;

    const seedKernel = buildBoundaryVolumeKernel({
      particles: this.particles,
      hashGrid: this.hashGrid,
      sph: this.sph,
      boundaryParticles: range,
      positionSource: 'committed',
    });
    await this.particles.renderer.computeAsync([...this.hashGrid.rebuildPipeline, seedKernel]);

    if (dynamic) {
      const perSubstepKernel = buildBoundaryVolumeKernel({
        particles: this.particles,
        hashGrid: this.hashGrid,
        sph: this.sph,
        boundaryParticles: range,
        positionSource: 'predicted',
      });
      this.#dynamicBoundaryKernels.push(perSubstepKernel);
    }

    // Phase 11 — fluid → solid Newton-3 reaction wiring. Lazy-allocate
    // on the first registration so pure-fluid scenes (which never call
    // this method) pay nothing. See `solidReaction.ts` for the per-pair
    // formula and paper provenance.
    if (this.solidReactionAccumulator === undefined) {
      const accumulator = new ContactAccumulator(
        this.particles,
        deriveAccumulatorScale(DEFAULT_SOLID_REACTION_MAX_CORRECTION_M),
      );
      this.solidReactionAccumulator = accumulator;

      // Reset goes into preIterKernels (once per substep, before any
      // per-iter scatter writes into the accumulator). Same convention
      // as the SimLoop-owned contact accumulator's reset placement.
      this.#baselinePreIterKernels.push(
        buildResetAccumulatorKernel(accumulator),
        buildResetOverflowFlagKernel(accumulator),
      );

      //
      // The legacy unfused `buildSolidReactionScatterKernel` stays
      // exported for callers that explicitly want the unfused form
      // (e.g. tests asserting kernel-level equivalence). Production
      // here uses the fused kernel.
      const coupledDeltaKernel = buildCoupledDeltaKernel({
        particles: this.particles,
        sph: this.sph,
        restDensity: this.restDensityUniform,
        lambda: this.lambda,
        deltaX: this.deltaX,
        contactAccumulator: accumulator,
        fluidParticles: this.fluidParticles,
        pairList: this.pairList,
        pairCount: this.pairCount,
      });
      // Swap the per-iter chain's `positionDelta` for the fused
      // `coupledDelta`. `#baselinePerIterKernels` was constructed as
      // [lambda, positionDelta, applyDelta]; entry [1] is positionDelta.
      // The swap is idempotent across multiple `registerBoundaryParticles`
      // calls because the second-and-later calls find an already-fused
      // kernel and skip — see the outer `solidReactionAccumulator`
      // existence guard.
      this.#baselinePerIterKernels[1] = coupledDeltaKernel;

      // Apply (accumulator drain) runs per iter — appended after
      // `applyDelta` by the `perIterKernels` getter so it commits
      // boundary deltas after the fluid's own Δp lands.
      this.#solidReactionKernels.push(buildApplyAccumulatorToPredictedKernel(accumulator));
    }
  }

  /**
   * Phase 14c — lazy-allocate the Yu & Turk 2010 §4 anisotropy storage
   * (`anisotropyDiag`, `anisotropyOff`, `smoothedPositions`). Idempotent;
   * subsequent calls are a no-op. Typically called once by
   * `FluidSurfaceRenderer`'s constructor when the renderer wires up its
   * anisotropy compute pass; pure-sim configurations never call this
   * and pay nothing.
   *
   * Initial values: zero-initialised by `instancedArray`. The compute
   * kernel writes valid values on the first render frame (before the
   * depth pass reads). The depth-pass ellipsoid path falls back to
   * spherical when `enabled = false` so this seeding latency is
   * invisible to the production rendering path.
   */
  enableAnisotropyBuffers(): void {
    if (this.anisotropyDiag !== undefined) return;
    // vec4 (not vec3) so the buffer's JS-side stride matches its
    // WGSL-side `array<vec3<f32>>` 16-byte alignment. With 'vec3'
    // three.js's `StorageInstancedBufferAttribute` allocates a packed
    // 12-byte stride that mismatches the GPU's vec3 layout, producing
    // misaligned reads. The .w lane carries no anisotropy data; the
    // depth pass and CPU readbacks index .xyz only.
    this.anisotropyDiag = instancedArray(this.particles.capacity, 'vec4');
    this.anisotropyOff = instancedArray(this.particles.capacity, 'vec4');
    this.smoothedPositions = instancedArray(this.particles.capacity, 'vec4');
    this.anisotropyDiagnostic = instancedArray(this.particles.capacity, 'vec4');
  }

  /**
   * Readback the per-substep pair-list overflow flag. Returns `true`
   * if any fluid particle had more than `MAX_NEIGHBORS` within-`h`
   * neighbors during the last build pass and its pair list was
   * truncated. A truncated pair list undercounts density (and
   * therefore underestimates `λ` and overshoots `Δx`), which can
   * manifest as visible jitter or instability in dense regions.
   *
   * Async (forces a CPU↔GPU readback) — not free; poll periodically
   * rather than every frame. The build kernel itself resets the
   * flag to 0 at the start of every dispatch, so this readback
   * reflects the most-recent substep's state.
   */
  async readbackPairOverflow(): Promise<boolean> {
    const buf = await this.particles.renderer.getArrayBufferAsync(this.pairOverflowFlag.value);
    return new Uint32Array(buf)[0] !== 0;
  }
}
