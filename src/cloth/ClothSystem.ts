import { Vector3 } from 'three';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ConstraintType, Material, ParticleSystem, XpbdUniforms } from '../core/index.js';

import { createClothAeroKernel, type ClothAeroKernel } from './aero.js';
import { createClothBendingConstraints } from './bending.js';
import { createClothDistanceConstraints } from './distance.js';
import type { ClothGraph } from './graph.js';
import { createClothTetherConstraints } from './tether.js';
import { buildTethers } from './tetherBuild.js';

/**
 * Cloth material — distance + bending (Phase 18) + tethers (Kim 2012)
 * + per-triangle aerodynamic drag / lift (Keckeisen 2004), Phase 19.
 * Implements {@link Material} so it plugs into `SimLoopOptions.materials`
 * exactly like `FluidSystem` and `SoftbodySystem`. Per-cloth state
 * (constraint storage, λ buffers, group inverted indices, aero
 * topology) is owned here; particle state lives in the shared
 * {@link ParticleSystem} the caller provides.
 *
 * **Lifecycle.** The caller is responsible for:
 *   1. Building the {@link ClothGraph} via {@link "./graph.js".fromBufferGeometry}.
 *   2. Allocating a `ParticleSystem` whose capacity covers the cloth's
 *      `graph.positions.length` slots starting at `particleOffset`.
 *   3. Calling `particleSystem.uploadParticles` with the positions and
 *      inverse masses from the graph (and any per-particle `phase`
 *      tag the scene uses for self-collision masking).
 *   4. Constructing `new ClothSystem({...})` with the graph + offset.
 *
 * **Pipeline placement.** Phase 19 adds two new dispatch points:
 *
 *   - `preIterKernels` — runs once per substep, after `predict`:
 *     `[distance.resetLambda, bending.resetLambda, tether.resetLambda,
 *       aero.kernel]`. The aero kernel adds a wind-driven Δx* on top
 *     of the gravity-only prediction, so the iter-loop constraint
 *     kernels see the wind-perturbed prediction. Paper §3 / §3.1
 *     (Macklin 2014 Algorithm 1 line 2 — external forces during
 *     predict).
 *   - `perIterKernels` — runs once per iter:
 *     `[distance groups, bending groups, tether groups]`. Order
 *     follows Kim 2012 Algorithm 1: local bilateral constraints
 *     (distance + bending) project first, then unilateral LRAs.
 *     Within each constraint type, graph-coloring groups dispatch in
 *     ascending color order.
 */
export interface ClothSystemOptions {
  readonly particles: ParticleSystem;
  /**
   * Shared XPBD uniforms — must be the *same instance* the
   * `SimLoop` constructs with so the `dt` uniform broadcasts
   * correctly. `SimLoop.xpbd` is the canonical source.
   */
  readonly xpbd: XpbdUniforms;
  /**
   * Pre-built cloth graph. Construct via
   * {@link "./graph.js".fromBufferGeometry}.
   */
  readonly graph: ClothGraph;
  /**
   * Absolute slot offset of this cloth's first particle inside
   * `particles`. The cloth's `i`-th vertex lives at slot
   * `particleOffset + i`. `particleOffset + graph.positions.length`
   * must be ≤ `particles.capacity`.
   */
  readonly particleOffset: number;
  /**
   * XPBD compliance for distance (stretch) constraints (s²/kg).
   * Default `1e-7` — Phase 18 G1 drape test target ("stretch < 1 % of
   * rest length" at MVP S/I).
   */
  readonly stretchCompliance?: number;
  /**
   * XPBD compliance for bending constraints (s²/(rad²·kg)).
   * Default `1e-5` — produces visible bend resistance at MVP S/I
   * without locking to a flat reference under gravity.
   */
  readonly bendCompliance?: number;
  /**
   * XPBD compliance for tether (LRA) constraints (s²/kg). Default
   * `0` (strict inextensibility — matches Kim 2012's "infinite
   * stiffness" PBD behaviour). Non-zero allows soft over-stretch.
   */
  readonly tetherCompliance?: number;
  /**
   * Kim 2012 §3.5 "Controlled Stretchiness" — every LRA rest radius
   * is multiplied by `(1 + stretchTolerance)`. Default `0` (strict).
   * Paper Fig. 5 shows `0.1`–`0.2` produces more natural-looking
   * folds on hanging cloth.
   */
  readonly stretchTolerance?: number;
  /**
   * Max LRAs per free particle (Kim 2012 §3.4). Default `4`. Scenes
   * with fewer attachment islands than `N` produce fewer constraints
   * per particle (a flag pinned along one edge degenerates to N=1).
   */
  readonly maxAttachmentsPerTether?: number;
  /**
   * Combined aerodynamic drag scalar `0.5 · C_D · ρ` (kg/m³). Default
   * `0.6125` (`0.5 · 1.0 · 1.225` — sea-level air × flat-plate `C_D = 1`).
   * Paper §3 — `dragCoeff` and `liftCoeff` are scene-tunable
   * artist parameters.
   */
  readonly dragCoeff?: number;
  /**
   * Combined aerodynamic lift scalar `0.5 · C_L · ρ` (kg/m³).
   * Default `0.3`.
   */
  readonly liftCoeff?: number;
  /**
   * Initial wind velocity in m/s (sampled at every triangle's
   * centroid in MVP). Live-mutable through {@link ClothSystem.wind}
   * after construction. Default `(0, 0, 0)` — no wind.
   */
  readonly wind?: Vector3;
}

const DEFAULT_STRETCH_COMPLIANCE = 1e-7;
const DEFAULT_BEND_COMPLIANCE = 1e-5;
const DEFAULT_TETHER_COMPLIANCE = 0;
const DEFAULT_STRETCH_TOLERANCE = 0;
const DEFAULT_MAX_ATTACHMENTS = 4;

export class ClothSystem implements Material {
  /** The graph this cloth was constructed from. Read-only after construction. */
  readonly graph: ClothGraph;
  /** First slot of this cloth in the shared `ParticleSystem`. */
  readonly particleOffset: number;
  /** Number of cloth particles ( = `graph.positions.length`). */
  readonly nParticles: number;
  /** Distance constraint type — exposed for tests / advanced inspection. */
  readonly distance: ConstraintType;
  /**
   * Bending constraint type, or `null` if the input mesh had no
   * shared edges (e.g. a single triangle, or a fully-disconnected
   * fan). Most real cloth meshes produce a non-null bending type.
   */
  readonly bending: ConstraintType | null;
  /**
   * Tether constraint type, or `null` if the cloth has no pinned
   * vertices (no attachment islands → no LRAs to build). Always
   * non-null in practice for cloth scenes — pinning is the whole
   * point of the LRA paper.
   */
  readonly tether: ConstraintType | null;
  /** Aero kernel + uniforms (live-mutable wind / coefficients). */
  readonly aero: ClothAeroKernel;
  /** Live-mutable wind uniform — alias for {@link aero.wind}. */
  readonly wind: UniformNode<'vec3', Vector3>;
  /** Live-mutable drag coefficient — alias for {@link aero.dragCoeff}. */
  readonly dragCoeff: UniformNode<'float', number>;
  /** Live-mutable lift coefficient — alias for {@link aero.liftCoeff}. */
  readonly liftCoeff: UniformNode<'float', number>;
  /** Number of LRA constraints actually emitted by `buildTethers`. */
  readonly nTethers: number;

  readonly preIterKernels: readonly ComputeNode[];
  readonly perIterKernels: readonly ComputeNode[];
  readonly postAdvectKernels: readonly ComputeNode[] = [];

  constructor(options: ClothSystemOptions) {
    const {
      particles,
      xpbd,
      graph,
      particleOffset,
      stretchCompliance = DEFAULT_STRETCH_COMPLIANCE,
      bendCompliance = DEFAULT_BEND_COMPLIANCE,
      tetherCompliance = DEFAULT_TETHER_COMPLIANCE,
      stretchTolerance = DEFAULT_STRETCH_TOLERANCE,
      maxAttachmentsPerTether = DEFAULT_MAX_ATTACHMENTS,
      dragCoeff,
      liftCoeff,
      wind,
    } = options;

    if (!Number.isInteger(particleOffset) || particleOffset < 0) {
      throw new Error(
        `ClothSystem: particleOffset must be a non-negative integer, got ${particleOffset}`,
      );
    }
    const nParticles = graph.positions.length;
    if (particleOffset + nParticles > particles.capacity) {
      throw new Error(
        `ClothSystem: particleOffset (${particleOffset}) + cloth particle count (${nParticles}) exceeds ParticleSystem capacity (${particles.capacity})`,
      );
    }
    if (!Number.isFinite(stretchCompliance) || stretchCompliance < 0) {
      throw new Error(
        `ClothSystem: stretchCompliance must be a non-negative finite number, got ${stretchCompliance}`,
      );
    }
    if (!Number.isFinite(bendCompliance) || bendCompliance < 0) {
      throw new Error(
        `ClothSystem: bendCompliance must be a non-negative finite number, got ${bendCompliance}`,
      );
    }
    if (!Number.isFinite(tetherCompliance) || tetherCompliance < 0) {
      throw new Error(
        `ClothSystem: tetherCompliance must be a non-negative finite number, got ${tetherCompliance}`,
      );
    }

    this.graph = graph;
    this.particleOffset = particleOffset;
    this.nParticles = nParticles;

    this.distance = createClothDistanceConstraints({
      particles,
      particleOffset,
      edges: graph.distancePairs,
      restLengths: graph.distanceRestLengths,
      compliance: stretchCompliance,
      xpbd,
    });

    this.bending =
      graph.bendingTuples.length > 0
        ? createClothBendingConstraints({
            particles,
            particleOffset,
            tuples: graph.bendingTuples,
            restAngles: graph.bendingRestAngles,
            compliance: bendCompliance,
            xpbd,
          })
        : null;

    // Tethers — built from the graph's pinned set + edge topology
    // (Kim 2012 §3.2 geodesic distance + §3.4 island assignment).
    const tethers = buildTethers({
      graph,
      options: {
        maxAttachmentsPerParticle: maxAttachmentsPerTether,
        stretchTolerance,
      },
    });
    this.nTethers = tethers.length;
    this.tether =
      tethers.length > 0
        ? createClothTetherConstraints({
            particles,
            particleOffset,
            tethers,
            compliance: tetherCompliance,
            xpbd,
          })
        : null;

    // Aero — runs every substep regardless of wind magnitude (a
    // zero wind produces zero force per the |v_rel| guard, so the
    // overhead is the per-vertex incident-triangle walk only).
    this.aero = createClothAeroKernel({
      particles,
      particleOffset,
      nClothParticles: nParticles,
      triangles: graph.triangles,
      xpbd,
      ...(dragCoeff !== undefined ? { initialDragCoeff: dragCoeff } : {}),
      ...(liftCoeff !== undefined ? { initialLiftCoeff: liftCoeff } : {}),
      ...(wind !== undefined ? { initialWind: wind } : {}),
    });
    this.wind = this.aero.wind;
    this.dragCoeff = this.aero.dragCoeff;
    this.liftCoeff = this.aero.liftCoeff;

    // preIter — reset λ for each constraint type (Macklin 2016 Alg.
    // 1 line 4) then apply aero. Aero adds Δx* on top of predict's
    // gravity-only prediction, so the iter-loop constraint kernels
    // see the wind-perturbed prediction.
    const preIter: ComputeNode[] = [this.distance.resetLambdaKernel];
    if (this.bending) preIter.push(this.bending.resetLambdaKernel);
    if (this.tether) preIter.push(this.tether.resetLambdaKernel);
    preIter.push(this.aero.kernel);
    this.preIterKernels = preIter;

    // perIter — local bilateral constraints first (distance, then
    // bending), then unilateral tethers — Kim 2012 Algorithm 1 order.
    // Within each constraint type, graph-coloring groups dispatch in
    // ascending color order.
    const perIter: ComputeNode[] = [];
    for (const g of this.distance.groups) perIter.push(g.solveKernel);
    if (this.bending) {
      for (const g of this.bending.groups) perIter.push(g.solveKernel);
    }
    if (this.tether) {
      for (const g of this.tether.groups) perIter.push(g.solveKernel);
    }
    this.perIterKernels = perIter;
  }
}
