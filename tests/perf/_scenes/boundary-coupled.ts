// Phase Perf-11 — boundary-coupled regression scenes.
//
// Three production scenes for fluid + boundary-coupled bodies, sharing
// the same fluid setup (column-pack falling onto a plane floor) at the
// same `fluidCount`:
//
//   - `fluid-static-boundary-Nk`: 1 static range (`{dynamic:false}`) →
//                                 Phase 11 lazy-alloc fires, no per-substep
//                                 boundary-volume recompute.
//   - `fluid-rigid-only-Nk`    : 1 dynamic range + RigidBodySystem with
//                                 3 cubes (~72 surface boundary particles).
//   - `fluid-dense-boundary-Nk`: 1 dynamic range, dense boundary slab
//                                 (625 particles, mirrors the production
//                                 wet-cloth scene's 24×24 quad cloth).
//                                 High totalCount × maxContacts × per-
//                                 substep boundary-volume recompute. The
//                                 regression gate for dispatch-count-
//                                 reducing perf phases — required by the
//                                 phase-perf-17 abort lesson
//                                 (`memory/phase_perf_17_aborted_2026_05_05.md`)
//                                 because the existing static + rigid-only
//                                 scenes did not catch the wet-cloth 3×
//                                 regression.
//

import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createXpbdUniforms,
  type ParticleInit,
  type ParticleRange,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';
import {
  RigidBodySystem,
  voxelize,
  type RigidBodyDef,
  type TriangleMesh,
} from '../../../src/softbody/index.js';

import type { PerfRenderer } from '../_helpers/PerfRenderer.js';
import type { PerfKernelSpec, PerfSceneSpec } from '../_helpers/PerfRunner.js';
import { fluidColumn, hashGridKernelSpecs } from './_helpers.js';

const SPACING = 0.025;
const H = 0.04;
const R = SPACING * 0.5;
const REST_DENSITY = 1000;
const SUBSTEPS = 3;
const ITERATIONS = 2;

const PHASE_FLUID = 1;
const PHASE_BOUNDARY = 2;
const phaseFor = (id: number): number => ((id & 0xffff) << 16) >>> 0;

export interface BuiltScene {
  readonly spec: PerfSceneSpec;
  readonly dispose: () => void;
}

/**
 * `boundaryMode`:
 *   - 'static'       : 1 range, `{dynamic:false}`. Phase 11 lazy-alloc
 *                      fires; no per-substep recompute.
 *   - 'rigid-only'   : 1 dynamic range backed by 3 RigidBody cubes
 *                      (~72 surface boundary particles).
 *   - 'dense-dynamic': 1 dynamic range, 625-particle slab. Mirrors the
 *                      production wet-cloth scene's particle count; the
 *                      regression gate per phase-perf-17 abort lesson.
 */
type BoundaryMode = 'static' | 'rigid-only' | 'dense-dynamic';

interface BuildArgs {
  readonly id: string;
  readonly fluidCount: number;
  readonly boundaryMode: BoundaryMode;
}

const BOUNDARY_TOTAL = 600;

/**
 * Particle count for `dense-dynamic` mode. 625 = 25×25, mirrors the
 * production wet-cloth scene at default 24-segment width/height
 * (vertices = (24+1)² = 625). See header comment for rationale.
 */
const DENSE_BOUNDARY_TOTAL = 625;

/**
 * Flat horizontal slab of boundary particles, centered at floor-level,
 * `count` particles in a roughly-square XZ grid spaced at `SPACING`.
 * `invMass = 0` (kinematic). Phase tag PHASE_BOUNDARY ensures fluid →
 * boundary contact pairs emit (boundary self-pairs are filtered).
 */
function boundarySlab(count: number, yOffset: number): ParticleInit[] {
  const side = Math.ceil(Math.sqrt(count));
  const half = (side * SPACING) / 2;
  const out: ParticleInit[] = [];
  let n = 0;
  for (let z = 0; z < side && n < count; z++) {
    for (let x = 0; x < side && n < count; x++) {
      out.push({
        position: [-half + (x + 0.5) * SPACING, yOffset, -half + (z + 0.5) * SPACING],
        velocity: [0, 0, 0],
        invMass: 0,
        phase: phaseFor(PHASE_BOUNDARY),
      });
      n++;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Procedural unit cube (mirrors unified-proof.ts).
// ---------------------------------------------------------------------------

function unitCubeMesh(): TriangleMesh {
  const vertices = new Float32Array([
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
  ]);
  const indices = new Uint32Array([
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3,
    5, 3, 7, 5,
  ]);
  return { vertices, indices };
}

function scaledCube(edge: number): TriangleMesh {
  const m = unitCubeMesh();
  return { vertices: m.vertices.map((v) => v * edge), indices: m.indices };
}

async function buildScene(perf: PerfRenderer, args: BuildArgs): Promise<BuiltScene> {
  const { id, fluidCount, boundaryMode } = args;

  // --- Body / boundary particle accounting.
  let bodyParticles: ParticleInit[] = [];
  let bodyRanges: ParticleRange[] = [];
  type RigidSlot = { start: number; count: number };
  const rigidSlots: RigidSlot[] = [];
  let rigidVox: ReturnType<typeof voxelize> | undefined;

  const fluidOrigin: readonly [number, number, number] = [-0.6, 0.5, -0.6];
  const boundaryYOffset = 0.0; // floor-aligned slab

  if (boundaryMode === 'static') {
    bodyParticles = boundarySlab(BOUNDARY_TOTAL, boundaryYOffset);
    bodyRanges = [{ start: fluidCount, count: bodyParticles.length }];
  } else if (boundaryMode === 'dense-dynamic') {
    bodyParticles = boundarySlab(DENSE_BOUNDARY_TOTAL, boundaryYOffset);
    bodyRanges = [{ start: fluidCount, count: bodyParticles.length }];
  } else if (boundaryMode === 'rigid-only') {
    rigidVox = voxelize(scaledCube(0.1), {
      particleRadius: R,
      bakeSdf: true,
      spacingFactor: 1.0,
    });
    const perBody = rigidVox.count;
    // Three rigid cubes side-by-side, well above the falling fluid column.
    const offsets: Array<readonly [number, number, number]> = [
      [-0.4, 0.1, 0],
      [0.0, 0.1, 0],
      [0.4, 0.1, 0],
    ];
    for (let b = 0; b < 3; b++) {
      const [ox, oy, oz] = offsets[b]!;
      const start = fluidCount + b * perBody;
      for (let i = 0; i < perBody; i++) {
        bodyParticles.push({
          position: [
            rigidVox.positions[3 * i + 0]! + ox,
            rigidVox.positions[3 * i + 1]! + oy,
            rigidVox.positions[3 * i + 2]! + oz,
          ],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: phaseFor(PHASE_BOUNDARY + b),
        });
      }
      rigidSlots.push({ start, count: perBody });
      bodyRanges.push({ start, count: rigidVox.surfaceCount });
    }
  }

  const totalCount = fluidCount + bodyParticles.length;

  // --- Fluid initial particles, phase-tagged.
  const fluidInits = fluidColumn({
    count: fluidCount,
    spacing: SPACING,
    origin: fluidOrigin,
    phase: phaseFor(PHASE_FLUID),
  });

  const particles = new ParticleSystem(perf.renderer, totalCount, R);
  particles.uploadParticles([...fluidInits, ...bodyParticles]);

  const hashGrid = new HashGrid(particles, { cellSize: H });

  const colliders = new PrimitiveSet(particles, { capacity: 1 });
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
  colliders.upload();

  const xpbd = createXpbdUniforms(1 / 60);
  const fluid = new FluidSystem({
    particles,
    hashGrid,
    xpbd,
    restDensity: REST_DENSITY,
    h: H,
    particleSpacing: SPACING,
    compliance: 1e-4,
    fluidParticles: { start: 0, count: fluidCount },
    vorticity: { strength: 0.1 },
    xsph: { c: 0.01 },
  });

  // --- Boundary registration BEFORE SimLoop construction.
  if (boundaryMode === 'static') {
    await fluid.registerBoundaryParticles(bodyRanges[0]!, { dynamic: false });
  } else if (boundaryMode === 'dense-dynamic') {
    await fluid.registerBoundaryParticles(bodyRanges[0]!, { dynamic: true });
  } else {
    for (const range of bodyRanges) {
      await fluid.registerBoundaryParticles(range, { dynamic: true });
    }
  }

  // --- Optional rigid bodies (rigid-only mode).
  let rigid: RigidBodySystem | undefined;
  if (boundaryMode === 'rigid-only' && rigidVox) {
    const defs: RigidBodyDef[] = rigidSlots.map((slot, i) => ({
      particleRange: { start: slot.start, count: slot.count },
      restPositions: rigidVox!.positions.slice(),
      restSDF: rigidVox!.restSDF!.slice(),
      phaseId: PHASE_BOUNDARY + i,
      compliance: 0,
    }));
    rigid = new RigidBodySystem({ particles, xpbd, bodies: defs });
  }

  // --- SimLoop. Both scenes use the contact pipeline so cross-pair
  //     contacts emit. Phase Perf-14 dispatch-shape reduction passes the
  //     body/boundary ranges (the only particles that can originate
  //     non-suppressed cross-pair contacts).
  const materials = rigid !== undefined ? [fluid, rigid] : [fluid];
  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations: ITERATIONS,
    xpbd,
    hashGrid,
    contact: {
      hashGrid,
      maxContacts: Math.max(8192, totalCount * 6),
      emittingRanges: bodyRanges,
    },
    colliders: { colliders },
    materials,
  });
  loop.gravity.set(0, -9.81, 0);

  // --- Per-kernel attribution. Both scenes expose:
  //     - predict, hash grid, fluid lambda/coupledDelta/applyDelta,
  //       solidReactionApply, advect, vorticity stack
  //     - fluid.boundaryVolume[k] (rigid-only — Akinci 2012 §2.2)
  //     - rigid.* (rigid-only)
  // FluidSystem.perIterKernels layout when boundaries are registered
  // (post-Phase-Perf-15):
  //   [lambda, coupledDelta, applyDelta, solidReactionApply]
  // The Phase 15 fusion folds the per-iter `solidReactionScatter` into
  // entry [1] (`coupledDelta`); only the apply kernel remains separate.
  const fpi = fluid.perIterKernels;
  const kernels: PerfKernelSpec[] = [
    { name: 'core.predict', kernel: loop.kernels.predict, dispatchesPerFrame: SUBSTEPS },
    ...hashGridKernelSpecs(hashGrid, SUBSTEPS),
    { name: 'fluid.lambda', kernel: fpi[0]!, dispatchesPerFrame: SUBSTEPS * ITERATIONS },
    { name: 'fluid.coupledDelta', kernel: fpi[1]!, dispatchesPerFrame: SUBSTEPS * ITERATIONS },
    { name: 'fluid.applyDelta', kernel: fpi[2]!, dispatchesPerFrame: SUBSTEPS * ITERATIONS },
    {
      name: 'fluid.solidReactionApply',
      kernel: fpi[3]!,
      dispatchesPerFrame: SUBSTEPS * ITERATIONS,
    },
    { name: 'core.advect', kernel: loop.kernels.advect, dispatchesPerFrame: SUBSTEPS },
    {
      name: 'fluid.fusedVorticityXsphWalk',
      kernel: fluid.postAdvectKernels[0]!,
      dispatchesPerFrame: SUBSTEPS,
    },
    {
      name: 'fluid.vorticityPass2',
      kernel: fluid.postAdvectKernels[1]!,
      dispatchesPerFrame: SUBSTEPS,
    },
    {
      name: 'fluid.fusedVorticityXsphApply',
      kernel: fluid.postAdvectKernels[2]!,
      dispatchesPerFrame: SUBSTEPS,
    },
  ];
  // Dynamic boundary-volume kernels (one per range, rigid-only). preIterKernels
  // composition is `[...dynamicBoundary, ...baselinePreIter]` per
  // FluidSystem.preIterKernels getter when ranges are dynamic.
  if (boundaryMode === 'rigid-only' || boundaryMode === 'dense-dynamic') {
    const numDyn = bodyRanges.length;
    for (let k = 0; k < numDyn; k++) {
      kernels.push({
        name: `fluid.boundaryVolume[${k}]`,
        kernel: fluid.preIterKernels[k]!,
        dispatchesPerFrame: SUBSTEPS,
      });
    }
  }
  if (rigid) {
    for (let k = 0; k < rigid.preIterKernels.length; k++) {
      kernels.push({
        name: `rigid.preIter[${k}]`,
        kernel: rigid.preIterKernels[k]!,
        dispatchesPerFrame: SUBSTEPS,
      });
    }
    for (let k = 0; k < rigid.perIterKernels.length; k++) {
      kernels.push({
        name: `rigid.perIter[${k}]`,
        kernel: rigid.perIterKernels[k]!,
        dispatchesPerFrame: SUBSTEPS * ITERATIONS,
      });
    }
  }

  // Phase Perf-16 — collider kernel attribution. Every scene in this
  // bench file uses an analytic plane collider for the floor.
  if (loop.colliderKernels?.primitive) {
    const c = loop.colliderKernels.primitive;
    kernels.push(
      { name: 'collider.resetLambda', kernel: c.resetLambda, dispatchesPerFrame: SUBSTEPS },
      { name: 'collider.solve', kernel: c.solve, dispatchesPerFrame: SUBSTEPS * ITERATIONS },
      {
        name: 'collider.frictionVelocity',
        kernel: c.frictionVelocity,
        dispatchesPerFrame: SUBSTEPS,
      },
    );
  }
  if (loop.colliderKernels?.sdf) {
    for (let k = 0; k < loop.colliderKernels.sdf.length; k++) {
      const s = loop.colliderKernels.sdf[k]!;
      kernels.push(
        {
          name: `collider.sdf[${k}].resetLambda`,
          kernel: s.resetLambda,
          dispatchesPerFrame: SUBSTEPS,
        },
        {
          name: `collider.sdf[${k}].solve`,
          kernel: s.solve,
          dispatchesPerFrame: SUBSTEPS * ITERATIONS,
        },
        {
          name: `collider.sdf[${k}].frictionVelocity`,
          kernel: s.frictionVelocity,
          dispatchesPerFrame: SUBSTEPS,
        },
      );
    }
  }

  // Phase Perf-12 — contact-pipeline kernel attribution.
  if (loop.contactKernels) {
    const ck = loop.contactKernels;
    const STAB_ITERS = 1; // SimLoop default
    kernels.push(
      { name: 'contact.copyInvMass', kernel: ck.copyInvMass, dispatchesPerFrame: SUBSTEPS },
      { name: 'contact.resetCounter', kernel: ck.resetCounter, dispatchesPerFrame: SUBSTEPS },
      { name: 'contact.generate', kernel: ck.generate, dispatchesPerFrame: SUBSTEPS },
      { name: 'contact.resetLambda', kernel: ck.resetLambda, dispatchesPerFrame: SUBSTEPS },
    );
    if (ck.stabilize) {
      kernels.push({
        name: 'contact.stabilize',
        kernel: ck.stabilize,
        dispatchesPerFrame: SUBSTEPS * STAB_ITERS,
      });
    }
    kernels.push(
      // applyAccumulatorToBoth runs once per stabilize + (separately) per
      // iter — the per-iter dispatch is OUTSIDE this kernel list because
      // it's appended after collider solves; the bench attribution here
      // measures the kernel's per-call cost, multiplied by the post-stab
      // dispatch count.
      {
        name: 'contact.applyAccumulatorToBoth',
        kernel: ck.applyAccumulatorToBoth,
        dispatchesPerFrame: SUBSTEPS * STAB_ITERS,
      },
      { name: 'contact.solve', kernel: ck.solve, dispatchesPerFrame: SUBSTEPS * ITERATIONS },
      {
        name: 'contact.frictionVelocity',
        kernel: ck.frictionVelocity,
        dispatchesPerFrame: SUBSTEPS,
      },
    );
  }

  const dt = 1 / 60;
  const contactPairCountReadback = loop.contacts ? () => loop.contacts!.readbackCount() : undefined;

  return {
    spec: {
      id,
      particleCount: totalCount,
      substeps: SUBSTEPS,
      iterations: ITERATIONS,
      stepFrame: () => loop.step(dt),
      kernels,
      ...(contactPairCountReadback ? { contactPairCountReadback } : {}),
    },
    dispose: () => {
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}

// ---------------------------------------------------------------------------
// Per-scene builders.
// ---------------------------------------------------------------------------

export function buildFluidStaticBoundary(perf: PerfRenderer, n: number): Promise<BuiltScene> {
  return buildScene(perf, {
    id: `fluid-static-boundary-${nLabel(n)}`,
    fluidCount: n,
    boundaryMode: 'static',
  });
}

export function buildFluidRigidOnly(perf: PerfRenderer, n: number): Promise<BuiltScene> {
  return buildScene(perf, {
    id: `fluid-rigid-only-${nLabel(n)}`,
    fluidCount: n,
    boundaryMode: 'rigid-only',
  });
}

export function buildFluidDenseBoundary(perf: PerfRenderer, n: number): Promise<BuiltScene> {
  return buildScene(perf, {
    id: `fluid-dense-boundary-${nLabel(n)}`,
    fluidCount: n,
    boundaryMode: 'dense-dynamic',
  });
}

function nLabel(n: number): string {
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
}
