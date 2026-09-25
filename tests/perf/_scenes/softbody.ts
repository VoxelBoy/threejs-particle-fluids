// Phase Perf — `softbody` scene type. NUM_BODIES × 1000-particle cubes
// (10 × 10 × 10 packing) drop onto an analytic plane floor with the
// contact pipeline enabled. Exports `buildSoftbody10kScene` (10 bodies)
// and `buildSoftbody100kScene` (100 bodies).
//
// Scope note: contact kernels are not individually exposed at Phase Perf
// landing time.

import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { SoftbodySystem, type SoftbodyDef } from '../../../src/softbody/index.js';

import { PerfRenderer } from '../_helpers/PerfRenderer.js';
import type { PerfKernelSpec, PerfSceneSpec } from '../_helpers/PerfRunner.js';
import { hashGridKernelSpecs } from './_helpers.js';

const PARTICLES_PER_BODY = 1000; // 10 × 10 × 10
const PARTICLE_RADIUS = 0.01;
const SPACING = PARTICLE_RADIUS * 2;
const SIDE = 10;
const SUBSTEPS = 4;
const ITERATIONS = 2;
// Contact cell size ≥ 2·radius·radiusExpansion. Use 4·radius to be safe.
const CELL_SIZE = PARTICLE_RADIUS * 4;

export interface BuiltScene {
  readonly spec: PerfSceneSpec;
  readonly dispose: () => void;
}

/**
 * Cubic-grid rest configuration for one body, centered at `origin`.
 * 10 × 10 × 10 = 1000 particles per body. Surface particles come first
 * — that's the SoftbodySystem invariant.
 */
function buildBody(
  bodyIndex: number,
  origin: Vector3,
): {
  readonly initial: ParticleInit[];
  readonly restPositions: Float32Array;
  readonly surfaceFlag: Uint8Array;
  readonly surfaceCount: number;
} {
  const cells: Array<{
    readonly position: readonly [number, number, number];
    readonly isSurface: boolean;
  }> = [];
  for (let z = 0; z < SIDE; z++) {
    for (let y = 0; y < SIDE; y++) {
      for (let x = 0; x < SIDE; x++) {
        const isSurface =
          x === 0 || x === SIDE - 1 || y === 0 || y === SIDE - 1 || z === 0 || z === SIDE - 1;
        cells.push({
          position: [
            origin.x + (x - (SIDE - 1) / 2) * SPACING,
            origin.y + (y - (SIDE - 1) / 2) * SPACING,
            origin.z + (z - (SIDE - 1) / 2) * SPACING,
          ],
          isSurface,
        });
      }
    }
  }
  cells.sort((a, b) => Number(b.isSurface) - Number(a.isSurface));

  const restPositions = new Float32Array(cells.length * 3);
  const surfaceFlag = new Uint8Array(cells.length);
  const initial: ParticleInit[] = [];
  let surfaceCount = 0;
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i]!;
    restPositions[i * 3] = c.position[0];
    restPositions[i * 3 + 1] = c.position[1];
    restPositions[i * 3 + 2] = c.position[2];
    surfaceFlag[i] = c.isSurface ? 1 : 0;
    if (c.isSurface) surfaceCount++;
    initial.push({
      position: c.position,
      velocity: [0, 0, 0],
      invMass: 1,
      phase: bodyIndex,
    });
  }
  return { initial, restPositions, surfaceFlag, surfaceCount };
}

async function buildSoftbodyScene(
  perf: PerfRenderer,
  id: string,
  numBodies: number,
): Promise<BuiltScene> {
  const total = numBodies * PARTICLES_PER_BODY;
  const particles = new ParticleSystem(perf.renderer, total, PARTICLE_RADIUS);

  const allInitial: ParticleInit[] = [];
  const bodies: SoftbodyDef[] = [];
  const bodySpacing = SIDE * SPACING * 2;
  const gridSide = Math.ceil(Math.sqrt(numBodies));
  for (let b = 0; b < numBodies; b++) {
    const gx = b % gridSide;
    const gz = Math.floor(b / gridSide);
    const origin = new Vector3(
      (gx - gridSide / 2) * bodySpacing,
      1.0 + ((b * 13) % 7) * 0.05,
      (gz - gridSide / 2) * bodySpacing,
    );
    const body = buildBody(b, origin);
    bodies.push({
      particleRange: { start: b * PARTICLES_PER_BODY, count: PARTICLES_PER_BODY },
      restPositions: body.restPositions,
      surfaceFlag: body.surfaceFlag,
      phaseId: b,
      matchCompliance: 1e-7,
    });
    for (const p of body.initial) allInitial.push(p);
  }
  particles.uploadParticles(allInitial);

  const hashGrid = new HashGrid(particles, { cellSize: CELL_SIZE });

  const colliders = new PrimitiveSet(particles, { capacity: 1 });
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
  colliders.upload();

  const xpbd = createXpbdUniforms(1 / 60);
  const softbody = new SoftbodySystem({ particles, xpbd, bodies });

  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations: ITERATIONS,
    xpbd,
    hashGrid,
    contact: { hashGrid, maxContacts: total * 8 },
    colliders: { colliders },
    materials: [softbody],
  });
  loop.gravity.set(0, -9.81, 0);

  const kernels: PerfKernelSpec[] = [
    { name: 'core.predict', kernel: loop.kernels.predict, dispatchesPerFrame: SUBSTEPS },
    ...hashGridKernelSpecs(hashGrid, SUBSTEPS),
    {
      name: 'softbody.resetLambda',
      kernel: softbody.preIterKernels[0]!,
      dispatchesPerFrame: SUBSTEPS,
    },
    {
      name: 'softbody.centerOfMass',
      kernel: softbody.preIterKernels[1]!,
      dispatchesPerFrame: SUBSTEPS,
    },
    {
      name: 'softbody.momentAndPolarDecomp',
      kernel: softbody.preIterKernels[2]!,
      dispatchesPerFrame: SUBSTEPS,
    },
    {
      name: 'softbody.shapeMatchApply',
      kernel: softbody.perIterKernels[0]!,
      dispatchesPerFrame: SUBSTEPS * ITERATIONS,
    },
    { name: 'core.advect', kernel: loop.kernels.advect, dispatchesPerFrame: SUBSTEPS },
  ];

  const dt = 1 / 60;
  return {
    spec: {
      id,
      particleCount: total,
      substeps: SUBSTEPS,
      iterations: ITERATIONS,
      stepFrame: () => loop.step(dt),
      kernels,
    },
    dispose: () => {
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}

export function buildSoftbody10kScene(perf: PerfRenderer): Promise<BuiltScene> {
  return buildSoftbodyScene(perf, 'softbody-10k', 10);
}

export function buildSoftbody100kScene(perf: PerfRenderer): Promise<BuiltScene> {
  return buildSoftbodyScene(perf, 'softbody-100k', 100);
}
