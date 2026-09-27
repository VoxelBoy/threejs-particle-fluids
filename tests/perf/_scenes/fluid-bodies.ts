// `fluid-bodies` scenes: the `fluid` scene's water column with eight
// soft-body cubes dropped into it, the way fluids and solids are coupled in
// practice. The cubes' surfaces are dynamic fluid boundaries (Akinci et al.
// 2012), so their boundary volumes are recomputed every substep and the fluid
// pushes back on them; particle contacts are on so the cubes collide with
// each other and the water. Fluid particles never contact each other (the
// pressure solve keeps them apart), so contact pairs come from the cubes.
// About a tenth of the particles belong to the cubes.

import { Vector3 } from 'three';
import {
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  SoftbodySystem,
  type ParticleInit,
  type SoftbodyDef,
} from '../../../src/index.js';

import type { PerfRenderer } from '../_helpers/PerfRenderer.js';
import { FRAME_DT, fluidColumn, latticeSide, particleCube, type BuiltScene } from './_helpers.js';

const SPACING = 0.025;
const SMOOTHING_RADIUS = 0.04;
const REST_DENSITY = 1000;
const SUBSTEPS = 4;
const ITERATIONS = 2;
const CUBE_COUNT = 8;
const CUBE_SHARE = 0.1;
/** Cube density relative to the water, so the cubes float. */
const CUBE_DENSITY = 0.5;

function buildFluidBodiesScene(
  perf: PerfRenderer,
  id: string,
  count: number,
  origin: readonly [number, number, number],
): BuiltScene {
  const cubeSide = Math.round(Math.cbrt((count * CUBE_SHARE) / CUBE_COUNT));
  const fluidCount = count - CUBE_COUNT * cubeSide ** 3;
  const initial: ParticleInit[] = fluidColumn({ count: fluidCount, spacing: SPACING, origin });

  // One layer of 4 × 2 cubes a cube width above the column, side by side
  // with small gaps. (Cubes dropped onto each other from higher up would hit
  // too fast for 4 substeps; see softbody.ts.)
  const columnWidth = latticeSide(fluidCount) * SPACING;
  const cubeWidth = cubeSide * SPACING;
  const invMass = 1 / (CUBE_DENSITY * REST_DENSITY * SPACING ** 3);
  const bodies: SoftbodyDef[] = [];
  for (let b = 0; b < CUBE_COUNT; b++) {
    const cube = particleCube({
      center: [
        origin[0] + (columnWidth * ((b % 4) + 0.5)) / 4,
        origin[1] + columnWidth + cubeWidth,
        origin[2] + (columnWidth * (Math.floor(b / 4) + 0.5)) / 2,
      ],
      side: cubeSide,
      spacing: SPACING,
      invMass,
    });
    bodies.push({
      range: { start: initial.length, count: cube.particles.length },
      surfaceCount: cube.surfaceCount,
      compliance: 1e-6,
    });
    initial.push(...cube.particles);
  }

  const particles = new ParticleSystem(perf.renderer, count, SPACING / 2);
  particles.uploadParticles(initial);

  const floor = new PrimitiveSet(particles);
  floor.addPlane(new Vector3(0, 1, 0), new Vector3());

  const cubes = new SoftbodySystem(particles, { bodies });
  const fluid = new FluidSystem(particles, {
    range: { start: 0, count: fluidCount },
    restDensity: REST_DENSITY,
    smoothingRadius: SMOOTHING_RADIUS,
    viscosity: 0.01,
    vorticity: 0.1,
  });
  for (let b = 0; b < CUBE_COUNT; b++) fluid.addBoundary(cubes.surfaceRange(b));

  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations: ITERATIONS,
    materials: [fluid, cubes],
    colliders: [floor],
    contact: true,
  });
  const contacts = loop.contacts!;

  return {
    spec: {
      id,
      particleCount: count,
      substeps: SUBSTEPS,
      iterations: ITERATIONS,
      stepFrame: () => loop.step(FRAME_DT),
      readContactCount: () => contacts.readbackCount(),
    },
    dispose: () => {
      loop.dispose();
      floor.dispose();
      particles.dispose();
    },
  };
}

export function buildFluidBodies10kScene(perf: PerfRenderer): BuiltScene {
  return buildFluidBodiesScene(perf, 'fluid-bodies-10k', 10_000, [-0.25, 0.5, -0.25]);
}

export function buildFluidBodies100kScene(perf: PerfRenderer): BuiltScene {
  return buildFluidBodiesScene(perf, 'fluid-bodies-100k', 100_000, [-0.6, 0.5, -0.6]);
}
