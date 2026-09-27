// `softbody` scenes: 10 or 100 cubes of 10 × 10 × 10 particles settle in
// stacks of two on a floor plane. Global shape matching per cube; particle
// contacts are on, so each top cube rests on the one below (a cube's own
// particles never contact each other).
//
// The scene stays inside what 4 substeps and 2 iterations resolve: cubes
// start 5 cm apart, because stiff cubes that meet much faster (a 1 m drop
// lands at over 4 m/s) push their lattices into each other and stay stuck,
// and stacks stop at two, because taller ones slowly sink into each other.

import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  SoftbodySystem,
  type ParticleInit,
  type SoftbodyDef,
} from '../../../src/index.js';

import type { PerfRenderer } from '../_helpers/PerfRenderer.js';
import { FRAME_DT, particleCube, type BuiltScene } from './_helpers.js';

const RADIUS = 0.01;
const SPACING = RADIUS * 2;
const CUBE_SIDE = 10;
const CUBE_WIDTH = CUBE_SIDE * SPACING;
const DROP = 0.05;
const SUBSTEPS = 4;
const ITERATIONS = 2;

function buildSoftbodyScene(perf: PerfRenderer, id: string, cubeCount: number): BuiltScene {
  // Stacks stand on a square grid, a cube width apart.
  const stacks = Math.ceil(cubeCount / 2);
  const gridSide = Math.ceil(Math.sqrt(stacks));
  const initial: ParticleInit[] = [];
  const bodies: SoftbodyDef[] = [];
  for (let b = 0; b < cubeCount; b++) {
    const stack = b % stacks;
    const level = Math.floor(b / stacks);
    const cube = particleCube({
      center: [
        ((stack % gridSide) - (gridSide - 1) / 2) * 2 * CUBE_WIDTH,
        CUBE_WIDTH / 2 + DROP + level * (CUBE_WIDTH + DROP),
        (Math.floor(stack / gridSide) - (gridSide - 1) / 2) * 2 * CUBE_WIDTH,
      ],
      side: CUBE_SIDE,
      spacing: SPACING,
    });
    bodies.push({
      range: { start: initial.length, count: cube.particles.length },
      surfaceCount: cube.surfaceCount,
      compliance: 1e-7,
    });
    initial.push(...cube.particles);
  }

  const particles = new ParticleSystem(perf.renderer, initial.length, RADIUS);
  particles.uploadParticles(initial);

  const floor = new PrimitiveSet(particles);
  floor.addPlane(new Vector3(0, 1, 0), new Vector3());

  // Rest shapes default to the uploaded positions.
  const softbody = new SoftbodySystem(particles, { bodies });
  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations: ITERATIONS,
    materials: [softbody],
    colliders: [floor],
    contact: true,
  });
  const contacts = loop.contacts!;

  return {
    spec: {
      id,
      particleCount: initial.length,
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

export function buildSoftbody10kScene(perf: PerfRenderer): BuiltScene {
  return buildSoftbodyScene(perf, 'softbody-10k', 10);
}

export function buildSoftbody100kScene(perf: PerfRenderer): BuiltScene {
  return buildSoftbodyScene(perf, 'softbody-100k', 100);
}
