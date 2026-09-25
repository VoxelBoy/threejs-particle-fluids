// Phase Perf — `fluid` scene type. PBF density + vorticity + XSPH on a
// plane floor. No contact pipeline, no surface tension, no boundary box.
// Exports `buildFluid10kScene` and `buildFluid100kScene`.

import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createXpbdUniforms,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';

import { PerfRenderer } from '../_helpers/PerfRenderer.js';
import type { PerfKernelSpec, PerfSceneSpec } from '../_helpers/PerfRunner.js';
import { fluidColumn, hashGridKernelSpecs } from './_helpers.js';

const SPACING = 0.025;
const H = 0.04;
const R = SPACING * 0.5;
const REST_DENSITY = 1000;
const SUBSTEPS = 4;
const ITERATIONS = 2;

export interface BuiltScene {
  readonly spec: PerfSceneSpec;
  readonly dispose: () => void;
}

async function buildFluidScene(
  perf: PerfRenderer,
  id: string,
  count: number,
  origin: readonly [number, number, number],
): Promise<BuiltScene> {
  const particles = new ParticleSystem(perf.renderer, count, R);
  particles.uploadParticles(fluidColumn({ count, spacing: SPACING, origin }));

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
    fluidParticles: { start: 0, count },
    vorticity: { strength: 0.1 },
    xsph: { c: 0.01 },
  });

  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations: ITERATIONS,
    xpbd,
    hashGrid,
    colliders: { colliders },
    materials: [fluid],
  });
  loop.gravity.set(0, -9.81, 0);

  const kernels: PerfKernelSpec[] = [
    { name: 'core.predict', kernel: loop.kernels.predict, dispatchesPerFrame: SUBSTEPS },
    ...hashGridKernelSpecs(hashGrid, SUBSTEPS),
    {
      name: 'fluid.lambda',
      kernel: fluid.perIterKernels[0]!,
      dispatchesPerFrame: SUBSTEPS * ITERATIONS,
    },
    {
      name: 'fluid.positionDelta',
      kernel: fluid.perIterKernels[1]!,
      dispatchesPerFrame: SUBSTEPS * ITERATIONS,
    },
    {
      name: 'fluid.applyDelta',
      kernel: fluid.perIterKernels[2]!,
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

  const dt = 1 / 60;
  return {
    spec: {
      id,
      particleCount: count,
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

export function buildFluid10kScene(perf: PerfRenderer): Promise<BuiltScene> {
  return buildFluidScene(perf, 'fluid-10k', 10_000, [-0.25, 0.5, -0.25]);
}

export function buildFluid100kScene(perf: PerfRenderer): Promise<BuiltScene> {
  return buildFluidScene(perf, 'fluid-100k', 100_000, [-0.6, 0.5, -0.6]);
}
