// Phase Perf — `fluid-surface-tension` scene type. PBF density + Akinci
// 2013 surface tension (γ=1) + adhesion (β=1) into an open-top
// kinematic box. No contact pipeline, no plane collider — the box is
// the only solid surface. Exports `buildFluidSurfaceTension10kScene` and
// `buildFluidSurfaceTension100kScene`.
//
// Note: vorticity + XSPH are off here because the FluidSystem layout
// switches to the pair-list path when surfaceTension > 0. The kernel
// list reflects the pair-list-driven layout (see FluidSystem.ts).

import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';

import { PerfRenderer } from '../_helpers/PerfRenderer.js';
import type { PerfKernelSpec, PerfSceneSpec } from '../_helpers/PerfRunner.js';
import { boxBoundary, fluidColumn, hashGridKernelSpecs } from './_helpers.js';

const SPACING = 0.025;
const H = 0.04;
const R = SPACING * 0.5;
const REST_DENSITY = 1000;
const SUBSTEPS = 4;
const ITERATIONS = 2;
const BOX_NXZ = 20;
const BOX_HY = 8;
const BOUNDARY_COUNT = BOX_NXZ * BOX_NXZ + 4 * BOX_NXZ * BOX_HY; // 1040

export interface BuiltScene {
  readonly spec: PerfSceneSpec;
  readonly dispose: () => void;
}

async function buildFluidSurfaceTensionScene(
  perf: PerfRenderer,
  id: string,
  fluidCount: number,
  origin: readonly [number, number, number],
): Promise<BuiltScene> {
  const total = fluidCount + BOUNDARY_COUNT;
  const particles = new ParticleSystem(perf.renderer, total, R);

  const fluidInit = fluidColumn({ count: fluidCount, spacing: SPACING, origin });
  const boundaryInit = boxBoundary({
    Nxz: BOX_NXZ,
    Hy: BOX_HY,
    spacing: SPACING,
    center: [0, 0],
    floorY: 0,
  });
  const initial: ParticleInit[] = [...fluidInit, ...boundaryInit];
  particles.uploadParticles(initial);

  const hashGrid = new HashGrid(particles, { cellSize: H });
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
    surfaceTension: 1.0,
    adhesion: 1.0,
  });

  await fluid.registerBoundaryParticles(
    { start: fluidCount, count: BOUNDARY_COUNT },
    { dynamic: false },
  );

  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations: ITERATIONS,
    xpbd,
    hashGrid,
    materials: [fluid],
  });
  loop.gravity.set(0, -9.81, 0);

  // FluidSystem.preIterKernels layout when surfaceTension > 0 AND
  // adhesion > 0 AND boundary range is registered with dynamic: false:
  //   [0] fluid.pairListBuild
  //   [1] fluid.colorFieldNormal
  //   [2] fluid.surfaceTensionScatter
  //   [3] fluid.adhesionScatter
  //   [4] fluid.applyVelocityImpulse
  const kernels: PerfKernelSpec[] = [
    { name: 'core.predict', kernel: loop.kernels.predict, dispatchesPerFrame: SUBSTEPS },
    ...hashGridKernelSpecs(hashGrid, SUBSTEPS),
    { name: 'fluid.pairListBuild', kernel: fluid.preIterKernels[0]!, dispatchesPerFrame: SUBSTEPS },
    {
      name: 'fluid.colorFieldNormal',
      kernel: fluid.preIterKernels[1]!,
      dispatchesPerFrame: SUBSTEPS,
    },
    {
      name: 'fluid.surfaceTensionScatter',
      kernel: fluid.preIterKernels[2]!,
      dispatchesPerFrame: SUBSTEPS,
    },
    {
      name: 'fluid.adhesionScatter',
      kernel: fluid.preIterKernels[3]!,
      dispatchesPerFrame: SUBSTEPS,
    },
    {
      name: 'fluid.applyVelocityImpulse',
      kernel: fluid.preIterKernels[4]!,
      dispatchesPerFrame: SUBSTEPS,
    },
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
  ];

  const dt = 1 / 60;
  return {
    spec: {
      id,
      particleCount: fluidCount,
      substeps: SUBSTEPS,
      iterations: ITERATIONS,
      stepFrame: () => loop.step(dt),
      kernels,
    },
    dispose: () => {
      particles.destroy();
      hashGrid.destroy();
    },
  };
}

export function buildFluidSurfaceTension10kScene(perf: PerfRenderer): Promise<BuiltScene> {
  return buildFluidSurfaceTensionScene(
    perf,
    'fluid-surface-tension-10k',
    10_000,
    [-0.25, 0.5, -0.25],
  );
}

export function buildFluidSurfaceTension100kScene(perf: PerfRenderer): Promise<BuiltScene> {
  return buildFluidSurfaceTensionScene(
    perf,
    'fluid-surface-tension-100k',
    100_000,
    [-0.6, 0.5, -0.6],
  );
}
