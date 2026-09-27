// `fluid-surface-tension` scenes: a column of water falls into an open-top
// box of pinned boundary particles (Akinci et al. 2012) standing on a floor
// plane. Surface tension (Akinci et al. 2013, γ = 1) and adhesion to the box
// (β = 1) are on; vorticity, viscosity, and particle contacts are off. The
// box is a static boundary, so its volumes are computed once, on the first
// step. The fluid column is wider than the box, so part of it spills over
// onto the floor.

import { Vector3 } from 'three';
import { FluidSystem, ParticleSystem, PrimitiveSet, SimLoop } from '../../../src/index.js';

import type { PerfRenderer } from '../_helpers/PerfRenderer.js';
import { FRAME_DT, boxBoundary, fluidColumn, type BuiltScene } from './_helpers.js';

const SPACING = 0.025;
const RADIUS = SPACING / 2;
const SMOOTHING_RADIUS = 0.04;
const SUBSTEPS = 4;
const ITERATIONS = 2;
const BOX_NXZ = 20;
const BOX_HEIGHT = 8;

function buildFluidSurfaceTensionScene(
  perf: PerfRenderer,
  id: string,
  fluidCount: number,
  origin: readonly [number, number, number],
): BuiltScene {
  const box = boxBoundary({
    nxz: BOX_NXZ,
    height: BOX_HEIGHT,
    spacing: SPACING,
    center: [0, 0],
    floorY: 0,
  });
  const total = fluidCount + box.length;
  const particles = new ParticleSystem(perf.renderer, total, RADIUS);
  particles.uploadParticles([
    ...fluidColumn({ count: fluidCount, spacing: SPACING, origin }),
    ...box,
  ]);

  // Catches the fluid that spills over the box. The plane sits one radius
  // down so particle centers can reach y = 0, level with the box floor.
  const floor = new PrimitiveSet(particles);
  floor.addPlane(new Vector3(0, 1, 0), new Vector3(0, -RADIUS, 0));

  const fluid = new FluidSystem(particles, {
    range: { start: 0, count: fluidCount },
    smoothingRadius: SMOOTHING_RADIUS,
    surfaceTension: 1,
    adhesion: 1,
  });
  fluid.addBoundary({ start: fluidCount, count: box.length }, { dynamic: false });

  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations: ITERATIONS,
    materials: [fluid],
    colliders: [floor],
  });

  return {
    spec: {
      id,
      particleCount: total,
      substeps: SUBSTEPS,
      iterations: ITERATIONS,
      stepFrame: () => loop.step(FRAME_DT),
    },
    dispose: () => {
      loop.dispose();
      floor.dispose();
      particles.dispose();
    },
  };
}

export function buildFluidSurfaceTension10kScene(perf: PerfRenderer): BuiltScene {
  return buildFluidSurfaceTensionScene(
    perf,
    'fluid-surface-tension-10k',
    10_000,
    [-0.25, 0.5, -0.25],
  );
}

export function buildFluidSurfaceTension100kScene(perf: PerfRenderer): BuiltScene {
  return buildFluidSurfaceTensionScene(
    perf,
    'fluid-surface-tension-100k',
    100_000,
    [-0.6, 0.5, -0.6],
  );
}
