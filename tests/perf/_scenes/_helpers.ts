// Phase Perf — shared scene utilities. Initial-state generators and
// kernel naming conventions.

import type { HashGrid, ParticleInit } from '../../../src/core/index.js';

import type { PerfKernelSpec } from '../_helpers/PerfRunner.js';

/**
 * Packed cubic column of particles, gravity-falling onto a plane floor.
 * `count` is the total particle count; the column auto-sizes to a cubic
 * grid with `spacing` between neighbors.
 *
 * Returns an array of `count` `ParticleInit`s with deterministic positions
 * and zero initial velocity. `phase` defaults to 0 (no self-collision
 * group) and `invMass` defaults to 1 (overwritten by FluidSystem at
 * construction).
 */
export function fluidColumn(args: {
  readonly count: number;
  readonly spacing: number;
  readonly origin: readonly [number, number, number];
  readonly phase?: number;
}): ParticleInit[] {
  const { count, spacing } = args;
  const phase = args.phase ?? 0;
  const [ox, oy, oz] = args.origin;
  const side = Math.ceil(Math.cbrt(count));
  const out: ParticleInit[] = [];
  let n = 0;
  for (let z = 0; z < side && n < count; z++) {
    for (let y = 0; y < side && n < count; y++) {
      for (let x = 0; x < side && n < count; x++) {
        out.push({
          position: [ox + (x + 0.5) * spacing, oy + (y + 0.5) * spacing, oz + (z + 0.5) * spacing],
          velocity: [0, 0, 0],
          invMass: 1,
          phase,
        });
        n++;
      }
    }
  }
  return out;
}

/**
 * Open-top box of kinematic boundary particles, intended as the
 * Akinci-2012 boundary set for a fluid scene. Emits floor cells +
 * four wall faces; total count = `Nxz² + 4·Nxz·Hy`. All particles
 * have `invMass = 0` (kinematic) and `velocity = [0,0,0]`.
 */
export function boxBoundary(args: {
  readonly Nxz: number;
  readonly Hy: number;
  readonly spacing: number;
  readonly center: readonly [number, number];
  readonly floorY: number;
  readonly phase?: number;
}): ParticleInit[] {
  const { Nxz, Hy, spacing, center, floorY } = args;
  const phase = args.phase ?? 0;
  const [cx, cz] = center;
  const half = (Nxz * spacing) / 2;
  const out: ParticleInit[] = [];

  for (let z = 0; z < Nxz; z++) {
    for (let x = 0; x < Nxz; x++) {
      out.push({
        position: [cx - half + (x + 0.5) * spacing, floorY, cz - half + (z + 0.5) * spacing],
        velocity: [0, 0, 0],
        invMass: 0,
        phase,
      });
    }
  }

  for (let h = 0; h < Hy; h++) {
    const y = floorY + (h + 0.5) * spacing;
    for (let i = 0; i < Nxz; i++) {
      const t = -half + (i + 0.5) * spacing;
      out.push({ position: [cx + t, y, cz - half], velocity: [0, 0, 0], invMass: 0, phase });
      out.push({ position: [cx + t, y, cz + half], velocity: [0, 0, 0], invMass: 0, phase });
      out.push({ position: [cx - half, y, cz + t], velocity: [0, 0, 0], invMass: 0, phase });
      out.push({ position: [cx + half, y, cz + t], velocity: [0, 0, 0], invMass: 0, phase });
    }
  }

  return out;
}

/** Standard kernel-naming positions for the 8-stage HashGrid pipeline. */
export const HASH_GRID_KERNEL_NAMES: readonly string[] = [
  'hashGrid.resetCounts',
  'hashGrid.resetOverflowFlag',
  'hashGrid.cellIndexAndHistogram',
  'hashGrid.blockScan',
  'hashGrid.blockSumScan',
  'hashGrid.finalizeCellRanges',
  'hashGrid.scatter',
  'hashGrid.sortedPositions',
];

export function hashGridKernelSpecs(hashGrid: HashGrid, substeps: number): PerfKernelSpec[] {
  const list = hashGrid.rebuildPipeline;
  if (list.length !== HASH_GRID_KERNEL_NAMES.length) {
    throw new Error(
      `hashGridKernelSpecs: expected ${HASH_GRID_KERNEL_NAMES.length} kernels, got ${list.length}`,
    );
  }
  return HASH_GRID_KERNEL_NAMES.map((name, i) => ({
    name,
    kernel: list[i]!,
    dispatchesPerFrame: substeps,
  }));
}
