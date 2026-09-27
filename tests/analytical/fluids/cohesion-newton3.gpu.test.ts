import { describe, expect, it } from 'vitest';
import { instancedArray, uniform } from 'three/tsl';
import {
  Accumulator,
  HashGrid,
  NeighborList,
  ParticleSystem,
  createParticleRenderer,
  createSphKernelUniforms,
  type ParticleInit,
} from '../../../src/index.js';
import {
  buildColorFieldNormalKernel,
  buildSurfaceTensionKernel,
} from '../../../src/fluids/sim/cohesion.js';
import type { FluidKernelContext } from '../../../src/fluids/sim/shared.js';

// Newton's 3rd law check on the surface-tension impulses. Akinci et al.
// 2013 (§2.3, §4) apply the cohesion + curvature force to particle pairs
// with full symmetrization, so the velocity changes of one surface-
// tension pass must sum to zero.
//
// Scene: an 8³ cube at rest, centred at the origin. The color-field
// normal and surface-tension kernels run once, exactly as in the first
// substep of a FluidSystem with surface tension (density still at its
// ρ0 seed). The surface-tension kernel scatters each pair's Δv into a
// fixed-point accumulator (+Δv on one side, −Δv on the other), so the
// per-axis tick sums over all particles must be exactly zero. A non-zero
// sum directly proves a pair's contributions don't cancel.

describe('surface tension conserves momentum', () => {
  it('Σ Δv = 0 after one surface-tension pass on a symmetric cube', async () => {
    const renderer = await createParticleRenderer();
    try {
      const spacing = 0.025;
      const h = 0.05;
      const r = spacing * 0.5;
      const restDensity = 1000;
      const gamma = 1.0;
      const mass = restDensity * spacing ** 3;
      const nx = 8;
      const ny = 8;
      const nz = 8;
      const count = nx * ny * nz;

      // Cube centred at origin — symmetric initial condition.
      // Index layout: idx = i + k·8 + j·64 (x fastest, then z, then y).
      const initial: ParticleInit[] = [];
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          for (let i = 0; i < nx; i++) {
            initial.push({
              position: [
                -((nx * spacing) / 2) + spacing * 0.5 + i * spacing,
                -((ny * spacing) / 2) + spacing * 0.5 + j * spacing,
                -((nz * spacing) / 2) + spacing * 0.5 + k * spacing,
              ],
              invMass: 1 / mass,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, count, r);
      particles.uploadParticles(initial);

      const range = { start: 0, count };
      const grid = new HashGrid(particles, { cellSize: h });
      const sph = createSphKernelUniforms(h);
      const neighbors = new NeighborList(particles, range);
      const context: FluidKernelContext = {
        particles,
        range,
        neighbors,
        sph,
        restDensity: uniform(restDensity, 'float'),
        particleVolume: uniform(spacing ** 3, 'float'),
        mass: uniform(mass, 'float'),
        dt: uniform(1 / 60, 'float'),
      };
      // Before the first pressure iteration the surface-tension kernel reads
      // the density FluidSystem seeds, ρ0.
      const density = instancedArray(new Float32Array(count).fill(restDensity), 'float');
      const normal = instancedArray(count, 'vec4');
      const impulses = new Accumulator(particles, 50);

      await renderer.computeAsync([
        ...grid.rebuildPipeline,
        ...neighbors.buildKernels(grid, sph.hSq),
        impulses.buildResetKernel(),
        buildColorFieldNormalKernel(context, normal),
        buildSurfaceTensionKernel(context, {
          gamma: uniform(gamma, 'float'),
          normal,
          density,
          accumulator: impulses,
        }),
      ]);

      const normalBuf = new Float32Array(await renderer.getArrayBufferAsync(normal.value));
      // Fixed-point Δv sums, xyz per particle, in accumulator ticks.
      const ticks = new Int32Array(await renderer.getArrayBufferAsync(impulses.delta.value));
      const dv = (idx: number): [number, number, number] => [
        ticks[3 * idx + 0]! / impulses.scale,
        ticks[3 * idx + 1]! / impulses.scale,
        ticks[3 * idx + 2]! / impulses.scale,
      ];

      // The color-field normal feeds the curvature force; it must be finite.
      let nonFiniteNormals = 0;
      for (let i = 0; i < count * 4; i++) {
        if (!Number.isFinite(normalBuf[i]!)) nonFiniteNormals++;
      }
      expect(nonFiniteNormals).toBe(0);

      // Σ Δv per axis, summed exactly in integer ticks.
      const tickSum = [0, 0, 0];
      let maxAbs = 0;
      let nonZeroCount = 0;
      for (let i = 0; i < count; i++) {
        for (let a = 0; a < 3; a++) tickSum[a]! += ticks[3 * i + a]!;
        const m = Math.max(...dv(i).map(Math.abs));
        if (m > maxAbs) maxAbs = m;
        if (m > 1e-6) nonZeroCount++;
      }
      const fmt = (v: readonly number[]) => v.map((x) => x.toExponential(3)).join(', ');
      console.info(
        `[cohesion-newton3] Σ Δv ticks = (${tickSum.join(', ')}) | max |Δv| = ${maxAbs.toExponential(3)} m/s | non-zero: ${nonZeroCount} / ${count}`,
      );
      // Mirror pairs across the cube's centre: corner (−,−,−) vs (+,+,+),
      // and the −x vs +x face centres (i = 0 / 7, k = 3, j = 3). By the
      // cube's symmetry each pair's Δv should be (nearly) opposite.
      console.info(
        `[cohesion-newton3] Δv[0] = (${fmt(dv(0))})  Δv[511] = (${fmt(dv(511))})  ` +
          `Δv[-x face] = (${fmt(dv(3 * 8 + 3 * 64))})  Δv[+x face] = (${fmt(dv(7 + 3 * 8 + 3 * 64))})`,
      );

      // Soundness: surface tension actually acted. A silently disabled
      // kernel would satisfy Σ Δv = 0 trivially.
      expect(maxAbs).toBeGreaterThan(0);
      // Newton's 3rd law: every pair's contributions cancel exactly.
      expect(tickSum).toEqual([0, 0, 0]);

      grid.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
