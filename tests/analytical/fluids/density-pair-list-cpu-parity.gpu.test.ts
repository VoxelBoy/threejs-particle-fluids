// Density through the neighbor list matches a CPU SPH sum.
//
// End-to-end check that the fused density+λ kernel (which reads the
// per-substep neighbor list instead of walking the hash grid) produces
// the same density values as a brute-force CPU SPH sum.
//
// The neighbor-list correctness test (`pair-list-correctness.gpu.test.ts`)
// checks that the list contents match a brute-force neighbor search.
// This test goes one step downstream: given a correct list, does the
// lambda kernel produce the same Poly6 sum as a direct SPH evaluation?
// If yes, the neighbor list is a faithful substitute for the 27-cell
// walk in the density summation.
//
// Density and λ come out of one neighbor walk in `buildLambdaKernel`.
// The kernel writes `density[i]` before the unilateral-clamp early
// return, so reading the density buffer exercises the production math.
//

import { describe, expect, it } from 'vitest';
import { instancedArray, uniform } from 'three/tsl';
import {
  HashGrid,
  NeighborList,
  ParticleSystem,
  createParticleRenderer,
  createSphKernelUniforms,
  type ParticleInit,
} from '../../../src/index.js';
import { buildLambdaKernel } from '../../../src/fluids/sim/lambda.js';
import type { FluidKernelContext } from '../../../src/fluids/sim/shared.js';

describe('density via neighbor list matches CPU SPH', () => {
  it('5×5×5 lattice: GPU density matches brute-force CPU sum within f32 rounding', async () => {
    const renderer = await createParticleRenderer();
    try {
      // 5³ = 125 particles. Same lattice geometry as the neighbor-list
      // correctness test: spacing 0.025, h = 0.04. Self is included
      // in the sum per the SPH density formula (`W_poly6(0, h) =
      // poly6Coef · h⁶`).
      const LATTICE = 5;
      const SPACING = 0.025;
      const H = 0.04;
      const REST_DENSITY = 1000;
      const N = LATTICE ** 3;

      // Per-particle mass derived as in `FluidSystem`:
      // `m = ρ_0 · spacing³`. The kernel's Akinci ψ-branch returns
      // `1/invMass = m` for fluid particles (no boundary in this
      // test), so `ψ = m` for every pair.
      const M = REST_DENSITY * SPACING * SPACING * SPACING;
      const INV_MASS = 1 / M;

      const initial: ParticleInit[] = [];
      for (let z = 0; z < LATTICE; z++) {
        for (let y = 0; y < LATTICE; y++) {
          for (let x = 0; x < LATTICE; x++) {
            initial.push({
              position: [x * SPACING, y * SPACING, z * SPACING],
              invMass: INV_MASS,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, N, SPACING * 0.5);
      particles.uploadParticles(initial);

      const range = { start: 0, count: N };
      const grid = new HashGrid(particles, { cellSize: H });
      const sph = createSphKernelUniforms(H);
      const neighbors = new NeighborList(particles, range);
      // `dt` and compliance only enter λ's α̃ term; they are immaterial
      // for the density this test inspects.
      const context: FluidKernelContext = {
        particles,
        range,
        neighbors,
        sph,
        restDensity: uniform(REST_DENSITY, 'float'),
        particleVolume: uniform(SPACING ** 3, 'float'),
        mass: uniform(M, 'float'),
        dt: uniform(1 / 60 / 4, 'float'),
      };
      const densityBuffer = instancedArray(N, 'float');
      const lambdaBuffer = instancedArray(N, 'float');
      const lambdaKernel = buildLambdaKernel(context, {
        compliance: uniform(1e-6, 'float'),
        density: densityBuffer,
        lambda: lambdaBuffer,
      });

      await renderer.computeAsync([
        ...grid.rebuildPipeline,
        ...neighbors.buildKernels(grid, sph.hSq),
        lambdaKernel,
      ]);

      const densityBuf = await renderer.getArrayBufferAsync(densityBuffer.value);
      const gpuDensity = new Float32Array(densityBuf);

      // CPU brute-force ground truth — direct SPH sum.
      // ρ_i = Σ_j m_j · W_poly6(|x_i − x_j|, h)
      // W_poly6(r, h) = (315 / (64 π h⁹)) · max(h² − r², 0)³
      const POLY6 = 315 / (64 * Math.PI * Math.pow(H, 9));
      const H_SQ = H * H;
      const cpuDensity = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        const xi = initial[i]!.position;
        let rho = 0;
        for (let j = 0; j < N; j++) {
          const xj = initial[j]!.position;
          const dx = xi[0] - xj[0];
          const dy = xi[1] - xj[1];
          const dz = xi[2] - xj[2];
          const rSq = dx * dx + dy * dy + dz * dz;
          if (rSq < H_SQ) {
            const d = H_SQ - rSq;
            rho += M * d * d * d * POLY6;
          }
        }
        cpuDensity[i] = rho;
      }

      // f32 summation order can vary, so compare against the rounding
      // envelope `N_neighbors · max_term · 1e-7`. Worst-case neighbor
      // count for a 5³ lattice with h ≈ 1.6·spacing is ~33 (centered
      // particle). Max term magnitude is `M · poly6 · h⁶` ≈ ρ_0 ·
      // spacing³ · (315/64π) · 1/h³ — roughly 600 kg/m³ for these
      // parameters. Envelope ≈ 33 · 600 · 1e-7 ≈ 2e-3 kg/m³. Use a
      // slightly slack 5e-3 tolerance.
      let maxAbsDelta = 0;
      let maxRelDelta = 0;
      for (let i = 0; i < N; i++) {
        const absD = Math.abs(gpuDensity[i]! - cpuDensity[i]!);
        const relD = absD / Math.max(cpuDensity[i]!, 1);
        if (absD > maxAbsDelta) maxAbsDelta = absD;
        if (relD > maxRelDelta) maxRelDelta = relD;
      }
      console.log(
        '[density-pair-list-cpu] max|Δρ|=' +
          maxAbsDelta.toExponential(3) +
          ' max relΔρ=' +
          maxRelDelta.toExponential(3),
      );

      // f32 rounding envelope per particle, ~5e-3 absolute.
      expect(maxAbsDelta).toBeLessThan(5e-3);
      // Relative: max|Δρ|/ρ < 1e-5 (rounding-level).
      expect(maxRelDelta).toBeLessThan(1e-5);

      grid.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
