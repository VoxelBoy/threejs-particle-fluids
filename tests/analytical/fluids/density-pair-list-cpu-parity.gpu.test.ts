// Phase Perf — pair-list density parity G1.
//
// End-to-end check that running the fused density+λ kernel (which
// reads from the per-substep pair list instead of walking the hash
// grid) produces the same density values as a paper-faithful CPU
// brute-force SPH sum.
//
// The pair-list correctness test (`pair-list-correctness.gpu.test.ts`)
// validates that the *list contents* match a brute-force neighbor
// search. This test extends that one step downstream: given that the
// list is correct, does iterating it through the fused kernel produce
// the same Poly6 sum as a direct SPH evaluation? If yes, the paper-
// amortization migration is end-to-end correct: the pair list IS a
// faithful substitute for the 27-cell walk in the downstream
// summation.
//
// Phase Perf-08 fused `buildDensityKernel` into `buildLambdaKernel`
// (one walk produces both `density[i]` and `lambda[i]`). The fused
// kernel always writes `density[i]` BEFORE the unilateral-clamp
// early return, so the test exercises the same density math via the
// production kernel.
//

import { describe, expect, it } from 'vitest';
import { instancedArray, uniform } from 'three/tsl';
import {
  HashGrid,
  ParticleSystem,
  allocatePairListStorage,
  buildPairListKernel,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';
import { buildLambdaKernel } from '../../../src/fluids/sim/lambda.js';
import { createSphKernelUniforms } from '../../../src/fluids/sim/kernels.js';

describe('Phase Perf — density via pair list matches CPU SPH', () => {
  it('5×5×5 lattice: GPU density matches brute-force CPU sum within tier-2 tol', async () => {
    const renderer = await createParticleRenderer();
    try {
      // 5³ = 125 particles. Same lattice geometry as the pair-list
      // correctness test: spacing 0.025, h = 0.04. Self is included
      // in the sum per paper density formula (`W_poly6(0, h) = poly6Coef
      // · h⁶`).
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
              velocity: [0, 0, 0],
              invMass: INV_MASS,
              phase: 0,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, N, SPACING * 0.5);
      particles.uploadParticles(initial);

      const grid = new HashGrid(particles, { cellSize: H });
      const sph = createSphKernelUniforms(H);
      const restDensityU = uniform(REST_DENSITY, 'float');
      // Phase Perf-08: lambda kernel needs `dt` and `compliance` to
      // compute α̃ — values are immaterial for the density side that
      // this test inspects (`density[i]` is written before the
      // unilateral-clamp branch on `Ci`).
      const dtU = uniform(1 / 60 / 4, 'float');
      const complianceU = uniform(1e-6, 'float');
      const pairStorage = allocatePairListStorage(N);
      const densityBuffer = instancedArray(N, 'float');
      const lambdaBuffer = instancedArray(N, 'float');

      const buildPairs = buildPairListKernel({
        particles,
        hashGrid: grid,
        hSq: sph.hSq,
        fluidParticles: { start: 0, count: N },
        ...pairStorage,
      });
      const fusedDensityLambdaKernel = buildLambdaKernel({
        particles,
        sph,
        restDensity: restDensityU,
        dt: dtU,
        compliance: complianceU,
        density: densityBuffer,
        lambda: lambdaBuffer,
        fluidParticles: { start: 0, count: N },
        pairList: pairStorage.pairList,
        pairCount: pairStorage.pairCount,
      });

      await renderer.computeAsync([...grid.rebuildPipeline, buildPairs, fusedDensityLambdaKernel]);

      const densityBuf = await renderer.getArrayBufferAsync(densityBuffer.value);
      const gpuDensity = new Float32Array(densityBuf);

      // CPU brute-force ground truth — paper-faithful SPH sum.
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

      // Tier-2 G4 tolerance: f32 sum order can vary, so compare with
      // the per-kernel ULP envelope `N_neighbors · max_term · 1e-7`.
      // Worst-case neighbor count for a 5³ lattice with h ≈ 1.6·spacing
      // is ~33 (centered particle). Max term magnitude is `M · poly6
      // · h⁶` ≈ ρ_0 · spacing³ · (315/64π) · 1/h³ — roughly 600 kg/m³
      // for our parameters. Envelope ≈ 33 · 600 · 1e-7 ≈ 2e-3 kg/m³.
      // Use a slightly slack 5e-3 tolerance.
      let maxAbsDelta = 0;
      let maxRelDelta = 0;
      for (let i = 0; i < N; i++) {
        const absD = Math.abs(gpuDensity[i]! - cpuDensity[i]!);
        const relD = absD / Math.max(cpuDensity[i]!, 1);
        if (absD > maxAbsDelta) maxAbsDelta = absD;
        if (relD > maxRelDelta) maxRelDelta = relD;
      }
      // eslint-disable-next-line no-console
      console.log(
        '[density-pair-list-cpu fused] max|Δρ|=' +
          maxAbsDelta.toExponential(3) +
          ' max relΔρ=' +
          maxRelDelta.toExponential(3),
      );

      // Tier-2 bounds: f32 ULP envelope per particle, ~5e-3 absolute.
      expect(maxAbsDelta).toBeLessThan(5e-3);
      // Relative: max|Δρ|/ρ < 1e-5 (ULP-class).
      expect(maxRelDelta).toBeLessThan(1e-5);

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
