// Phase 14c — Yu & Turk 2010 anisotropy kernel correctness G1.
//

//
//   1. **Uniform 3D grid → near-isotropic G_i^-1.** For the centre
//      particle of a dense 7×7×7 cube lattice (interior, N > N_ε),
//      the local neighbour distribution is symmetric across all
//      three axes, so eigenvalues of `C_i` are equal (mod fp noise),
//      eq. 15 clamp is a no-op (σ_2 = σ_3 = σ_1 = σ̄ ≥ σ_1/k_r),
//      and `G_i^-1 = h · k_s · σ̄ · I`. We assert eigenvalue ratio
//      σ_max / σ_min < 1.05 (the plan's stated tolerance).
//
//   2. **Flat sheet (thin in z) → smallest eigenvector along z.**
//      For a 9×9×3 lattice (z-extent one third of x/y extent), the
//      centre particle's covariance has σ_z < σ_xy = σ_xy. Eq. 15
//      clamp drives σ̃_3 = σ_xy / k_r (since σ_3 ≈ 0), and
//      `G_i^-1`'s smallest eigenvalue corresponds to the z-axis. We
//      assert |v_min · ẑ| > 0.9.
//
// Both tests use spacing s = h = 0.025; 2h/s = 2, neighbours-within-
// 2h ≈ 33 in a full lattice ( > N_ε = 25 ), 31 in the 3-layer sheet
// (still > N_ε). The kernel runs as a single dispatch alongside the
// hash-grid rebuild; we read back the anisotropy buffers and run a
// CPU 3×3 symmetric eigensolver (Kopp 2008 closed-form, mirroring
// the GPU kernel's algorithm so a kernel bug wouldn't be cancelled
// out by an identical CPU bug — but the eigenvalue / eigenvector
// shape assertions are robust to either implementation).

import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/FluidSystem.js';
import {
  buildAnisotropyKernel,
  createAnisotropyKernelUniforms,
} from '../../../src/fluids/render/passes/anisotropy.js';

interface SymmetricEig3 {
  /** Eigenvalues sorted descending: σ_1 ≥ σ_2 ≥ σ_3. */
  readonly sigma: readonly [number, number, number];
  /** Eigenvectors as rows; v_k corresponds to σ_k. */
  readonly v: readonly [
    readonly [number, number, number],
    readonly [number, number, number],
    readonly [number, number, number],
  ];
}

/**
 * CPU 3×3 symmetric eigendecomposition (Kopp 2008 closed-form,
 * mirroring the GPU kernel's algorithm). For test verification only;
 * eigenvector ordering is descending by eigenvalue.
 */
function eigSymmetric3(
  c00: number,
  c11: number,
  c22: number,
  c01: number,
  c02: number,
  c12: number,
): SymmetricEig3 {
  const trC = c00 + c11 + c22;
  const q = trC / 3;
  const p1 = c01 * c01 + c02 * c02 + c12 * c12;
  const aq = c00 - q;
  const bq = c11 - q;
  const cq = c22 - q;
  const p2 = aq * aq + bq * bq + cq * cq + 2 * p1;
  const p = Math.sqrt(Math.max(p2 / 6, 1e-30));
  const detAmQI =
    aq * (bq * cq - c12 * c12) - c01 * (c01 * cq - c12 * c02) + c02 * (c01 * c12 - bq * c02);
  const r = detAmQI / (p * p * p * 2);
  const rClamped = Math.max(-1, Math.min(1, r));
  const phi = Math.acos(rClamped) / 3;
  const sigma1 = q + 2 * p * Math.cos(phi);
  const sigma3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  const sigma2 = trC - sigma1 - sigma3;

  const eigvec = (lambda: number): [number, number, number] => {
    const m00 = c00 - lambda;
    const m11 = c11 - lambda;
    const m22 = c22 - lambda;
    const c1: [number, number, number] = [
      c01 * c12 - c02 * m11,
      c02 * c01 - m00 * c12,
      m00 * m11 - c01 * c01,
    ];
    const c2: [number, number, number] = [
      c01 * m22 - c02 * c12,
      c02 * c02 - m00 * m22,
      m00 * c12 - c01 * c02,
    ];
    const c3: [number, number, number] = [
      m11 * m22 - c12 * c12,
      c12 * c02 - c01 * m22,
      c01 * c12 - m11 * c02,
    ];
    const m1 = c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2];
    const m2 = c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2];
    const m3 = c3[0] * c3[0] + c3[1] * c3[1] + c3[2] * c3[2];
    const chosen = m1 >= m2 && m1 >= m3 ? c1 : m2 >= m3 ? c2 : c3;
    const norm = Math.sqrt(
      Math.max(chosen[0] * chosen[0] + chosen[1] * chosen[1] + chosen[2] * chosen[2], 1e-30),
    );
    return [chosen[0] / norm, chosen[1] / norm, chosen[2] / norm];
  };

  const v1 = eigvec(sigma1);
  const v3 = eigvec(sigma3);
  const v2: [number, number, number] = [
    v3[1] * v1[2] - v3[2] * v1[1],
    v3[2] * v1[0] - v3[0] * v1[2],
    v3[0] * v1[1] - v3[1] * v1[0],
  ];
  return {
    sigma: [sigma1, sigma2, sigma3],
    v: [v1, v2, v3],
  };
}

interface AnisoBuffers {
  readonly diag: Float32Array;
  readonly off: Float32Array;
  readonly smoothed: Float32Array;
}

async function runAnisotropyOnLattice(
  shape: readonly [number, number, number],
  spacing: number,
  h: number,
): Promise<{
  readonly buffers: AnisoBuffers;
  readonly N: number;
  readonly centreIdx: number;
}> {
  const renderer = await createParticleRenderer();
  try {
    const [LX, LY, LZ] = shape;
    const N = LX * LY * LZ;
    const initial: ParticleInit[] = [];
    for (let z = 0; z < LZ; z++) {
      for (let y = 0; y < LY; y++) {
        for (let x = 0; x < LX; x++) {
          initial.push({
            position: [x * spacing, y * spacing, z * spacing],
            velocity: [0, 0, 0],
            invMass: 1,
            phase: 0,
          });
        }
      }
    }
    const centreIdx = Math.floor(LZ / 2) * (LX * LY) + Math.floor(LY / 2) * LX + Math.floor(LX / 2);
    const particles = new ParticleSystem(renderer, N, spacing * 0.5);
    particles.uploadParticles(initial);

    const grid = new HashGrid(particles, { cellSize: h });
    const xpbd = createXpbdUniforms(1 / 60);
    const fluid = new FluidSystem({
      particles,
      hashGrid: grid,
      xpbd,
      restDensity: 1000,
      h,
      particleSpacing: spacing,
      compliance: 1e-4,
      fluidParticles: { start: 0, count: N },
    });
    fluid.enableAnisotropyBuffers();

    const aniso = createAnisotropyKernelUniforms({
      kr: 4,
      ks: 1, // tests verify shape, not absolute scale; k_s = 1 is fine.
      kn: 0.5,
      nEpsilon: 25,
      lambda: 0.95,
    });
    const kernel = buildAnisotropyKernel({ fluidSystem: fluid, aniso });

    await renderer.computeAsync([...grid.rebuildPipeline, kernel]);

    const [diagBuf, offBuf, smoothedBuf] = await Promise.all([
      renderer.getArrayBufferAsync(fluid.anisotropyDiag!.value),
      renderer.getArrayBufferAsync(fluid.anisotropyOff!.value),
      renderer.getArrayBufferAsync(fluid.smoothedPositions!.value),
    ]);
    return {
      buffers: {
        diag: new Float32Array(diagBuf),
        off: new Float32Array(offBuf),
        smoothed: new Float32Array(smoothedBuf),
      },
      N,
      centreIdx,
    };
  } finally {
    renderer.dispose();
  }
}

/**
 * Read the symmetric 3×3 G_i^-1 for particle `idx` from the diag/off
 * vec3 buffers. WebGPU pads vec3 to 16 bytes, so each particle has
 * stride 4 floats; the .w slot is unused.
 */
function readGInv(
  buffers: AnisoBuffers,
  idx: number,
): { c00: number; c11: number; c22: number; c01: number; c02: number; c12: number } {
  const base = idx * 4;
  return {
    c00: buffers.diag[base + 0]!,
    c11: buffers.diag[base + 1]!,
    c22: buffers.diag[base + 2]!,
    c01: buffers.off[base + 0]!,
    c02: buffers.off[base + 1]!,
    c12: buffers.off[base + 2]!,
  };
}

describe('Phase 14c — Yu & Turk 2010 anisotropy kernel', () => {
  it('uniform 3D grid → near-isotropic G_i^-1 at the interior centre', async () => {
    // 7×7×7 lattice — centre particle has ~33 neighbours within r_i = 2h
    // (well above N_ε = 25), and the local distribution is symmetric
    // across all three axes.
    const SPACING = 0.025;
    const H = 0.025;
    const { buffers, centreIdx } = await runAnisotropyOnLattice([7, 7, 7], SPACING, H);
    const g = readGInv(buffers, centreIdx);
    const e = eigSymmetric3(g.c00, g.c11, g.c22, g.c01, g.c02, g.c12);
    const ratio = e.sigma[0] / Math.max(Math.abs(e.sigma[2]), 1e-20);
    // Plan's stated tolerance: eigenvalues within 5% of each other.
    expect(ratio).toBeGreaterThan(0.95);
    expect(ratio).toBeLessThan(1.05);
  });

  it('flat sheet (z-thin) → smallest eigenvector aligned with z-axis', async () => {
    // 11×11×1 lattice — single z-layer, centre at (5, 5, 0). With
    // h = 2·spacing, r_i = 4·spacing so the xy neighbourhood within
    // r_i counts ~45 lattice points (well above N_ε = 25), while the
    // z extent is identically zero (no z-neighbours). The covariance
    // C is diag(σ_xy, σ_xy, 0); eq. 15 clamps σ_3 to σ_1/k_r, and
    // G_i^-1's smallest eigenvalue is at the z-axis.
    const SPACING = 0.025;
    const H = 2 * SPACING; // r_i = 2h = 4·spacing, ~45 xy neighbours
    const { buffers, centreIdx } = await runAnisotropyOnLattice([11, 11, 1], SPACING, H);
    const g = readGInv(buffers, centreIdx);
    const e = eigSymmetric3(g.c00, g.c11, g.c22, g.c01, g.c02, g.c12);
    // Smallest eigenvalue is e.sigma[2]; its eigenvector is e.v[2].
    // The xy-plane has the two larger eigenvalues; the z-axis is the
    // smallest. We assert |v_3 · ẑ| > 0.9 (sign-ambiguous).
    const vMin = e.v[2];
    const dotZ = Math.abs(vMin[2]!);
    expect(dotZ).toBeGreaterThan(0.9);
  });
});
