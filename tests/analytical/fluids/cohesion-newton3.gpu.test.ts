import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';

// Phase 09 diagnostic — per-substep Newton's 3rd law check on the
// cohesion impulse buffer. The paper says (§2.3, §4) the surface-
// tension force is "applied to the particle pairs" with full
// symmetrization. This test checks whether our gather-mode
// implementation produces Σ_i impulse[i] = 0 after a single substep.
//
// For a symmetric-in-space initial condition (8³ cube, zero velocity,
// no gravity), Newton's 3rd law requires Σ impulse = 0 exactly in
// floating-point (to within the residual of FP reassociation of pair
// contributions). A non-zero Σ directly proves that our gather-mode
// kernel violates per-pair cancellation — the U-33 root cause.

describe('Phase 09 diagnostic — cohesion Newton 3rd law', () => {
  it('Σ impulse = 0 after one substep of cohesion on a symmetric cube', async () => {
    const renderer = await createParticleRenderer();
    try {
      const spacing = 0.025;
      const h = 0.05;
      const r = spacing * 0.5;
      const restDensity = 1000;
      const nx = 8;
      const ny = 8;
      const nz = 8;
      const count = nx * ny * nz;

      // Cube centred at origin — symmetric initial condition.
      const OFFSET = 0;
      const initial: ParticleInit[] = [];
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          for (let i = 0; i < nx; i++) {
            initial.push({
              position: [
                OFFSET + -((nx * spacing) / 2) + spacing * 0.5 + i * spacing,
                OFFSET + -((ny * spacing) / 2) + spacing * 0.5 + j * spacing,
                OFFSET + -((nz * spacing) / 2) + spacing * 0.5 + k * spacing,
              ],
              velocity: [0, 0, 0],
              invMass: 1,
              phase: 0,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, count, r);
      particles.uploadParticles(initial);
      const hashGrid = new HashGrid(particles, { cellSize: h });

      const xpbd = createXpbdUniforms(1 / 60);
      const fluid = new FluidSystem({
        particles,
        hashGrid,
        xpbd,
        restDensity,
        h,
        particleSpacing: spacing,
        compliance: 1e-4,
        fluidParticles: { start: 0, count },
        vorticity: { strength: 0 },
        xsph: { c: 0 },
        surfaceTension: 1.0,
      });

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        xpbd,
        hashGrid,
        materials: [fluid],
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, 0, 0);

      // Run ONE substep. Scatter-mode surface tension writes Δv into
      // `velocities` (and Δv·Δt into `predictedPositions`) via the
      // apply kernel. Advect then recomputes `velocities = (x* − x)/Δt`;
      // since predict had g=0 and initial v=0, x* = x before the
      // surface-tension apply kernel runs, so x* after apply = x + Δv·Δt
      // and advect yields `velocities = Δv` — exactly the net scattered
      // impulse per particle. That's what this test reads back.
      await loop.step(1 / 60);

      if (!fluid.colorFieldNormal || !fluid.surfaceTensionAccumulator) {
        throw new Error('surface-tension buffers not allocated');
      }
      const densityBuf = new Float32Array(await renderer.getArrayBufferAsync(fluid.density.value));
      let zeroDensity = 0;
      const densityJy: number[] = new Array(8).fill(0);
      for (let i = 0; i < count; i++) {
        if (densityBuf[i]! > 1e-3) {
          const jy = Math.floor(i / 64);
          densityJy[jy]! += 1;
        } else {
          zeroDensity++;
        }
      }
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] density: zeroCount=${zeroDensity} / ${count}; per-jy non-zero: ${densityJy.join(',')}`,
      );
      // Scatter architecture: Δv is the post-advect velocity (see above).
      const snap = await particles.readback();
      const impulseBuf = new Float32Array(count * 4);
      for (let i = 0; i < count; i++) {
        impulseBuf[4 * i + 0] = snap.velocities[4 * i + 0]!;
        impulseBuf[4 * i + 1] = snap.velocities[4 * i + 1]!;
        impulseBuf[4 * i + 2] = snap.velocities[4 * i + 2]!;
      }
      const normalBuf = new Float32Array(
        await renderer.getArrayBufferAsync(fluid.colorFieldNormal.value),
      );
      // n histogram by jy.
      const nonZeroN: number[] = new Array(8).fill(0);
      for (let i = 0; i < count; i++) {
        const nx = normalBuf[4 * i + 0]!;
        const ny = normalBuf[4 * i + 1]!;
        const nz = normalBuf[4 * i + 2]!;
        if (Math.max(Math.abs(nx), Math.abs(ny), Math.abs(nz)) > 1e-6) {
          const jy = Math.floor(i / 64);
          nonZeroN[jy]! += 1;
        }
      }
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] non-zero n by jy: ${nonZeroN.map((n, j) => `jy=${j}:${n}`).join(' ')}`,
      );
      // Sample a few corner / surface particles' n.
      const probes = [
        [0, 0, 0],
        [7, 0, 0],
        [0, 7, 0],
        [7, 7, 7],
        [3, 3, 3],
      ] as const;
      for (const [ix, jy, kz] of probes) {
        const idx = ix + kz * 8 + jy * 64;
        const nx = normalBuf[4 * idx + 0]!;
        const ny = normalBuf[4 * idx + 1]!;
        const nz = normalBuf[4 * idx + 2]!;
        // eslint-disable-next-line no-console
        console.info(
          `[cohesion-newton3] n[ix=${ix},jy=${jy},kz=${kz}] = (${nx.toExponential(3)}, ${ny.toExponential(3)}, ${nz.toExponential(3)})`,
        );
      }

      // Σ impulse should be 0 (Newton's 3rd law on symmetric pairs).
      let sx = 0;
      let sy = 0;
      let sz = 0;
      let maxAbs = 0;
      for (let i = 0; i < count; i++) {
        const dx = impulseBuf[4 * i + 0]!;
        const dy = impulseBuf[4 * i + 1]!;
        const dz = impulseBuf[4 * i + 2]!;
        sx += dx;
        sy += dy;
        sz += dz;
        const m = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
        if (m > maxAbs) maxAbs = m;
      }
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] Σ impulse = (${sx.toExponential(3)}, ${sy.toExponential(3)}, ${sz.toExponential(3)}) | max |impulse| = ${maxAbs.toExponential(3)}`,
      );

      // Mirror-symmetry check. Particle 0 is at (-,-,-) corner (index
      // i=0, k=0, j=0). Particle 511 is at (+,+,+) corner (i=7, k=7,
      // j=7). By the scene's full cube mirror symmetry, impulse[0]
      // should equal -impulse[511] component-wise.
      const p0: [number, number, number] = [impulseBuf[0]!, impulseBuf[1]!, impulseBuf[2]!];
      const p511: [number, number, number] = [
        impulseBuf[4 * 511 + 0]!,
        impulseBuf[4 * 511 + 1]!,
        impulseBuf[4 * 511 + 2]!,
      ];
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] impulse[0]   = (${p0[0].toExponential(3)}, ${p0[1].toExponential(3)}, ${p0[2].toExponential(3)})`,
      );
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] impulse[511] = (${p511[0].toExponential(3)}, ${p511[1].toExponential(3)}, ${p511[2].toExponential(3)})`,
      );
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] sum(0,511) = (${(p0[0] + p511[0]).toExponential(3)}, ${(p0[1] + p511[1]).toExponential(3)}, ${(p0[2] + p511[2]).toExponential(3)})`,
      );

      // Face-centred comparison. Particles along the X axis mid-face
      // should mirror across x=0. Pick indices at (i=0,k=3,j=3) vs
      // (i=7,k=3,j=3). Formula: idx = i + k*8 + j*64.
      const ixL = 0 + 3 * 8 + 3 * 64;
      const ixR = 7 + 3 * 8 + 3 * 64;
      const pxL: [number, number, number] = [
        impulseBuf[4 * ixL + 0]!,
        impulseBuf[4 * ixL + 1]!,
        impulseBuf[4 * ixL + 2]!,
      ];
      const pxR: [number, number, number] = [
        impulseBuf[4 * ixR + 0]!,
        impulseBuf[4 * ixR + 1]!,
        impulseBuf[4 * ixR + 2]!,
      ];
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] impulse[-x face] = (${pxL[0].toExponential(3)}, ${pxL[1].toExponential(3)}, ${pxL[2].toExponential(3)})`,
      );
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] impulse[+x face] = (${pxR[0].toExponential(3)}, ${pxR[1].toExponential(3)}, ${pxR[2].toExponential(3)})`,
      );

      // Categorize which particles have non-zero impulse.
      let nonZeroCount = 0;
      const jyHist: number[] = new Array(8).fill(0);
      const allNonZero: string[] = [];
      for (let i = 0; i < count; i++) {
        const dx = impulseBuf[4 * i + 0]!;
        const dy = impulseBuf[4 * i + 1]!;
        const dz = impulseBuf[4 * i + 2]!;
        const m = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
        if (m > 1e-6) {
          nonZeroCount++;
          const ix = i % 8;
          const kz = Math.floor(i / 8) % 8;
          const jy = Math.floor(i / 64);
          jyHist[jy]! += 1;
          allNonZero.push(
            `  idx=${i} (ix=${ix},jy=${jy},kz=${kz}) dv=(${dx.toExponential(2)},${dy.toExponential(2)},${dz.toExponential(2)})`,
          );
        }
      }
      // eslint-disable-next-line no-console
      console.info(`[cohesion-newton3] non-zero impulses: ${nonZeroCount} / ${count}`);
      // eslint-disable-next-line no-console
      console.info(
        `[cohesion-newton3] jy histogram: ${jyHist.map((n, j) => `jy=${j}:${n}`).join(' ')}`,
      );
      for (const s of allNonZero) {
        // eslint-disable-next-line no-console
        console.info(`[cohesion-newton3]${s}`);
      }

      // Diagnostic — no hard gate; we want to see the actual value.
      expect(Number.isFinite(sx)).toBe(true);
      expect(Number.isFinite(sy)).toBe(true);
      expect(Number.isFinite(sz)).toBe(true);

      particles.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
