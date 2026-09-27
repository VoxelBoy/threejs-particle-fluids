import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FluidSystem,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

/*
 * Newton's 3rd law check on the fluid → solid reaction: the position
 * correction the fluid's pressure solve applies to a dynamic boundary
 * particle must balance the corrections on its fluid neighbours.
 *
 * **Why this scene chooses `m_fluid = 1`:** with equal fluid and boundary
 * masses, any mass-weighting mismatch between the fluid's correction and
 * the boundary's reaction collapses to FP noise, so the check isolates
 * the per-pair pairing itself.
 *
 * Tuning: `m_fluid = ρ_0 · spacing³`. Choosing spacing = 0.1 m and
 * ρ_0 = 1000 kg/m³ gives `m_fluid = 1 kg`. Boundary `invMass` is set
 * to 1 explicitly via `uploadParticles` (FluidSystem only overwrites
 * the fluid range).
 *
 * **Scene:**
 *   - 4³ fluid cube at rest density (64 fluid particles, spacing 0.1,
 *     centred at origin, span [−0.15, +0.15]³).
 *   - One boundary particle placed at (0.07, 0.04, 0.02) — between
 *     fluid grid points along every axis, breaks the cube's 3-fold
 *     symmetry so the reaction is non-trivial in all axes (a centred
 *     boundary would by symmetry produce zero net Δp).
 *   - Both invMass = 1 (m = 1).
 *   - Gravity off, zero initial velocity, NO particle contacts.
 *
 * **What the kernel does:**
 *   The boundary inside the fluid spikes the density of its fluid
 *   neighbours (boundary contributes `ψ_j = ρ_0 · V_j` to fluid density,
 *   in addition to the fluid neighbours). At those fluid particles
 *   `C_i = ρ_i / ρ_0 − 1 > 0`, so `λ_i < 0` (lambda kernel: λ = −C/denom).
 *   The position-delta kernel pushes those fluid particles AWAY from the
 *   boundary (Δp_i ∝ λ_i · ψ_j · ∇W; for λ_i < 0 and ∇W oriented toward
 *   the boundary, the result points away), and scatters the equal-and-
 *   opposite per-pair reaction onto the boundary.
 *
 * **What we measure:**
 *   After one substep with no other forces, `advect` computes
 *   `v = (x* − x) / dt`, so each particle's velocity is its accumulated
 *   net Δx / dt. Total system momentum `Σ m_i v_i` should be 0 within
 *   FP noise. We also assert the boundary actually moved — otherwise a
 *   silently-disabled reaction would trivially satisfy Σ = 0 at zero.
 *
 * The reaction accumulator's i32 atomicAdd is exact, but the f32
 * intermediates (the fluid's gather sum and the reaction coefficient)
 * are order-dependent, so the tolerance is `2e-5 · max(|v_i|)` per axis.
 */

describe('fluid → solid Newton-3 conservation', () => {
  it('Σ m·v ≈ 0 after one substep with one boundary inside a fluid cube', async () => {
    const renderer = await createParticleRenderer();
    try {
      // Tune so m_fluid = ρ_0 · spacing³ = 1 kg.
      const spacing = 0.1;
      const h = 2 * spacing;
      const r = spacing * 0.5;
      const restDensity = 1 / (spacing * spacing * spacing); // = 1000
      const mFluid = restDensity * spacing * spacing * spacing;
      // Self-check the tuning.
      expect(mFluid).toBeCloseTo(1, 12);

      const nx = 4;
      const ny = 4;
      const nz = 4;
      const fluidCount = nx * ny * nz;
      const boundaryCount = 1;
      const total = fluidCount + boundaryCount;

      // Fluid block centred at origin, grid points at
      // {−0.15, −0.05, +0.05, +0.15} along each axis.
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
              // invMass is overwritten by FluidSystem to 1/m_fluid = 1.
            });
          }
        }
      }
      // Boundary: between fluid grid points along every axis. Off-
      // symmetry by construction.
      initial.push({
        position: [0.07, 0.04, 0.02],
        invMass: 1, // m_boundary = 1 (matches m_fluid for strict Newton-3)
      });

      const particles = new ParticleSystem(renderer, total, r);
      particles.uploadParticles(initial);

      const fluid = new FluidSystem(particles, {
        range: { start: 0, count: fluidCount },
        restDensity,
        smoothingRadius: h,
        particleSpacing: spacing,
        // Higher compliance (softer constraint) keeps λ in a sane
        // range under the boundary's density spike. With α̃ → 0 the
        // boundary's `ψ_j W(0)` term dominates the denominator and
        // λ blows up; α = 1 keeps the test deterministic.
        compliance: 1.0,
      });

      // The boundary moves, so its volume is recomputed every substep.
      // Boundaries must be added before the SimLoop is built.
      fluid.addBoundary({ start: fluidCount, count: boundaryCount }, { dynamic: true });

      // No particle contacts — we want to observe ONLY the density-Δp
      // reaction. With contacts on, the contact solve would also act on
      // the fluid-boundary pairs.
      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        gravity: new Vector3(0, 0, 0),
        materials: [fluid],
      });

      await loop.step(1 / 60);

      const snap = await particles.readback();

      let pxSum = 0;
      let pySum = 0;
      let pzSum = 0;
      let maxAbsV = 0;
      let movedCount = 0;
      for (let i = 0; i < total; i++) {
        const w = snap.invMass[i]!;
        const m = w > 0 ? 1 / w : 0;
        const vx = snap.velocities[4 * i + 0]!;
        const vy = snap.velocities[4 * i + 1]!;
        const vz = snap.velocities[4 * i + 2]!;
        pxSum += m * vx;
        pySum += m * vy;
        pzSum += m * vz;
        const av = Math.max(Math.abs(vx), Math.abs(vy), Math.abs(vz));
        if (av > maxAbsV) maxAbsV = av;
        if (av > 1e-8) movedCount++;
      }

      const vbx = snap.velocities[4 * fluidCount + 0]!;
      const vby = snap.velocities[4 * fluidCount + 1]!;
      const vbz = snap.velocities[4 * fluidCount + 2]!;
      const vbMag = Math.hypot(vbx, vby, vbz);

      console.info(
        `[fluid-solid-newton3] Σm·v = (${pxSum.toExponential(3)}, ${pySum.toExponential(3)}, ${pzSum.toExponential(3)}); max|v| = ${maxAbsV.toExponential(3)}; movedCount = ${movedCount}/${total}`,
      );
      console.info(
        `[fluid-solid-newton3] boundary v = (${vbx.toExponential(3)}, ${vby.toExponential(3)}, ${vbz.toExponential(3)}); |v| = ${vbMag.toExponential(3)}`,
      );

      // Soundness — boundary actually moved. Trivially-satisfied
      // Newton-3 (everything = 0) would otherwise pass.
      expect(vbMag).toBeGreaterThan(1e-6);

      // Direction soundness — the spiky kernel weights closer
      // neighbours much more, so the boundary is pushed primarily
      // away from its NEAREST fluid neighbour. Boundary at
      // (+0.07, +0.04, +0.02) is closest to the fluid grid point at
      // (+0.05, +0.05, +0.05); the displacement vector
      // (boundary - nearest_fluid) = (+0.02, -0.01, -0.03), and the
      // reaction's signs should match. We assert the dot-product of
      // the boundary's velocity with this displacement is positive
      // (i.e. the reaction has a positive component along
      // "away from the nearest fluid neighbour").
      const boundaryPos: [number, number, number] = [0.07, 0.04, 0.02];
      const nearestFluid: [number, number, number] = [0.05, 0.05, 0.05];
      const awayDir: [number, number, number] = [
        boundaryPos[0] - nearestFluid[0],
        boundaryPos[1] - nearestFluid[1],
        boundaryPos[2] - nearestFluid[2],
      ];
      const awayDot = vbx * awayDir[0] + vby * awayDir[1] + vbz * awayDir[2];
      console.info(
        `[fluid-solid-newton3] v · (boundary − nearest fluid) = ${awayDot.toExponential(3)} (should be > 0)`,
      );
      expect(awayDot).toBeGreaterThan(0);

      // Newton-3: total system momentum within `2e-5 · max |v|` per
      // axis. With m_fluid = m_boundary = 1 the mass-weighting residual
      // collapses to FP noise; the residual ceiling is set by the i32
      // quantization of the reaction accumulator (one scatter
      // contributes a 9.3e-9 m tick rounded toward zero; ~30 scatters
      // per substep gives ~5e-7 m/s residual at dt = 1/60). A
      // `1e-5 · max|v|` target, rounded up to 2e-5 to absorb the
      // quantization noise and a small FP-summation envelope.
      const tol = Math.max(2e-5 * maxAbsV, 1e-9);
      expect(Math.abs(pxSum)).toBeLessThan(tol);
      expect(Math.abs(pySum)).toBeLessThan(tol);
      expect(Math.abs(pzSum)).toBeLessThan(tol);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
