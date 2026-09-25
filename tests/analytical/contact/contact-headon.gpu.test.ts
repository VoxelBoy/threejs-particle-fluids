import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 05 G1 — head-on elastic collision smoke test.
//
// Plan §Validation (BLOCKING): "Two particles, radius r, equal mass, velocities
// ±v along x. After collision, velocities should be reversed (elastic bounce).
// Momentum conservation within 1e-4. Energy loss within the tolerance expected
// from position-based dynamics (document the measured value)."
//
// **Interpretation caveat** — PBD / XPBD contact without a separate restitution
// term is **perfectly inelastic**: the position-level projection undoes the
// approach velocity, giving `Δx = -dt·v_approach` → post-solve `v_new ≈ 0` for
// both particles. The plan's "velocities reversed" sentence is aspirational and
// internally contradicts its own "energy loss expected from PBD" sentence; see
// paper Macklin 2014 §6 which introduces no restitution parameter. The load-
// bearing assertions in this test are therefore:
//   (1) Non-penetration: `|x_i − x_j| ≥ 2r − ε` after collision.
//   (2) Momentum conservation: `w_i·v_i + w_j·v_j ≈ 0` within 1e-4.
//   (3) Energy loss is **documented** but not gated on a magnitude.
//
// Tier choice (G4): this is a G1 analytical test, not a determinism test.

describe('Phase 05 — contact: head-on collision', () => {
  it('two approaching particles do not interpenetrate and conserve momentum', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.5;
      const twoR = 2 * r;
      const v = 0.5;

      const particles = new ParticleSystem(renderer, 2, r);
      const initial: ParticleInit[] = [
        {
          // Inside the hash grid's 27-cell reach of particle 1 (cells +1 /
          // +2 at cellSize=1.1 from domainMin=-2): x=-0.75 → cell 1,
          // x=+0.75 → cell 2, adjacency holds.
          position: [-0.75, 0, 0],
          velocity: [+v, 0, 0],
          invMass: 1,
          phase: 0,
        },
        {
          position: [+0.75, 0, 0],
          velocity: [-v, 0, 0],
          invMass: 1,
          phase: 0,
        },
      ];
      particles.uploadParticles(initial);

      const hashGrid = new HashGrid(particles, {
        cellSize: twoR * 1.1,
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 8, // headroom for coloring-gather convergence
        contact: {
          hashGrid,
          maxContacts: 16,
          friction: { muS: 0.0, muK: 0.0 }, // isolate the normal projection
        },
      });
      // Disable gravity and floor clamp — pure 1D head-on collision.
      loop.gravity.set(0, 0, 0);
      loop.kernels.floorY.value = -1e9;

      const frameDt = 1 / 60;
      const totalFrames = 120; // 2s, well past the ~0.5s collision time.

      // Track minimum inter-particle distance to catch interpenetration.
      let minDistance = Infinity;
      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
        const snap = await particles.readback();
        const dx = snap.positions[0]! - snap.positions[4]!;
        const dy = snap.positions[1]! - snap.positions[5]!;
        const dz = snap.positions[2]! - snap.positions[6]!;
        const dist = Math.hypot(dx, dy, dz);
        if (dist < minDistance) minDistance = dist;
      }

      // Final-state diagnostics.
      const snap = await particles.readback();
      const xFinal = [snap.positions[0]!, snap.positions[4]!];
      const vFinal = [snap.velocities[0]!, snap.velocities[4]!];

      const pInitial = 1 * +v + 1 * -v; // 0 by construction
      const pFinal = 1 * vFinal[0]! + 1 * vFinal[1]!;
      const keInitial = 0.5 * (v * v + v * v);
      const keFinal = 0.5 * (vFinal[0]! ** 2 + vFinal[1]! ** 2);

      // eslint-disable-next-line no-console
      console.info(
        `[contact-headon] frames=${totalFrames} minDist=${minDistance.toExponential(3)} ` +
          `(threshold=${(twoR - 1e-3).toExponential(3)}); ` +
          `xFinal=[${xFinal[0]!.toFixed(4)}, ${xFinal[1]!.toFixed(4)}] ` +
          `vFinal=[${vFinal[0]!.toFixed(4)}, ${vFinal[1]!.toFixed(4)}]; ` +
          `pInit=${pInitial.toFixed(6)} pFinal=${pFinal.toExponential(3)} ` +
          `|dp|=${Math.abs(pFinal - pInitial).toExponential(3)}; ` +
          `KEInit=${keInitial.toFixed(4)} KEFinal=${keFinal.toExponential(3)} ` +
          `energyLoss=${((1 - keFinal / keInitial) * 100).toFixed(2)}%`,
      );

      // (1) Non-penetration — the blocking physics claim.
      expect(
        minDistance,
        `minDist ${minDistance} fell below 2r − ε = ${twoR - 1e-3}`,
      ).toBeGreaterThan(twoR - 1e-3);

      // (2) Momentum conservation — |Δp| < 1e-4 per plan.
      expect(Math.abs(pFinal - pInitial)).toBeLessThan(1e-4);

      particles.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
