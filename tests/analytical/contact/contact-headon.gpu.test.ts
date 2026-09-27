import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Head-on collision of two particles.
//
// Two particles, radius r, equal mass, velocities ±v along x. An elastic
// bounce would reverse both velocities, but PBD / XPBD contact without a
// separate restitution term is perfectly inelastic: the position-level
// projection undoes the approach velocity (`Δx = −dt·v_approach`), so after
// the solve both particles end up with `v ≈ 0` (Macklin 2014 §6 has no
// restitution parameter). The load-bearing assertions are therefore:
//   (1) Non-penetration: `|x_i − x_j| ≥ 2r − ε` on every frame.
//   (2) Momentum conservation: `w_i·v_i + w_j·v_j ≈ 0` within 1e-4.
//   (3) Energy loss is logged, not gated on a magnitude.

describe('contact: head-on collision', () => {
  it('two approaching particles do not interpenetrate and conserve momentum', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.5;
      const twoR = 2 * r;
      const v = 0.5;

      const particles = new ParticleSystem(renderer, 2, r);
      const initial: ParticleInit[] = [
        { position: [-0.75, 0, 0], velocity: [+v, 0, 0], invMass: 1 },
        { position: [+0.75, 0, 0], velocity: [-v, 0, 0], invMass: 1 },
      ];
      particles.uploadParticles(initial);

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 8,
        // Frictionless, to isolate the normal projection.
        contact: { maxContacts: 16, muS: 0, muK: 0 },
      });
      // No gravity: a pure 1D head-on collision.
      loop.gravity.set(0, 0, 0);

      const frameDt = 1 / 60;
      const totalFrames = 120; // 2 s, well past the ~0.5 s collision time.

      // Track the minimum inter-particle distance to catch interpenetration.
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

      // (1) Non-penetration.
      expect(
        minDistance,
        `minDist ${minDistance} fell below 2r − ε = ${twoR - 1e-3}`,
      ).toBeGreaterThan(twoR - 1e-3);

      // (2) Momentum conservation.
      expect(Math.abs(pFinal - pInitial)).toBeLessThan(1e-4);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
