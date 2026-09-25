import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 02 G1 — energy drift sanity check. Semi-implicit Euler has a known
// per-step energy loss of exactly `0.5 · g² · dt²` per particle (derivable
// algebraically from `v += -g·dt ; y += v·dt`; independent of current
// state). We don't correct for it at this phase.
//
// This test doesn't chase ULP accuracy. It asserts the drift has the
// expected sign and magnitude — which catches: wrong sign of gravity,
// gravity-not-applied, exploding integration, or the wrong integrator
// entirely. Floor clamp is disabled so no energy is lost to the
// inelastic ground contact.

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

describe('Phase 02 — integrate: energy drift', () => {
  it('matches semi-implicit Euler truncation ~ -½g²·dt²·steps over 5 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const N = 1000;
      const particles = new ParticleSystem(renderer, N, 0.05);
      const rand = lcg(0xc0ffee);
      const between = (lo: number, hi: number): number => lo + (hi - lo) * rand();

      const data: ParticleInit[] = [];
      for (let i = 0; i < N; i++) {
        data.push({
          position: [between(-5, 5), between(10, 50), between(-5, 5)],
          velocity: [between(-2, 2), between(-2, 2), between(-2, 2)],
          invMass: 1,
          phase: 0,
        });
      }
      particles.uploadParticles(data);

      // Pinned at `substeps: 1` to preserve the Phase 02 energy-drift
      // formula `-½·g²·dt²·steps`, which is derived per-step at the frame
      // dt — multi-substepping would scale this by 1/S.
      const loop = new SimLoop(particles, { substeps: 1 });
      loop.kernels.floorY.value = -1e9;

      const g = 9.81;
      const dt = 1 / 60;
      const steps = 300;

      const meanEnergy = (positions: ArrayLike<number>, velocities: ArrayLike<number>): number => {
        let total = 0;
        for (let i = 0; i < N; i++) {
          const base = i * 4;
          const vx = velocities[base]!;
          const vy = velocities[base + 1]!;
          const vz = velocities[base + 2]!;
          const y = positions[base + 1]!;
          total += 0.5 * (vx * vx + vy * vy + vz * vz) + g * y;
        }
        return total / N;
      };

      const initialPositions = new Float32Array(N * 4);
      const initialVelocities = new Float32Array(N * 4);
      for (let i = 0; i < N; i++) {
        const base = i * 4;
        const p = data[i]!;
        initialPositions[base] = p.position[0];
        initialPositions[base + 1] = p.position[1];
        initialPositions[base + 2] = p.position[2];
        initialVelocities[base] = p.velocity[0];
        initialVelocities[base + 1] = p.velocity[1];
        initialVelocities[base + 2] = p.velocity[2];
      }
      const e0 = meanEnergy(initialPositions, initialVelocities);

      for (let n = 0; n < steps; n++) await loop.step(dt);
      const snap = await particles.readback();
      const e1 = meanEnergy(snap.positions, snap.velocities);

      const drift = e1 - e0;
      const expected = -0.5 * g * g * dt * dt * steps;

      // eslint-disable-next-line no-console
      console.info(
        `[energy-drift] initial=${e0.toFixed(2)} final=${e1.toFixed(2)} drift=${drift.toFixed(3)} (expected ~${expected.toFixed(3)})`,
      );

      expect(drift).toBeLessThan(0);
      expect(Math.abs(drift - expected)).toBeLessThan(0.5);

      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
