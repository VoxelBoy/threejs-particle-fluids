import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Energy drift sanity check. Semi-implicit Euler has a known per-step
// energy loss of exactly `0.5 · g² · dt²` per particle (derivable
// algebraically from `v += -g·dt ; y += v·dt`; independent of current
// state). The integrator does not correct for it.
//
// This test doesn't chase ULP accuracy. It asserts the drift has the
// expected sign and magnitude — which catches: wrong sign of gravity,
// gravity-not-applied, exploding integration, or the wrong integrator
// entirely. There is no floor collider, so no energy is lost to ground
// contact.

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

describe('integrate: energy drift', () => {
  it('matches semi-implicit Euler truncation ~ -½g²·dt²·steps over 5 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const N = 1000;
      const particles = new ParticleSystem(renderer, N, 0.05);
      const rand = lcg(0xc0ffee);
      const between = (lo: number, hi: number): number => lo + (hi - lo) * rand();

      const positions: [number, number, number][] = [];
      const velocities: [number, number, number][] = [];
      for (let i = 0; i < N; i++) {
        positions.push([between(-5, 5), between(10, 50), between(-5, 5)]);
        velocities.push([between(-2, 2), between(-2, 2), between(-2, 2)]);
      }
      const data: ParticleInit[] = positions.map((position, i) => ({
        position,
        velocity: velocities[i]!,
        invMass: 1,
      }));
      particles.uploadParticles(data);

      // Pinned at `substeps: 1` to keep the energy-drift formula
      // `-½·g²·dt²·steps`, which is derived per step at the frame dt —
      // multi-substepping would scale this by 1/S.
      const loop = new SimLoop(particles, { substeps: 1 });

      const g = 9.81;
      const dt = 1 / 60;
      const steps = 300;

      const meanEnergy = (pos: ArrayLike<number>, vel: ArrayLike<number>): number => {
        let total = 0;
        for (let i = 0; i < N; i++) {
          const base = i * 4;
          const vx = vel[base]!;
          const vy = vel[base + 1]!;
          const vz = vel[base + 2]!;
          const y = pos[base + 1]!;
          total += 0.5 * (vx * vx + vy * vy + vz * vz) + g * y;
        }
        return total / N;
      };

      const initialPositions = new Float32Array(N * 4);
      const initialVelocities = new Float32Array(N * 4);
      for (let i = 0; i < N; i++) {
        initialPositions.set(positions[i]!, i * 4);
        initialVelocities.set(velocities[i]!, i * 4);
      }
      const e0 = meanEnergy(initialPositions, initialVelocities);

      for (let n = 0; n < steps; n++) await loop.step(dt);
      const snap = await particles.readback();
      const e1 = meanEnergy(snap.positions, snap.velocities);

      const drift = e1 - e0;
      const expected = -0.5 * g * g * dt * dt * steps;

      console.info(
        `[energy-drift] initial=${e0.toFixed(2)} final=${e1.toFixed(2)} drift=${drift.toFixed(3)} (expected ~${expected.toFixed(3)})`,
      );

      expect(drift).toBeLessThan(0);
      expect(Math.abs(drift - expected)).toBeLessThan(0.5);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
