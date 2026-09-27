import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// No NaN/Inf in any integrated buffer after 1000 frames of gravity-only
// simulation; particle count unchanged. A floor plane at y = 0 catches the
// falling particles, so the collider path is exercised too.

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

describe('integrate: invariants', () => {
  it('produces no NaN/Inf after 1000 frames and preserves particle count', async () => {
    const renderer = await createParticleRenderer();
    try {
      const N = 2048;
      const particles = new ParticleSystem(renderer, N, 0.05);
      const rand = lcg(0x51de5eed);
      const between = (lo: number, hi: number): number => lo + (hi - lo) * rand();

      const data: ParticleInit[] = [];
      for (let i = 0; i < N; i++) {
        data.push({
          position: [between(-10, 10), between(1, 20), between(-10, 10)],
          velocity: [between(-3, 3), between(-3, 3), between(-3, 3)],
          invMass: 1,
        });
      }
      particles.uploadParticles(data);
      const floor = new PrimitiveSet(particles);
      floor.addPlane(new Vector3(0, 1, 0), new Vector3());
      const loop = new SimLoop(particles, { colliders: [floor] });

      const dt = 1 / 60;
      for (let n = 0; n < 1000; n++) await loop.step(dt);

      const snap = await particles.readback();
      expect(snap.capacity).toBe(N);

      const checkFinite = (name: string, arr: Float32Array): void => {
        for (let i = 0; i < arr.length; i++) {
          if (!Number.isFinite(arr[i]!)) {
            throw new Error(`${name}[${i}] = ${arr[i]} (not finite)`);
          }
        }
      };
      checkFinite('positions', snap.positions);
      checkFinite('predictedPositions', snap.predictedPositions);
      checkFinite('velocities', snap.velocities);
      checkFinite('invMass', snap.invMass);

      loop.dispose();
      floor.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
