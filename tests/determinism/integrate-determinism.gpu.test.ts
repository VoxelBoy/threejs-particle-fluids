import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../src/core/index.js';

// Phase 02 G4 — same-GPU same-seed repeatability. Two identical simulation
// runs (constructed from scratch each time) must produce bit-identical
// positions at frame 1000. Any non-determinism in TSL's kernel generation,
// three.js's dispatch ordering, or driver behavior would show here.
//

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function buildScene(): ParticleInit[] {
  const N = 512;
  const rand = lcg(0xdeadbeef);
  const between = (lo: number, hi: number): number => lo + (hi - lo) * rand();
  const data: ParticleInit[] = [];
  for (let i = 0; i < N; i++) {
    data.push({
      position: [between(-5, 5), between(5, 20), between(-5, 5)],
      velocity: [between(-1, 1), between(-1, 1), between(-1, 1)],
      invMass: 1,
      phase: 0,
    });
  }
  return data;
}

async function runScene(): Promise<Float32Array> {
  const renderer = await createParticleRenderer();
  try {
    const data = buildScene();
    const particles = new ParticleSystem(renderer, data.length, 0.05);
    particles.uploadParticles(data);
    const loop = new SimLoop(particles);
    const dt = 1 / 60;
    for (let n = 0; n < 1000; n++) await loop.step(dt);
    const snap = await particles.readback();
    particles.destroy();
    return snap.positions;
  } finally {
    renderer.dispose();
  }
}

describe('Phase 02 — integrate: determinism', () => {
  it('produces bit-identical positions after 1000 frames on repeat runs', async () => {
    const runA = await runScene();
    const runB = await runScene();

    expect(runA.length).toBe(runB.length);
    let firstMismatch = -1;
    for (let i = 0; i < runA.length; i++) {
      if (runA[i] !== runB[i]) {
        firstMismatch = i;
        break;
      }
    }
    if (firstMismatch !== -1) {
      // eslint-disable-next-line no-console
      console.error(
        `First mismatch at index ${firstMismatch}: runA=${runA[firstMismatch]} runB=${runB[firstMismatch]}`,
      );
    }
    expect(firstMismatch).toBe(-1);
  }, 120_000);
});
