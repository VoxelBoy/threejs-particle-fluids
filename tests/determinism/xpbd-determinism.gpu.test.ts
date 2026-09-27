import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  constraintKernels,
  createDistanceConstraints,
  createParticleRenderer,
  type Material,
  type ParticleInit,
} from '../../src/index.js';

// Bit-exact determinism for XPBD distance constraints (tier 1 of the
// two-tier determinism policy).
//
// Gather-mode XPBD is order-independent at the kernel level: every per-
// particle write lands in a single, predetermined slot and every per-
// constraint λ accumulation is written by exactly one "leader" thread
// (see `distance.ts` JSDoc). The residual sums computed inside
// `xpbdDeltaLambda` are 2-term reductions (wi + wj) — not the unbounded
// neighbor-sum reductions that tier 2 exists for — so there is no expected
// source of run-to-run variation. Tier 1 (bit-identical) applies.
//
// Same-seed repeatability is asserted on `positions` after 300 frames of a
// random chain-connected distance-constraint graph under gravity.

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

interface Scene {
  readonly data: readonly ParticleInit[];
  readonly pairs: readonly [number, number][];
  readonly restLengths: readonly number[];
}

function buildScene(): Scene {
  const rand = lcg(0xfacade01);
  const N = 64;
  const data: ParticleInit[] = [];
  for (let i = 0; i < N; i++) {
    data.push({
      position: [rand() * 2 - 1, 5 + rand() * 2, rand() * 2 - 1],
      velocity: [rand() - 0.5, rand() - 0.5, rand() - 0.5],
      invMass: i === 0 ? 0 : 1, // particle 0 pinned
    });
  }
  // Chain: 0-1-2-...-(N-1)
  const pairs: [number, number][] = [];
  const restLengths: number[] = [];
  for (let i = 0; i < N - 1; i++) {
    pairs.push([i, i + 1]);
    const a = data[i]!.position;
    const b = data[i + 1]!.position;
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    restLengths.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  return { data, pairs, restLengths };
}

async function runOnce(scene: Scene): Promise<Float32Array> {
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, scene.data.length, 0.02);
    particles.uploadParticles([...scene.data]);

    const chain: Material = {
      build: ({ dt }) =>
        constraintKernels([
          createDistanceConstraints({
            particles,
            pairs: [...scene.pairs],
            compliance: 1e-4,
            restLength: [...scene.restLengths],
            dt,
          }),
        ]),
    };
    const loop = new SimLoop(particles, {
      materials: [chain],
      substeps: 2,
      iterations: 2,
    });

    const dt = 1 / 60;
    for (let n = 0; n < 300; n++) await loop.step(dt);

    const snap = await particles.readback();
    loop.dispose();
    particles.dispose();
    return snap.positions;
  } finally {
    renderer.dispose();
  }
}

// Each constraint is solved by one thread that moves both endpoints, so no
// thread reads a position another is writing and repeat runs match exactly.
describe('XPBD: same-seed determinism', () => {
  it('bit-identical positions after 300 frames across repeat runs', async () => {
    const scene = buildScene();
    const runA = await runOnce(scene);
    const runB = await runOnce(scene);

    expect(runA.length).toBe(runB.length);
    let firstMismatch = -1;
    for (let i = 0; i < runA.length; i++) {
      if (runA[i] !== runB[i]) {
        firstMismatch = i;
        break;
      }
    }
    if (firstMismatch !== -1) {
      console.error(
        `First mismatch at index ${firstMismatch}: runA=${runA[firstMismatch]} runB=${runB[firstMismatch]}`,
      );
    }
    expect(firstMismatch).toBe(-1);
  }, 180_000);
});
