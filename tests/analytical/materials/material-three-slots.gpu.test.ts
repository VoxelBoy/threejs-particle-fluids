import { describe, expect, it } from 'vitest';
import { Fn, atomicAdd, instancedArray, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import {
  type Material,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
} from '../../../src/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

// Dispatch cadence of a material's three per-substep kernel slots:
//   preSolve   — once per substep (before the solver iterations).
//   solve      — once per iteration (the solver loop runs `I` times).
//   postSolve  — once per substep (after velocities are updated, before
//                friction).
//
// A single mock material registers a counter kernel in each slot. After one
// `step()` with substeps = 1 and iterations = 3, the counters read pre = 1,
// solve = 3, post = 1. With substeps = 2 and iterations = 3 they read
// pre = 2, solve = 6, post = 2.

function makeCounterKernel(buf: StorageBufferNode<'uint'>): ComputeNode {
  return Fn(() => {
    const unused: Any = (atomicAdd(buf.element(0), uint(1)) as Any).toVar();
    // `.toVar()` anchors the atomic in the compute graph; `void` silences
    // the unused-variable lint without adding a side effect.
    void unused;
  })().compute(1);
}

async function runWithCadence(
  substeps: number,
  iterations: number,
): Promise<{ pre: number; solve: number; post: number }> {
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, 1, 0.05);
    particles.uploadParticles([{ position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1 }]);

    const preBuf = instancedArray(1, 'uint').toAtomic();
    const solveBuf = instancedArray(1, 'uint').toAtomic();
    const postBuf = instancedArray(1, 'uint').toAtomic();

    const material: Material = {
      build: () => ({
        preSolve: [makeCounterKernel(preBuf)],
        solve: [makeCounterKernel(solveBuf)],
        postSolve: [makeCounterKernel(postBuf)],
      }),
    };

    const loop = new SimLoop(particles, {
      substeps,
      iterations,
      materials: [material],
    });
    await loop.step(1 / 60);

    const [pre, solve, post] = await Promise.all([
      renderer.getArrayBufferAsync(preBuf.value),
      renderer.getArrayBufferAsync(solveBuf.value),
      renderer.getArrayBufferAsync(postBuf.value),
    ]);
    const result = {
      pre: new Uint32Array(pre)[0]!,
      solve: new Uint32Array(solve)[0]!,
      post: new Uint32Array(post)[0]!,
    };
    loop.dispose();
    particles.dispose();
    return result;
  } finally {
    renderer.dispose();
  }
}

describe('materials: kernel slot dispatch cadence', () => {
  it('substeps=1, iterations=3: preSolve=1, solve=3, postSolve=1', async () => {
    const { pre, solve, post } = await runWithCadence(1, 3);
    console.info(`[mat-cadence-1x3] pre=${pre} solve=${solve} post=${post}`);
    expect(pre).toBe(1);
    expect(solve).toBe(3);
    expect(post).toBe(1);
  }, 30_000);

  it('substeps=2, iterations=3: preSolve=2, solve=6, postSolve=2', async () => {
    const { pre, solve, post } = await runWithCadence(2, 3);
    console.info(`[mat-cadence-2x3] pre=${pre} solve=${solve} post=${post}`);
    expect(pre).toBe(2);
    expect(solve).toBe(6);
    expect(post).toBe(2);
  }, 30_000);
});
