import { describe, expect, it } from 'vitest';
import { Fn, atomicAdd, instancedArray, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import {
  type Material,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
} from '../../../src/core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

// Phase 07b G1 — three-slot dispatch cadence.
//
// A material's three kernel arrays run at different cadences inside one
// substep:
//   preIterKernels     — once per substep (before the iter loop).
//   perIterKernels     — once per iter (the inner solver loop runs `I` times).
//   postAdvectKernels  — once per substep (after advect, before velocity-friction).
//
// This test pins the cadence: a single mock material registers a counter
// kernel in each slot. After one `step()` with substeps=1, iterations=3,
// the counters read pre=1, per=3, post=1. With substeps=2, iterations=3,
// they read pre=2, per=6, post=2.

function makeCounterKernel(buf: StorageBufferNode<'uint'>): ComputeNode {
  return Fn(() => {
    const unused: Any = (atomicAdd(buf.element(0), uint(1)) as Any).toVar();
    // Silence "unused var" without a side effect. `.toVar()` already anchors
    // the node in the compute graph via the atomic side-effect.
    void unused;
  })().compute(1);
}

async function runWithCadence(
  substeps: number,
  iterations: number,
): Promise<{ pre: number; per: number; post: number }> {
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, 1, 0.05);
    particles.uploadParticles([{ position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 }]);

    const preBuf = instancedArray(1, 'uint').toAtomic();
    const perBuf = instancedArray(1, 'uint').toAtomic();
    const postBuf = instancedArray(1, 'uint').toAtomic();

    const material: Material = {
      preIterKernels: [makeCounterKernel(preBuf)],
      perIterKernels: [makeCounterKernel(perBuf)],
      postAdvectKernels: [makeCounterKernel(postBuf)],
    };

    const loop = new SimLoop(particles, {
      substeps,
      iterations,
      materials: [material],
    });
    loop.kernels.floorY.value = -1e9;
    await loop.step(1 / 60);

    const [pre, per, post] = await Promise.all([
      renderer.getArrayBufferAsync(preBuf.value),
      renderer.getArrayBufferAsync(perBuf.value),
      renderer.getArrayBufferAsync(postBuf.value),
    ]);
    const result = {
      pre: new Uint32Array(pre)[0]!,
      per: new Uint32Array(per)[0]!,
      post: new Uint32Array(post)[0]!,
    };
    particles.destroy();
    return result;
  } finally {
    renderer.dispose();
  }
}

describe('Phase 07b — materials: three-slot dispatch cadence', () => {
  it('substeps=1, iterations=3: preIter=1, perIter=3, postAdvect=1', async () => {
    const { pre, per, post } = await runWithCadence(1, 3);
    // eslint-disable-next-line no-console
    console.info(`[mat-cadence-1x3] pre=${pre} per=${per} post=${post}`);
    expect(pre).toBe(1);
    expect(per).toBe(3);
    expect(post).toBe(1);
  }, 30_000);

  it('substeps=2, iterations=3: preIter=2, perIter=6, postAdvect=2', async () => {
    const { pre, per, post } = await runWithCadence(2, 3);
    // eslint-disable-next-line no-console
    console.info(`[mat-cadence-2x3] pre=${pre} per=${per} post=${post}`);
    expect(pre).toBe(2);
    expect(per).toBe(6);
    expect(post).toBe(2);
  }, 30_000);
});
