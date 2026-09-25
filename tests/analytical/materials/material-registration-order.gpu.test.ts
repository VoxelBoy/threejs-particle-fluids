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

// TSL's @types surface many nodes as bare `Node`, dropping proxy methods.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

// Phase 07b G1 — material registration order.
//
// SimLoop iterates `options.materials` in array order, appending each
// material's `perIterKernels` to the per-iter dispatch chain. This test
// pins the ordering policy: a material registered earlier in the array
// MUST run before a material registered later within each iter.
//
// Mechanism: two mock materials share a single atomic counter. Each
// material's perIterKernel does `seq = atomicAdd(counter, 1)` and stores
// `seq` in its own 1-slot output buffer. After one step with substeps=1,
// iterations=1, the output slots reveal the sequence in which the kernels
// ran: the earlier-registered material sees seq=0, the later sees seq=1.

function makeMockMaterial(
  counter: StorageBufferNode<'uint'>,
  orderSlot: StorageBufferNode<'uint'>,
): Material & { readonly _kernel: ComputeNode } {
  const kernel: ComputeNode = Fn(() => {
    const seq: Any = (atomicAdd(counter.element(0), uint(1)) as Any).toVar();
    orderSlot.element(0).assign(seq);
  })().compute(1);
  return { perIterKernels: [kernel], _kernel: kernel };
}

async function runWithOrder(
  orderA: 'first' | 'second',
): Promise<{ orderA: number; orderB: number }> {
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, 1, 0.05);
    particles.uploadParticles([{ position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 }]);

    const counter = instancedArray(1, 'uint').toAtomic();
    const outA = instancedArray(1, 'uint');
    const outB = instancedArray(1, 'uint');

    const matA = makeMockMaterial(counter, outA);
    const matB = makeMockMaterial(counter, outB);

    const materials = orderA === 'first' ? ([matA, matB] as const) : ([matB, matA] as const);

    const loop = new SimLoop(particles, {
      substeps: 1,
      iterations: 1,
      materials,
    });
    loop.kernels.floorY.value = -1e9;
    await loop.step(1 / 60);

    const [a, b] = await Promise.all([
      renderer.getArrayBufferAsync(outA.value),
      renderer.getArrayBufferAsync(outB.value),
    ]);
    const result = {
      orderA: new Uint32Array(a)[0]!,
      orderB: new Uint32Array(b)[0]!,
    };
    particles.destroy();
    return result;
  } finally {
    renderer.dispose();
  }
}

describe('Phase 07b — materials: registration order', () => {
  it('material[0] runs before material[1] within each iter', async () => {
    const { orderA, orderB } = await runWithOrder('first');
    // eslint-disable-next-line no-console
    console.info(`[mat-order] [A, B]: orderA=${orderA} orderB=${orderB}`);
    expect(orderA).toBe(0);
    expect(orderB).toBe(1);
  }, 30_000);

  it('reversing registration order reverses dispatch order', async () => {
    const { orderA, orderB } = await runWithOrder('second');
    // eslint-disable-next-line no-console
    console.info(`[mat-order] [B, A]: orderA=${orderA} orderB=${orderB}`);
    expect(orderB).toBe(0);
    expect(orderA).toBe(1);
  }, 30_000);
});
