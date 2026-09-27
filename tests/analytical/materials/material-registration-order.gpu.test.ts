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

// TSL's @types surface many nodes as bare `Node`, dropping proxy methods.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

// Material registration order.
//
// SimLoop builds `options.materials` in array order and concatenates each
// material's kernels per slot. This test pins the ordering policy: within
// each slot (`preSolve`, `solve`, `postSolve`), a material registered
// earlier in the array MUST run before a material registered later.
//
// Mechanism: each slot has one atomic counter shared by two mock materials.
// Each material's kernel in that slot does `seq = atomicAdd(counter, 1)` and
// stores `seq` in the material's own output slot. After one step with
// substeps = 1 and iterations = 1, the outputs reveal the order in which the
// kernels ran: the earlier-registered material sees seq = 0, the later one
// seq = 1.

const SLOTS = ['preSolve', 'solve', 'postSolve'] as const;
type Slot = (typeof SLOTS)[number];
type SlotBuffers = Record<Slot, StorageBufferNode<'uint'>>;

/** A kernel that takes the next number from `counter` and stores it in `out`. */
function ticketKernel(
  counter: StorageBufferNode<'uint'>,
  out: StorageBufferNode<'uint'>,
): ComputeNode {
  return Fn(() => {
    const seq: Any = (atomicAdd(counter.element(0), uint(1)) as Any).toVar();
    out.element(0).assign(seq);
  })().compute(1);
}

function makeMockMaterial(counters: SlotBuffers, out: SlotBuffers): Material {
  return {
    build: () => ({
      preSolve: [ticketKernel(counters.preSolve, out.preSolve)],
      solve: [ticketKernel(counters.solve, out.solve)],
      postSolve: [ticketKernel(counters.postSolve, out.postSolve)],
    }),
  };
}

function slotBuffers(make: () => StorageBufferNode<'uint'>): SlotBuffers {
  return { preSolve: make(), solve: make(), postSolve: make() };
}

async function runWithOrder(
  orderA: 'first' | 'second',
): Promise<Record<Slot, { orderA: number; orderB: number }>> {
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, 1, 0.05);
    particles.uploadParticles([{ position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1 }]);

    const counters = slotBuffers(() => instancedArray(1, 'uint').toAtomic());
    const outA = slotBuffers(() => instancedArray(1, 'uint'));
    const outB = slotBuffers(() => instancedArray(1, 'uint'));

    const matA = makeMockMaterial(counters, outA);
    const matB = makeMockMaterial(counters, outB);

    const materials = orderA === 'first' ? [matA, matB] : [matB, matA];

    const loop = new SimLoop(particles, {
      substeps: 1,
      iterations: 1,
      materials,
    });
    await loop.step(1 / 60);

    const read = async (buffer: StorageBufferNode<'uint'>): Promise<number> =>
      new Uint32Array(await renderer.getArrayBufferAsync(buffer.value))[0]!;
    const result = {} as Record<Slot, { orderA: number; orderB: number }>;
    for (const slot of SLOTS) {
      result[slot] = { orderA: await read(outA[slot]), orderB: await read(outB[slot]) };
    }
    loop.dispose();
    particles.dispose();
    return result;
  } finally {
    renderer.dispose();
  }
}

describe('materials: registration order', () => {
  it('material[0] runs before material[1] in every slot', async () => {
    const result = await runWithOrder('first');
    for (const slot of SLOTS) {
      const { orderA, orderB } = result[slot];
      console.info(`[mat-order] [A, B] ${slot}: orderA=${orderA} orderB=${orderB}`);
      expect(orderA, slot).toBe(0);
      expect(orderB, slot).toBe(1);
    }
  }, 30_000);

  it('reversing registration order reverses dispatch order', async () => {
    const result = await runWithOrder('second');
    for (const slot of SLOTS) {
      const { orderA, orderB } = result[slot];
      console.info(`[mat-order] [B, A] ${slot}: orderA=${orderA} orderB=${orderB}`);
      expect(orderB, slot).toBe(0);
      expect(orderA, slot).toBe(1);
    }
  }, 30_000);
});
