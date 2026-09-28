import { Fn, atomicStore, instanceIndex, instancedArray, int, struct, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { WebGPURenderer } from 'three/webgpu';

import { releaseStorageBuffers } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Fixed-point scale for the accumulated Lagrange multipliers λ_n and λ_t, in
 * ticks per kg·m. 2^20 gives ~1e-6 kg·m resolution and ±2048 kg·m of range,
 * far beyond the per-substep impulses of particle scenes.
 */
export const LAMBDA_SCALE = 1 << 20;

/**
 * One particle–particle contact. `lambdaN`/`lambdaT` accumulate the normal and
 * tangential multipliers across solver iterations (Macklin et al. 2020 §3.5);
 * `normal` is the last normal the solve used, read by the friction pass.
 *
 * Layout: i (0), j (4), lambdaN (8), lambdaT (12), normal (16), stride 32 bytes.
 */
export const ContactRecord = struct(
  {
    i: 'uint',
    j: 'uint',
    lambdaN: { type: 'int', atomic: true },
    lambdaT: { type: 'int', atomic: true },
    normal: 'vec3',
  },
  'ContactRecord',
);
const RECORD_STRIDE_WORDS = 8;

/** Contact pairs found each substep, plus their accumulated multipliers. */
export class ContactBuffer {
  readonly renderer: WebGPURenderer;
  /** Pairs past this count are dropped. The counter still counts them. */
  readonly maxContacts: number;
  readonly records: StorageBufferNode<'struct'>;
  /** Atomic count of pairs emitted this substep. */
  readonly counter: StorageBufferNode<'uint'>;
  readonly resetCounterKernel: ComputeNode;
  readonly resetLambdaKernel: ComputeNode;

  constructor(renderer: WebGPURenderer, maxContacts: number) {
    if (!Number.isInteger(maxContacts) || maxContacts <= 0) {
      throw new Error(`ContactBuffer: maxContacts must be a positive integer, got ${maxContacts}`);
    }
    this.renderer = renderer;
    this.maxContacts = maxContacts;
    // three r184 declares a one-element struct buffer as a bare struct, which
    // can't be indexed, so always allocate at least two records.
    this.records = instancedArray(Math.max(maxContacts, 2), ContactRecord as Any) as Any;
    this.counter = instancedArray(1, 'uint').toAtomic();

    const counter = this.counter;
    this.resetCounterKernel = Fn(() => {
      atomicStore(counter.element(uint(0)), uint(0));
    })().compute(1);

    const records: Any = this.records;
    this.resetLambdaKernel = Fn(() => {
      const record: Any = records.element(instanceIndex);
      atomicStore(record.get('lambdaN'), int(0));
      atomicStore(record.get('lambdaT'), int(0));
    })().compute(maxContacts);
  }

  /** Number of pairs emitted in the last substep, including dropped ones. */
  async readbackCount(): Promise<number> {
    const buffer = await this.renderer.getArrayBufferAsync(this.counter.value);
    return new Uint32Array(buffer)[0]!;
  }

  /** Free the contact storage on the GPU. Kernels built on it can't run afterwards. */
  dispose(): void {
    releaseStorageBuffers(this.renderer, [this.records, this.counter]);
  }

  /** The stored `(i, j)` pairs, flattened. For tests and debugging. */
  async readbackPairs(): Promise<{ readonly nContacts: number; readonly pairs: Uint32Array }> {
    const [countBuffer, recordsBuffer] = await Promise.all([
      this.renderer.getArrayBufferAsync(this.counter.value),
      this.renderer.getArrayBufferAsync(this.records.value),
    ]);
    const nContacts = Math.min(new Uint32Array(countBuffer)[0]!, this.maxContacts);
    const words = new Uint32Array(recordsBuffer);
    const pairs = new Uint32Array(2 * nContacts);
    for (let c = 0; c < nContacts; c++) {
      pairs[2 * c] = words[c * RECORD_STRIDE_WORDS]!;
      pairs[2 * c + 1] = words[c * RECORD_STRIDE_WORDS + 1]!;
    }
    return { nContacts, pairs };
  }
}
