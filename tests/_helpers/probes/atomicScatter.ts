import { Fn, atomicAdd, instanceIndex, instancedArray, uint } from 'three/tsl';
import { createWebGPURenderer } from './_renderer.js';

type Any = any; // eslint-disable-line @typescript-eslint/no-explicit-any

const N = 1_000_000;
const B = 1024;

function lcgU32(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
}

export interface AtomicScatterResult {
  readonly n: number;
  readonly bins: number;
  readonly matched: boolean;
  readonly totalCountMatches: boolean;
  readonly elapsedMs: number;
}

export async function runAtomicScatterProbe(seed: number): Promise<AtomicScatterResult> {
  const renderer = await createWebGPURenderer();

  const keysNode = instancedArray(N, 'uint');
  const countsNode = instancedArray(B, 'uint').toAtomic();

  const keysArr = keysNode.value.array as Uint32Array;
  const rand = lcgU32(seed);
  for (let i = 0; i < N; i++) keysArr[i] = rand() & (B - 1);

  const scatterKernel = Fn(() => {
    const i: Any = instanceIndex;
    const key: Any = keysNode.element(i);
    atomicAdd(countsNode.element(key), uint(1));
  })().compute(N);

  const t0 = performance.now();
  await renderer.computeAsync(scatterKernel);
  const gpuCounts = new Uint32Array(await renderer.getArrayBufferAsync(countsNode.value));
  const elapsedMs = performance.now() - t0;

  const reference = new Uint32Array(B);
  for (let i = 0; i < N; i++) reference[keysArr[i]!]! += 1;

  let matched = true;
  for (let b = 0; b < B; b++) {
    if (gpuCounts[b] !== reference[b]) {
      matched = false;
      break;
    }
  }

  let totalGpu = 0;
  for (let b = 0; b < B; b++) totalGpu += gpuCounts[b]!;

  renderer.dispose();
  return {
    n: N,
    bins: B,
    matched,
    totalCountMatches: totalGpu === N,
    elapsedMs,
  };
}
