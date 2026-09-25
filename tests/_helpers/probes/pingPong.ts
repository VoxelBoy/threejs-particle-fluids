import { Fn, instanceIndex, instancedArray, uint } from 'three/tsl';
import { createWebGPURenderer } from './_renderer.js';

type Any = any; // eslint-disable-line @typescript-eslint/no-explicit-any

const N = 1 << 20; // 1,048,576

export interface PingPongResult {
  readonly n: number;
  readonly iters: number;
  readonly allCorrect: boolean;
  readonly firstMismatchAt: number;
  readonly elapsedMs: number;
}

export async function runPingPongProbe(iters = 100): Promise<PingPongResult> {
  const renderer = await createWebGPURenderer();

  const a = instancedArray(N, 'uint');
  const b = instancedArray(N, 'uint');

  const aArr = a.value.array as Uint32Array;
  const initial = 7; // arbitrary non-zero start so zero-init doesn't hide bugs
  for (let i = 0; i < N; i++) aArr[i] = initial;

  const aToB = Fn(() => {
    const i: Any = instanceIndex;
    b.element(i).assign(a.element(i).add(uint(1)));
  })().compute(N);

  const bToA = Fn(() => {
    const i: Any = instanceIndex;
    a.element(i).assign(b.element(i).add(uint(1)));
  })().compute(N);

  const sequence = [];
  for (let k = 0; k < iters; k++) {
    sequence.push(k % 2 === 0 ? aToB : bToA);
  }

  const t0 = performance.now();
  await renderer.computeAsync(sequence);
  // Final result lands in `b` when iters is odd, `a` when even.
  const finalBuffer = iters % 2 === 1 ? b.value : a.value;
  const result = new Uint32Array(await renderer.getArrayBufferAsync(finalBuffer));
  const elapsedMs = performance.now() - t0;

  const expected = (initial + iters) >>> 0;
  let firstMismatchAt = -1;
  for (let i = 0; i < N; i++) {
    if (result[i] !== expected) {
      firstMismatchAt = i;
      break;
    }
  }

  renderer.dispose();
  return {
    n: N,
    iters,
    allCorrect: firstMismatchAt === -1,
    firstMismatchAt,
    elapsedMs,
  };
}
