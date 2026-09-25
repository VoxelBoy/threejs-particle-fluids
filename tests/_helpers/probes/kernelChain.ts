import { Fn, instanceIndex, instancedArray, uint } from 'three/tsl';
import { createWebGPURenderer } from './_renderer.js';

type Any = any; // eslint-disable-line @typescript-eslint/no-explicit-any

const N = 1 << 20; // 1,048,576

export interface KernelChainResult {
  readonly n: number;
  readonly stageBMatches: boolean;
  readonly stageCMatches: boolean;
  readonly stageDMatches: boolean;
  readonly elapsedMs: number;
}

export async function runKernelChainProbe(): Promise<KernelChainResult> {
  const renderer = await createWebGPURenderer();

  const aNode = instancedArray(N, 'uint');
  const bNode = instancedArray(N, 'uint');
  const cNode = instancedArray(N, 'uint');
  const dNode = instancedArray(N, 'uint');

  const aArr = aNode.value.array as Uint32Array;
  for (let i = 0; i < N; i++) aArr[i] = i >>> 0;

  // Stage 1: b[i] = a[i] * 2
  const k1 = Fn(() => {
    const i: Any = instanceIndex;
    bNode.element(i).assign(aNode.element(i).mul(uint(2)));
  })().compute(N);

  // Stage 2: c[i] = b[i] + 10  — reads stage-1 output
  const k2 = Fn(() => {
    const i: Any = instanceIndex;
    cNode.element(i).assign(bNode.element(i).add(uint(10)));
  })().compute(N);

  // Stage 3: d[i] = c[i] * c[i]  — reads stage-2 output
  const k3 = Fn(() => {
    const i: Any = instanceIndex;
    const c: Any = cNode.element(i);
    dNode.element(i).assign(c.mul(c));
  })().compute(N);

  const t0 = performance.now();
  await renderer.computeAsync([k1, k2, k3]);
  const [bRead, cRead, dRead] = await Promise.all([
    renderer.getArrayBufferAsync(bNode.value),
    renderer.getArrayBufferAsync(cNode.value),
    renderer.getArrayBufferAsync(dNode.value),
  ]);
  const bGpu = new Uint32Array(bRead);
  const cGpu = new Uint32Array(cRead);
  const dGpu = new Uint32Array(dRead);
  const elapsedMs = performance.now() - t0;

  let stageBMatches = true;
  let stageCMatches = true;
  let stageDMatches = true;
  for (let i = 0; i < N; i++) {
    const a = aArr[i]!;
    const bRef = (a * 2) >>> 0;
    const cRef = (bRef + 10) >>> 0;
    const dRef = Math.imul(cRef, cRef) >>> 0;
    if (bGpu[i] !== bRef) stageBMatches = false;
    if (cGpu[i] !== cRef) stageCMatches = false;
    if (dGpu[i] !== dRef) stageDMatches = false;
    if (!stageBMatches && !stageCMatches && !stageDMatches) break;
  }

  renderer.dispose();
  return { n: N, stageBMatches, stageCMatches, stageDMatches, elapsedMs };
}
