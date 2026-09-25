import { Fn, instanceIndex, instancedArray, float } from 'three/tsl';
import { createWebGPURenderer } from './_renderer.js';

export interface SaxpyResult {
  readonly n: number;
  readonly maxAbsError: number;
  readonly elapsedMs: number;
}

export async function runSaxpyProbe(n: number, a: number): Promise<SaxpyResult> {
  const renderer = await createWebGPURenderer();

  const xNode = instancedArray(n, 'float');
  const yNode = instancedArray(n, 'float');

  const xArr = xNode.value.array as Float32Array;
  const yArr = yNode.value.array as Float32Array;
  for (let i = 0; i < n; i++) {
    xArr[i] = Math.sin(i * 1e-3) * 2;
    yArr[i] = Math.cos(i * 1e-3) * 3;
  }

  const reference = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    reference[i] = Math.fround(Math.fround(a * xArr[i]!) + yArr[i]!);
  }

  const kernel = Fn(() => {
    const i = instanceIndex;
    const x = xNode.element(i);
    const y = yNode.element(i);
    y.assign(x.mul(float(a)).add(y));
  })().compute(n);

  const t0 = performance.now();
  await renderer.computeAsync(kernel);
  const readBack = new Float32Array(await renderer.getArrayBufferAsync(yNode.value));
  const elapsedMs = performance.now() - t0;

  let maxAbsError = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.abs(readBack[i]! - reference[i]!);
    if (e > maxAbsError) maxAbsError = e;
  }

  renderer.dispose();
  return { n, maxAbsError, elapsedMs };
}
