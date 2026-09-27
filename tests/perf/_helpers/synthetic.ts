// Synthetic scenes for testing the runner itself: known GPU work with no
// simulation behind it.

import { Fn, float, instanceIndex, instancedArray } from 'three/tsl';

import type { PerfRenderer } from './PerfRenderer.js';
import type { PerfSceneSpec } from './PerfRunner.js';

// TSL's type declarations drop the operator methods on storage elements.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * A scene whose frame runs `y = 2.5·x + y` over `count` floats `dispatches`
 * times in a single compute pass, so frame cost grows with `dispatches`.
 */
export function saxpyScene(
  perf: PerfRenderer,
  args: { readonly id: string; readonly count: number; readonly dispatches: number },
): PerfSceneSpec {
  const { id, count, dispatches } = args;
  const x = instancedArray(count, 'float');
  const y = instancedArray(count, 'float');
  const xs = x.value.array as Float32Array;
  const ys = y.value.array as Float32Array;
  for (let i = 0; i < count; i++) {
    xs[i] = Math.sin(i * 1e-3);
    ys[i] = Math.cos(i * 1e-3);
  }
  const saxpy = Fn(() => {
    const yi: Any = y.element(instanceIndex);
    yi.assign((x.element(instanceIndex) as Any).mul(float(2.5)).add(yi));
  })().compute(count);
  const frame = Array.from({ length: dispatches }, () => saxpy);
  return {
    id,
    particleCount: count,
    substeps: 1,
    iterations: 1,
    stepFrame: () => perf.renderer.computeAsync(frame),
  };
}
