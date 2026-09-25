// Phase Perf — Runner forced gross-timing fallback (Guardrail 1).
//
// Constructs a `PerfRenderer` with `forceFallback: true` to exercise the
// `gross-only` timing path on a platform that natively supports
// Outcome A. Asserts:
//   1. `timingMethod === 'gross-only'`
//   2. Per-pipeline-segment timings (frame-total + per-kernel) are
//      produced and non-zero — proves the fallback's
//      `performance.now() + onSubmittedWorkDone()` path works end-to-end.
//

import { Fn, float, instanceIndex, instancedArray } from 'three/tsl';
import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './PerfRenderer.js';
import { PerfRunner, type PerfSceneSpec } from './PerfRunner.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

function buildScene(perf: PerfRenderer): PerfSceneSpec {
  const N = 1 << 17;

  const xNode = instancedArray(N, 'float');
  const yNode = instancedArray(N, 'float');
  const xArr = xNode.value.array as Float32Array;
  const yArr = yNode.value.array as Float32Array;
  for (let i = 0; i < N; i++) {
    xArr[i] = Math.sin(i * 1e-3);
    yArr[i] = Math.cos(i * 1e-3);
  }

  const saxpy = Fn(() => {
    const i = instanceIndex;
    const x = (xNode as Any).element(i);
    const y = (yNode as Any).element(i);
    y.assign(x.mul(float(2.5)).add(y));
  })().compute(N);

  return {
    id: 'fallback-synthetic',
    particleCount: N,
    substeps: 1,
    iterations: 1,
    stepFrame: async () => {
      await perf.stepChain([saxpy]);
    },
    kernels: [{ name: 'self-test.saxpy', kernel: saxpy, dispatchesPerFrame: 1 }],
  };
}

describe('Phase Perf — runner forced fallback', () => {
  it('reports timingMethod=gross-only and produces non-zero timings', async () => {
    const perf = await PerfRenderer.create({ forceFallback: true });
    try {
      expect(perf.timingMethod).toBe('gross-only');

      const runner = new PerfRunner(perf);
      const result = await runner.runScene(buildScene(perf), {
        warmup: 3,
        measure: 10,
      });

      expect(result.kernels).toHaveLength(1);
      expect(result.kernels[0]!.p50Ms).toBeGreaterThan(0);
      expect(result.frameTotalMs.p50).toBeGreaterThan(0);

      // eslint-disable-next-line no-console
      console.log(
        '[fallback] timingMethod=' +
          perf.timingMethod +
          ' kernel.p50=' +
          result.kernels[0]!.p50Ms.toFixed(3) +
          'ms frame.p50=' +
          result.frameTotalMs.p50.toFixed(3) +
          'ms',
      );
    } finally {
      perf.dispose();
    }
  }, 120_000);
});
