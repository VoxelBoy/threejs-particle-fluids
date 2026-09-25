// Phase Perf — Runner same-process determinism smoke (Guardrail 1).
//

//
// `frameTotalMs` is wall-clock, susceptible to host-side scheduling and
// thermal jitter. On a 128k-element saxpy where GPU compute is ~30 µs
// per dispatch, a single slow fence dominates the median, producing
// ~50–70% run-to-run swings even when the harness is behaving correctly
// (see the warmup-sweep diagnosis in scripts/perf-warmup-sweep.ts).
// Threshold is 90% — catastrophic-regression gate, not a tightness gate.
// Per-kernel timings are GPU-timestamp-based and the right place to
// gate on tighter bounds; this test exists only to catch order-of-
// magnitude harness breakage.

import { Fn, float, instanceIndex, instancedArray } from 'three/tsl';
import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './PerfRenderer.js';
import { PerfRunner, type PerfSceneSpec } from './PerfRunner.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

function buildScene(perf: PerfRenderer): PerfSceneSpec {
  const N = 1 << 17; // 128k elements — small to keep the test fast

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
    id: 'determinism-synthetic',
    particleCount: N,
    substeps: 1,
    iterations: 1,
    stepFrame: async () => {
      await perf.stepChain([saxpy]);
    },
    kernels: [{ name: 'self-test.saxpy', kernel: saxpy, dispatchesPerFrame: 1 }],
  };
}

describe('Phase Perf — runner determinism smoke', () => {
  it('reports frame-total p50 within 90% across two same-process runs', async () => {
    const perf = await PerfRenderer.create();
    try {
      const runner = new PerfRunner(perf);

      const r1 = await runner.runScene(buildScene(perf), {
        warmup: 5,
        measure: 30,
      });
      const r2 = await runner.runScene(buildScene(perf), {
        warmup: 5,
        measure: 30,
      });

      const p1 = r1.frameTotalMs.p50;
      const p2 = r2.frameTotalMs.p50;
      const ratio = Math.abs(p1 - p2) / Math.max(p1, p2);

      // eslint-disable-next-line no-console
      console.log(
        '[determinism] r1.p50=' +
          p1.toFixed(3) +
          'ms r2.p50=' +
          p2.toFixed(3) +
          'ms ratio=' +
          (ratio * 100).toFixed(1) +
          '%',
      );

      expect(p1).toBeGreaterThan(0);
      expect(p2).toBeGreaterThan(0);
      expect(ratio).toBeLessThan(0.9);
    } finally {
      perf.dispose();
    }
  }, 180_000);
});
