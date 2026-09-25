// Phase Perf — Runner self-test (Guardrail 1).
//
// Synthetic two-kernel scene that exercises the harness's own correctness:
// each registered kernel becomes a separate JSON entry with non-zero p50
// and the statistics fields are populated.
//
// Picks deliberately distinct workloads so the two kernels report
// distinguishable p50 (heavier kernel slower) — that protects against
// accidental swaps in the harness.

import { Fn, float, instanceIndex, instancedArray } from 'three/tsl';
import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './PerfRenderer.js';
import { PerfRunner, type PerfSceneSpec } from './PerfRunner.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

function buildSyntheticScene(perf: PerfRenderer): PerfSceneSpec {
  const N = 1 << 18; // 256k elements
  const a = 2.5;

  const xNode = instancedArray(N, 'float');
  const yNode = instancedArray(N, 'float');

  const xArr = xNode.value.array as Float32Array;
  const yArr = yNode.value.array as Float32Array;
  for (let i = 0; i < N; i++) {
    xArr[i] = Math.sin(i * 1e-3);
    yArr[i] = Math.cos(i * 1e-3);
  }

  // Saxpy: y = a*x + y. Light per-element work.
  const saxpy = Fn(() => {
    const i = instanceIndex;
    const x = (xNode as Any).element(i);
    const y = (yNode as Any).element(i);
    y.assign(x.mul(float(a)).add(y));
  })().compute(N);

  // Heavier kernel — strictly more work than saxpy so durations differ.
  const heavy = Fn(() => {
    const i = instanceIndex;
    const x = (xNode as Any).element(i);
    const y = (yNode as Any).element(i);
    const r = x.mul(x).add(y.mul(y)).add(float(1.0)).sqrt().add(x.sin()).add(y.cos());
    y.assign(r);
  })().compute(N);

  return {
    id: 'self-test-synthetic',
    particleCount: N,
    substeps: 1,
    iterations: 1,
    stepFrame: async () => {
      // Advance state by running saxpy once per "frame" — same role as
      // `simLoop.step` in real scenes. Drains the timestamp pool so the
      // per-kernel isolation that follows starts from a clean slate.
      await perf.stepChain([saxpy]);
    },
    kernels: [
      { name: 'self-test.saxpy', kernel: saxpy, dispatchesPerFrame: 1 },
      { name: 'self-test.heavy', kernel: heavy, dispatchesPerFrame: 1 },
    ],
  };
}

describe('Phase Perf — runner self-test', () => {
  it('measures two synthetic kernels as separate entries with non-zero p50', async () => {
    const perf = await PerfRenderer.create();
    try {
      const runner = new PerfRunner(perf);
      const scene = buildSyntheticScene(perf);
      const result = await runner.runScene(scene, {
        warmup: 5,
        measure: 20,
      });

      expect(result.id).toBe('self-test-synthetic');
      expect(result.kernels).toHaveLength(2);

      const saxpy = result.kernels.find((k) => k.name === 'self-test.saxpy');
      const heavy = result.kernels.find((k) => k.name === 'self-test.heavy');
      expect(saxpy).toBeDefined();
      expect(heavy).toBeDefined();

      // Both kernels must report non-zero p50 (Outcome A) or non-zero
      // wall-clock (gross-only fallback). Either way > 0.
      expect(saxpy!.p50Ms).toBeGreaterThan(0);
      expect(heavy!.p50Ms).toBeGreaterThan(0);

      // Sample counts: measure × dispatchesPerFrame = 20 × 1 = 20.
      expect(saxpy!.samples).toBe(20);
      expect(heavy!.samples).toBe(20);

      // Frame total > 0.
      expect(result.frameTotalMs.p50).toBeGreaterThan(0);

      // Distinguishability: heavier kernel should report at least as
      // much p50 as saxpy. Allow 50% slack for noise, but flag a hard
      // crossover (heavy < 0.5 × saxpy means the two were swapped).
      expect(heavy!.p50Ms).toBeGreaterThan(saxpy!.p50Ms * 0.5);

      // eslint-disable-next-line no-console
      console.log(
        '[self-test] timing method=' +
          perf.timingMethod +
          ' saxpy.p50=' +
          saxpy!.p50Ms.toFixed(4) +
          'ms heavy.p50=' +
          heavy!.p50Ms.toFixed(4) +
          'ms frame.p50=' +
          result.frameTotalMs.p50.toFixed(2) +
          'ms',
      );
    } finally {
      perf.dispose();
    }
  }, 120_000);
});
