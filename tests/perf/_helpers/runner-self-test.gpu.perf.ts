// Checks that the runner times whole frames: every measured frame yields one
// sample, the statistics are ordered, and a frame with 16× the GPU work
// reports clearly more GPU time than a light one (which would catch swapped
// or stale timestamp reads).

import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './PerfRenderer.js';
import { PerfRunner, type PerfStats } from './PerfRunner.js';
import { saxpyScene } from './synthetic.js';

const COUNT = 1 << 20;
const WINDOW = { warmup: 5, measure: 20 };

function expectOrdered(stats: PerfStats): void {
  expect(stats.min).toBeGreaterThan(0);
  expect(stats.min).toBeLessThanOrEqual(stats.p10);
  expect(stats.p10).toBeLessThanOrEqual(stats.p50);
  expect(stats.p50).toBeLessThanOrEqual(stats.p90);
  expect(stats.p90).toBeLessThanOrEqual(stats.max);
}

describe('PerfRunner', () => {
  it('reports one sample per measured frame and separates light from heavy frames', async () => {
    const perf = await PerfRenderer.create();
    try {
      const runner = new PerfRunner(perf);
      const light = await runner.runScene(
        saxpyScene(perf, { id: 'light', count: COUNT, dispatches: 1 }),
        WINDOW,
      );
      const heavy = await runner.runScene(
        saxpyScene(perf, { id: 'heavy', count: COUNT, dispatches: 16 }),
        WINDOW,
      );

      for (const result of [light, heavy]) {
        expect(result.framesWarmup).toBe(WINDOW.warmup);
        expect(result.framesMeasure).toBe(WINDOW.measure);
        expect(result.stepFrameMs.samples).toBe(WINDOW.measure);
        expectOrdered(result.stepFrameMs);
        expect(result.contactCount).toBeUndefined();
      }

      console.log(
        `[self-test] timing=${perf.timingMethod} ` +
          `light GPU p50=${light.gpuFrameMs?.p50.toFixed(4)} ms, ` +
          `heavy GPU p50=${heavy.gpuFrameMs?.p50.toFixed(4)} ms, ` +
          `light wall p50=${light.stepFrameMs.p50.toFixed(3)} ms`,
      );

      if (perf.timingMethod === 'timestamp') {
        expect(light.gpuFrameMs?.samples).toBe(WINDOW.measure);
        expect(heavy.gpuFrameMs?.samples).toBe(WINDOW.measure);
        expectOrdered(light.gpuFrameMs!);
        expectOrdered(heavy.gpuFrameMs!);
        expect(heavy.gpuFrameMs!.p50).toBeGreaterThan(4 * light.gpuFrameMs!.p50);
      } else {
        expect(light.gpuFrameMs).toBeUndefined();
      }
    } finally {
      perf.dispose();
    }
  }, 120_000);
});
