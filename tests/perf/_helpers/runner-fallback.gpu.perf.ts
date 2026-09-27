// Forces the wall-clock fallback that devices without `timestamp-query` use,
// and checks that the runner still times frames and leaves GPU times out.

import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './PerfRenderer.js';
import { PerfRunner } from './PerfRunner.js';
import { saxpyScene } from './synthetic.js';

describe('PerfRunner wall-clock fallback', () => {
  it('times frames on the CPU and reports no GPU times', async () => {
    const perf = await PerfRenderer.create({ forceFallback: true });
    try {
      expect(perf.timingMethod).toBe('wall-clock');
      await expect(perf.readGpuMs()).rejects.toThrow(/wall-clock/);

      const runner = new PerfRunner(perf);
      const result = await runner.runScene(
        saxpyScene(perf, { id: 'fallback', count: 1 << 17, dispatches: 1 }),
        { warmup: 3, measure: 10 },
      );

      expect(result.gpuFrameMs).toBeUndefined();
      expect(result.stepFrameMs.samples).toBe(10);
      expect(result.stepFrameMs.p50).toBeGreaterThan(0);

      console.log(`[fallback] wall-clock frame p50=${result.stepFrameMs.p50.toFixed(3)} ms`);
    } finally {
      perf.dispose();
    }
  }, 120_000);
});
