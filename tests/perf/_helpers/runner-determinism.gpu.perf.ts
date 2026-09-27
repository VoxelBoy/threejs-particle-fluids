// Runs the same synthetic scene twice in one process and checks that the
// frame times agree within a factor of ten. It's a gate for broken timing
// (stale reads, samples from the wrong frame), not for measurement noise:
// small workloads swing by tens of percent between runs as GPU clocks ramp
// up and down.

import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './PerfRenderer.js';
import { PerfRunner, type PerfSceneResult } from './PerfRunner.js';
import { saxpyScene } from './synthetic.js';

const WINDOW = { warmup: 5, measure: 30 };

/** GPU time when the device has timestamps, wall-clock otherwise. */
function frameP50(result: PerfSceneResult): number {
  return (result.gpuFrameMs ?? result.stepFrameMs).p50;
}

describe('PerfRunner determinism', () => {
  it('reports frame p50s within a factor of ten across two runs', async () => {
    const perf = await PerfRenderer.create();
    try {
      const runner = new PerfRunner(perf);
      const scene = () => saxpyScene(perf, { id: 'determinism', count: 1 << 20, dispatches: 4 });
      const p1 = frameP50(await runner.runScene(scene(), WINDOW));
      const p2 = frameP50(await runner.runScene(scene(), WINDOW));
      const ratio = Math.abs(p1 - p2) / Math.max(p1, p2);

      console.log(
        `[determinism] timing=${perf.timingMethod} run 1 p50=${p1.toFixed(3)} ms, ` +
          `run 2 p50=${p2.toFixed(3)} ms, difference ${(ratio * 100).toFixed(1)}%`,
      );

      expect(p1).toBeGreaterThan(0);
      expect(p2).toBeGreaterThan(0);
      expect(ratio).toBeLessThan(0.9);
    } finally {
      perf.dispose();
    }
  }, 180_000);
});
