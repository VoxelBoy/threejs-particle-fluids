// Runs one benchmark scene: warm it up, then time a window of frames and
// summarize the samples.

import type { PerfRenderer } from './PerfRenderer.js';

export interface PerfSceneSpec {
  readonly id: string;
  /** Every particle in the simulation. */
  readonly particleCount: number;
  readonly substeps: number;
  readonly iterations: number;
  /**
   * Advance the simulation one frame with a single compute submission,
   * typically `loop.step(dt)`.
   */
  readonly stepFrame: () => Promise<void>;
  /**
   * Contact pairs found in the last substep, e.g.
   * `loop.contacts.readbackCount()`. Read once per measured frame, after the
   * frame is timed.
   */
  readonly readContactCount?: () => Promise<number>;
}

export interface PerfFrameWindow {
  readonly warmup: number;
  readonly measure: number;
}

export interface PerfQuantiles {
  readonly p10: number;
  readonly p50: number;
  readonly p90: number;
}

export interface PerfStats extends PerfQuantiles {
  readonly min: number;
  readonly max: number;
  readonly samples: number;
}

export interface PerfSceneResult {
  readonly id: string;
  readonly particleCount: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly framesWarmup: number;
  readonly framesMeasure: number;
  /** GPU time per frame from timestamp queries. Absent in wall-clock mode. */
  readonly gpuFrameMs?: PerfStats;
  /** Wall-clock time from starting the frame until the GPU reports it done. */
  readonly stepFrameMs: PerfStats;
  /** Contact pairs per substep, when the scene reads them. */
  readonly contactCount?: PerfQuantiles;
}

/** Quantile of a sorted array, linearly interpolated; `q` in [0, 1]. */
function quantileSorted(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  const idx = q * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const t = idx - lo;
  return sorted[lo]! * (1 - t) + sorted[hi]! * t;
}

function statsOf(samples: readonly number[]): PerfStats {
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    min: sorted[0] ?? 0,
    p10: quantileSorted(sorted, 0.1),
    p50: quantileSorted(sorted, 0.5),
    p90: quantileSorted(sorted, 0.9),
    max: sorted[sorted.length - 1] ?? 0,
    samples: sorted.length,
  };
}

export class PerfRunner {
  constructor(private readonly perf: PerfRenderer) {}

  async runScene(scene: PerfSceneSpec, frames: PerfFrameWindow): Promise<PerfSceneResult> {
    if (!(frames.warmup >= 0) || !(frames.measure > 0)) {
      throw new Error(
        `PerfRunner.runScene: invalid window { warmup: ${frames.warmup}, measure: ${frames.measure} }`,
      );
    }
    const timestamps = this.perf.timingMethod === 'timestamp';
    // Start from an empty timestamp pool so scene setup can't leak into the first sample.
    if (timestamps) await this.perf.discardGpuTimings();

    // Warmup compiles pipelines and lets the scene move past its first frames.
    // Its timestamps are dropped every frame so the query pool never fills up.
    for (let i = 0; i < frames.warmup; i++) {
      await scene.stepFrame();
      if (timestamps) await this.perf.discardGpuTimings();
    }

    const gpu: number[] = [];
    const step: number[] = [];
    const contacts: number[] = [];
    for (let i = 0; i < frames.measure; i++) {
      const start = performance.now();
      await scene.stepFrame();
      // stepFrame resolves once the work is queued; wait for the GPU to finish it.
      await this.perf.device.queue.onSubmittedWorkDone();
      step.push(performance.now() - start);
      if (timestamps) gpu.push(await this.perf.readGpuMs());
      if (scene.readContactCount) contacts.push(await scene.readContactCount());
    }

    const contactStats = contacts.length > 0 ? statsOf(contacts) : undefined;
    return {
      id: scene.id,
      particleCount: scene.particleCount,
      substeps: scene.substeps,
      iterations: scene.iterations,
      framesWarmup: frames.warmup,
      framesMeasure: frames.measure,
      ...(timestamps ? { gpuFrameMs: statsOf(gpu) } : {}),
      stepFrameMs: statsOf(step),
      ...(contactStats
        ? { contactCount: { p10: contactStats.p10, p50: contactStats.p50, p90: contactStats.p90 } }
        : {}),
    };
  }
}
