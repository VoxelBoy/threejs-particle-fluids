// Phase Perf — `PerfRunner` orchestrates one perf-bench run: warmup,
// per-frame `simLoop.step` advance, per-kernel isolation timing, and
// quantile statistics.

import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type { PerfRenderer } from './PerfRenderer.js';

export interface PerfKernelSpec {
  /**
   * Stable, dotted name (e.g. `hashGrid.cellIndexAndHistogram`,
   * `fluid.density`, `contact.solve`). Stable across phases — adding a
   * kernel adds a row in the JSON; renaming an existing kernel breaks
   * comparison and requires a baseline re-snapshot.
   */
  readonly name: string;
  readonly kernel: ComputeNode;
  readonly dispatchesPerFrame: number;
}

export interface PerfFrameWindow {
  readonly warmup: number;
  readonly measure: number;
}

export interface PerfQuantileBlock {
  readonly p10: number;
  readonly p50: number;
  readonly p90: number;
}

export interface PerfKernelStats {
  readonly name: string;
  readonly dispatchesPerFrame: number;
  readonly minMs: number;
  readonly p10Ms: number;
  readonly p50Ms: number;
  readonly p90Ms: number;
  readonly maxMs: number;
  readonly samples: number;
}

export interface PerfSceneResult {
  readonly id: string;
  readonly particleCount: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly framesWarmup: number;
  readonly framesMeasure: number;
  /**
   * Wall-clock around `stepFrame` AND the per-kernel-isolation loop.
   * Inflated by the isolation overhead.
   */
  readonly frameTotalMs: PerfQuantileBlock;
  /**
   * Wall-clock bracketing only `await scene.stepFrame()` — i.e. the
   * production-equivalent frame cost without the harness's per-kernel
   * isolation dispatches. Compare against `frameTotalMs` to quantify
   * the isolation overhead inflation.
   */
  readonly frameStepMs: PerfQuantileBlock;
  /**
   * Total number of `runKernelInIsolation` dispatches per measure
   * frame = `Σ k.dispatchesPerFrame for k in scene.kernels`.
   */
  readonly dispatchCount: number;
  /**
   * Phase Perf-11 H3: contact-pair count quantiles, present only when
   * the scene supplied `contactPairCountReadback`. Sampled once per
   * measure frame.
   */
  readonly contactPairCount?: PerfQuantileBlock;
  readonly kernels: readonly PerfKernelStats[];
}

export interface PerfSceneSpec {
  readonly id: string;
  readonly particleCount: number;
  readonly substeps: number;
  readonly iterations: number;
  /** Advance simulation state by one frame (typically `await simLoop.step(dt)`). */
  readonly stepFrame: () => Promise<void>;
  /** Kernels to time in isolation. Order is preserved in JSON output. */
  readonly kernels: readonly PerfKernelSpec[];
  /**
   * Phase Perf-11 H3: optional readback of the per-frame contact-pair
   * count. When present the runner samples this once per measure frame
   * and reports `contactPairCount` quantiles. Reads are after
   * `stepFrame()` resolves; the readback stalls the GPU but is gated
   * behind opt-in so non-contact scenes pay nothing.
   */
  readonly contactPairCountReadback?: () => Promise<number>;
}

/** Quantile of a sorted array. Linear interpolation; q in [0, 1]. */
function quantileSorted(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0]!;
  const idx = q * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  const t = idx - lo;
  return sorted[lo]! * (1 - t) + sorted[hi]! * t;
}

function quantileBlockFrom(samples: readonly number[]): PerfQuantileBlock {
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    p10: quantileSorted(sorted, 0.1),
    p50: quantileSorted(sorted, 0.5),
    p90: quantileSorted(sorted, 0.9),
  };
}

function statsOf(samples: readonly number[]): {
  minMs: number;
  p10Ms: number;
  p50Ms: number;
  p90Ms: number;
  maxMs: number;
  samples: number;
} {
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    minMs: sorted[0] ?? 0,
    p10Ms: quantileSorted(sorted, 0.1),
    p50Ms: quantileSorted(sorted, 0.5),
    p90Ms: quantileSorted(sorted, 0.9),
    maxMs: sorted[sorted.length - 1] ?? 0,
    samples: sorted.length,
  };
}

export class PerfRunner {
  constructor(private readonly perfRenderer: PerfRenderer) {}

  async runScene(scene: PerfSceneSpec, window: PerfFrameWindow): Promise<PerfSceneResult> {
    if (window.warmup < 0 || window.measure <= 0) {
      throw new Error(
        `PerfRunner.runScene: invalid window { warmup: ${window.warmup}, measure: ${window.measure} }`,
      );
    }

    // Warmup — pipelines build, caches warm, sleeping particles wake.
    for (let i = 0; i < window.warmup; i++) {
      await scene.stepFrame();
    }

    const kernelTimings = new Map<string, number[]>();
    for (const k of scene.kernels) kernelTimings.set(k.name, []);
    const frameTotals: number[] = [];
    const frameSteps: number[] = [];
    const contactPairSamples: number[] = [];

    for (let i = 0; i < window.measure; i++) {
      const tFrameStart = performance.now();

      // Bracket stepFrame separately so frameStepMs reports the
      // production-equivalent frame cost without harness overhead.
      // `await scene.stepFrame()` resolves when the WebGPU work has
      // been QUEUED, not when the GPU has finished executing it. Add
      // `onSubmittedWorkDone()` to bracket GPU completion explicitly.
      const tStepStart = performance.now();
      await scene.stepFrame();
      await this.perfRenderer.device.queue.onSubmittedWorkDone();
      frameSteps.push(performance.now() - tStepStart);

      if (scene.contactPairCountReadback) {
        contactPairSamples.push(await scene.contactPairCountReadback());
      }

      // Per-kernel isolation. Each kernel runs `dispatchesPerFrame` times
      // per measure frame so total samples per kernel = measure ×
      // dispatchesPerFrame. These dispatches are pure measurement
      // overhead; they are NOT part of frameStepMs.
      for (const k of scene.kernels) {
        const samples = kernelTimings.get(k.name)!;
        for (let d = 0; d < k.dispatchesPerFrame; d++) {
          const ms = await this.perfRenderer.runKernelInIsolation(k.kernel);
          samples.push(ms);
        }
      }
      frameTotals.push(performance.now() - tFrameStart);
    }

    const dispatchCount = scene.kernels.reduce((sum, k) => sum + k.dispatchesPerFrame, 0);

    const kernelStats: PerfKernelStats[] = scene.kernels.map((k) => {
      const samples = kernelTimings.get(k.name)!;
      return {
        name: k.name,
        dispatchesPerFrame: k.dispatchesPerFrame,
        ...statsOf(samples),
      };
    });

    return {
      id: scene.id,
      particleCount: scene.particleCount,
      substeps: scene.substeps,
      iterations: scene.iterations,
      framesWarmup: window.warmup,
      framesMeasure: window.measure,
      frameTotalMs: quantileBlockFrom(frameTotals),
      frameStepMs: quantileBlockFrom(frameSteps),
      dispatchCount,
      ...(contactPairSamples.length > 0
        ? { contactPairCount: quantileBlockFrom(contactPairSamples) }
        : {}),
      kernels: kernelStats,
    };
  }
}
