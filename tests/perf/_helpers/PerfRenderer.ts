import { TimestampQuery } from 'three/src/constants.js';
import type { WebGPURenderer } from 'three/webgpu';

import { createParticleRenderer } from '../../../src/index.js';

export interface PerfRendererOptions {
  /**
   * Measure wall-clock time only, even when the device supports
   * `timestamp-query`. Lets `runner-fallback.gpu.perf.ts` exercise the
   * fallback path on any machine.
   */
  readonly forceFallback?: boolean;
}

/**
 * - `timestamp`: three.js writes GPU timestamps at the start and end of every
 *   compute pass, and {@link PerfRenderer.readGpuMs} reads them back.
 * - `wall-clock`: no GPU timestamps; the runner times frames on the CPU.
 */
export type PerfTimingMethod = 'timestamp' | 'wall-clock';

/** A WebGPU renderer set up for benchmarking, and the GPU timing it supports. */
export class PerfRenderer {
  readonly renderer: WebGPURenderer;
  readonly device: GPUDevice;
  readonly timingMethod: PerfTimingMethod;

  static async create(options: PerfRendererOptions = {}): Promise<PerfRenderer> {
    const renderer = await createParticleRenderer({ trackTimestamp: !options.forceFallback });
    const device = (renderer.backend as { readonly device?: GPUDevice }).device;
    if (!device) {
      renderer.dispose();
      throw new Error('PerfRenderer.create: the renderer has no GPUDevice after init');
    }
    const timestamps = !options.forceFallback && device.features.has('timestamp-query');
    return new PerfRenderer(renderer, device, timestamps ? 'timestamp' : 'wall-clock');
  }

  /** WebGPU errors no error scope caught, such as a pipeline that failed to build. */
  readonly errors: string[] = [];

  private constructor(renderer: WebGPURenderer, device: GPUDevice, timingMethod: PerfTimingMethod) {
    this.renderer = renderer;
    this.device = device;
    this.timingMethod = timingMethod;
    // A failed pipeline makes WebGPU skip its dispatches silently, which
    // would read as a speedup. Collect the errors so runs can fail on them.
    device.addEventListener('uncapturederror', (event) => {
      this.errors.push((event as GPUUncapturedErrorEvent).error.message);
    });
  }

  /** Throw if WebGPU reported an error since the last call. */
  assertNoErrors(context: string): void {
    if (this.errors.length === 0) return;
    const messages = this.errors.splice(0);
    throw new Error(`${context}: WebGPU reported ${messages.length} error(s): ${messages[0]}`);
  }

  /**
   * GPU milliseconds of the compute pass submitted since the previous read.
   * `SimLoop.step` submits a whole frame as one pass, so this is the frame's
   * GPU time. Read after every submission: three.js files every array
   * submission's queries under the same key, so of two array submissions
   * between reads only the last would be counted.
   */
  async readGpuMs(): Promise<number> {
    this.assertTimestamps();
    const ms = await this.renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
    if (ms === undefined || !Number.isFinite(ms)) {
      throw new Error(`PerfRenderer.readGpuMs: timestamp resolve returned ${ms}`);
    }
    return ms;
  }

  /** Drop pending GPU timings, e.g. from warmup frames. Fine to call when none are pending. */
  async discardGpuTimings(): Promise<void> {
    this.assertTimestamps();
    await this.renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
  }

  dispose(): void {
    this.renderer.dispose();
  }

  private assertTimestamps(): void {
    if (this.timingMethod !== 'timestamp') {
      throw new Error('PerfRenderer: GPU timestamps are unavailable in wall-clock mode');
    }
  }
}
