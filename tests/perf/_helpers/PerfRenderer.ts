import { TimestampQuery } from 'three/src/constants.js';
import type { WebGPURenderer } from 'three/webgpu';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import { createParticleRenderer } from '../../../src/core/index.js';

export interface PerfRendererOptions {
  /**
   * Force the gross-timing fallback path even if `timestamp-query` is
   * available on the device. Used by `runner-fallback.gpu.perf.ts` to
   * exercise the fallback path on a platform that natively supports
   * Outcome A.
   */
  readonly forceFallback?: boolean;
}

export type PerfTimingMethod = 'per-kernel-pass' | 'gross-only';

export class PerfRenderer {
  readonly renderer: WebGPURenderer;
  readonly device: GPUDevice;
  readonly timingMethod: PerfTimingMethod;

  static async create(options: PerfRendererOptions = {}): Promise<PerfRenderer> {
    const trackTimestamp = !options.forceFallback;
    const renderer = await createParticleRenderer({ trackTimestamp });

    const backend = renderer.backend as { readonly device?: GPUDevice };
    const device = backend.device;
    if (!device) {
      renderer.dispose();
      throw new Error(
        'PerfRenderer.create: renderer.backend.device unavailable after init (U-43 surface check failed)',
      );
    }

    const hasTimestampQuery = !options.forceFallback && device.features.has('timestamp-query');
    const timingMethod: PerfTimingMethod = hasTimestampQuery ? 'per-kernel-pass' : 'gross-only';

    return new PerfRenderer(renderer, device, timingMethod);
  }

  private constructor(renderer: WebGPURenderer, device: GPUDevice, timingMethod: PerfTimingMethod) {
    this.renderer = renderer;
    this.device = device;
    this.timingMethod = timingMethod;
  }

  /**
   * Dispatch a single kernel and return its measured GPU cost in ms.
   *
   * Per-kernel-pass mode: relies on three.js's `WebGPUBackend.beginCompute`
   * auto-injecting `timestampWrites` into the compute-pass descriptor when
   * `trackTimestamp: true`. Each `computeAsync(kernel)` runs ONE compute
   * pass; `resolveTimestampsAsync(TimestampQuery.COMPUTE)` reads back the
   * per-pass duration in ms. State carries forward — when the same kernel
   * is dispatched repeatedly (per the plan's `dispatchesPerFrame` design)
   * each timing reflects realistic per-frame state.
   *
   * Gross-only mode: brackets `computeAsync` with `performance.now()` and
   * waits for GPU completion via `device.queue.onSubmittedWorkDone()`.
   * Returns wall-clock ms which includes JS dispatch overhead — coarser
   * but works on any platform, including those without `timestamp-query`.
   */
  async runKernelInIsolation(kernel: ComputeNode): Promise<number> {
    if (this.timingMethod === 'per-kernel-pass') {
      await this.renderer.computeAsync(kernel);
      const ms = await this.renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
      if (ms === undefined || !Number.isFinite(ms)) {
        throw new Error(
          `PerfRenderer.runKernelInIsolation: timestamp resolve returned ${ms}; expected a finite ms value`,
        );
      }
      return ms;
    } else {
      const t0 = performance.now();
      await this.renderer.computeAsync(kernel);
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    }
  }

  /**
   * Run a chain of kernels (e.g. `simLoop.computeNodes`) in one
   * `computeAsync` call. Used by the runner to advance simulation state
   * each frame between per-kernel isolation measurements. Per-kernel-pass
   * mode also resolves the timestamp pool here so the per-kernel
   * measurements that follow start from a clean slate.
   */
  async stepChain(kernels: readonly ComputeNode[]): Promise<void> {
    if (kernels.length === 0) return;
    await this.renderer.computeAsync(kernels as ComputeNode[]);
    if (this.timingMethod === 'per-kernel-pass') {
      // Drain the pool so the next per-kernel-pass measurement is not
      // contaminated by the chain's accumulated timestamps.
      await this.renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
    }
  }

  dispose(): void {
    this.renderer.dispose();
  }
}
