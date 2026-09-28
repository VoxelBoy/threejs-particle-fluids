// JSON written by the benchmark run. `run.ts` saves one report per run and
// inlines it into the HTML report template.

export interface PerfQuantileBlock {
  readonly p10: number;
  readonly p50: number;
  readonly p90: number;
}

export interface PerfStatsJson extends PerfQuantileBlock {
  readonly min: number;
  readonly max: number;
}

export interface PerfPlatformInfo {
  readonly gpu: string;
  readonly browser: string;
  readonly os: string;
}

/**
 * - `timestamp`: the GPU reports each frame's compute time through
 *   timestamp queries, so scenes carry `frame_gpu_ms`.
 * - `wall-clock`: the device has no `timestamp-query` feature; only
 *   `frame_step_ms` is measured.
 */
export type PerfTimingMethodJson = 'timestamp' | 'wall-clock';

export interface PerfSceneJson {
  readonly id: string;
  /** Every particle in the simulation, including pinned boundary particles. */
  readonly particle_count: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly frames_warmup: number;
  readonly frames_measure: number;
  /**
   * GPU time of one `SimLoop.step` (the whole frame runs as one compute
   * pass), from timestamp queries. Present when `timing_method` is
   * `timestamp`.
   */
  readonly frame_gpu_ms?: PerfStatsJson;
  /**
   * Wall-clock time from calling `SimLoop.step` until the GPU reports the
   * frame done. Includes CPU-side encoding and scheduling noise.
   */
  readonly frame_step_ms: PerfStatsJson;
  /** Contact pairs found per substep, sampled once per measured frame. Scenes with contacts only. */
  readonly contact_count?: PerfQuantileBlock;
}

export interface PerfReportJson {
  readonly version: 2;
  readonly commit: string;
  readonly date: string;
  readonly platform: PerfPlatformInfo;
  readonly timing_method: PerfTimingMethodJson;
  readonly scenes: readonly PerfSceneJson[];
}

/** One kernel's GPU time per frame, from the profile suite. */
export interface ProfileKernelJson {
  readonly name: string;
  /** Mean GPU milliseconds per frame, summed over the kernel's dispatches. */
  readonly ms: number;
  /** Dispatches per frame. */
  readonly calls: number;
}

export interface ProfileSceneJson {
  readonly id: string;
  readonly particle_count: number;
  readonly substeps: number;
  readonly iterations: number;
  /** Median GPU time of one frame's simulation, submitted as the demo submits it. */
  readonly sim_gpu_ms: number;
  /** Median GPU time of one frame's render preparation (liquid surface, smoke volume). */
  readonly render_gpu_ms?: number;
  /** Simulation kernels, slowest first, each timed in its own pass. */
  readonly kernels: readonly ProfileKernelJson[];
  readonly render_kernels?: readonly ProfileKernelJson[];
}

export interface ProfileReportJson {
  readonly version: 1;
  readonly commit: string;
  readonly date: string;
  /** Particle level the demo presets were built at. */
  readonly level: string;
  /** `warmup` is −1 when each preset warmed up for half its demo duration. */
  readonly frames: { readonly warmup: number; readonly measure: number; readonly profile: number };
  readonly scenes: readonly ProfileSceneJson[];
}
