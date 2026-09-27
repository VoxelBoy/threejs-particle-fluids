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
