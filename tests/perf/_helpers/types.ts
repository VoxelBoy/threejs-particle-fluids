// Phase Perf — JSON output schema. The harness writes one of these per
// run; the HTML report consumes it via inline `__PARTICLE_FLUIDS_PERF_DATA__`
// substitution.

export interface PerfQuantileBlock {
  readonly p10: number;
  readonly p50: number;
  readonly p90: number;
}

export interface PerfPlatformInfo {
  readonly gpu: string;
  readonly browser: string;
  readonly os: string;
}

export type PerfTimingMethodJson = 'per-kernel-pass' | 'per-kernel-stage-boundary' | 'gross-only';

export interface PerfKernelJson {
  readonly name: string;
  readonly dispatches_per_frame: number;
  readonly min_ms: number;
  readonly p10_ms: number;
  readonly p50_ms: number;
  readonly p90_ms: number;
  readonly max_ms: number;
  readonly samples: number;
}

export interface PerfSceneJson {
  readonly id: string;
  readonly particle_count: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly frames_warmup: number;
  readonly frames_measure: number;
  /** stepFrame + per-kernel-isolation overhead. */
  readonly frame_total_ms: PerfQuantileBlock;
  /** Production-equivalent frame cost (stepFrame only). */
  readonly frame_step_ms: PerfQuantileBlock;
  /** Per-frame isolation-loop dispatch count. */
  readonly dispatch_count: number;
  /**
   * Phase Perf-11 H3: contact-pair count per frame, sampled once per
   * measure frame from `ContactBuffer.readbackCount()`. Optional —
   * present only for scenes that opt in via `contactPairCountReadback`
   * on `PerfSceneSpec`. Pure-fluid scenes that never construct a
   * contact pipeline omit this field.
   */
  readonly contact_pair_count?: PerfQuantileBlock;
  readonly kernels: readonly PerfKernelJson[];
}

export interface PerfReportJson {
  readonly version: 1;
  readonly commit: string;
  readonly date: string;
  readonly platform: PerfPlatformInfo;
  readonly timing_method: PerfTimingMethodJson;
  readonly scenes: readonly PerfSceneJson[];
}
