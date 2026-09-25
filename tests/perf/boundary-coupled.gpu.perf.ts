// Phase Perf-11 — boundary-coupled regression bench.
//
// Runs the production boundary-coupled scenes (static-boundary at 50k,
// rigid-only at 50k, dense-boundary at 50k) and emits a `PerfReportJson`
// to stdout between `__PARTICLE_FLUIDS_BOUNDARY_COUPLED_BEGIN__` /
// `__PARTICLE_FLUIDS_BOUNDARY_COUPLED_END__` sentinels for the
// `tests/perf/_helpers/run-boundary-coupled.ts` driver.
//

import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './_helpers/PerfRenderer.js';
import { PerfRunner, type PerfFrameWindow, type PerfSceneResult } from './_helpers/PerfRunner.js';
import type {
  PerfKernelJson,
  PerfReportJson,
  PerfSceneJson,
  PerfTimingMethodJson,
} from './_helpers/types.js';
import {
  buildFluidStaticBoundary,
  buildFluidRigidOnly,
  buildFluidDenseBoundary,
  type BuiltScene,
} from './_scenes/boundary-coupled.js';

// Body-coupled scenes' wall-clock noise floor is documented as ~7 % per
// U-50 — increasing measure beyond ~50 has diminishing returns for
// frame_step_ms attribution. Per-kernel samples are GPU-timestamp-based
// so they remain stable across the smaller window.
const DEFAULT_WINDOW: PerfFrameWindow = {
  warmup: 60,
  measure: 50,
};

const JSON_BEGIN = '__PARTICLE_FLUIDS_BOUNDARY_COUPLED_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_BOUNDARY_COUPLED_END__';

type Builder = (perf: PerfRenderer) => Promise<BuiltScene>;

const SCENE_BUILDERS: readonly Builder[] = [
  // Phase 11 production path: static boundary slab at 50k.
  (p) => buildFluidStaticBoundary(p, 50_000),
  // Rigid coupling production path: 3 cubes at 50k.
  (p) => buildFluidRigidOnly(p, 50_000),
  // Phase Perf-17 wet-cloth-shape regression gate: 625 dynamic boundary
  // particles at 50k (mirrors production wet-cloth scene).
  (p) => buildFluidDenseBoundary(p, 50_000),
];

function toKernelJson(k: PerfSceneResult['kernels'][number]): PerfKernelJson {
  return {
    name: k.name,
    dispatches_per_frame: k.dispatchesPerFrame,
    min_ms: k.minMs,
    p10_ms: k.p10Ms,
    p50_ms: k.p50Ms,
    p90_ms: k.p90Ms,
    max_ms: k.maxMs,
    samples: k.samples,
  };
}

function toSceneJson(r: PerfSceneResult): PerfSceneJson {
  return {
    id: r.id,
    particle_count: r.particleCount,
    substeps: r.substeps,
    iterations: r.iterations,
    frames_warmup: r.framesWarmup,
    frames_measure: r.framesMeasure,
    frame_total_ms: r.frameTotalMs,
    frame_step_ms: r.frameStepMs,
    dispatch_count: r.dispatchCount,
    ...(r.contactPairCount ? { contact_pair_count: r.contactPairCount } : {}),
    kernels: r.kernels.map(toKernelJson),
  };
}

function readEnvWindow(): PerfFrameWindow {
  const env =
    (typeof globalThis !== 'undefined' &&
      (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env) ||
    {};
  const warmup = Number(env['PARTICLE_FLUIDS_PERF_WARMUP'] ?? DEFAULT_WINDOW.warmup);
  const measure = Number(env['PARTICLE_FLUIDS_PERF_MEASURE'] ?? DEFAULT_WINDOW.measure);
  return {
    warmup: Number.isFinite(warmup) && warmup >= 0 ? warmup : DEFAULT_WINDOW.warmup,
    measure: Number.isFinite(measure) && measure > 0 ? measure : DEFAULT_WINDOW.measure,
  };
}

async function probePlatform(perf: PerfRenderer): Promise<{
  readonly gpu: string;
  readonly browser: string;
}> {
  let gpu = 'unknown';
  try {
    const backend = perf.renderer.backend as {
      readonly adapter?: {
        requestAdapterInfo?: () => Promise<{
          readonly description?: string;
          readonly vendor?: string;
          readonly architecture?: string;
          readonly device?: string;
        }>;
      };
    };
    const info = await backend.adapter?.requestAdapterInfo?.();
    if (info) {
      gpu = info.description || info.architecture || info.device || info.vendor || 'unknown';
    }
  } catch {
    // best-effort
  }
  const browser =
    typeof navigator !== 'undefined' && navigator.userAgent ? navigator.userAgent : 'unknown';
  return { gpu, browser };
}

describe('Phase Perf-11 — boundary-coupled bench', () => {
  it(
    'runs every boundary-coupled scene and emits a PerfReportJson',
    async () => {
      const window = readEnvWindow();
      const perf = await PerfRenderer.create();
      const platform = await probePlatform(perf);
      const sceneJsons: PerfSceneJson[] = [];

      try {
        const runner = new PerfRunner(perf);
        for (const build of SCENE_BUILDERS) {
          const built = await build(perf);
          try {
            const result = await runner.runScene(built.spec, window);
            sceneJsons.push(toSceneJson(result));
            // eslint-disable-next-line no-console
            console.log(
              '[Phase Perf-11] scene ' +
                result.id +
                ': frame_step_ms.p50=' +
                result.frameStepMs.p50.toFixed(2) +
                'ms (' +
                result.kernels.length +
                ' kernels' +
                (result.contactPairCount
                  ? ', pairs.p50=' + result.contactPairCount.p50.toFixed(0)
                  : '') +
                ')',
            );
          } finally {
            built.dispose();
          }
        }
      } finally {
        perf.dispose();
      }

      const timingMethod: PerfTimingMethodJson =
        perf.timingMethod === 'gross-only' ? 'gross-only' : 'per-kernel-pass';
      const report: PerfReportJson = {
        version: 1,
        commit: 'COMMIT_PLACEHOLDER',
        date: new Date().toISOString(),
        platform: {
          gpu: platform.gpu,
          browser: platform.browser,
          os: 'OS_PLACEHOLDER',
        },
        timing_method: timingMethod,
        scenes: sceneJsons,
      };

      // eslint-disable-next-line no-console
      console.log(JSON_BEGIN);
      // eslint-disable-next-line no-console
      console.log(JSON.stringify(report));
      // eslint-disable-next-line no-console
      console.log(JSON_END);

      expect(report.scenes.length).toBe(SCENE_BUILDERS.length);
      for (const s of report.scenes) {
        expect(s.kernels.length).toBeGreaterThan(0);
        expect(s.frame_step_ms.p50).toBeGreaterThan(0);
      }
    },
    60 * 60_000,
  );
});
