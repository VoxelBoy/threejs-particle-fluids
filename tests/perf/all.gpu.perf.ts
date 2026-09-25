// Phase Perf — top-level perf bench. Runs every scene in the library,
// collects results into a single `PerfReportJson`, and emits the JSON
// to stdout between sentinel markers so the `run.ts` driver can extract
// it and write to disk.
//
// Window: 20 warmup + 50 measure frames per scene. Tunable via
// PARTICLE_FLUIDS_PERF_WARMUP / PARTICLE_FLUIDS_PERF_MEASURE env vars.

import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './_helpers/PerfRenderer.js';
import { PerfRunner, type PerfFrameWindow, type PerfSceneResult } from './_helpers/PerfRunner.js';
import type {
  PerfKernelJson,
  PerfReportJson,
  PerfSceneJson,
  PerfTimingMethodJson,
} from './_helpers/types.js';
import { buildFluid10kScene, buildFluid100kScene } from './_scenes/fluid.js';
import { buildFluidContact10kScene, buildFluidContact100kScene } from './_scenes/fluid-contact.js';
import {
  buildFluidSurfaceTension10kScene,
  buildFluidSurfaceTension100kScene,
} from './_scenes/fluid-surface-tension.js';
import { buildSoftbody10kScene, buildSoftbody100kScene } from './_scenes/softbody.js';

// 4 scene types × {10k, 100k} = 8 scenes. Order: 10k of every type
// first, then 100k of every type. Lets the user spot small-scale
// regressions before the long-running 100k scenes finish.
const SCENE_BUILDERS = [
  buildFluid10kScene,
  buildFluidContact10kScene,
  buildFluidSurfaceTension10kScene,
  buildSoftbody10kScene,
  buildFluid100kScene,
  buildFluidContact100kScene,
  buildFluidSurfaceTension100kScene,
  buildSoftbody100kScene,
];

// Scenes step at dt = 1/60, so 60 warmup frames = 1 simulated second —
// enough for a 0.5 m drop to land and the column to start compressing.
//
// measure=50 is a deliberate choice: per-kernel timings are GPU-timestamp-
// based (zero jitter), and 50 frames × dispatchesPerFrame gives 200-400
// samples per kernel — plenty for a stable p50. measure=200 would only
// improve frame_step_ms stability, which is wall-clock and noise-prone
// enough that it isn't trusted as a comparison metric anyway.
const DEFAULT_WINDOW: PerfFrameWindow = {
  warmup: 60,
  measure: 50,
};

const JSON_BEGIN = '__PARTICLE_FLUIDS_PERF_JSON_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_PERF_JSON_END__';

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
    // adapter info is best-effort; fall through.
  }
  const browser =
    typeof navigator !== 'undefined' && navigator.userAgent ? navigator.userAgent : 'unknown';
  return { gpu, browser };
}

describe('Phase Perf — all scenes', () => {
  it(
    'runs every scene and emits a PerfReportJson to stdout',
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
              '[Phase Perf] scene ' +
                result.id +
                ': frame_step_ms.p50=' +
                result.frameStepMs.p50.toFixed(2) +
                'ms (' +
                result.kernels.length +
                ' kernels measured)',
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
        // run.ts substitutes commit + os into the placeholders below.
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
    30 * 60_000,
  );
});
