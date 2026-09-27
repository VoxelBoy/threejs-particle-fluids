// Benchmark suite: runs every scene, then prints one `PerfReportJson`
// between sentinel lines on stdout, where `_helpers/run.ts` picks it up and
// writes the JSON and HTML reports.
//
// Each scene runs 60 warmup frames and 100 measured frames at 60 fps.
// Override with the VITE_PERF_WARMUP / VITE_PERF_MEASURE environment
// variables, e.g. `VITE_PERF_MEASURE=20 npm run test:perf`.

import { describe, expect, it } from 'vitest';

import { PerfRenderer } from './_helpers/PerfRenderer.js';
import {
  PerfRunner,
  type PerfFrameWindow,
  type PerfSceneResult,
  type PerfStats,
} from './_helpers/PerfRunner.js';
import type { PerfReportJson, PerfSceneJson, PerfStatsJson } from './_helpers/types.js';
import type { BuiltScene } from './_scenes/_helpers.js';
import { buildFluid10kScene, buildFluid100kScene } from './_scenes/fluid.js';
import { buildFluidBodies10kScene, buildFluidBodies100kScene } from './_scenes/fluid-bodies.js';
import {
  buildFluidSurfaceTension10kScene,
  buildFluidSurfaceTension100kScene,
} from './_scenes/fluid-surface-tension.js';
import { buildSoftbody10kScene, buildSoftbody100kScene } from './_scenes/softbody.js';

// Every 10k scene first, so small-scale regressions show up before the
// slower 100k scenes finish.
const SCENE_BUILDERS: readonly ((perf: PerfRenderer) => BuiltScene)[] = [
  buildFluid10kScene,
  buildFluidBodies10kScene,
  buildFluidSurfaceTension10kScene,
  buildSoftbody10kScene,
  buildFluid100kScene,
  buildFluidBodies100kScene,
  buildFluidSurfaceTension100kScene,
  buildSoftbody100kScene,
];

// One simulated second of warmup lets every scene's falling water and bodies
// land, so the measured frames time a settling scene rather than free fall.
const DEFAULT_WINDOW: PerfFrameWindow = { warmup: 60, measure: 100 };

const JSON_BEGIN = '__PARTICLE_FLUIDS_PERF_JSON_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_PERF_JSON_END__';

function toStatsJson(s: PerfStats): PerfStatsJson {
  return { min: s.min, p10: s.p10, p50: s.p50, p90: s.p90, max: s.max };
}

function toSceneJson(r: PerfSceneResult): PerfSceneJson {
  return {
    id: r.id,
    particle_count: r.particleCount,
    substeps: r.substeps,
    iterations: r.iterations,
    frames_warmup: r.framesWarmup,
    frames_measure: r.framesMeasure,
    ...(r.gpuFrameMs ? { frame_gpu_ms: toStatsJson(r.gpuFrameMs) } : {}),
    frame_step_ms: toStatsJson(r.stepFrameMs),
    ...(r.contactCount ? { contact_count: r.contactCount } : {}),
  };
}

/** The frame window. Vite passes only `VITE_` variables through to the browser. */
function readWindow(): PerfFrameWindow {
  const read = (name: string, fallback: number, min: number): number => {
    const value = Number(import.meta.env[name] ?? fallback);
    return Number.isInteger(value) && value >= min ? value : fallback;
  };
  return {
    warmup: read('VITE_PERF_WARMUP', DEFAULT_WINDOW.warmup, 0),
    measure: read('VITE_PERF_MEASURE', DEFAULT_WINDOW.measure, 1),
  };
}

function describeGpu(device: GPUDevice): string {
  // Older browsers don't have `GPUDevice.adapterInfo`.
  const info = device.adapterInfo as GPUAdapterInfo | undefined;
  const parts = [info?.vendor, info?.architecture, info?.device, info?.description];
  return parts.filter(Boolean).join(' ') || 'unknown';
}

function formatSample(label: string, stats: PerfStats | undefined): string {
  return stats ? `${label} p50 ${stats.p50.toFixed(2)} ms` : `${label} n/a`;
}

describe('benchmark scenes', () => {
  it(
    'times every scene and prints the report JSON',
    async () => {
      const frames = readWindow();
      const perf = await PerfRenderer.create();
      const gpu = describeGpu(perf.device);
      const scenes: PerfSceneJson[] = [];
      try {
        const runner = new PerfRunner(perf);
        for (const build of SCENE_BUILDERS) {
          const built = build(perf);
          try {
            const result = await runner.runScene(built.spec, frames);
            scenes.push(toSceneJson(result));
            console.log(
              `[perf] ${result.id}: ${formatSample('GPU', result.gpuFrameMs)}, ` +
                `${formatSample('wall-clock', result.stepFrameMs)}`,
            );
          } finally {
            built.dispose();
          }
        }
      } finally {
        perf.dispose();
      }

      const report: PerfReportJson = {
        version: 2,
        // run.ts fills in the commit and OS, which the browser can't read.
        commit: 'COMMIT_PLACEHOLDER',
        date: new Date().toISOString(),
        platform: { gpu, browser: navigator.userAgent, os: 'OS_PLACEHOLDER' },
        timing_method: perf.timingMethod,
        scenes,
      };
      console.log(JSON_BEGIN);
      console.log(JSON.stringify(report));
      console.log(JSON_END);

      expect(report.scenes.length).toBe(SCENE_BUILDERS.length);
      const settled = frames.warmup >= DEFAULT_WINDOW.warmup;
      for (const s of report.scenes) {
        expect(s.frame_step_ms.p50).toBeGreaterThan(0);
        if (report.timing_method === 'timestamp') expect(s.frame_gpu_ms?.p50).toBeGreaterThan(0);
        // After a full warmup every contact scene's bodies have landed; one
        // that finds no pairs would be timing an idle contact pipeline.
        if (s.contact_count && settled) expect(s.contact_count.p50).toBeGreaterThan(0);
      }
    },
    30 * 60_000,
  );
});
