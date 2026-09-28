// Profile suite: times each scene's frames as usual, then again with every
// kernel in its own timed pass, and prints one `ProfileReportJson` between
// sentinel lines for `_helpers/profile.ts` to save and compare.
//
// Scenes are the demo presets (built exactly as the demo builds them, at a
// particle level) and the benchmark scenes. Select them with
// VITE_PROFILE_SCENES, a comma-separated list of ids; the default is every
// preset. VITE_PROFILE_LEVEL picks the presets' particle level (default
// `ultra`). Presets warm up for half their demo duration, so scenes that
// build up are measured mid-run; VITE_PERF_WARMUP overrides that for every
// scene. VITE_PERF_MEASURE and VITE_PROFILE_FRAMES set the measured windows.

import { describe, expect, it } from 'vitest';

import { PARTICLE_LEVELS, type ParticleLevel } from '../../demo/types.js';
import { KernelProfiler } from './_helpers/KernelProfiler.js';
import { PerfRenderer } from './_helpers/PerfRenderer.js';
import type { ProfileReportJson, ProfileSceneJson } from './_helpers/types.js';
import { PRESET_IDS, SCENES } from './_scenes/registry.js';

const JSON_BEGIN = '__PARTICLE_FLUIDS_PROFILE_JSON_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_PROFILE_JSON_END__';

function readInt(name: string, fallback: number, min: number): number {
  const value = Number(import.meta.env[name] ?? fallback);
  return Number.isInteger(value) && value >= min ? value : fallback;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

describe('kernel profile', () => {
  it(
    'times every kernel of every selected scene and prints the report JSON',
    async () => {
      const ids = String(import.meta.env['VITE_PROFILE_SCENES'] ?? '')
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const selected = ids.length > 0 ? ids : PRESET_IDS;
      for (const id of selected) if (!SCENES[id]) throw new Error(`Unknown scene ${id}`);
      const levelName = String(import.meta.env['VITE_PROFILE_LEVEL'] ?? 'ultra');
      const level = (PARTICLE_LEVELS.find((l) => l.id === levelName)?.id ??
        'ultra') as ParticleLevel;
      // Without VITE_PERF_WARMUP, presets warm up for half their demo duration.
      const warmupOverride = import.meta.env['VITE_PERF_WARMUP'] !== undefined;
      const warmupDefault = readInt('VITE_PERF_WARMUP', 60, 0);
      const measure = readInt('VITE_PERF_MEASURE', 60, 1);
      const profileFrames = readInt('VITE_PROFILE_FRAMES', 10, 1);

      const perf = await PerfRenderer.create();
      expect(perf.timingMethod).toBe('timestamp');
      const scenes: ProfileSceneJson[] = [];
      try {
        for (const id of selected) {
          const scene = await SCENES[id]!(perf, level);
          try {
            const frame = async () => {
              await scene.simulate();
              await scene.prepareRender?.();
            };
            const warmup = warmupOverride ? warmupDefault : (scene.warmupFrames ?? warmupDefault);
            for (let i = 0; i < warmup; i++) {
              await frame();
              await perf.discardGpuTimings();
            }

            // Batched, as the demo runs it.
            const simMs: number[] = [];
            const renderMs: number[] = [];
            for (let i = 0; i < measure; i++) {
              await scene.simulate();
              simMs.push(await perf.readGpuMs());
              if (scene.prepareRender) {
                await scene.prepareRender();
                renderMs.push(await perf.readGpuMs());
              }
            }

            // One pass per kernel.
            const profiler = new KernelProfiler(perf.renderer);
            profiler.install();
            let kernels;
            let renderKernels;
            try {
              await profiler.reset();
              for (let i = 0; i < profileFrames; i++) {
                await scene.simulate();
                await profiler.endFrame();
              }
              kernels = profiler.results();
              await profiler.reset();
              if (scene.prepareRender) {
                for (let i = 0; i < profileFrames; i++) {
                  await scene.prepareRender();
                  await profiler.endFrame();
                }
                renderKernels = profiler.results();
              }
            } finally {
              profiler.uninstall();
            }

            await perf.device.queue.onSubmittedWorkDone();
            perf.assertNoErrors(id);
            const result: ProfileSceneJson = {
              id: scene.id,
              particle_count: scene.particleCount,
              substeps: scene.substeps,
              iterations: scene.iterations,
              sim_gpu_ms: median(simMs),
              ...(renderMs.length > 0 && { render_gpu_ms: median(renderMs) }),
              kernels,
              ...(renderKernels && { render_kernels: renderKernels }),
            };
            scenes.push(result);
            console.log(
              `[profile] ${scene.id}: sim ${result.sim_gpu_ms.toFixed(2)} ms` +
                (result.render_gpu_ms !== undefined
                  ? `, render prep ${result.render_gpu_ms.toFixed(2)} ms`
                  : ''),
            );
          } finally {
            scene.dispose();
          }
        }
      } finally {
        perf.dispose();
      }

      const report: ProfileReportJson = {
        version: 1,
        commit: 'COMMIT_PLACEHOLDER',
        date: new Date().toISOString(),
        level,
        frames: { warmup: warmupOverride ? warmupDefault : -1, measure, profile: profileFrames },
        scenes,
      };
      console.log(JSON_BEGIN);
      console.log(JSON.stringify(report));
      console.log(JSON_END);
    },
    60 * 60_000,
  );
});
