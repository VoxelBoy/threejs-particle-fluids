// Driver for `npm run perf:fingerprint`.
//
//   npm run perf:fingerprint -- --save-baseline [--scenes a,b] [--level ultra]
//   npm run perf:fingerprint -- [--scenes a,b] [--level ultra]
//
// `--save-baseline` runs the fingerprint suite four times and saves the runs
// to `tests/perf/results/fingerprint-baseline-<level>.json`; their range is
// the run-to-run noise. Without it, the suite runs once and every statistic
// is compared with the baseline runs' mean. A statistic is flagged when it
// lands further from the mean than twice the noise, plus 3% and a small
// absolute floor.

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { Fingerprint } from '../fingerprint.gpu.perf.js';

interface Report {
  readonly level: string;
  readonly frames: number;
  readonly scenes: Fingerprint[];
}

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..');
const resultsDir = join(repoRoot, 'tests', 'perf', 'results');
const JSON_BEGIN = '__PARTICLE_FLUIDS_FINGERPRINT_JSON_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_FINGERPRINT_JSON_END__';

const BASELINE_RUNS = 4;

/** Absolute floor per statistic: metres for positions, m/s for speeds. */
const FLOOR: Record<string, number> = {
  centroid: 0.002,
  spread: 0.002,
  minY: 0.003,
  maxY: 0.01,
  meanSpeed: 0.01,
  rmsSpeed: 0.01,
  p99Speed: 0.05,
};

function arg(name: string): string | undefined {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function runOnce(env: Record<string, string>): Promise<Report> {
  return new Promise((done, fail) => {
    const proc = spawn(
      'npx',
      ['vitest', 'run', '--reporter=default', 'tests/perf/fingerprint.gpu.perf.ts'],
      {
        cwd: repoRoot,
        env: { ...process.env, VITEST_SUITE: 'perf', ...env },
        stdio: ['inherit', 'pipe', 'inherit'],
      },
    );
    let out = '';
    proc.stdout.on('data', (chunk: Buffer) => {
      out += chunk.toString();
    });
    proc.on('error', fail);
    proc.on('close', (code) => {
      if (code !== 0) return fail(new Error('vitest exited with code ' + code + '\n' + out));
      const begin = out.lastIndexOf(JSON_BEGIN);
      const end = out.lastIndexOf(JSON_END);
      const line = out
        .slice(begin + JSON_BEGIN.length, end)
        .split(/\r?\n/)
        .map((l) => l.trim())
        .find((l) => l.startsWith('{'));
      if (begin < 0 || !line) return fail(new Error('fingerprint: no report in the output'));
      done(JSON.parse(line) as Report);
    });
  });
}

function values(f: Fingerprint): Record<string, number> {
  return {
    'centroid.x': f.centroid[0],
    'centroid.y': f.centroid[1],
    'centroid.z': f.centroid[2],
    spread: f.spread,
    minY: f.minY,
    maxY: f.maxY,
    meanSpeed: f.meanSpeed,
    rmsSpeed: f.rmsSpeed,
    p99Speed: f.p99Speed,
  };
}

async function main(): Promise<void> {
  const level = arg('level') ?? 'ultra';
  const env: Record<string, string> = { VITE_PROFILE_LEVEL: level };
  const scenes = arg('scenes');
  if (scenes) env['VITE_PROFILE_SCENES'] = scenes;
  const frames = process.env['VITE_FINGERPRINT_FRAMES'];
  if (frames) env['VITE_FINGERPRINT_FRAMES'] = frames;
  if (!existsSync(resultsDir)) mkdirSync(resultsDir, { recursive: true });
  const baselinePath = join(resultsDir, `fingerprint-baseline-${level}.json`);

  if (process.argv.includes('--save-baseline')) {
    const runs: Report[] = [];
    for (let i = 0; i < BASELINE_RUNS; i++) runs.push(await runOnce(env));
    writeFileSync(baselinePath, JSON.stringify({ runs }, null, 2) + '\n');
    console.log('[fingerprint] saved ' + relative(repoRoot, baselinePath));
    return;
  }

  const { runs } = JSON.parse(readFileSync(baselinePath, 'utf8')) as { runs: Report[] };
  const now = await runOnce(env);
  let flagged = 0;
  for (const scene of now.scenes) {
    const base = runs.map((run) => run.scenes.find((s) => s.id === scene.id));
    if (base.some((s) => !s)) {
      console.log(`${scene.id}: no baseline`);
      continue;
    }
    const lines: string[] = [];
    if (scene.nonFinite > 0) lines.push(`  ${scene.nonFinite} particles are NaN or infinite`);
    if (scene.moving !== base[0]!.moving) {
      lines.push(`  moving ${base[0]!.moving} → ${scene.moving}`);
    }
    const baseValues = base.map((s) => values(s!));
    const vn = values(scene);
    for (const key of Object.keys(vn)) {
      const samples = baseValues.map((v) => v[key]!);
      const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
      const noise = Math.max(...samples) - Math.min(...samples);
      const diff = Math.abs(vn[key]! - mean);
      const allowed = 2 * noise + 0.03 * Math.abs(mean) + FLOOR[key.split('.')[0]!]!;
      if (diff > allowed) {
        lines.push(
          `  ${key.padEnd(11)} ${mean.toFixed(4)} → ${vn[key]!.toFixed(4)}` +
            `  (moved ${diff.toFixed(4)}, noise ${noise.toFixed(4)})`,
        );
      }
    }
    flagged += lines.length > 0 ? 1 : 0;
    console.log(`${scene.id}: ${lines.length === 0 ? 'ok' : 'CHANGED'}`);
    for (const line of lines) console.log(line);
  }
  console.log(`[fingerprint] ${flagged} of ${now.scenes.length} scenes changed beyond noise`);
  if (flagged > 0) process.exitCode = 1;
}

main().catch((error: Error) => {
  console.error('[fingerprint] failed:', error.message);
  process.exit(1);
});
