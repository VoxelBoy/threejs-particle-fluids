// Driver for `npm run perf:profile`.
//
//   npm run perf:profile -- [--scenes a,b] [--level ultra] [--top 12]
//                           [--compare <file> | --no-compare] [--label name]
//
// Runs `profile.gpu.perf.ts` in the browser, saves its report to
// `tests/perf/results/profile-<stamp>__<sha>[__label].json`, and prints each
// scene's frame times and slowest kernels. By default the report is compared
// with the newest earlier profile of the same level that covered the scene.

import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { ProfileKernelJson, ProfileReportJson, ProfileSceneJson } from './types.js';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..');
const resultsDir = join(repoRoot, 'tests', 'perf', 'results');
const JSON_BEGIN = '__PARTICLE_FLUIDS_PROFILE_JSON_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_PROFILE_JSON_END__';

function arg(name: string): string | undefined {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function stamp(): string {
  const d = new Date();
  const p = (n: number): string => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function commit(): string {
  try {
    const sha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' });
    const dirty = execFileSync('git', ['status', '--porcelain', '--', 'src'], {
      cwd: repoRoot,
      encoding: 'utf8',
    });
    return sha.trim() + (dirty.trim() ? '+dirty' : '');
  } catch {
    return 'unknown';
  }
}

function run(env: Record<string, string>): Promise<string> {
  return new Promise((done, fail) => {
    const proc = spawn(
      'npx',
      ['vitest', 'run', '--reporter=default', 'tests/perf/profile.gpu.perf.ts'],
      {
        cwd: repoRoot,
        env: { ...process.env, VITEST_SUITE: 'perf', ...env },
        stdio: ['inherit', 'pipe', 'inherit'],
      },
    );
    let out = '';
    proc.stdout.on('data', (chunk: Buffer) => {
      const s = chunk.toString();
      out += s;
      // Keep the console to progress lines; the JSON is long.
      for (const line of s.split('\n')) if (line.includes('[profile]')) console.log(line.trim());
    });
    proc.on('error', fail);
    proc.on('close', (code) =>
      code === 0 ? done(out) : fail(new Error('vitest exited with code ' + code + '\n' + out)),
    );
  });
}

function extract(stdout: string): ProfileReportJson {
  const begin = stdout.lastIndexOf(JSON_BEGIN);
  const end = stdout.lastIndexOf(JSON_END);
  if (begin < 0 || end <= begin) throw new Error('profile: no report in the vitest output');
  const line = stdout
    .slice(begin + JSON_BEGIN.length, end)
    .split(/\r?\n/)
    .map((l) => l.trim())
    .find((l) => l.startsWith('{"version"'));
  if (!line) throw new Error('profile: found the sentinels but no JSON between them');
  return JSON.parse(line) as ProfileReportJson;
}

/** Newest earlier profile at the same level with each scene, keyed by scene id. */
function baselines(level: string, exclude: string): Map<string, ProfileSceneJson> {
  const found = new Map<string, ProfileSceneJson>();
  if (!existsSync(resultsDir)) return found;
  const files = readdirSync(resultsDir)
    .filter((f) => f.startsWith('profile-') && f.endsWith('.json') && f !== exclude)
    .sort()
    .reverse();
  for (const file of files) {
    const report = JSON.parse(readFileSync(join(resultsDir, file), 'utf8')) as ProfileReportJson;
    if (report.level !== level) continue;
    for (const scene of report.scenes) if (!found.has(scene.id)) found.set(scene.id, scene);
  }
  return found;
}

function delta(now: number, before: number | undefined): string {
  if (before === undefined || before === 0) return '';
  const pct = ((now - before) / before) * 100;
  const sign = pct > 0 ? '+' : '';
  return `${sign}${pct.toFixed(0)}%`.padStart(6);
}

function table(
  title: string,
  kernels: readonly ProfileKernelJson[],
  before: readonly ProfileKernelJson[] | undefined,
  top: number,
): void {
  const total = kernels.reduce((s, k) => s + k.ms, 0);
  const previous = new Map(before?.map((k) => [k.name, k.ms]));
  console.log(`  ${title}: kernel sum ${total.toFixed(3)} ms, ${kernels.length} kernels`);
  for (const k of kernels.slice(0, top)) {
    console.log(
      '    ' +
        k.ms.toFixed(3).padStart(7) +
        ' ms ' +
        ((k.ms / total) * 100).toFixed(1).padStart(5) +
        '% ' +
        delta(k.ms, previous.get(k.name)) +
        '  ×' +
        String(Math.round(k.calls)).padEnd(4) +
        k.name,
    );
  }
}

async function main(): Promise<void> {
  const scenes = arg('scenes');
  const level = arg('level') ?? 'ultra';
  const top = Number(arg('top') ?? 12);
  const label = arg('label');
  const env: Record<string, string> = { VITE_PROFILE_LEVEL: level };
  if (scenes) env['VITE_PROFILE_SCENES'] = scenes;
  for (const key of ['VITE_PERF_WARMUP', 'VITE_PERF_MEASURE', 'VITE_PROFILE_FRAMES']) {
    const value = process.env[key];
    if (value) env[key] = value;
  }

  const report = { ...extract(await run(env)), commit: commit() };
  if (!existsSync(resultsDir)) mkdirSync(resultsDir, { recursive: true });
  const name =
    `profile-${stamp()}__${report.commit.slice(0, 8)}` + (label ? `__${label}` : '') + '.json';
  const path = join(resultsDir, name);
  writeFileSync(path, JSON.stringify(report, null, 2) + '\n');

  const compareFile = arg('compare');
  let previous: Map<string, ProfileSceneJson>;
  if (process.argv.includes('--no-compare')) previous = new Map();
  else if (compareFile) {
    const other = JSON.parse(readFileSync(compareFile, 'utf8')) as ProfileReportJson;
    previous = new Map(other.scenes.map((s) => [s.id, s]));
  } else previous = baselines(report.level, name);

  console.log('');
  for (const scene of report.scenes) {
    const before = previous.get(scene.id);
    console.log(
      `${scene.id}  (${scene.particle_count} particles, ${scene.substeps}×${scene.iterations})` +
        `  sim ${scene.sim_gpu_ms.toFixed(3)} ms${delta(scene.sim_gpu_ms, before?.sim_gpu_ms)}` +
        (scene.render_gpu_ms !== undefined
          ? `  render prep ${scene.render_gpu_ms.toFixed(3)} ms${delta(scene.render_gpu_ms, before?.render_gpu_ms)}`
          : ''),
    );
    table('sim', scene.kernels, before?.kernels, top);
    if (scene.render_kernels) {
      table('render prep', scene.render_kernels, before?.render_kernels, Math.min(top, 6));
    }
    console.log('');
  }
  console.log('[profile] saved ' + relative(repoRoot, path));
}

main().catch((error: Error) => {
  console.error('[profile] failed:', error.message);
  process.exit(1);
});
