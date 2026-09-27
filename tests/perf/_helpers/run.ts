// Driver for `npm run test:perf` and `npm run bench:baseline`.
//
// Runs the benchmark suite in the browser through vitest, pulls the
// `PerfReportJson` out of its stdout (between sentinel lines), fills in the
// commit hash and OS, which the browser can't read, and writes:
//   - the JSON to `tests/perf/results/<stamp>__<sha>__<scenes>.json`, or to
//     `tests/perf/baseline-<date>__<sha>__<scenes>.json` with --baseline;
//   - a self-contained HTML report to `tests/perf/results/`.
// Both locations are git-ignored. Load a baseline JSON into any later
// report to compare runs.

import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { platform, release } from 'node:os';

import { generateHtmlReport } from './report-generator.js';
import type { PerfReportJson } from './types.js';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..');
const resultsDir = join(repoRoot, 'tests', 'perf', 'results');
const templatePath = join(here, 'report-template.html');

const JSON_BEGIN = '__PARTICLE_FLUIDS_PERF_JSON_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_PERF_JSON_END__';

function timestamp(): string {
  const d = new Date();
  const pad = (n: number): string => String(n).padStart(2, '0');
  return (
    d.getFullYear() +
    '-' +
    pad(d.getMonth() + 1) +
    '-' +
    pad(d.getDate()) +
    '-' +
    pad(d.getHours()) +
    pad(d.getMinutes()) +
    pad(d.getSeconds())
  );
}

function dateOnly(): string {
  const d = new Date();
  const pad = (n: number): string => String(n).padStart(2, '0');
  return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
}

function commitHash(): string {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: repoRoot,
      encoding: 'utf8',
    }).trim();
  } catch {
    return 'unknown';
  }
}

function osDescription(): string {
  return platform() + ' ' + release();
}

function runVitestAndCapture(): Promise<string> {
  return new Promise((resolveP, rejectP) => {
    const env = { ...process.env, VITEST_SUITE: 'perf' };
    // Only the scene benchmarks: the runner's own tests under _helpers/
    // check the harness and add nothing to a report.
    const proc = spawn(
      'npx',
      ['vitest', 'run', '--passWithNoTests', '--reporter=default', 'tests/perf/all.gpu.perf.ts'],
      { cwd: repoRoot, env, stdio: ['inherit', 'pipe', 'inherit'] },
    );
    let captured = '';
    proc.stdout.on('data', (chunk: Buffer) => {
      const s = chunk.toString();
      captured += s;
      process.stdout.write(s);
    });
    proc.on('error', rejectP);
    proc.on('close', (code) => {
      if (code !== 0) {
        rejectP(new Error('vitest exited with code ' + code));
      } else {
        resolveP(captured);
      }
    });
  });
}

function extractJson(stdout: string): string {
  const beginIdx = stdout.lastIndexOf(JSON_BEGIN);
  const endIdx = stdout.lastIndexOf(JSON_END);
  if (beginIdx < 0 || endIdx < 0 || endIdx <= beginIdx) {
    throw new Error(
      'run.ts: did not find ' +
        JSON_BEGIN +
        '…' +
        JSON_END +
        ' sentinels in vitest stdout. Did the perf test run? (begin=' +
        beginIdx +
        ', end=' +
        endIdx +
        ')',
    );
  }
  const slice = stdout.slice(beginIdx + JSON_BEGIN.length, endIdx);
  const lines = slice.split(/\r?\n/);
  let bestLine = '';
  for (const raw of lines) {
    const line = raw.trim();
    if (!line.startsWith('{"version"')) continue;
    if (line.length > bestLine.length) bestLine = line;
  }
  if (bestLine === '') {
    throw new Error(
      'run.ts: found sentinels but could not locate a JSON line starting with `{"version"` between them.',
    );
  }
  return bestLine;
}

/**
 * Short tag for the run's scene set, e.g.
 * `8scenes-fluid+fluid-bodies+fluid-surface-tension+softbody`. Goes into the
 * file name so reports side by side are recognizable without opening them.
 */
function sceneSetTag(report: PerfReportJson): string {
  const ids = report.scenes.map((s) => s.id);
  const types = new Set<string>();
  for (const id of ids) {
    // Strip the trailing size (-10k, -100k) to get the scene type.
    const t = id.replace(/-(?:\d+k|\d+x\d+k|\d+)$/i, '');
    types.add(t);
  }
  return ids.length + 'scenes-' + Array.from(types).sort().join('+');
}

async function main(): Promise<void> {
  const isBaseline = process.argv.includes('--baseline');
  if (!existsSync(resultsDir)) mkdirSync(resultsDir, { recursive: true });

  console.log(
    isBaseline ? '[perf] running benchmarks (baseline mode)…' : '[perf] running benchmarks…',
  );

  const stdout = await runVitestAndCapture();
  const jsonText = extractJson(stdout);
  const report = JSON.parse(jsonText) as PerfReportJson;

  const sha = commitHash();
  const shortSha = sha === 'unknown' ? 'nogit' : sha.slice(0, 8);

  const finalReport: PerfReportJson = {
    ...report,
    commit: sha,
    platform: { ...report.platform, os: osDescription() },
  };

  const stamp = timestamp();
  const tag = sceneSetTag(finalReport);
  const stem = isBaseline
    ? 'baseline-' + dateOnly() + '__' + shortSha + '__' + tag
    : stamp + '__' + shortSha + '__' + tag;

  const jsonPath = isBaseline
    ? join(repoRoot, 'tests', 'perf', stem + '.json')
    : join(resultsDir, stem + '.json');
  const htmlPath = join(resultsDir, stem + '.html');

  writeFileSync(jsonPath, JSON.stringify(finalReport, null, 2) + '\n');
  const template = readFileSync(templatePath, 'utf8');
  writeFileSync(htmlPath, generateHtmlReport(template, finalReport));

  console.log('');
  console.log('[perf] JSON written to ' + relative(repoRoot, jsonPath));
  console.log('[perf] open ' + relative(repoRoot, htmlPath) + ' in your browser.');
}

main().catch((err: Error) => {
  console.error('[perf] run.ts failed:', err.message);
  process.exit(1);
});
