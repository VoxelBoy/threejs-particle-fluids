// Phase Perf-11 — driver for the boundary-coupled bench.
//
// Spawns vitest against `tests/perf/boundary-coupled.gpu.perf.ts`,
// captures the JSON between `__PARTICLE_FLUIDS_BOUNDARY_COUPLED_BEGIN__` /
// `__PARTICLE_FLUIDS_BOUNDARY_COUPLED_END__` sentinels, substitutes commit + os,
// writes both JSON and HTML to `tests/perf/results/`.

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

const JSON_BEGIN = '__PARTICLE_FLUIDS_BOUNDARY_COUPLED_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_BOUNDARY_COUPLED_END__';

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
    const proc = spawn(
      'npx',
      [
        'vitest',
        'run',
        '--passWithNoTests',
        '--reporter=default',
        'tests/perf/boundary-coupled.gpu.perf.ts',
      ],
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
      if (code !== 0) rejectP(new Error('vitest exited with code ' + code));
      else resolveP(captured);
    });
  });
}

function extractJson(stdout: string): string {
  const beginIdx = stdout.lastIndexOf(JSON_BEGIN);
  const endIdx = stdout.lastIndexOf(JSON_END);
  if (beginIdx < 0 || endIdx < 0 || endIdx <= beginIdx) {
    throw new Error(
      'run-boundary-coupled: did not find ' + JSON_BEGIN + '…' + JSON_END + ' sentinels.',
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
    throw new Error('run-boundary-coupled: sentinels found but no JSON line between them.');
  }
  return bestLine;
}

async function main(): Promise<void> {
  if (!existsSync(resultsDir)) mkdirSync(resultsDir, { recursive: true });
  // eslint-disable-next-line no-console
  console.log('[Phase Perf-11] running boundary-coupled bench…');
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
  const stem = 'boundary-coupled-' + stamp + '__' + shortSha;
  const jsonPath = join(resultsDir, stem + '.json');
  const htmlPath = join(resultsDir, stem + '.html');

  writeFileSync(jsonPath, JSON.stringify(finalReport, null, 2) + '\n');
  const template = readFileSync(templatePath, 'utf8');
  writeFileSync(htmlPath, generateHtmlReport(template, finalReport));

  // eslint-disable-next-line no-console
  console.log('');
  // eslint-disable-next-line no-console
  console.log('[Phase Perf-11] JSON written to ' + relative(repoRoot, jsonPath));
}

main().catch((err: Error) => {
  // eslint-disable-next-line no-console
  console.error('[Phase Perf-11] run-boundary-coupled.ts failed:', err.message);
  process.exit(1);
});
