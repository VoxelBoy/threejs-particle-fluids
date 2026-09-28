// Driver for `npm run perf:ab`: compares the working tree with a git ref.
//
//   npm run perf:ab -- [--ref HEAD] [--rounds 3] [--scenes a,b] [--level ultra]
//
// Separate profile runs drift by several percent (clocks, heat), so this
// checks the ref out into a temporary worktree, copies the current perf
// harness into it, and alternates runs of the two trees. It prints each
// scene's median simulation and render-preparation times per tree.

import { execFileSync, spawn } from 'node:child_process';
import { cpSync, existsSync, mkdtempSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { ProfileReportJson } from './types.js';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..');
const JSON_BEGIN = '__PARTICLE_FLUIDS_PROFILE_JSON_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_PROFILE_JSON_END__';

function arg(name: string): string | undefined {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function profile(cwd: string, env: Record<string, string>): Promise<ProfileReportJson> {
  return new Promise((done, fail) => {
    const proc = spawn(
      'npx',
      ['vitest', 'run', '--reporter=default', 'tests/perf/profile.gpu.perf.ts'],
      {
        cwd,
        // Only whole-frame times are compared, so skip the per-kernel pass.
        env: { ...process.env, VITEST_SUITE: 'perf', VITE_PROFILE_FRAMES: '1', ...env },
        stdio: ['inherit', 'pipe', 'inherit'],
      },
    );
    let out = '';
    proc.stdout.on('data', (chunk: Buffer) => (out += chunk.toString()));
    proc.on('error', fail);
    proc.on('close', (code) => {
      if (code !== 0) return fail(new Error(`vitest exited with code ${code} in ${cwd}\n${out}`));
      const line = out
        .slice(out.lastIndexOf(JSON_BEGIN) + JSON_BEGIN.length, out.lastIndexOf(JSON_END))
        .split(/\r?\n/)
        .map((l) => l.trim())
        .find((l) => l.startsWith('{"version"'));
      if (!line) return fail(new Error('ab: no report in the output'));
      done(JSON.parse(line) as ProfileReportJson);
    });
  });
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

async function main(): Promise<void> {
  const ref = arg('ref') ?? 'HEAD';
  const rounds = Number(arg('rounds') ?? 3);
  const env: Record<string, string> = { VITE_PROFILE_LEVEL: arg('level') ?? 'ultra' };
  const scenes = arg('scenes');
  if (scenes) env['VITE_PROFILE_SCENES'] = scenes;
  for (const key of ['VITE_PERF_WARMUP', 'VITE_PERF_MEASURE']) {
    if (process.env[key]) env[key] = process.env[key]!;
  }

  const worktree = mkdtempSync(join(tmpdir(), 'particle-fluids-ab-'));
  execFileSync('git', ['worktree', 'add', '--detach', worktree, ref], {
    cwd: repoRoot,
    stdio: 'ignore',
  });
  try {
    symlinkSync(join(repoRoot, 'node_modules'), join(worktree, 'node_modules'));
    // The ref may predate the harness; run the current one against its code.
    rmSync(join(worktree, 'tests', 'perf'), { recursive: true, force: true });
    cpSync(join(repoRoot, 'tests', 'perf'), join(worktree, 'tests', 'perf'), {
      recursive: true,
      filter: (source) => !source.includes(join('tests', 'perf', 'results')),
    });
    if (!existsSync(join(worktree, 'public'))) {
      symlinkSync(join(repoRoot, 'public'), join(worktree, 'public'));
    }

    const times = new Map<string, { ref: number[][]; now: number[][] }>();
    for (let round = 0; round < rounds; round++) {
      // Alternate which tree goes first, so warm-up drift hits both.
      const order: ['ref' | 'now', string][] =
        round % 2 === 0
          ? [
              ['ref', worktree],
              ['now', repoRoot],
            ]
          : [
              ['now', repoRoot],
              ['ref', worktree],
            ];
      for (const [which, cwd] of order) {
        const report = await profile(cwd, env);
        for (const scene of report.scenes) {
          const entry = times.get(scene.id) ?? { ref: [], now: [] };
          entry[which].push([scene.sim_gpu_ms, scene.render_gpu_ms ?? 0]);
          times.set(scene.id, entry);
        }
        console.log(`[ab] round ${round + 1}/${rounds}: ${which} done`);
      }
    }

    console.log(`\nMedian GPU ms per frame, ${ref} → working tree (${rounds} runs each)`);
    console.log('scene'.padEnd(28) + 'sim'.padStart(22) + 'render prep'.padStart(26));
    for (const [id, { ref: a, now: b }] of times) {
      const cell = (k: number): string => {
        const before = median(a.map((t) => t[k]!));
        const after = median(b.map((t) => t[k]!));
        if (before === 0) return '-'.padStart(24);
        const pct = ((after - before) / before) * 100;
        return `${before.toFixed(2)} → ${after.toFixed(2)} ${(pct > 0 ? '+' : '') + pct.toFixed(0)}%`.padStart(
          24,
        );
      };
      console.log(id.padEnd(28) + cell(0) + '  ' + cell(1));
    }
  } finally {
    execFileSync('git', ['worktree', 'remove', '--force', worktree], { cwd: repoRoot });
  }
}

main().catch((error: Error) => {
  console.error('[ab] failed:', error.message);
  process.exit(1);
});
