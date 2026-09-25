import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const [suite, ...args] = process.argv.slice(2);
if (!['gpu', 'perf'].includes(suite)) throw new Error('Expected gpu or perf test suite.');
const result = spawnSync(
  process.execPath,
  [fileURLToPath(new URL('../node_modules/vitest/vitest.mjs', import.meta.url)), 'run', ...args],
  { stdio: 'inherit', env: { ...process.env, VITEST_SUITE: suite } },
);
if (result.error) throw result.error;
process.exit(result.status ?? 1);
