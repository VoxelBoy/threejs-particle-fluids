import { describe, expect, it } from 'vitest';
import { runScanProbe } from '../../_helpers/probes/scan.js';

// Phase 01 Probe 2 — workgroup shared memory + barriers via Blelloch scan.
// Exact integer equality to CPU reference across 10 random inputs.

describe('Phase 01 Probe 2 — Blelloch exclusive prefix-scan', () => {
  const seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23];
  for (const seed of seeds) {
    it(`matches CPU reference exactly (N=1,048,576, seed=${seed})`, async () => {
      const r = await runScanProbe(seed);
      // eslint-disable-next-line no-console
      console.info(
        `[probe-scan] seed=${seed} matched=${r.matched} firstMismatchAt=${r.firstMismatchAt} elapsed=${r.elapsedMs.toFixed(1)}ms`,
      );
      expect(r.firstMismatchAt).toBe(-1);
      expect(r.matched).toBe(true);
    }, 60_000);
  }
});
