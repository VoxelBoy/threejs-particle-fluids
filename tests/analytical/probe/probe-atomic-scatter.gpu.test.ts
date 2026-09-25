import { describe, expect, it } from 'vitest';
import { runAtomicScatterProbe } from '../../_helpers/probes/atomicScatter.js';

// Phase 01 Probe 3 — atomicAdd correctness over 1M keys into a 1024-bin histogram.

describe('Phase 01 Probe 3 — atomic scatter', () => {
  it('matches CPU histogram exactly and conserves total count (N=1,000,000)', async () => {
    const r = await runAtomicScatterProbe(42);
    // eslint-disable-next-line no-console
    console.info(
      `[probe-atomic-scatter] matched=${r.matched} totalCountOk=${r.totalCountMatches} elapsed=${r.elapsedMs.toFixed(1)}ms`,
    );
    expect(r.matched).toBe(true);
    expect(r.totalCountMatches).toBe(true);
  }, 30_000);
});
