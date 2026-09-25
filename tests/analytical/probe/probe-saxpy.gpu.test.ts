import { describe, expect, it } from 'vitest';
import { runSaxpyProbe } from '../../_helpers/probes/saxpy.js';

// Phase 01 Probe 1 — baseline compute correctness.
// Verified if saxpy on 1M floats matches the CPU reference within f32 roundoff.

describe('Phase 01 Probe 1 — saxpy', () => {
  it('matches CPU reference within 1e-6 over N=1,000,000', async () => {
    const result = await runSaxpyProbe(1_000_000, 2.5);
    // eslint-disable-next-line no-console
    console.info(
      `[probe-saxpy] N=${result.n} maxAbsError=${result.maxAbsError.toExponential(3)} elapsed=${result.elapsedMs.toFixed(1)}ms`,
    );
    expect(result.maxAbsError).toBeLessThan(1e-6);
  }, 30_000);
});
