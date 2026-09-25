import { describe, expect, it } from 'vitest';
import { runKernelChainProbe } from '../../_helpers/probes/kernelChain.js';

// Phase 01 Probe 5 — three sequential compute dispatches with data dependencies.

describe('Phase 01 Probe 5 — kernel chain', () => {
  it('each stage matches CPU reference (N=1,048,576)', async () => {
    const r = await runKernelChainProbe();
    // eslint-disable-next-line no-console
    console.info(
      `[probe-kernel-chain] B=${r.stageBMatches} C=${r.stageCMatches} D=${r.stageDMatches} elapsed=${r.elapsedMs.toFixed(1)}ms`,
    );
    expect(r.stageBMatches).toBe(true);
    expect(r.stageCMatches).toBe(true);
    expect(r.stageDMatches).toBe(true);
  }, 30_000);
});
