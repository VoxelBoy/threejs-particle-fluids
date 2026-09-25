import { describe, expect, it } from 'vitest';
import { runPingPongProbe } from '../../_helpers/probes/pingPong.js';

// Phase 01 Probe 4 — 100 dispatches alternating between two storage buffers.

describe('Phase 01 Probe 4 — ping-pong stability', () => {
  it('every element equals initial + 100 after 100 alternating dispatches', async () => {
    const r = await runPingPongProbe(100);
    // eslint-disable-next-line no-console
    console.info(
      `[probe-ping-pong] iters=${r.iters} allCorrect=${r.allCorrect} firstMismatchAt=${r.firstMismatchAt} elapsed=${r.elapsedMs.toFixed(1)}ms`,
    );
    expect(r.firstMismatchAt).toBe(-1);
    expect(r.allCorrect).toBe(true);
  }, 60_000);
});
