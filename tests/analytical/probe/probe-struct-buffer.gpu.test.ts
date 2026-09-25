import { describe, expect, it } from 'vitest';
import { runStructBufferProbe } from '../../_helpers/probes/structBuffer.js';

// Phase 21a Probe — TSL struct buffer with atomic fields.
//
// Validates `instancedArray(N, struct({...}))` plus `atomicAdd` / `atomicLoad`
// / `atomicStore` on struct fields declared with `{ atomic: true }`. This is
// the gate before the Phase 21a contact-record refactor lands; if any check
// fails, the refactor stops and we fall back to bitcast-into-pairs.
//
// File pointers exercised:
//   - three/src/nodes/accessors/Arrays.js:47 — instancedArray + struct.
//   - three/src/nodes/core/StructTypeNode.js:13 — atomic field flag.
//   - three/src/nodes/utils/StorageArrayElementNode.js:62 — getMemberType.

describe('Phase 21a Probe — TSL struct buffer with atomic fields', () => {
  it('supports field write/read, atomicAdd, atomic race, and atomicLoad on a struct member', async () => {
    const r = await runStructBufferProbe();
    // eslint-disable-next-line no-console
    console.info(
      `[probe-struct-buffer] stride=${r.recordStrideBytes}B ` +
        `u32[0..]=${r.firstRecordWordsU32.join(',')} ` +
        `i32[0..]=${r.firstRecordWordsI32.join(',')} ` +
        `plainField=${r.plainFieldMatched} ` +
        `atomicAddSingle=${r.atomicAddSingleThreadMatched} ` +
        `race=${r.atomicRaceTotal}/${r.atomicRaceExpected} ` +
        `atomicLoad=${r.atomicLoadMatched} ` +
        `elapsed=${r.elapsedMs.toFixed(1)}ms`,
    );
    expect(r.plainFieldMatched, 'Check 1: plain struct-field write/read').toBe(true);
    expect(r.atomicAddSingleThreadMatched, 'Check 2: single-thread atomicAdd on struct field').toBe(
      true,
    );
    expect(r.atomicRaceTotal, 'Check 3: multi-thread atomicAdd race').toBe(r.atomicRaceExpected);
    expect(r.atomicLoadMatched, 'Check 4: atomicLoad on struct field').toBe(true);
  }, 30_000);
});
