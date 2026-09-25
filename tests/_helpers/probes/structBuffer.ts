// Phase 21a probe — TSL struct buffer with atomic fields. Minimal isolation
// version: try one write at a time and dump WGSL/buffer state.

import {
  Fn,
  If,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  instanceIndex,
  instancedArray,
  int,
  struct,
  uint,
} from 'three/tsl';
import { createWebGPURenderer } from './_renderer.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface StructBufferProbeResult {
  readonly recordStrideBytes: number;
  readonly firstRecordWordsU32: readonly number[];
  readonly firstRecordWordsI32: readonly number[];
  readonly plainFieldMatched: boolean;
  readonly atomicAddSingleThreadMatched: boolean;
  readonly atomicRaceTotal: number;
  readonly atomicRaceExpected: number;
  readonly atomicLoadMatched: boolean;
  readonly elapsedMs: number;
}

const N_RECORDS = 4;
const N_RACE_THREADS = 64;

export async function runStructBufferProbe(): Promise<StructBufferProbeResult> {
  const renderer = await createWebGPURenderer();

  const ContactRecordProbe = struct(
    {
      i: 'uint',
      j: 'uint',
      lambdaN: { type: 'int', atomic: true },
      lambdaT: { type: 'int', atomic: true },
      normal: 'vec3',
    },
    'ContactRecordProbe',
  );

  const records: Any = instancedArray(N_RECORDS, ContactRecordProbe as Any);

  // Run kernels SEPARATELY to isolate which one fails.

  // Step 1 — plain field write only.
  const writeKernel = Fn(() => {
    const idx: Any = instanceIndex;
    If(idx.greaterThanEqual(uint(N_RECORDS)), () => {
      Return();
    });
    const rec: Any = records.element(idx);
    rec.get('i').assign(idx);
    rec.get('j').assign(idx.add(uint(1)));
  })().compute(N_RECORDS);

  // Step 2 — atomicAdd 7 to lambdaN per slot.
  const atomicAddKernel = Fn(() => {
    const idx: Any = instanceIndex;
    If(idx.greaterThanEqual(uint(N_RECORDS)), () => {
      Return();
    });
    const rec: Any = records.element(idx);
    atomicAdd(rec.get('lambdaN'), int(7));
  })().compute(N_RECORDS);

  // Step 3 — race: N threads atomicAdd 1 into records[0].lambdaN. We're
  // re-using the records buffer (already verified working for plain + single-
  // thread atomic). Pre-state at slot 0: lambdaN = 7 (from atomicAddKernel
  // above). Expected post-race lambdaN = 7 + N_RACE_THREADS.
  const raceKernel = Fn(() => {
    atomicAdd(records.element(uint(0)).get('lambdaN'), int(1));
  })().compute(N_RACE_THREADS);

  // Step 4 — atomicLoad and copy lambdaN -> lambdaT.
  const loadKernel = Fn(() => {
    const idx: Any = instanceIndex;
    If(idx.greaterThanEqual(uint(N_RECORDS)), () => {
      Return();
    });
    const rec: Any = records.element(idx);
    const v: Any = atomicLoad(rec.get('lambdaN'));
    atomicStore(rec.get('lambdaT'), v);
  })().compute(N_RECORDS);

  const t0 = performance.now();
  // Mirror SimLoop.step()'s array-form dispatch: this is the production
  // shape and the refactor will use it. Sequential per-kernel dispatches
  // were verified working in earlier probe iterations; this is the path
  // that matters for production.
  await renderer.computeAsync([writeKernel, atomicAddKernel, raceKernel, loadKernel]);

  const recordsBuf = await renderer.getArrayBufferAsync(records.value);
  const elapsedMs = performance.now() - t0;

  const recordStrideBytes = recordsBuf.byteLength / N_RECORDS;
  const u32 = new Uint32Array(recordsBuf);
  const i32 = new Int32Array(recordsBuf);
  const stride = recordStrideBytes / 4;

  let plainFieldMatched = true;
  let atomicAddSingleThreadMatched = true;
  let atomicLoadMatched = true;
  // Slot 0: post-race lambdaN should be 7 + N_RACE_THREADS; loadKernel ran
  // AFTER race so lambdaT[0] should equal post-race lambdaN (= 7 + 64 = 71).
  // Slots 1..: lambdaN = 7, lambdaT = 7.
  for (let idx = 0; idx < N_RECORDS; idx++) {
    const base = idx * stride;
    if (u32[base] !== idx || u32[base + 1] !== idx + 1) plainFieldMatched = false;
    const expectLambdaN = idx === 0 ? 7 + N_RACE_THREADS : 7;
    if (i32[base + 2] !== expectLambdaN) atomicAddSingleThreadMatched = false;
    const expectLambdaT = idx === 0 ? 7 + N_RACE_THREADS : 7;
    if (i32[base + 3] !== expectLambdaT) atomicLoadMatched = false;
  }

  // Race target is records[0].lambdaN. Pre-race lambdaN was 7; post-race
  // expected = 7 + N_RACE_THREADS. loadKernel ran AFTER raceKernel so
  // records[0].lambdaT also = 7 + N_RACE_THREADS.
  const racePostValue = i32[2] ?? 0;
  const racePreValue = 7;

  console.info(
    `[probe-struct-buffer] all records i32 stride=${stride}: ${Array.from(
      i32.subarray(0, N_RECORDS * stride),
    ).join(',')}`,
  );
  const atomicRaceTotal = racePostValue - racePreValue;

  renderer.dispose();
  return {
    recordStrideBytes,
    firstRecordWordsU32: Array.from(u32.subarray(0, Math.min(stride, 8))),
    firstRecordWordsI32: Array.from(i32.subarray(0, Math.min(stride, 8))),
    plainFieldMatched,
    atomicAddSingleThreadMatched,
    atomicRaceTotal,
    atomicRaceExpected: N_RACE_THREADS,
    atomicLoadMatched,
    elapsedMs,
  };
}
