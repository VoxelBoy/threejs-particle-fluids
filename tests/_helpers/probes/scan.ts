// Phase 01 Probe 2 — Blelloch exclusive prefix-scan (u32) via TSL compute.
// Verifies workgroup shared memory, workgroupBarrier, and multi-kernel chaining.
//

import {
  Fn,
  If,
  globalId,
  instancedArray,
  instanceIndex,
  localId,
  uint,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from 'three/tsl';
import { createWebGPURenderer } from './_renderer.js';

// TSL's @types/three surface types many GPGPU nodes as bare `Node`, which drops
// the proxy-provided `.element()/.add()/.toVar()` methods. Probes use this
// loose alias to keep the code readable without module-augmentation scaffolding
// that'd live beyond the probe's lifespan.
type Any = any; // eslint-disable-line @typescript-eslint/no-explicit-any

const W = 1024;
const NUM_BLOCKS = 1024;
const N = W * NUM_BLOCKS; // 1,048,576

// Simple LCG for deterministic input generation, matched in CPU reference.
function lcgU32(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
}

export interface ScanResult {
  readonly n: number;
  readonly seed: number;
  readonly matched: boolean;
  readonly firstMismatchAt: number; // -1 if matched
  readonly elapsedMs: number;
}

export async function runScanProbe(seed: number): Promise<ScanResult> {
  const renderer = await createWebGPURenderer();

  const inputNode = instancedArray(N, 'uint');
  const scannedNode = instancedArray(N, 'uint');
  const blockSumsNode = instancedArray(NUM_BLOCKS, 'uint');

  const inputArr = inputNode.value.array as Uint32Array;
  const rand = lcgU32(seed);
  // Cap values so final scan stays within u32 (1M * 16 = 16M < 2^32).
  for (let i = 0; i < N; i++) inputArr[i] = rand() & 0xf;

  // Pass 1 kernel — per-block Blelloch scan + block-sum write.
  const blockScanKernel = Fn(() => {
    const tid: Any = localId.x;
    const wg: Any = workgroupId.x;
    const gi: Any = globalId.x;

    const s: Any = workgroupArray('uint', W);
    s.element(tid).assign(inputNode.element(gi));
    workgroupBarrier();

    // Up-sweep (reduce). Snapshot `offset` into `off` each iteration so the
    // If() callback — which TSL invokes later during shader build — sees the
    // correct per-stage value instead of the final post-loop JS value.
    let offset = 1;
    for (let d = W >> 1; d > 0; d >>= 1) {
      const off = offset;
      If(tid.lessThan(uint(d)), () => {
        const ai: Any = tid.mul(uint(off * 2)).add(uint(off - 1));
        const bi: Any = ai.add(uint(off));
        s.element(bi).assign(s.element(bi).add(s.element(ai)));
      });
      workgroupBarrier();
      offset *= 2;
    }

    // Save the total (inclusive sum of block) for the block-sums pass,
    // then clear the last element to start the down-sweep (exclusive).
    If(tid.equal(uint(0)), () => {
      blockSumsNode.element(wg).assign(s.element(uint(W - 1)));
      s.element(uint(W - 1)).assign(uint(0));
    });
    workgroupBarrier();

    // Down-sweep. Same closure-capture hazard as up-sweep.
    for (let d = 1; d < W; d *= 2) {
      offset >>= 1;
      const off = offset;
      If(tid.lessThan(uint(d)), () => {
        const ai: Any = tid.mul(uint(off * 2)).add(uint(off - 1));
        const bi: Any = ai.add(uint(off));
        const t: Any = s.element(ai).toVar();
        s.element(ai).assign(s.element(bi));
        s.element(bi).assign(s.element(bi).add(t));
      });
      workgroupBarrier();
    }

    scannedNode.element(gi).assign(s.element(tid));
  })().compute(N, [W]);

  // Pass 2 kernel — single-workgroup Blelloch scan of block sums.
  // Block sums array has exactly NUM_BLOCKS (=W) entries, so one workgroup.
  const blockSumScanKernel = Fn(() => {
    const tid: Any = localId.x;

    const s: Any = workgroupArray('uint', W);
    s.element(tid).assign(blockSumsNode.element(tid));
    workgroupBarrier();

    let offset = 1;
    for (let d = W >> 1; d > 0; d >>= 1) {
      const off = offset;
      If(tid.lessThan(uint(d)), () => {
        const ai: Any = tid.mul(uint(off * 2)).add(uint(off - 1));
        const bi: Any = ai.add(uint(off));
        s.element(bi).assign(s.element(bi).add(s.element(ai)));
      });
      workgroupBarrier();
      offset *= 2;
    }

    If(tid.equal(uint(0)), () => {
      s.element(uint(W - 1)).assign(uint(0));
    });
    workgroupBarrier();

    for (let d = 1; d < W; d *= 2) {
      offset >>= 1;
      const off = offset;
      If(tid.lessThan(uint(d)), () => {
        const ai: Any = tid.mul(uint(off * 2)).add(uint(off - 1));
        const bi: Any = ai.add(uint(off));
        const t: Any = s.element(ai).toVar();
        s.element(ai).assign(s.element(bi));
        s.element(bi).assign(s.element(bi).add(t));
      });
      workgroupBarrier();
    }

    blockSumsNode.element(tid).assign(s.element(tid));
  })().compute(W, [W]);

  // Pass 3 kernel — add each block's scanned prefix to every element of the block.
  const addPrefixKernel = Fn(() => {
    const i: Any = instanceIndex;
    const wg: Any = i.div(uint(W));
    scannedNode.element(i).assign(scannedNode.element(i).add(blockSumsNode.element(wg)));
  })().compute(N);

  const t0 = performance.now();
  await renderer.computeAsync([blockScanKernel, blockSumScanKernel, addPrefixKernel]);
  const gpuResult = new Uint32Array(await renderer.getArrayBufferAsync(scannedNode.value));
  const elapsedMs = performance.now() - t0;

  // CPU reference (exclusive prefix scan, u32 wrap).
  const reference = new Uint32Array(N);
  let acc = 0;
  for (let i = 0; i < N; i++) {
    reference[i] = acc >>> 0;
    acc = (acc + inputArr[i]!) >>> 0;
  }

  let firstMismatchAt = -1;
  for (let i = 0; i < N; i++) {
    if (gpuResult[i] !== reference[i]) {
      firstMismatchAt = i;
      break;
    }
  }

  renderer.dispose();
  return {
    n: N,
    seed,
    matched: firstMismatchAt === -1,
    firstMismatchAt,
    elapsedMs,
  };
}
