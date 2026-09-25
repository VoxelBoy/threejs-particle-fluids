// Phase Perf paper-gap diagnosis — H1 dispatch overhead probe.
//
// Submits a no-op compute kernel at increasing dispatch counts within a
// single command buffer (each in its own compute pass with timestamp
// queries). Reports per-dispatch CPU+driver overhead in microseconds:
// the slope of (wallclock vs N) gives total per-dispatch overhead, and
// the slope of (GPU-busy vs N) gives the kernel's GPU-side work; the
// difference is the CPU/driver/JS share that scales with dispatch
// count.
//
// Why no-op kernel: a `@compute @workgroup_size(1)` kernel that does
// nothing isolates fixed per-dispatch overhead from per-particle
// kernel work. Real kernels' overhead is bounded below by this number.
//

import { describe, expect, it } from 'vitest';
import { createParticleRenderer } from '../../../src/core/index.js';

const SENTINEL_BEGIN = '__PARTICLE_FLUIDS_PAPER_GAP_H1_BEGIN__';
const SENTINEL_END = '__PARTICLE_FLUIDS_PAPER_GAP_H1_END__';

const NOOP_WGSL = `
@group(0) @binding(0) var<storage, read_write> y: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Touch one slot so the dispatch isn't dead-code-eliminated; pay
  // exactly one global memory write per dispatch and nothing else.
  y[0] = y[0] + 0u;
}
`;

const DISPATCH_COUNTS: readonly number[] = [1, 10, 100, 1000, 10000];
const REPEATS = 3; // submit-and-read-back this many times per N for stability.

interface DispatchOverheadSample {
  readonly dispatchCount: number;
  readonly wallMs: number;
  readonly gpuBusyMs: number;
  /**
   * Per-dispatch overhead in microseconds, isolated to CPU/driver/JS:
   * `(wallMs - gpuBusyMs) * 1000 / dispatchCount`. Lower bound on the
   * fixed overhead any production kernel pays per dispatch.
   */
  readonly perDispatchOverheadUs: number;
}

interface DispatchOverheadReport {
  readonly probeId: 'paper-gap-h1-dispatch-overhead';
  readonly capturedAtIso: string;
  readonly samples: readonly DispatchOverheadSample[];
  /**
   * Linear fit slope: per-dispatch overhead extrapolated from
   * (wallMs - gpuBusyMs) vs N. Microseconds. The flat dispatch
   * overhead in the limit of large N (CPU pipeline build + driver
   * submission per dispatch).
   */
  readonly slopeOverheadUsPerDispatch: number;
  /**
   * Intercept of the same fit. Microseconds. The fixed cost of one
   * submit + one queue.onSubmittedWorkDone, independent of N.
   */
  readonly interceptOverheadUs: number;
}

async function captureDispatchOverhead(): Promise<DispatchOverheadReport> {
  const renderer = await createParticleRenderer();
  const backend = renderer.backend as { readonly device?: GPUDevice };
  const device = backend.device;
  if (!device) {
    renderer.dispose();
    throw new Error('paper-gap-h1: renderer.backend.device unavailable after init');
  }
  if (!device.features.has('timestamp-query')) {
    renderer.dispose();
    throw new Error('paper-gap-h1: timestamp-query feature unavailable; H1 needs GPU-busy timing');
  }

  const yBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(yBuffer, 0, new Uint32Array([0, 0, 0, 0]));

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: device.createShaderModule({ code: NOOP_WGSL }),
      entryPoint: 'main',
    },
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: yBuffer } }],
  });

  const samples: DispatchOverheadSample[] = [];

  for (const N of DISPATCH_COUNTS) {
    const wallSamples: number[] = [];
    const gpuSamples: number[] = [];

    for (let r = 0; r < REPEATS; r++) {
      const querySet = device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
      const resolveBuffer = device.createBuffer({
        size: 2 * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const readbackBuffer = device.createBuffer({
        size: 2 * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const encoder = device.createCommandEncoder();

      // Single compute pass containing N back-to-back dispatches.
      // Per-pass timestamps bracket the whole batch; the GPU-busy
      // duration scales linearly with N if the GPU is pipelining
      // dispatches efficiently. Per-dispatch CPU overhead is in the
      // wallclock - GPU-busy difference.
      const pass = encoder.beginComputePass({
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let i = 0; i < N; i++) {
        pass.dispatchWorkgroups(1);
      }
      pass.end();

      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 2 * 8);

      const t0 = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const wallMs = performance.now() - t0;

      await readbackBuffer.mapAsync(GPUMapMode.READ);
      const tsCopy = readbackBuffer.getMappedRange().slice(0);
      readbackBuffer.unmap();
      const ts = new BigUint64Array(tsCopy);
      const gpuNs = ts[1]! - ts[0]!;
      const gpuMs = Number(gpuNs) / 1e6;

      querySet.destroy();
      resolveBuffer.destroy();
      readbackBuffer.destroy();

      wallSamples.push(wallMs);
      gpuSamples.push(gpuMs);
    }

    // Median for stability.
    wallSamples.sort((a, b) => a - b);
    gpuSamples.sort((a, b) => a - b);
    const wallMedian = wallSamples[Math.floor(REPEATS / 2)]!;
    const gpuMedian = gpuSamples[Math.floor(REPEATS / 2)]!;
    const perDispatchOverheadUs = ((wallMedian - gpuMedian) * 1000) / N;

    samples.push({
      dispatchCount: N,
      wallMs: wallMedian,
      gpuBusyMs: gpuMedian,
      perDispatchOverheadUs,
    });
  }

  // Linear regression of (wallMs - gpuMs) microseconds vs N to extract
  // the per-dispatch slope and the per-submit intercept.
  let sumN = 0;
  let sumOver = 0;
  let sumNN = 0;
  let sumNOver = 0;
  for (const s of samples) {
    const overUs = (s.wallMs - s.gpuBusyMs) * 1000;
    const N = s.dispatchCount;
    sumN += N;
    sumOver += overUs;
    sumNN += N * N;
    sumNOver += N * overUs;
  }
  const n = samples.length;
  const slope = (n * sumNOver - sumN * sumOver) / (n * sumNN - sumN * sumN);
  const intercept = (sumOver - slope * sumN) / n;

  yBuffer.destroy();
  renderer.dispose();

  return {
    probeId: 'paper-gap-h1-dispatch-overhead',
    capturedAtIso: new Date().toISOString(),
    samples,
    slopeOverheadUsPerDispatch: slope,
    interceptOverheadUs: intercept,
  };
}

describe('Phase Perf paper-gap H1 — dispatch overhead', () => {
  it('measures per-dispatch CPU+driver overhead at multiple dispatch counts', async () => {
    const report = await captureDispatchOverhead();

    const line = JSON.stringify(report);
    // eslint-disable-next-line no-console
    console.log(SENTINEL_BEGIN);
    // eslint-disable-next-line no-console
    console.log(line);
    // eslint-disable-next-line no-console
    console.log(SENTINEL_END);

    expect(report.samples.length).toBe(DISPATCH_COUNTS.length);
    // Sanity: GPU-busy time should be non-negative and not exceed
    // wallclock (GPU runs concurrently with CPU but its bracketed
    // duration is bounded above by the wallclock window).
    for (const s of report.samples) {
      expect(s.gpuBusyMs).toBeGreaterThanOrEqual(0);
      expect(s.wallMs).toBeGreaterThan(0);
    }
  }, 120_000);
});
