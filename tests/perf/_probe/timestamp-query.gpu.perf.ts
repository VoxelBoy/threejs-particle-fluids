// Phase Perf — Step 0 timestamp-query capability probe.
//

//
//   A — per-pass timestamps work (non-zero, monotonic, distinguishable,
//       and the timestamp domain is reasonable vs. wall-clock). Harness
//       uses per-kernel-pass timing.
//   B — `timestamp-query` is reported as supported but the readback fails
//       one of the four sanity checks (zeros, all-equal, non-monotonic,
//       or wildly off from wall-clock). Harness falls back to gross
//       (per-pipeline) timing.
//   C — `timestamp-query` not present on the device. Harness falls back
//       to gross timing.
//
// The probe asserts only that an outcome was classified — it does NOT
// assert that the platform produced Outcome A. The plan's exit criteria
// require the *outcome* to be documented, not that any specific outcome
// was achieved.
//

import { describe, expect, it } from 'vitest';
import { createParticleRenderer } from '../../../src/core/index.js';

const KERNEL_A_WGSL = `
struct Params { a: f32, n: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  y[i] = params.a * x[i] + y[i];
}
`;

const KERNEL_B_WGSL = `
struct Params { a: f32, n: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  // Heavier per-element work than kernel A so the two pass durations
  // are distinguishable on any reasonable timing source.
  let xi = x[i];
  let yi = y[i];
  y[i] = sqrt(xi * xi + yi * yi + 1.0) + sin(xi) + cos(yi);
}
`;

interface ProbeOutcomeA {
  readonly kind: 'A';
  readonly passDurationsMs: readonly [number, number];
  readonly wallMs: number;
  readonly rawNs: readonly [string, string, string, string];
}

interface ProbeOutcomeB {
  readonly kind: 'B';
  readonly reason: string;
  readonly rawNs: readonly [string, string, string, string];
  readonly wallMs: number;
}

interface ProbeOutcomeC {
  readonly kind: 'C';
  readonly reason: string;
}

type ProbeOutcome = ProbeOutcomeA | ProbeOutcomeB | ProbeOutcomeC;

async function runTimestampQueryProbe(): Promise<ProbeOutcome> {
  const renderer = await createParticleRenderer();
  const backend = renderer.backend as { readonly device?: GPUDevice };
  const device = backend.device;

  if (!device) {
    renderer.dispose();
    return {
      kind: 'C',
      reason: 'renderer.backend.device unavailable after init',
    };
  }

  if (!device.features.has('timestamp-query')) {
    renderer.dispose();
    return {
      kind: 'C',
      reason: "GPUDevice does not expose the 'timestamp-query' feature",
    };
  }

  const N = 1 << 20; // 1M elements — large enough for measurable cost
  const dispatch = Math.ceil(N / 64);

  const x = new Float32Array(N);
  const y = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    x[i] = Math.sin(i * 1e-3);
    y[i] = Math.cos(i * 1e-3);
  }

  const xBuffer = device.createBuffer({
    size: x.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const yBuffer = device.createBuffer({
    size: y.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(xBuffer, 0, x);
  device.queue.writeBuffer(yBuffer, 0, y);

  // Uniform buffer must be at least 16 bytes for std140-style alignment.
  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const paramsScratch = new ArrayBuffer(16);
  new Float32Array(paramsScratch, 0, 1)[0] = 2.5;
  new Uint32Array(paramsScratch, 4, 1)[0] = N;
  device.queue.writeBuffer(paramsBuffer, 0, paramsScratch);

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipelineA = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: device.createShaderModule({ code: KERNEL_A_WGSL }),
      entryPoint: 'main',
    },
  });
  const pipelineB = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: device.createShaderModule({ code: KERNEL_B_WGSL }),
      entryPoint: 'main',
    },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: xBuffer } },
      { binding: 1, resource: { buffer: yBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  const querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
  const resolveBuffer = device.createBuffer({
    size: 4 * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: 4 * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();

  {
    const pass = encoder.beginComputePass({
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      },
    });
    pass.setPipeline(pipelineA);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatch);
    pass.end();
  }
  {
    const pass = encoder.beginComputePass({
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: 2,
        endOfPassWriteIndex: 3,
      },
    });
    pass.setPipeline(pipelineB);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatch);
    pass.end();
  }

  encoder.resolveQuerySet(querySet, 0, 4, resolveBuffer, 0);
  encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 4 * 8);

  const t0 = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const wallMs = performance.now() - t0;

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const tsCopy = readbackBuffer.getMappedRange().slice(0);
  readbackBuffer.unmap();
  const ts = new BigUint64Array(tsCopy);

  querySet.destroy();
  resolveBuffer.destroy();
  readbackBuffer.destroy();
  xBuffer.destroy();
  yBuffer.destroy();
  paramsBuffer.destroy();
  renderer.dispose();

  const t = [ts[0]!, ts[1]!, ts[2]!, ts[3]!] as const;
  const rawNs = [t[0].toString(), t[1].toString(), t[2].toString(), t[3].toString()] as const;

  const allNonZero = t.every((v) => v > 0n);
  const monotonic = t[0] <= t[1] && t[1] <= t[2] && t[2] <= t[3];
  const durA = t[1] - t[0];
  const durB = t[3] - t[2];
  const durAms = Number(durA) / 1e6;
  const durBms = Number(durB) / 1e6;

  if (!allNonZero) {
    return {
      kind: 'B',
      reason: `One or more timestamps are zero: ${rawNs.join(', ')}`,
      rawNs,
      wallMs,
    };
  }
  if (!monotonic) {
    return {
      kind: 'B',
      reason: `Timestamps not monotonically non-decreasing: ${rawNs.join(', ')}`,
      rawNs,
      wallMs,
    };
  }
  if (durA === durB) {
    return {
      kind: 'B',
      reason: `Pass durations are exactly equal (${durA.toString()} ns) — kernel B does strictly more work than kernel A so this indicates the timestamp domain is collapsed`,
      rawNs,
      wallMs,
    };
  }
  // Wall-clock includes JS / submit / queue.onSubmittedWorkDone overhead, so
  // wallMs typically exceeds passSumMs. We only flag the inverse: GPU pass
  // durations exceeding wall-clock by >2× means the timestamp domain is
  // garbage (e.g. some unknown unit, or a leftover counter from a prior run).
  if (durAms > wallMs * 2 || durBms > wallMs * 2) {
    return {
      kind: 'B',
      reason: `Pass durations exceed wall-clock by >2×: passA=${durAms.toFixed(3)}ms, passB=${durBms.toFixed(3)}ms, wall=${wallMs.toFixed(3)}ms`,
      rawNs,
      wallMs,
    };
  }

  return {
    kind: 'A',
    passDurationsMs: [durAms, durBms],
    wallMs,
    rawNs,
  };
}

describe('Phase Perf — Step 0 timestamp-query probe', () => {
  it('classifies the reference platform as Outcome A / B / C', async () => {
    const result = await runTimestampQueryProbe();
    // Print outcome to the test log so the user can read it back when
    // running `npm run test:perf`. The probe records the outcome — it does
    // not assert any specific path.
    // eslint-disable-next-line no-console
    console.log('[Phase Perf probe] outcome: ' + JSON.stringify(result, null, 2));
    expect(result.kind).toMatch(/^[ABC]$/);
  }, 60_000);
});
