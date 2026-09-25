// Phase Perf paper-gap diagnosis — H3 TSL vs raw WGSL kernel cost.
//
// Microbenchmark: brute-force O(N²) Poly6 density evaluation at N=2000
// particles, run once as a TSL kernel and once as a hand-rolled raw
// WGSL kernel reading from the same predictedPositions buffer. Both
// kernels do mathematically identical work — same Poly6 polynomial,
// same self-clamp, same accumulator pattern.
//
// Hypothesis: TSL transpiles to WGSL via three.js's node-graph
// codegen. The emitted WGSL may be substantially less efficient than
// hand-written WGSL — extra register moves, indirect access patterns,
// missed compiler optimizations, redundant bounds checks, etc. If the
// TSL kernel costs >2x what the raw-WGSL kernel costs on the same
// workload, that delta multiplies into every neighbor-walking
// production kernel.
//
// Why brute-force O(N²) instead of replicating the production hash-
// grid walk: a faithful raw-WGSL replica of `emitForEachNeighbor`
// would re-implement Morton hashing, the 27-cell walk, the bucket-
// dedup chain, and the within-h filter — hundreds of lines of code
// that themselves could harbor independent inefficiencies. The
// brute-force microbenchmark eliminates the neighbor structure as a
// confounding variable: both kernels iterate over an identical flat
// loop of N candidates per particle, so any cost difference is
// attributable to the kernel emitter alone.
//
// At N=2000, work is 2000 × 2000 = 4M poly6 evaluations per dispatch
// — small enough to run in a few ms, large enough that per-thread
// overhead is amortized.
//

import { Fn, Loop, float, instanceIndex, instancedArray, uniform, uint } from 'three/tsl';
import { TimestampQuery } from 'three/src/constants.js';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import { describe, expect, it } from 'vitest';
import { createParticleRenderer } from '../../../src/core/index.js';

const SENTINEL_BEGIN = '__PARTICLE_FLUIDS_PAPER_GAP_H3_BEGIN__';
const SENTINEL_END = '__PARTICLE_FLUIDS_PAPER_GAP_H3_END__';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const N = 2000;
const SPACING = 0.025;
const H = 0.04;
const REST_DENSITY = 1000;
const MEASURE_RUNS = 20;

interface H3Sample {
  readonly impl: 'tsl' | 'wgsl';
  readonly p10Ms: number;
  readonly p50Ms: number;
  readonly p90Ms: number;
  readonly perParticleNs: number;
}

interface H3Report {
  readonly probeId: 'paper-gap-h3-tsl-vs-wgsl-density';
  readonly capturedAtIso: string;
  readonly particleCount: number;
  readonly measureRuns: number;
  readonly tsl: H3Sample;
  readonly wgsl: H3Sample;
  /**
   * Ratio TSL p50 / raw-WGSL p50.
   *   ≥ 3 ⇒ TSL inefficiency is load-bearing for the gap.
   *   1.5 – 3 ⇒ contributory.
   *   < 1.5 ⇒ TSL is fine, look elsewhere.
   */
  readonly tslOverWgslRatio: number;
}

const POLY6_COEF = 315 / (64 * Math.PI * Math.pow(H, 9));
const H_SQ = H * H;

function buildSettledCubePositions(n: number): Float32Array {
  const side = Math.ceil(Math.cbrt(n));
  const out = new Float32Array(n * 4);
  let k = 0;
  for (let z = 0; z < side && k < n; z++) {
    for (let y = 0; y < side && k < n; y++) {
      for (let x = 0; x < side && k < n; x++) {
        out[k * 4 + 0] = (x + 0.5) * SPACING;
        out[k * 4 + 1] = (y + 0.5) * SPACING;
        out[k * 4 + 2] = (z + 0.5) * SPACING;
        out[k * 4 + 3] = 0;
        k++;
      }
    }
  }
  return out;
}

function quantile(samples: number[], q: number): number {
  if (samples.length === 0) return 0;
  const sorted = [...samples].sort((a, b) => a - b);
  const idx = q * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! * (1 - (idx - lo)) + sorted[hi]! * (idx - lo);
}

const RAW_WGSL = `
struct Uniforms {
  hSq: f32,
  poly6Coef: f32,
  particleCount: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> density: array<f32>;
@group(0) @binding(2) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.particleCount) { return; }
  let xi = positions[i].xyz;
  var rho: f32 = 0.0;
  for (var j: u32 = 0u; j < u.particleCount; j = j + 1u) {
    let xj = positions[j].xyz;
    let diff = xi - xj;
    let rSq = dot(diff, diff);
    if (rSq < u.hSq) {
      let d = max(u.hSq - rSq, 0.0);
      rho = rho + d * d * d * u.poly6Coef;
    }
  }
  density[i] = rho;
}
`;

async function captureH3(): Promise<H3Report> {
  const renderer = await createParticleRenderer({ trackTimestamp: true });
  const backend = renderer.backend as { readonly device?: GPUDevice };
  const device = backend.device;
  if (!device) {
    renderer.dispose();
    throw new Error('paper-gap-h3: renderer.backend.device unavailable after init');
  }
  if (!device.features.has('timestamp-query')) {
    renderer.dispose();
    throw new Error('paper-gap-h3: timestamp-query feature unavailable; H3 needs GPU-busy timing');
  }

  const initialPositions = buildSettledCubePositions(N);

  // ---- TSL kernel ----
  const tslPositions: StorageBufferNode<'vec4'> = instancedArray(N, 'vec4');
  (tslPositions.value as Any).array.set(initialPositions);
  (tslPositions.value as Any).needsUpdate = true;

  const tslDensity: StorageBufferNode<'float'> = instancedArray(N, 'float');

  const hSqU: UniformNode<'float', number> = uniform(H_SQ, 'float');
  const poly6U: UniformNode<'float', number> = uniform(POLY6_COEF, 'float');
  const countU: UniformNode<'uint', number> = uniform(uint(N));

  const tslDensityKernel: ComputeNode = Fn(() => {
    const i: Any = (instanceIndex as Any).toVar();
    const xi: Any = (tslPositions as Any).element(i).xyz.toVar();
    const rho: Any = float(0.0).toVar();
    // Brute-force O(N²) loop. TSL's `Loop({ start, end, type })`
    // emits a WGSL `for` loop with a dynamic upper bound from the
    // uniform — matches the raw-WGSL kernel's structure.
    (Loop as Any)({ start: 0, end: countU as Any, type: 'uint' }, ({ i: j }: { i: Any }) => {
      const xj: Any = (tslPositions as Any).element(j).xyz;
      const diff: Any = xi.sub(xj).toVar();
      const rSq: Any = diff.dot(diff).toVar();
      const d: Any = (hSqU as Any).sub(rSq).max(float(0.0)).toVar();
      rho.addAssign(
        d
          .mul(d)
          .mul(d)
          .mul(poly6U as Any),
      );
    });
    (tslDensity as Any).element(i).assign(rho);
  })().compute(N);

  // Warmup: pipeline build + first dispatch.
  for (let i = 0; i < 3; i++) {
    await renderer.computeAsync(tslDensityKernel);
  }
  await device.queue.onSubmittedWorkDone();

  // Measure TSL via three's resolveTimestampsAsync.
  const tslTimings: number[] = [];
  for (let i = 0; i < MEASURE_RUNS; i++) {
    await renderer.computeAsync(tslDensityKernel);
    const ms = await renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
    if (typeof ms === 'number' && Number.isFinite(ms)) {
      tslTimings.push(ms);
    }
  }

  // ---- Raw WGSL kernel ----
  const positionsBuffer = device.createBuffer({
    size: initialPositions.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positionsBuffer, 0, initialPositions);

  const densityBuffer = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(densityBuffer, 0, new Float32Array(N));

  const uniformsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniformsScratch = new ArrayBuffer(16);
  new Float32Array(uniformsScratch, 0, 1)[0] = H_SQ;
  new Float32Array(uniformsScratch, 4, 1)[0] = POLY6_COEF;
  new Uint32Array(uniformsScratch, 8, 1)[0] = N;
  device.queue.writeBuffer(uniformsBuffer, 0, uniformsScratch);

  const bgLayout = device.createBindGroupLayout({
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
    bindGroupLayouts: [bgLayout],
  });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: device.createShaderModule({ code: RAW_WGSL }),
      entryPoint: 'main',
    },
  });
  const bindGroup = device.createBindGroup({
    layout: bgLayout,
    entries: [
      { binding: 0, resource: { buffer: positionsBuffer } },
      { binding: 1, resource: { buffer: densityBuffer } },
      { binding: 2, resource: { buffer: uniformsBuffer } },
    ],
  });

  const dispatchCount = Math.ceil(N / 64);

  // Warmup.
  for (let i = 0; i < 3; i++) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchCount);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const wgslTimings: number[] = [];
  for (let i = 0; i < MEASURE_RUNS; i++) {
    const querySet = device.createQuerySet({
      type: 'timestamp',
      count: 2,
    });
    const resolveBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchCount);
    pass.end();
    encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 16);

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const tsCopy = readbackBuffer.getMappedRange().slice(0);
    readbackBuffer.unmap();
    const ts = new BigUint64Array(tsCopy);
    const ms = Number(ts[1]! - ts[0]!) / 1e6;
    wgslTimings.push(ms);

    querySet.destroy();
    resolveBuffer.destroy();
    readbackBuffer.destroy();
  }

  positionsBuffer.destroy();
  densityBuffer.destroy();
  uniformsBuffer.destroy();
  renderer.dispose();

  const tslSample: H3Sample = {
    impl: 'tsl',
    p10Ms: quantile(tslTimings, 0.1),
    p50Ms: quantile(tslTimings, 0.5),
    p90Ms: quantile(tslTimings, 0.9),
    perParticleNs: (quantile(tslTimings, 0.5) * 1e6) / N,
  };
  const wgslSample: H3Sample = {
    impl: 'wgsl',
    p10Ms: quantile(wgslTimings, 0.1),
    p50Ms: quantile(wgslTimings, 0.5),
    p90Ms: quantile(wgslTimings, 0.9),
    perParticleNs: (quantile(wgslTimings, 0.5) * 1e6) / N,
  };

  return {
    probeId: 'paper-gap-h3-tsl-vs-wgsl-density',
    capturedAtIso: new Date().toISOString(),
    particleCount: N,
    measureRuns: MEASURE_RUNS,
    tsl: tslSample,
    wgsl: wgslSample,
    tslOverWgslRatio: wgslSample.p50Ms > 0 ? tslSample.p50Ms / wgslSample.p50Ms : 0,
  };
}

describe('Phase Perf paper-gap H3 — TSL vs raw WGSL density', () => {
  it('compares brute-force O(N²) Poly6 density: TSL emit vs hand-rolled WGSL', async () => {
    const report = await captureH3();

    const line = JSON.stringify(report);
    // eslint-disable-next-line no-console
    console.log(SENTINEL_BEGIN);
    // eslint-disable-next-line no-console
    console.log(line);
    // eslint-disable-next-line no-console
    console.log(SENTINEL_END);

    expect(report.tsl.p50Ms).toBeGreaterThan(0);
    expect(report.wgsl.p50Ms).toBeGreaterThan(0);
  }, 180_000);
});
