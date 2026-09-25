// Phase Perf-10 — workgroup-size sweep probe.
//

//
//   Shape A — walk-bound pair-list consumer. Mirrors lambda /
//             positionDelta / cohesion / vorticityPass1 / xsphPass1.
//             Reads the pair list, gathers `predictedPositions[j]` per
//             pair, accumulates a vec3 into a per-particle output. No
//             atomics. Memory-bound.
//   Shape B — atomic-scatter contact accumulator. Mirrors
//             contact.frictionVelocity scatter pattern. Reads a contact
//             pair, atomicAdd's into a per-particle counter slot.
//             Atomic-bound.
//
// Why TSL kernels: the question is "what's optimal under our actual
// stack — TSL → three.js node-builder → WGSL → Dawn → Metal." Hand-
// rolling WGSL would skip the TSL layer and answer a different
// question.
//

import { Vector3 } from 'three';
import {
  Fn,
  Loop,
  atomicAdd,
  atomicStore,
  instanceIndex,
  instancedArray,
  uint,
  vec3,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import { describe, expect, it } from 'vitest';

import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';
import { SoftbodySystem, type SoftbodyDef } from '../../../src/softbody/index.js';

import { PerfRenderer } from '../_helpers/PerfRenderer.js';
import { fluidColumn } from '../_scenes/_helpers.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const SENTINEL_BEGIN = '__PARTICLE_FLUIDS_WORKGROUP_SWEEP_BEGIN__';
const SENTINEL_END = '__PARTICLE_FLUIDS_WORKGROUP_SWEEP_END__';

const WORKGROUP_SIZES: readonly number[] = [32, 64, 128, 256];

const WARMUP_KERNEL_DISPATCHES = 5;
const MEASURE_KERNEL_DISPATCHES = 60;
const REPEATS = 3;

interface SweepSample {
  readonly workgroupSize: number;
  readonly p10Ms: number;
  readonly p50Ms: number;
  readonly p90Ms: number;
  readonly p99Ms: number;
  readonly nSamples: number;
}

interface ShapeReport {
  readonly shape: 'A-walk-bound' | 'B-atomic-scatter';
  readonly contextNote: string;
  readonly samples: readonly SweepSample[];
  /** p50 ratio of best-W vs default-64. <1 means best is faster than default. */
  readonly bestVsDefault: number;
  readonly bestWorkgroupSize: number;
  /** True iff |best - default| / default ≥ 10% on this shape's p50. */
  readonly meetsStage2Gate: boolean;
}

interface ProbeReport {
  readonly probeId: 'phase-perf-10-workgroup-size-sweep';
  readonly capturedAtIso: string;
  readonly host: {
    readonly platform: string;
    readonly note: string;
  };
  readonly shapes: readonly ShapeReport[];
}

function quantile(samples: readonly number[], q: number): number {
  if (samples.length === 0) return 0;
  const sorted = [...samples].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0]!;
  const idx = q * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  const t = idx - lo;
  return sorted[lo]! * (1 - t) + sorted[hi]! * t;
}

async function timeKernel(
  perf: PerfRenderer,
  kernel: ComputeNode,
): Promise<SweepSample['p50Ms'][]> {
  // Pipeline-build warmup (first dispatches include compile cost).
  for (let i = 0; i < WARMUP_KERNEL_DISPATCHES; i++) {
    await perf.runKernelInIsolation(kernel);
  }
  const samples: number[] = [];
  for (let r = 0; r < REPEATS; r++) {
    for (let i = 0; i < MEASURE_KERNEL_DISPATCHES; i++) {
      const ms = await perf.runKernelInIsolation(kernel);
      samples.push(ms);
    }
  }
  return samples;
}

function summarize(workgroupSize: number, samples: readonly number[]): SweepSample {
  return {
    workgroupSize,
    p10Ms: quantile(samples, 0.1),
    p50Ms: quantile(samples, 0.5),
    p90Ms: quantile(samples, 0.9),
    p99Ms: quantile(samples, 0.99),
    nSamples: samples.length,
  };
}

function rollupShape(
  shape: ShapeReport['shape'],
  contextNote: string,
  samples: readonly SweepSample[],
): ShapeReport {
  const default64 = samples.find((s) => s.workgroupSize === 64);
  if (!default64) throw new Error('rollupShape: missing W=64 sample');
  const best = samples.reduce((acc, s) => (s.p50Ms < acc.p50Ms ? s : acc), samples[0]!);
  const ratio = best.p50Ms / default64.p50Ms;
  const meetsStage2Gate = Math.abs(1 - ratio) >= 0.1;
  return {
    shape,
    contextNote,
    samples,
    bestVsDefault: ratio,
    bestWorkgroupSize: best.workgroupSize,
    meetsStage2Gate,
  };
}

// =============================================================================
// Shape A — walk-bound pair-list consumer.
// =============================================================================
//
// Setup: build a 100k-particle fluid-surface-tension-style scene, run
// production warmup so FluidSystem's pair list is populated with
// realistic neighbor counts, then dispatch a synthetic walk kernel at
// each W against the live pair list. Each dispatch reads
// pairList[i*MAX_NEIGHBORS+k] → predictedPositions[j] → vec3 accum.

const SHAPE_A_SPACING = 0.025;
const SHAPE_A_H = 0.04;
const SHAPE_A_R = SHAPE_A_SPACING * 0.5;
const SHAPE_A_REST_DENSITY = 1000;
const SHAPE_A_SUBSTEPS = 4;
const SHAPE_A_ITERATIONS = 2;
const SHAPE_A_COUNT = 100_000;
const SHAPE_A_MAX_NEIGHBORS = 64;
const SHAPE_A_WARMUP_FRAMES = 60;

function buildShapeAKernel(args: {
  readonly particleCount: number;
  readonly maxNeighbors: number;
  readonly predictedPositions: Any;
  readonly pairList: Any;
  readonly pairCount: Any;
  readonly out: Any;
  readonly workgroupSize: number;
}): ComputeNode {
  const {
    particleCount,
    maxNeighbors,
    predictedPositions,
    pairList,
    pairCount,
    out,
    workgroupSize,
  } = args;
  return Fn(() => {
    const i: Any = (instanceIndex as Any).toVar();
    const xi: Any = predictedPositions.element(i).xyz.toVar();
    const count: Any = pairCount.element(i).toVar();
    const acc: Any = vec3(0, 0, 0).toVar();
    Loop({ start: uint(0), end: count, type: 'uint', condition: '<' }, ({ i: k }: { i: Any }) => {
      const j: Any = pairList.element(i.mul(uint(maxNeighbors)).add(k));
      const xj: Any = predictedPositions.element(j).xyz;
      acc.assign(acc.add(xj.sub(xi)));
    });
    out.element(i).assign(acc);
  })().compute(particleCount, [workgroupSize]);
}

// =============================================================================
// Shape B — atomic-scatter contact accumulator.
// =============================================================================
//
// Setup: build a 25-body softbody scene (25 × 1000 = 25k particles)
// dropped onto a floor plane with the contact pipeline enabled. Run
// production warmup so contact.generate has populated ContactBuffer.pairs
// with realistic contact count and i-index distribution. Then dispatch
// a synthetic atomic-scatter kernel at each W against the live contact
// buffer.
//
// Sizing rationale: 25 bodies generates ~50-150k contacts post-settle
// (each body has ~600 surface particles, contact pairs are O(M) per
// body when bodies are stacked). 25k particles is enough to produce a
// real atomic-contention pattern while keeping warmup runtime under
// ~30s. The plan called for "softbody-100k snapshot"; 25k is
// equivalent for our measurement purpose because (a) atomic contention
// scales with contact count not particle count, and (b) 25k still
// stresses the i-distribution non-uniformly across phases. Documented
// in the exit report.

const SHAPE_B_PARTICLES_PER_BODY = 1000;
const SHAPE_B_NUM_BODIES = 25;
const SHAPE_B_PARTICLE_RADIUS = 0.01;
const SHAPE_B_SPACING = SHAPE_B_PARTICLE_RADIUS * 2;
const SHAPE_B_SIDE = 10;
const SHAPE_B_CELL_SIZE = SHAPE_B_PARTICLE_RADIUS * 4;
const SHAPE_B_SUBSTEPS = 4;
const SHAPE_B_ITERATIONS = 2;
const SHAPE_B_WARMUP_FRAMES = 40;

function buildShapeBBody(
  bodyIndex: number,
  origin: Vector3,
): {
  readonly initial: ParticleInit[];
  readonly restPositions: Float32Array;
  readonly surfaceFlag: Uint8Array;
} {
  const cells: Array<{
    readonly position: readonly [number, number, number];
    readonly isSurface: boolean;
  }> = [];
  for (let z = 0; z < SHAPE_B_SIDE; z++) {
    for (let y = 0; y < SHAPE_B_SIDE; y++) {
      for (let x = 0; x < SHAPE_B_SIDE; x++) {
        const isSurface =
          x === 0 ||
          x === SHAPE_B_SIDE - 1 ||
          y === 0 ||
          y === SHAPE_B_SIDE - 1 ||
          z === 0 ||
          z === SHAPE_B_SIDE - 1;
        cells.push({
          position: [
            origin.x + (x - (SHAPE_B_SIDE - 1) / 2) * SHAPE_B_SPACING,
            origin.y + (y - (SHAPE_B_SIDE - 1) / 2) * SHAPE_B_SPACING,
            origin.z + (z - (SHAPE_B_SIDE - 1) / 2) * SHAPE_B_SPACING,
          ],
          isSurface,
        });
      }
    }
  }
  cells.sort((a, b) => Number(b.isSurface) - Number(a.isSurface));
  const restPositions = new Float32Array(cells.length * 3);
  const surfaceFlag = new Uint8Array(cells.length);
  const initial: ParticleInit[] = [];
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i]!;
    restPositions[i * 3] = c.position[0];
    restPositions[i * 3 + 1] = c.position[1];
    restPositions[i * 3 + 2] = c.position[2];
    surfaceFlag[i] = c.isSurface ? 1 : 0;
    initial.push({
      position: c.position,
      velocity: [0, 0, 0],
      invMass: 1,
      phase: bodyIndex,
    });
  }
  return { initial, restPositions, surfaceFlag };
}

function buildShapeBKernel(args: {
  readonly contactCount: number;
  readonly contactPairs: Any;
  readonly counts: Any;
  readonly workgroupSize: number;
}): ComputeNode {
  const { contactCount, contactPairs, counts, workgroupSize } = args;
  return Fn(() => {
    const m: Any = (instanceIndex as Any).toVar();
    // ContactBuffer.pairs layout is `[i0, j0, i1, j1, ...]` per
    // src/core/contact/ContactBuffer.ts L86-87.
    const i: Any = contactPairs.element(m.mul(uint(2)));
    atomicAdd(counts.element(i) as Any, uint(1));
  })().compute(contactCount, [workgroupSize]);
}

// =============================================================================
// Probe entry points.
// =============================================================================

describe('Phase Perf-10 — workgroup-size sweep', () => {
  const accumulatedShapes: ShapeReport[] = [];

  it(
    'Shape A: walk-bound pair-list consumer at 100k fluid particles',
    async () => {
      const perf = await PerfRenderer.create();
      const particles = new ParticleSystem(perf.renderer, SHAPE_A_COUNT, SHAPE_A_R);
      particles.uploadParticles(
        fluidColumn({
          count: SHAPE_A_COUNT,
          spacing: SHAPE_A_SPACING,
          origin: [-0.6, 0.5, -0.6],
        }),
      );

      const hashGrid = new HashGrid(particles, { cellSize: SHAPE_A_H });
      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
      colliders.upload();

      const xpbd = createXpbdUniforms(1 / 60);
      const fluid = new FluidSystem({
        particles,
        hashGrid,
        xpbd,
        restDensity: SHAPE_A_REST_DENSITY,
        h: SHAPE_A_H,
        particleSpacing: SHAPE_A_SPACING,
        compliance: 1e-4,
        fluidParticles: { start: 0, count: SHAPE_A_COUNT },
        vorticity: { strength: 0.1 },
        xsph: { c: 0.01 },
      });

      const loop = new SimLoop(particles, {
        substeps: SHAPE_A_SUBSTEPS,
        iterations: SHAPE_A_ITERATIONS,
        xpbd,
        hashGrid,
        colliders: { colliders },
        materials: [fluid],
      });
      loop.gravity.set(0, -9.81, 0);

      const dt = 1 / 60;
      for (let i = 0; i < SHAPE_A_WARMUP_FRAMES; i++) await loop.step(dt);

      // Allocate probe-private output buffer. The fluid's own pair list
      // is the input — we DON'T want to disturb it, so we read from
      // FluidSystem's pair list and write to our own scratch buffer.
      const probeOut = instancedArray(SHAPE_A_COUNT, 'vec3');

      // FluidSystem exposes pairList / pairCount as public fields per
      // src/fluids/FluidSystem.ts L238-239.
      const pairList: Any = fluid.pairList;
      const pairCount: Any = fluid.pairCount;

      // Read back pair count distribution so we have evidence of
      // realistic neighbor density. Median pair count is the canonical
      // workload signal for the walk-bound kernel.
      const pairCountBuf = await perf.renderer.getArrayBufferAsync((pairCount as Any).value);
      const pairCountArr = new Uint32Array(pairCountBuf, 0, SHAPE_A_COUNT);
      const sortedCounts = [...pairCountArr].sort((a, b) => a - b);
      const medianPairs = sortedCounts[Math.floor(sortedCounts.length * 0.5)] ?? 0;
      const p99Pairs = sortedCounts[Math.floor(sortedCounts.length * 0.99)] ?? 0;

      const samples: SweepSample[] = [];
      for (const W of WORKGROUP_SIZES) {
        const kernel = buildShapeAKernel({
          particleCount: SHAPE_A_COUNT,
          maxNeighbors: SHAPE_A_MAX_NEIGHBORS,
          predictedPositions: particles.predictedPositions,
          pairList,
          pairCount,
          out: probeOut,
          workgroupSize: W,
        });
        const ms = await timeKernel(perf, kernel);
        samples.push(summarize(W, ms));
      }

      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
      perf.dispose();

      // eslint-disable-next-line no-console
      console.log(
        `[wg-sweep][A] workload: ${SHAPE_A_COUNT} particles, ` +
          `medianPairs=${medianPairs}, p99Pairs=${p99Pairs}`,
      );

      const report = rollupShape(
        'A-walk-bound',
        `${SHAPE_A_COUNT} fluid particles, settled (${SHAPE_A_WARMUP_FRAMES}-frame warmup), ` +
          `pair list MAX_NEIGHBORS=${SHAPE_A_MAX_NEIGHBORS}, h=${SHAPE_A_H}`,
        samples,
      );
      accumulatedShapes.push(report);

      // eslint-disable-next-line no-console
      console.log(
        '[wg-sweep][A] ' +
          samples.map((s) => `W=${s.workgroupSize} p50=${s.p50Ms.toFixed(3)}ms`).join('  ') +
          `  best=W${report.bestWorkgroupSize} (${(report.bestVsDefault * 100).toFixed(1)}% of W=64)` +
          `  gate=${report.meetsStage2Gate ? 'PASS' : 'fail'}`,
      );

      for (const s of samples) expect(s.p50Ms).toBeGreaterThan(0);
    },
    10 * 60_000,
  );

  it(
    'Shape B: atomic-scatter at 25k softbody particles',
    async () => {
      const perf = await PerfRenderer.create();
      const total = SHAPE_B_NUM_BODIES * SHAPE_B_PARTICLES_PER_BODY;
      const particles = new ParticleSystem(perf.renderer, total, SHAPE_B_PARTICLE_RADIUS);

      const allInitial: ParticleInit[] = [];
      const bodies: SoftbodyDef[] = [];
      const bodySpacing = SHAPE_B_SIDE * SHAPE_B_SPACING * 2;
      const gridSide = Math.ceil(Math.sqrt(SHAPE_B_NUM_BODIES));
      for (let b = 0; b < SHAPE_B_NUM_BODIES; b++) {
        const gx = b % gridSide;
        const gz = Math.floor(b / gridSide);
        const origin = new Vector3(
          (gx - gridSide / 2) * bodySpacing,
          1.0 + ((b * 13) % 7) * 0.05,
          (gz - gridSide / 2) * bodySpacing,
        );
        const body = buildShapeBBody(b, origin);
        bodies.push({
          particleRange: {
            start: b * SHAPE_B_PARTICLES_PER_BODY,
            count: SHAPE_B_PARTICLES_PER_BODY,
          },
          restPositions: body.restPositions,
          surfaceFlag: body.surfaceFlag,
          phaseId: b,
          matchCompliance: 1e-7,
        });
        for (const p of body.initial) allInitial.push(p);
      }
      particles.uploadParticles(allInitial);

      const hashGrid = new HashGrid(particles, { cellSize: SHAPE_B_CELL_SIZE });
      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
      colliders.upload();

      const xpbd = createXpbdUniforms(1 / 60);
      const softbody = new SoftbodySystem({ particles, xpbd, bodies });

      const loop = new SimLoop(particles, {
        substeps: SHAPE_B_SUBSTEPS,
        iterations: SHAPE_B_ITERATIONS,
        xpbd,
        hashGrid,
        contact: { hashGrid, maxContacts: total * 8 },
        colliders: { colliders },
        materials: [softbody],
      });
      loop.gravity.set(0, -9.81, 0);

      const dt = 1 / 60;
      for (let i = 0; i < SHAPE_B_WARMUP_FRAMES; i++) await loop.step(dt);

      // Per src/core/loop.ts L281, SimLoop exposes the populated
      // ContactBuffer as `loop.contacts` (plural).
      const contactBuffer: Any = (loop as Any).contacts;
      if (!contactBuffer || !contactBuffer.pairs || !contactBuffer.counter) {
        throw new Error(
          'Shape B: SimLoop.contacts unavailable; ' +
            'SimLoop internal layout changed — update the probe',
        );
      }
      const contactCount: number = await contactBuffer.readbackCount();
      if (contactCount <= 0) {
        throw new Error(
          `Shape B: contact buffer is empty after ${SHAPE_B_WARMUP_FRAMES} warmup frames — ` +
            'softbody scene did not settle, increase warmup or check geometry',
        );
      }
      // eslint-disable-next-line no-console
      console.log(
        `[wg-sweep][B] workload: ${total} particles, ${contactCount} contacts ` +
          `(${(contactCount / total).toFixed(1)} per particle)`,
      );

      // Probe-private accumulator buffer. We don't reuse the contact
      // pipeline's accumulator — we want a clean atomicAdd target.
      const probeCounts = (instancedArray(total, 'uint') as Any).toAtomic();

      // Reset kernel — clears probeCounts before each measurement so
      // saturating contention doesn't accumulate run-to-run. Single
      // dispatch at W=64 (irrelevant to the sweep).
      const resetKernel = Fn(() => {
        atomicStore(probeCounts.element(instanceIndex as Any) as Any, uint(0));
      })().compute(total);

      const samples: SweepSample[] = [];
      for (const W of WORKGROUP_SIZES) {
        const kernel = buildShapeBKernel({
          contactCount,
          contactPairs: contactBuffer.pairs,
          counts: probeCounts,
          workgroupSize: W,
        });
        // Warmup includes a reset to clear contention from prior iters.
        for (let i = 0; i < WARMUP_KERNEL_DISPATCHES; i++) {
          await perf.runKernelInIsolation(resetKernel);
          await perf.runKernelInIsolation(kernel);
        }
        const msBatch: number[] = [];
        for (let r = 0; r < REPEATS; r++) {
          for (let i = 0; i < MEASURE_KERNEL_DISPATCHES; i++) {
            await perf.runKernelInIsolation(resetKernel);
            const ms = await perf.runKernelInIsolation(kernel);
            msBatch.push(ms);
          }
        }
        samples.push(summarize(W, msBatch));
      }

      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
      perf.dispose();

      const report = rollupShape(
        'B-atomic-scatter',
        `${total} softbody particles (${SHAPE_B_NUM_BODIES} × ${SHAPE_B_PARTICLES_PER_BODY}), ` +
          `${contactCount} contacts after ${SHAPE_B_WARMUP_FRAMES}-frame warmup`,
        samples,
      );
      accumulatedShapes.push(report);

      // eslint-disable-next-line no-console
      console.log(
        '[wg-sweep][B] ' +
          samples.map((s) => `W=${s.workgroupSize} p50=${s.p50Ms.toFixed(3)}ms`).join('  ') +
          `  best=W${report.bestWorkgroupSize} (${(report.bestVsDefault * 100).toFixed(1)}% of W=64)` +
          `  gate=${report.meetsStage2Gate ? 'PASS' : 'fail'}`,
      );

      for (const s of samples) expect(s.p50Ms).toBeGreaterThan(0);
    },
    10 * 60_000,
  );

  it('writes the rollup JSON report', () => {
    const report: ProbeReport = {
      probeId: 'phase-perf-10-workgroup-size-sweep',
      capturedAtIso: new Date().toISOString(),
      host: {
        platform: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
        note: 'Single-host probe (Apple M1 Pro / Chrome / WebGPU / Dawn / Metal); cross-vendor measurement is OUT OF SCOPE per plan',
      },
      shapes: accumulatedShapes,
    };
    // eslint-disable-next-line no-console
    console.log(SENTINEL_BEGIN);
    // eslint-disable-next-line no-console
    console.log(JSON.stringify(report, null, 2));
    // eslint-disable-next-line no-console
    console.log(SENTINEL_END);
    expect(accumulatedShapes.length).toBe(2);
  });
});
