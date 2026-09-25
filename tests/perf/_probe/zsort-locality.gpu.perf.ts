// Phase Perf — Z-sort locality probe.
//
// Measures the speedup attributable to two separable improvements:
//
//   Improvement 1 (within-bucket SoA permutation)
//     Sort particle data so that particles in the same hash bucket sit
//     contiguous in memory, removing the `sortedIndices[k] -> positions[id]`
//     pointer chase from the inner loop.
//
//   Improvement 2 (Morton bucketing)
//     Replace Teschner XOR-mix with Morton (Z-curve) cell encoding so that
//     adjacent cells in 3D space have nearby bucket indices. The 27-cell
//     walk now visits a few short bucket-index ranges instead of 27
//     scattered ones, improving cellStart/cellEnd cache behavior across
//     the walk.
//
// The four corners of the design space are timed at three particle counts
// (10k, 50k, 100k):
//
//   T+indir   Teschner bucketing + sortedIndices indirection (production)
//   T+perm    Teschner bucketing + permuted SoA (Improvement 1 alone)
//   M+indir   Morton bucketing  + sortedIndices indirection (Improvement 2 alone)
//   M+perm    Morton bucketing  + permuted SoA (full Z-sort: 1 + 2)
//
// Two work shapes per corner — synthetic (sum neighbor x) isolates pure
// memory-access cost; Poly6-loaded approximates production fluid-kernel
// ALU mass. The bucket-dedup chain, the 27-cell unroll, and the per-
// candidate distance filter are identical across all variants — the only
// differences are the bucket function and the inner-loop neighbor read.
//
// Per-frame overhead is also measured: the GPU permute kernel cost (the
// per-substep tax for Improvement 1) and the cellIndexAndHistogram cost
// for each bucket function (the per-substep tax for Improvement 2 vs.
// the production Teschner baseline).
//
// The probe makes no architectural commitment. Its only assertion is that
// data was produced. Read the printed table to decide which improvement
// to land first, both, or neither.

import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicStore,
  float,
  instanceIndex,
  instancedArray,
  int,
  uint,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { ParticleSystem, padToScanWorkgroup } from '../../../src/core/index.js';
import { uniform } from 'three/tsl';

// Deep imports — the probe needs the count-sort kernels without going
// through src/core's public surface, because we are constructing
// two parallel hash grids (Teschner vs Morton) and the production
// HashGrid class hardcodes Teschner. Phase Perf-17 fused the
// post-scan finalize pass (addPrefix + resetWriteCursor + cellBounds)
// into a single `finalizeCellRanges` kernel returned by
// `buildCountSortKernels`.
import { buildCountSortKernels } from '../../../src/core/hashGrid/sort.js';

import { PerfRenderer } from '../_helpers/PerfRenderer.js';
import { fluidColumn } from '../_scenes/_helpers.js';

// Teschner hash primes — must match `src/core/hashGrid/cellIndex.ts`.
const P1 = 73856093;
const P2 = 19349663;
const P3 = 83492791;

// Morton bias. Cell coordinates can be negative; we shift by this before
// the 10-bit-per-axis Morton encoding so all coords land in the encodable
// range [0, 1023]. For the probe scenes (cells span ~−10 to ~+50) this
// bias places everything around 512 with comfortable headroom.
const MORTON_BIAS = 512;

const SPACING = 0.025;
const H = 0.04;
const R = SPACING * 0.5;
const HSQ = H * H;
const POLY6_COEF = 315 / (64 * Math.PI * Math.pow(H, 9));

const WARMUP_FRAMES = 20;
const MEASURE_FRAMES = 50;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

// A bucket function takes three signed-int cell-coord TSL nodes and
// returns an unsigned-int TSL node holding the unmasked bucket index.
// The caller masks with `hashTableSize - 1` to clamp to table range.
type BucketFn = (cx: Any, cy: Any, cz: Any) => Any;

// Teschner XOR-mix bucket function: ((cx · P1) ⊕ (cy · P2) ⊕ (cz · P3)).
// Mirrors `src/core/hashGrid/cellIndex.ts`.
const teschnerBucket: BucketFn = (cx, cy, cz) => {
  const m1: Any = cx.mul(int(P1));
  const m2: Any = cy.mul(int(P2));
  const m3: Any = cz.mul(int(P3));
  return m1.bitXor(m2).bitXor(m3).toUint();
};

/**
 * Spread the lower 10 bits of `n` over every third bit, leaving zeros
 * in between. Standard Morton "part1by2" pattern. Implemented inline
 * because TSL does not expose user-defined helper functions outside of
 * `Fn(...)` bodies, and we need this called from inside the bucket
 * computation.
 */
function part1by2(n: Any): Any {
  const m0: Any = n.bitAnd(uint(0x000003ff)); // 10 bits
  const m1a: Any = m0.bitOr(m0.shiftLeft(uint(16)));
  const m1: Any = m1a.bitAnd(uint(0xff0000ff));
  const m2a: Any = m1.bitOr(m1.shiftLeft(uint(8)));
  const m2: Any = m2a.bitAnd(uint(0x0300f00f));
  const m3a: Any = m2.bitOr(m2.shiftLeft(uint(4)));
  const m3: Any = m3a.bitAnd(uint(0x030c30c3));
  const m4a: Any = m3.bitOr(m3.shiftLeft(uint(2)));
  const m4: Any = m4a.bitAnd(uint(0x09249249));
  return m4;
}

// Morton bucket function: bias to non-negative, encode with bits
// interleaved x|y|z so adjacent cells in 3D have nearby bucket indices.
const mortonBucket: BucketFn = (cx, cy, cz) => {
  const ux: Any = cx.add(int(MORTON_BIAS)).toUint();
  const uy: Any = cy.add(int(MORTON_BIAS)).toUint();
  const uz: Any = cz.add(int(MORTON_BIAS)).toUint();
  const sx: Any = part1by2(ux);
  const sy: Any = part1by2(uy);
  const sz: Any = part1by2(uz);
  return sx.bitOr(sy.shiftLeft(uint(1))).bitOr(sz.shiftLeft(uint(2)));
};

interface Stats {
  readonly min: number;
  readonly p10: number;
  readonly p50: number;
  readonly p90: number;
  readonly max: number;
}

function summarize(samples: readonly number[]): Stats {
  if (samples.length === 0) throw new Error('summarize: empty samples');
  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  const pct = (p: number): number => sorted[Math.floor(p * (n - 1))]!;
  return {
    min: sorted[0]!,
    p10: pct(0.1),
    p50: pct(0.5),
    p90: pct(0.9),
    max: sorted[n - 1]!,
  };
}

/**
 * A minimal hash grid built around a parameterized bucket function. The
 * count-sort and cellBounds kernels come from the production module so
 * the only thing different between Teschner and Morton variants is the
 * cellIndexAndHistogram pass and the matching walk-side bucket function.
 */
interface ProbeHashGrid {
  readonly hashTableSize: number;
  readonly hashTableSizePadded: number;
  readonly cellSize: number;
  readonly hashOriginUniform: UniformNode<'vec3', Vector3>;
  readonly cellSizeUniform: UniformNode<'float', number>;
  readonly cellIndex: StorageBufferNode<'uint'>;
  readonly cellStart: StorageBufferNode<'uint'>;
  readonly cellEnd: StorageBufferNode<'uint'>;
  readonly sortedIndices: StorageBufferNode<'uint'>;
  readonly cellIndexAndHistogram: ComputeNode;
  readonly rebuildPipeline: readonly ComputeNode[];
}

function buildProbeHashGrid(
  particles: ParticleSystem,
  cellSize: number,
  hashTableSize: number,
  bucketFn: BucketFn,
): ProbeHashGrid {
  if ((hashTableSize & (hashTableSize - 1)) !== 0) {
    throw new Error('hashTableSize must be a power of two');
  }
  const hashTableSizePadded = padToScanWorkgroup(hashTableSize);
  const bucketMask = hashTableSize - 1;

  const hashOriginUniform = uniform(new Vector3(0, 0, 0));
  const cellSizeUniform = uniform(cellSize, 'float');

  const cellIndex = instancedArray(particles.capacity, 'uint') as Any;
  const counts = (instancedArray(hashTableSizePadded, 'uint') as Any).toAtomic();
  const cellStart = instancedArray(hashTableSizePadded, 'uint') as Any;
  const cellEnd = instancedArray(hashTableSizePadded, 'uint') as Any;
  const sortedIndices = instancedArray(particles.capacity, 'uint') as Any;
  const blockSums = instancedArray(1024, 'uint') as Any; // SCAN_WORKGROUP_SIZE
  const writeCursor = (instancedArray(hashTableSizePadded, 'uint') as Any).toAtomic();

  // Reset the histogram.
  const resetCounts = Fn(() => {
    const i: Any = instanceIndex;
    atomicStore(counts.element(i), uint(0));
  })().compute(hashTableSizePadded);

  // Compute per-particle bucket and histogram.
  const cellIndexAndHistogram = Fn(() => {
    const i: Any = instanceIndex;
    const p: Any = (particles.positions as Any).element(i);
    const rel: Any = p.xyz.sub(hashOriginUniform as Any).div(cellSizeUniform as Any);
    const cx: Any = rel.x.floor().toInt();
    const cy: Any = rel.y.floor().toInt();
    const cz: Any = rel.z.floor().toInt();
    const bucket: Any = bucketFn(cx, cy, cz).bitAnd(uint(bucketMask));
    cellIndex.element(i).assign(bucket);
    atomicAdd(counts.element(bucket), uint(1));
  })().compute(particles.capacity);

  const sort = buildCountSortKernels(
    counts,
    cellStart,
    cellEnd,
    blockSums,
    writeCursor,
    cellIndex,
    sortedIndices,
    particles.capacity,
    hashTableSizePadded,
  );

  const rebuildPipeline: ComputeNode[] = [
    resetCounts,
    cellIndexAndHistogram,
    sort.blockScan,
    sort.blockSumScan,
    sort.finalizeCellRanges,
    sort.scatter,
  ];

  return {
    hashTableSize,
    hashTableSizePadded,
    cellSize,
    hashOriginUniform,
    cellSizeUniform,
    cellIndex,
    cellStart,
    cellEnd,
    sortedIndices,
    cellIndexAndHistogram,
    rebuildPipeline,
  };
}

/**
 * Build the 27-cell neighbor-walk inner loop, parameterized on:
 *   bucketFn       — which hash function to use for the 27 cell lookups
 *   readNeighbor   — how to fetch the neighbor's position (indirection
 *                    via sortedIndices, or direct from a permuted buffer)
 *   accumulate     — the per-candidate work shape (synthetic or Poly6)
 *
 * The bucket-dedup chain is included in every variant so all timed
 * walks pay the same compile-time-unrolled overhead.
 */
function buildWalkKernel(args: {
  readonly particles: ParticleSystem;
  readonly grid: ProbeHashGrid;
  readonly output: StorageBufferNode<'float'>;
  readonly bucketFn: BucketFn;
  readonly readNeighbor: (slot: Any) => Any;
  readonly accumulate: (accum: Any, neighborPos: Any, selfPos: Any) => void;
}): ComputeNode {
  const { particles, grid, output, bucketFn, readNeighbor, accumulate } = args;
  const bucketMask = grid.hashTableSize - 1;

  return Fn(() => {
    const i: Any = instanceIndex;
    const selfPos: Any = (particles.positions as Any).element(i).xyz;
    const accum: Any = float(0).toVar();

    const rel: Any = selfPos.sub(grid.hashOriginUniform as Any).div(grid.cellSizeUniform as Any);
    const qcx: Any = rel.x.floor().toInt().toVar();
    const qcy: Any = rel.y.floor().toInt().toVar();
    const qcz: Any = rel.z.floor().toInt().toVar();

    const previousBuckets: Any[] = [];
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ncx: Any = qcx.add(int(dx));
          const ncy: Any = qcy.add(int(dy));
          const ncz: Any = qcz.add(int(dz));
          const bucket: Any = bucketFn(ncx, ncy, ncz).bitAnd(uint(bucketMask)).toVar();

          let alreadySeen: Any = null;
          for (const prev of previousBuckets) {
            const eq: Any = bucket.equal(prev);
            alreadySeen = alreadySeen === null ? eq : alreadySeen.or(eq);
          }

          const walkBucket = (): void => {
            Loop(
              {
                start: (grid.cellStart as Any).element(bucket),
                end: (grid.cellEnd as Any).element(bucket),
                type: 'uint',
                condition: '<',
              },
              ({ i: slot }: { i: Any }) => {
                const neighborPos: Any = readNeighbor(slot);
                accumulate(accum, neighborPos, selfPos);
              },
            );
          };

          if (alreadySeen === null) {
            walkBucket();
          } else {
            If(alreadySeen.not(), walkBucket);
          }
          previousBuckets.push(bucket);
        }
      }
    }

    (output as Any).element(i).assign(accum);
  })().compute(particles.capacity);
}

function buildPermuteKernel(
  particles: ParticleSystem,
  sortedIndices: StorageBufferNode<'uint'>,
  permutedPositions: StorageBufferNode<'vec4'>,
): ComputeNode {
  return Fn(() => {
    const k: Any = instanceIndex;
    const p: Any = (sortedIndices as Any).element(k);
    (permutedPositions as Any).element(k).assign((particles.positions as Any).element(p));
  })().compute(particles.capacity);
}

interface CornerStats {
  readonly synth: Stats;
  readonly poly6: Stats;
}

interface ScaleResult {
  readonly count: number;
  readonly hashTableSize: number;
  readonly t_indir: CornerStats;
  readonly t_perm: CornerStats;
  readonly m_indir: CornerStats;
  readonly m_perm: CornerStats;
  readonly permute: Stats;
  readonly cellIndex_t: Stats;
  readonly cellIndex_m: Stats;
}

async function runProbeAtScale(perf: PerfRenderer, count: number): Promise<ScaleResult> {
  const particles = new ParticleSystem(perf.renderer, count, R);
  particles.uploadParticles(fluidColumn({ count, spacing: SPACING, origin: [-0.25, 0.5, -0.25] }));

  // Mirror the production hash-table sizing: 2·capacity, rounded to a
  // power of two. Same headroom characteristics as `HashGrid`.
  let hashTableSize = 1;
  while (hashTableSize < count * 2) hashTableSize <<= 1;

  const gridT = buildProbeHashGrid(particles, H, hashTableSize, teschnerBucket);
  const gridM = buildProbeHashGrid(particles, H, hashTableSize, mortonBucket);

  // Permuted-position mirrors — one per grid since each grid produces
  // a different `sortedIndices` ordering.
  const permPosT = instancedArray(count, 'vec4') as StorageBufferNode<'vec4'>;
  const permPosM = instancedArray(count, 'vec4') as StorageBufferNode<'vec4'>;

  // Output accumulators — one per timed walk so dispatches stay independent.
  const out = (): StorageBufferNode<'float'> =>
    instancedArray(count, 'float') as StorageBufferNode<'float'>;

  const synthAccum = (accum: Any, neighborPos: Any, _selfPos: Any): void => {
    accum.assign(accum.add(neighborPos.x));
  };
  const poly6Accum = (accum: Any, neighborPos: Any, selfPos: Any): void => {
    const diff: Any = neighborPos.sub(selfPos);
    const rsq: Any = diff.dot(diff);
    If(rsq.lessThan(float(HSQ)), () => {
      const k: Any = float(HSQ).sub(rsq);
      const w: Any = float(POLY6_COEF).mul(k).mul(k).mul(k);
      accum.assign(accum.add(w));
    });
  };

  const readIndir =
    (grid: ProbeHashGrid): ((slot: Any) => Any) =>
    (slot: Any) => {
      const neighborIdx: Any = (grid.sortedIndices as Any).element(slot);
      return (particles.positions as Any).element(neighborIdx).xyz;
    };
  const readPerm =
    (perm: StorageBufferNode<'vec4'>): ((slot: Any) => Any) =>
    (slot: Any) =>
      (perm as Any).element(slot).xyz;

  // Build all eight walk kernels. Naming: <bucket>_<read>_<work>.
  const walkT_indir_synth = buildWalkKernel({
    particles,
    grid: gridT,
    output: out(),
    bucketFn: teschnerBucket,
    readNeighbor: readIndir(gridT),
    accumulate: synthAccum,
  });
  const walkT_perm_synth = buildWalkKernel({
    particles,
    grid: gridT,
    output: out(),
    bucketFn: teschnerBucket,
    readNeighbor: readPerm(permPosT),
    accumulate: synthAccum,
  });
  const walkM_indir_synth = buildWalkKernel({
    particles,
    grid: gridM,
    output: out(),
    bucketFn: mortonBucket,
    readNeighbor: readIndir(gridM),
    accumulate: synthAccum,
  });
  const walkM_perm_synth = buildWalkKernel({
    particles,
    grid: gridM,
    output: out(),
    bucketFn: mortonBucket,
    readNeighbor: readPerm(permPosM),
    accumulate: synthAccum,
  });
  const walkT_indir_poly6 = buildWalkKernel({
    particles,
    grid: gridT,
    output: out(),
    bucketFn: teschnerBucket,
    readNeighbor: readIndir(gridT),
    accumulate: poly6Accum,
  });
  const walkT_perm_poly6 = buildWalkKernel({
    particles,
    grid: gridT,
    output: out(),
    bucketFn: teschnerBucket,
    readNeighbor: readPerm(permPosT),
    accumulate: poly6Accum,
  });
  const walkM_indir_poly6 = buildWalkKernel({
    particles,
    grid: gridM,
    output: out(),
    bucketFn: mortonBucket,
    readNeighbor: readIndir(gridM),
    accumulate: poly6Accum,
  });
  const walkM_perm_poly6 = buildWalkKernel({
    particles,
    grid: gridM,
    output: out(),
    bucketFn: mortonBucket,
    readNeighbor: readPerm(permPosM),
    accumulate: poly6Accum,
  });

  const permuteT = buildPermuteKernel(particles, gridT.sortedIndices, permPosT);
  const permuteM = buildPermuteKernel(particles, gridM.sortedIndices, permPosM);

  // Populate both grids and both permuted-position buffers.
  await perf.renderer.computeAsync([...gridT.rebuildPipeline, ...gridM.rebuildPipeline]);
  await perf.renderer.computeAsync([permuteT, permuteM]);
  await perf.stepChain([]); // drain timestamp pool

  async function timeKernel(kernel: ComputeNode): Promise<Stats> {
    for (let i = 0; i < WARMUP_FRAMES; i++) {
      await perf.runKernelInIsolation(kernel);
    }
    const samples: number[] = [];
    for (let i = 0; i < MEASURE_FRAMES; i++) {
      samples.push(await perf.runKernelInIsolation(kernel));
    }
    return summarize(samples);
  }

  const t_indir_synth = await timeKernel(walkT_indir_synth);
  const t_indir_poly6 = await timeKernel(walkT_indir_poly6);
  const t_perm_synth = await timeKernel(walkT_perm_synth);
  const t_perm_poly6 = await timeKernel(walkT_perm_poly6);
  const m_indir_synth = await timeKernel(walkM_indir_synth);
  const m_indir_poly6 = await timeKernel(walkM_indir_poly6);
  const m_perm_synth = await timeKernel(walkM_perm_synth);
  const m_perm_poly6 = await timeKernel(walkM_perm_poly6);
  const permuteStats = await timeKernel(permuteT);
  const cellIndex_t = await timeKernel(gridT.cellIndexAndHistogram);
  const cellIndex_m = await timeKernel(gridM.cellIndexAndHistogram);

  particles.destroy();

  return {
    count,
    hashTableSize,
    t_indir: { synth: t_indir_synth, poly6: t_indir_poly6 },
    t_perm: { synth: t_perm_synth, poly6: t_perm_poly6 },
    m_indir: { synth: m_indir_synth, poly6: m_indir_poly6 },
    m_perm: { synth: m_perm_synth, poly6: m_perm_poly6 },
    permute: permuteStats,
    cellIndex_t,
    cellIndex_m,
  };
}

function fmt(ms: number): string {
  if (ms < 0.01) return ms.toFixed(4);
  if (ms < 1) return ms.toFixed(3);
  return ms.toFixed(2);
}

function formatResultsAsMarkdown(results: readonly ScaleResult[]): string {
  const lines: string[] = [];
  lines.push('');
  lines.push('## Z-sort locality probe results — four-corner sweep');
  lines.push('');
  lines.push(
    'Baseline = Teschner bucketing + sortedIndices indirection (current production). Speedups shown are p50 ratios vs that baseline.',
  );
  lines.push('');

  const renderTable = (title: string, pick: (c: CornerStats) => Stats): void => {
    lines.push(`### ${title}`);
    lines.push('');
    lines.push('| Scene | T+indir (ms) | T+perm (ms) [×] | M+indir (ms) [×] | M+perm (ms) [×] |');
    lines.push('|-------|--------------|-----------------|-------------------|------------------|');
    for (const r of results) {
      const t_i = pick(r.t_indir).p50;
      const t_p = pick(r.t_perm).p50;
      const m_i = pick(r.m_indir).p50;
      const m_p = pick(r.m_perm).p50;
      lines.push(
        `| ${r.count.toLocaleString()} | ${fmt(t_i)} | ${fmt(t_p)} [${(t_i / t_p).toFixed(2)}×] | ${fmt(m_i)} [${(t_i / m_i).toFixed(2)}×] | ${fmt(m_p)} [${(t_i / m_p).toFixed(2)}×] |`,
      );
    }
    lines.push('');
  };

  renderTable('Synthetic walk (sum neighbor x)', (c) => c.synth);
  renderTable('Poly6-loaded walk (production-shape work)', (c) => c.poly6);

  // Per-frame overhead summary.
  lines.push('### Per-frame overhead (one dispatch per substep)');
  lines.push('');
  lines.push(
    '| Scene | permute (ms) | cellIndex Teschner (ms) | cellIndex Morton (ms) | Morton overhead vs T |',
  );
  lines.push(
    '|-------|--------------|--------------------------|------------------------|------------------------|',
  );
  for (const r of results) {
    const overhead = r.cellIndex_m.p50 - r.cellIndex_t.p50;
    lines.push(
      `| ${r.count.toLocaleString()} | ${fmt(r.permute.p50)} | ${fmt(r.cellIndex_t.p50)} | ${fmt(r.cellIndex_m.p50)} | +${fmt(overhead)} |`,
    );
  }
  lines.push('');

  // Verdict block based on the largest scene's Poly6 numbers.
  const big = results[results.length - 1]!;
  const baseline = big.t_indir.poly6.p50;
  const tPerm = baseline / big.t_perm.poly6.p50;
  const mIndir = baseline / big.m_indir.poly6.p50;
  const mPerm = baseline / big.m_perm.poly6.p50;

  lines.push('### Verdict (largest scene, Poly6-loaded)');
  lines.push('');
  lines.push(`- T+perm  speedup: **${tPerm.toFixed(2)}×** (Improvement 1 alone)`);
  lines.push(`- M+indir speedup: **${mIndir.toFixed(2)}×** (Improvement 2 alone)`);
  lines.push(`- M+perm  speedup: **${mPerm.toFixed(2)}×** (full Z-sort: 1 + 2)`);
  lines.push('');

  // Reasoning hints based on the relative sizes of the three speedups.
  const dominant =
    mPerm > Math.max(tPerm, mIndir) * 1.15
      ? 'combined'
      : tPerm > mIndir * 1.15
        ? 'permutation'
        : mIndir > tPerm * 1.15
          ? 'morton'
          : 'roughly equal';
  if (dominant === 'combined') {
    lines.push(
      '**Combined dominates.** Permutation and Morton bucketing each contribute meaningfully and stack. The plan should land both in one phase to capture the full win.',
    );
  } else if (dominant === 'morton') {
    lines.push(
      '**Morton dominates.** Across-bucket locality (Improvement 2) carries most of the speedup; within-bucket permutation alone is incremental. Reshape the plan to lead with Morton bucketing.',
    );
  } else if (dominant === 'permutation') {
    lines.push(
      '**Permutation dominates.** Within-bucket locality (Improvement 1) carries most of the speedup; Morton bucketing adds little on top. The plan as drafted is justified; defer Morton.',
    );
  } else {
    lines.push(
      '**Roughly equal contributions.** Both improvements help by similar amounts. Either land them together or pick whichever has the smaller engineering surface; the order matters less than expected.',
    );
  }
  lines.push('');
  lines.push(`(Window: ${WARMUP_FRAMES} warmup + ${MEASURE_FRAMES} measure frames per kernel.)`);
  lines.push('');
  return lines.join('\n');
}

describe('Phase Perf — Z-sort locality probe (four corners)', () => {
  it('measures Teschner/Morton × indirection/permuted at 10k / 50k / 100k', async () => {
    const perf = await PerfRenderer.create();

    if (perf.timingMethod !== 'per-kernel-pass') {
      perf.dispose();
      throw new Error(
        `Z-sort locality probe requires per-kernel-pass timing. PerfRenderer reported timingMethod="${perf.timingMethod}".`,
      );
    }

    const results: ScaleResult[] = [];
    for (const count of [10_000, 50_000, 100_000]) {
      results.push(await runProbeAtScale(perf, count));
    }
    perf.dispose();

    const md = formatResultsAsMarkdown(results);
    // eslint-disable-next-line no-console
    console.log(md);

    for (const r of results) {
      expect(r.t_indir.poly6.p50).toBeGreaterThan(0);
      expect(r.t_perm.poly6.p50).toBeGreaterThan(0);
      expect(r.m_indir.poly6.p50).toBeGreaterThan(0);
      expect(r.m_perm.poly6.p50).toBeGreaterThan(0);
      expect(r.permute.p50).toBeGreaterThan(0);
      expect(r.cellIndex_t.p50).toBeGreaterThan(0);
      expect(r.cellIndex_m.p50).toBeGreaterThan(0);
    }
  }, 900_000);
});
