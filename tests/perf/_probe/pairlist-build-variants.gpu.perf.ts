// Phase Perf — pair-list build dedup-cost diagnostic.
//
// `buildPairListKernel` is the heaviest single kernel in surface-tension
// scenes (~16 ms p50 × 4 dispatches/frame on Apple M-series at 100k
// particles). The kernel walks 27 neighbor cells per fluid particle,
// each gated by a "have we already iterated this bucket?" check
// implemented as a JS-unrolled OR-chain that grows from 0 to 26
// equality comparisons across the 27-cell unroll (351 ops total).
//
// Hypothesis: the dedup chain is significant. If we drop or shrink it,
// the kernel gets meaningfully faster.
//
// This probe builds three variants of the pair-list build kernel, each
// using a custom local copy of `emitForEachNeighbor` with a different
// dedup strategy, and times them on the same 100k-particle settled
// fluid scene:
//
//   FULL     — current production: OR-chain across all previous buckets.
//   NONE     — no dedup at all. Will produce duplicate pairs when buckets
//              collide (correctness-broken; for measurement only).
//   ADJACENT — compare only against the immediately previous bucket.
//              Catches Morton-adjacent collisions; a 1-comparison check.
//
// Pass criteria interpretation:
//   FULL ≈ NONE ⇒ dedup chain is in the noise → optimize elsewhere
//                 (likely the random-access predictedPositions[j] reads).
//   FULL >> NONE ⇒ dedup chain is real → ship ADJACENT if also fast.
//   ADJACENT ≈ FULL ⇒ shorter dedup doesn't help → the iteration cost,
//                     not the comparison cost, dominates.

import { Vector3 } from 'three';
import { Fn, If, Loop, atomicStore, instanceIndex, instancedArray, int, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import { describe, expect, it } from 'vitest';

import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createXpbdUniforms,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';

import { PerfRenderer } from '../_helpers/PerfRenderer.js';
import { fluidColumn } from '../_scenes/_helpers.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const SENTINEL_BEGIN = '__PARTICLE_FLUIDS_PAIRLIST_VARIANTS_BEGIN__';
const SENTINEL_END = '__PARTICLE_FLUIDS_PAIRLIST_VARIANTS_END__';

const SPACING = 0.025;
const H = 0.04;
const R = SPACING * 0.5;
const REST_DENSITY = 1000;
const SUBSTEPS = 4;
const ITERATIONS = 2;
const COUNT = 100_000;
const MAX_NEIGHBORS = 64;

const WARMUP = 60;
const MEASURE = 50;

type DedupMode = 'full' | 'none' | 'adjacent';

/**
 * Local copy of the production Morton bucket function. Inlined here so
 * the probe variants can mix-and-match dedup strategies without
 * polluting the production `emitForEachNeighbor` API.
 */
function mortonBucketUnmasked(cx: Any, cy: Any, cz: Any): Any {
  // Same bit-spread as `src/core/hashGrid/mortonHash.ts`.
  // Keep this in sync if production ever changes.
  const spread = (v: Any): Any => {
    let x: Any = v.bitAnd(int(0x3ff));
    x = x.bitOr(x.shiftLeft(int(16))).bitAnd(int(0x030000ff));
    x = x.bitOr(x.shiftLeft(int(8))).bitAnd(int(0x0300f00f));
    x = x.bitOr(x.shiftLeft(int(4))).bitAnd(int(0x030c30c3));
    x = x.bitOr(x.shiftLeft(int(2))).bitAnd(int(0x09249249));
    return x.toUint();
  };
  const bx = spread(cx);
  const by = spread(cy).shiftLeft(uint(1));
  const bz = spread(cz).shiftLeft(uint(2));
  return bx.bitOr(by).bitOr(bz);
}

function buildVariantKernel(args: {
  readonly particles: ParticleSystem;
  readonly hashGrid: HashGrid;
  readonly hSq: UniformNode<'float', number>;
  readonly fluidCount: number;
  readonly pairList: StorageBufferNode<'uint'>;
  readonly pairCount: StorageBufferNode<'uint'>;
  readonly pairOverflowFlag: StorageBufferNode<'uint'>;
  readonly mode: DedupMode;
}): ComputeNode {
  const { particles, hashGrid, hSq, fluidCount, pairList, pairCount, pairOverflowFlag, mode } =
    args;
  const bucketMask = hashGrid.hashTableSize - 1;
  const maxN = MAX_NEIGHBORS;

  return Fn(() => {
    const localIdx: Any = (instanceIndex as Any).toVar();
    const xi: Any = particles.predictedPositions.element(localIdx).xyz.toVar();
    const count: Any = uint(0).toVar();

    If(localIdx.equal(uint(0)), () => {
      atomicStore(pairOverflowFlag.element(uint(0)) as Any, uint(0));
    });

    const rel: Any = xi.sub(hashGrid.hashOriginUniform as Any).div(hashGrid.cellSizeUniform as Any);
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
          const bucket: Any = mortonBucketUnmasked(ncx, ncy, ncz).bitAnd(uint(bucketMask)).toVar();

          // Build dedup predicate per `mode`.
          let alreadySeen: Any = null;
          if (mode === 'full') {
            for (const prev of previousBuckets) {
              const eq: Any = bucket.equal(prev);
              alreadySeen = alreadySeen === null ? eq : alreadySeen.or(eq);
            }
          } else if (mode === 'adjacent' && previousBuckets.length > 0) {
            alreadySeen = bucket.equal(previousBuckets[previousBuckets.length - 1]);
          }
          // mode === 'none' leaves alreadySeen === null → no gate.

          const walkBucket = (): void => {
            Loop(
              {
                start: hashGrid.cellStart.element(bucket),
                end: hashGrid.cellEnd.element(bucket),
                type: 'uint',
                condition: '<',
              },
              ({ i: k }: { i: Any }) => {
                const j: Any = hashGrid.sortedIndices.element(k);
                const xj: Any = particles.predictedPositions.element(j).xyz;
                const diff: Any = xi.sub(xj).toVar();
                const rSq: Any = diff.dot(diff).toVar();
                If(rSq.lessThan(hSq as Any), () => {
                  If(count.lessThan(uint(maxN)), () => {
                    const slotIdx: Any = localIdx.mul(uint(maxN)).add(count);
                    pairList.element(slotIdx).assign(j);
                    count.assign(count.add(uint(1)));
                  }).Else(() => {
                    atomicStore(pairOverflowFlag.element(uint(0)) as Any, uint(1));
                  });
                });
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

    pairCount.element(localIdx).assign(count);
  })().compute(fluidCount);
}

interface VariantResult {
  readonly mode: DedupMode;
  readonly p10Ms: number;
  readonly p50Ms: number;
  readonly p90Ms: number;
}

interface ProbeReport {
  readonly probeId: 'pairlist-build-variants';
  readonly capturedAtIso: string;
  readonly particleCount: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly warmup: number;
  readonly measure: number;
  readonly results: readonly VariantResult[];
  readonly noneVsFull: number;
  readonly adjacentVsFull: number;
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

describe('Phase Perf — pair-list build dedup variants', () => {
  it(
    'compares full / none / adjacent dedup at 100k particles after settling',
    async () => {
      const perf = await PerfRenderer.create();
      const particles = new ParticleSystem(perf.renderer, COUNT, R);
      particles.uploadParticles(
        fluidColumn({ count: COUNT, spacing: SPACING, origin: [-0.6, 0.5, -0.6] }),
      );

      const hashGrid = new HashGrid(particles, { cellSize: H });
      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
      colliders.upload();

      const xpbd = createXpbdUniforms(1 / 60);
      const fluid = new FluidSystem({
        particles,
        hashGrid,
        xpbd,
        restDensity: REST_DENSITY,
        h: H,
        particleSpacing: SPACING,
        compliance: 1e-4,
        fluidParticles: { start: 0, count: COUNT },
        vorticity: { strength: 0.1 },
        xsph: { c: 0.01 },
      });

      const loop = new SimLoop(particles, {
        substeps: SUBSTEPS,
        iterations: ITERATIONS,
        xpbd,
        hashGrid,
        colliders: { colliders },
        materials: [fluid],
      });
      loop.gravity.set(0, -9.81, 0);

      // Warmup — let the column settle so we measure the steady-state
      // (compressed) regime where pair-list build is dominant.
      const dt = 1 / 60;
      for (let i = 0; i < WARMUP; i++) await loop.step(dt);

      // Allocate a private pair-list trio for the probe variants to
      // write into. Doesn't disturb the FluidSystem's own pair list.
      const probePairList = instancedArray(COUNT * MAX_NEIGHBORS, 'uint');
      const probePairCount = instancedArray(COUNT, 'uint');
      const probeOverflow = (instancedArray(1, 'uint') as Any).toAtomic();

      const variants: DedupMode[] = ['full', 'none', 'adjacent'];
      const results: VariantResult[] = [];
      for (const mode of variants) {
        const kernel = buildVariantKernel({
          particles,
          hashGrid,
          hSq: fluid.sph.hSq,
          fluidCount: COUNT,
          pairList: probePairList,
          pairCount: probePairCount,
          pairOverflowFlag: probeOverflow as Any,
          mode,
        });

        const samples: number[] = [];
        // Warmup the kernel itself (pipeline build).
        for (let i = 0; i < 3; i++) {
          await perf.runKernelInIsolation(kernel);
        }
        for (let i = 0; i < MEASURE; i++) {
          const ms = await perf.runKernelInIsolation(kernel);
          samples.push(ms);
        }
        results.push({
          mode,
          p10Ms: quantile(samples, 0.1),
          p50Ms: quantile(samples, 0.5),
          p90Ms: quantile(samples, 0.9),
        });
      }

      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
      perf.dispose();

      const fullP50 = results.find((r) => r.mode === 'full')!.p50Ms;
      const noneP50 = results.find((r) => r.mode === 'none')!.p50Ms;
      const adjP50 = results.find((r) => r.mode === 'adjacent')!.p50Ms;

      const report: ProbeReport = {
        probeId: 'pairlist-build-variants',
        capturedAtIso: new Date().toISOString(),
        particleCount: COUNT,
        substeps: SUBSTEPS,
        iterations: ITERATIONS,
        warmup: WARMUP,
        measure: MEASURE,
        results,
        noneVsFull: noneP50 / fullP50,
        adjacentVsFull: adjP50 / fullP50,
      };

      // eslint-disable-next-line no-console
      console.log(SENTINEL_BEGIN);
      // eslint-disable-next-line no-console
      console.log(JSON.stringify(report));
      // eslint-disable-next-line no-console
      console.log(SENTINEL_END);

      // eslint-disable-next-line no-console
      console.log(
        '[pairlist-variants] full  p50=' +
          fullP50.toFixed(3) +
          ' ms\n' +
          '[pairlist-variants] none  p50=' +
          noneP50.toFixed(3) +
          ' ms (' +
          (report.noneVsFull * 100).toFixed(1) +
          '% of full)\n' +
          '[pairlist-variants] adj   p50=' +
          adjP50.toFixed(3) +
          ' ms (' +
          (report.adjacentVsFull * 100).toFixed(1) +
          '% of full)',
      );

      for (const r of results) expect(r.p50Ms).toBeGreaterThan(0);
    },
    10 * 60_000,
  );
});
