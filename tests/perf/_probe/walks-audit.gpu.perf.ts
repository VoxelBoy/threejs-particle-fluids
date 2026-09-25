// Phase Perf paper-gap diagnosis — H2 per-iteration walk audit.
//
// Builds a 50k-particle PBF-only scene (no contact, no SDF, no surface
// tension, no XSPH, no vorticity) twice — once at S=4 I=1, once at
// S=4 I=2 — and measures density / lambda / positionDelta per-frame
// costs in each.
//
// Hypothesis: PBF Algorithm 1 builds neighbor lists once per substep
// (line 6) and reuses them across iterations (lines 8-19). Our impl
// rebuilds the hash grid once per substep but each density / lambda /
// positionDelta kernel re-walks the 27 neighbor cells per iteration.
// At I=2, that's 2x the walks of paper.
//
// Pass criterion: ratio of (sum of fluid kernel times at I=2) to
// (sum of fluid kernel times at I=1).
//   ratio ≈ 2.0 ⇒ per-iteration walks (Verlet skin-list optimization
//                 would help — confirms hypothesis).
//   ratio ≈ 1.0 ⇒ walks already amortized at substep level
//                 (hypothesis refuted).
//   ratio in between ⇒ partial amortization, partial per-iter cost.
//

import { Vector3 } from 'three';
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
import {
  PerfRunner,
  type PerfKernelSpec,
  type PerfQuantileBlock,
  type PerfSceneSpec,
} from '../_helpers/PerfRunner.js';
import { fluidColumn, hashGridKernelSpecs } from '../_scenes/_helpers.js';

const SENTINEL_BEGIN = '__PARTICLE_FLUIDS_PAPER_GAP_H2_BEGIN__';
const SENTINEL_END = '__PARTICLE_FLUIDS_PAPER_GAP_H2_END__';

const SPACING = 0.025;
const H = 0.04;
const R = SPACING * 0.5;
const REST_DENSITY = 1000;
const SUBSTEPS = 4;
const COUNT = 50_000;

interface KernelMeasurement {
  readonly name: string;
  readonly p50Ms: number;
  readonly dispatchesPerFrame: number;
  readonly perFrameP50Ms: number;
}

interface IterationConfigResult {
  readonly iterations: number;
  readonly frameTotalMs: PerfQuantileBlock;
  readonly frameStepMs: PerfQuantileBlock;
  readonly kernels: readonly KernelMeasurement[];
}

interface WalksAuditReport {
  readonly probeId: 'paper-gap-h2-walks';
  readonly capturedAtIso: string;
  readonly particleCount: number;
  readonly substeps: number;
  readonly results: readonly IterationConfigResult[];
  /**
   * Sum of (density + lambda + positionDelta) per-frame p50 cost,
   * per iteration count. The per-frame cost = p50 × dispatches/frame.
   */
  readonly walkKernelSumPerFrame: { readonly i1: number; readonly i2: number };
  /**
   * Ratio of walkKernelSumPerFrame at I=2 to I=1. Pass criterion:
   *   ≈ 2.0 ⇒ walks repeat per iter (hypothesis confirmed).
   *   ≈ 1.0 ⇒ walks amortized per substep (hypothesis refuted).
   */
  readonly walkAmortizationRatio: number;
}

async function buildSceneAtIterations(
  perf: PerfRenderer,
  iterations: number,
): Promise<{
  readonly spec: PerfSceneSpec;
  readonly dispose: () => void;
}> {
  const particles = new ParticleSystem(perf.renderer, COUNT, R);
  const initial = fluidColumn({
    count: COUNT,
    spacing: SPACING,
    origin: [-0.5, 0.5, -0.5],
  });
  particles.uploadParticles(initial);

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
    // Vorticity / XSPH OFF — pure PBF density-iteration regime so the
    // (I=1 vs I=2) comparison isolates per-iter walk cost without
    // post-advect kernels muddying the per-frame total.
  });

  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations,
    xpbd,
    hashGrid,
    colliders: { colliders },
    materials: [fluid],
  });
  loop.gravity.set(0, -9.81, 0);

  const dispatchesPerIter = SUBSTEPS * iterations;
  const kernels: PerfKernelSpec[] = [
    {
      name: 'core.predict',
      kernel: loop.kernels.predict,
      dispatchesPerFrame: SUBSTEPS,
    },
    ...hashGridKernelSpecs(hashGrid, SUBSTEPS),
    // Phase Perf-08: density fused into lambda. Per-iter dispatches drop
    // from 4 to 3.
    {
      name: 'fluid.lambda',
      kernel: fluid.perIterKernels[0]!,
      dispatchesPerFrame: dispatchesPerIter,
    },
    {
      name: 'fluid.positionDelta',
      kernel: fluid.perIterKernels[1]!,
      dispatchesPerFrame: dispatchesPerIter,
    },
    {
      name: 'fluid.applyDelta',
      kernel: fluid.perIterKernels[2]!,
      dispatchesPerFrame: dispatchesPerIter,
    },
    {
      name: 'core.advect',
      kernel: loop.kernels.advect,
      dispatchesPerFrame: SUBSTEPS,
    },
  ];

  const dt = 1 / 60;
  const spec: PerfSceneSpec = {
    id: 'fluid-only-50k-i' + iterations,
    particleCount: COUNT,
    substeps: SUBSTEPS,
    iterations,
    stepFrame: () => loop.step(dt),
    kernels,
  };

  return {
    spec,
    dispose: () => {
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}

async function captureWalksAudit(): Promise<WalksAuditReport> {
  const perf = await PerfRenderer.create();
  const runner = new PerfRunner(perf);
  // Smaller measure window than the full bench (which uses 50) — the
  // probe runs two configs and we want it to stay under ~2 minutes.
  const window = { warmup: 10, measure: 25 };

  const results: IterationConfigResult[] = [];
  try {
    for (const iterations of [1, 2]) {
      const built = await buildSceneAtIterations(perf, iterations);
      try {
        const result = await runner.runScene(built.spec, window);
        results.push({
          iterations,
          frameTotalMs: result.frameTotalMs,
          frameStepMs: result.frameStepMs,
          kernels: result.kernels.map((k) => ({
            name: k.name,
            p50Ms: k.p50Ms,
            dispatchesPerFrame: k.dispatchesPerFrame,
            perFrameP50Ms: k.p50Ms * k.dispatchesPerFrame,
          })),
        });
      } finally {
        built.dispose();
      }
    }
  } finally {
    perf.dispose();
  }

  const sumWalkKernels = (cfg: IterationConfigResult): number => {
    // Phase Perf-08: `fluid.density` was fused into `fluid.lambda`. The
    // walk-audit sum tracks the same three pair-walking dispatches as
    // before — `fluid.lambda` (now also doing density's Poly6) +
    // `fluid.positionDelta`. The walk-overhead H2 finding is unchanged
    // by the fusion: the sum measures total per-pair walk time across
    // the per-iter kernels.
    const targets = ['fluid.lambda', 'fluid.positionDelta'];
    return cfg.kernels
      .filter((k) => targets.includes(k.name))
      .reduce((sum, k) => sum + k.perFrameP50Ms, 0);
  };

  const i1 = results.find((r) => r.iterations === 1)!;
  const i2 = results.find((r) => r.iterations === 2)!;
  const sumI1 = sumWalkKernels(i1);
  const sumI2 = sumWalkKernels(i2);
  const ratio = sumI1 > 0 ? sumI2 / sumI1 : 0;

  return {
    probeId: 'paper-gap-h2-walks',
    capturedAtIso: new Date().toISOString(),
    particleCount: COUNT,
    substeps: SUBSTEPS,
    results,
    walkKernelSumPerFrame: { i1: sumI1, i2: sumI2 },
    walkAmortizationRatio: ratio,
  };
}

describe('Phase Perf paper-gap H2 — per-iteration walk audit', () => {
  it(
    'compares fluid kernel costs at I=1 vs I=2 on a 50k-particle PBF-only scene',
    async () => {
      const report = await captureWalksAudit();

      const line = JSON.stringify(report);
      // eslint-disable-next-line no-console
      console.log(SENTINEL_BEGIN);
      // eslint-disable-next-line no-console
      console.log(line);
      // eslint-disable-next-line no-console
      console.log(SENTINEL_END);

      expect(report.results.length).toBe(2);
      expect(report.walkKernelSumPerFrame.i1).toBeGreaterThan(0);
      expect(report.walkKernelSumPerFrame.i2).toBeGreaterThan(0);
    },
    20 * 60_000,
  );
});
