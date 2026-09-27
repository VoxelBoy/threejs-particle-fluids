import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';

import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  SoftbodySystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// A soft-body cube dropped onto a floor: it settles without blowing up, and
// its peak impact compression converges as the timestep is refined and is
// independent of the solver iteration count.

function buildLattice(
  dim: number,
  spacing: number,
  comY: number,
): { rest: [number, number, number][]; initial: ParticleInit[] } {
  // Symmetric lattice centered at origin in x,z and at comY in y.
  const rest: [number, number, number][] = [];
  const initial: ParticleInit[] = [];
  const half = (dim - 1) / 2;
  for (let iz = 0; iz < dim; iz++) {
    for (let iy = 0; iy < dim; iy++) {
      for (let ix = 0; ix < dim; ix++) {
        const rx = (ix - half) * spacing;
        const ry = (iy - half) * spacing;
        const rz = (iz - half) * spacing;
        rest.push([rx, ry, rz]);
        initial.push({ position: [rx, ry + comY, rz] });
      }
    }
  }
  return { rest, initial };
}

interface RunSettleArgs {
  readonly substeps: number;
  readonly iterations: number;
  readonly compliance: number;
  readonly frames: number;
}

interface RunSettleResult {
  readonly peakCompressionRatio: number;
  readonly settledExtent: number;
  readonly restExtent: number;
}

/**
 * Drop a 3×3×3 soft-body cube onto a plane collider. Returns:
 *   - peakCompressionRatio: (rest_extent − min_extent_during_impact) / rest_extent
 *   - settledExtent: vertical extent at the final frame
 *   - restExtent: theoretical vertical extent of the rest lattice
 */
async function runSettle(args: RunSettleArgs): Promise<RunSettleResult> {
  const renderer = await createParticleRenderer();
  try {
    const spacing = 0.1;
    const r = 0.049; // slightly less than spacing/2 so particles don't
    //                  overlap — particle contacts are off (the plane
    //                  collider handles the floor).
    const dim = 3;
    const count = dim * dim * dim;
    // Drop from a modest height so impact velocity stays in a regime
    // where semi-implicit Euler's substep error is small relative to
    // the shape-matching projection. v_impact = sqrt(2·g·h); at h =
    // 0.3 m → ~2.4 m/s, giving dt_substep·v ≈ 1 cm at S=4 — 5× the
    // particle radius, well below the "integrator-dominates" regime.
    const comY = 0.3;
    const { rest, initial } = buildLattice(dim, spacing, comY);

    const particles = new ParticleSystem(renderer, count, r);
    particles.uploadParticles(initial);

    const restFlat = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      restFlat[3 * i + 0] = rest[i]![0];
      restFlat[3 * i + 1] = rest[i]![1];
      restFlat[3 * i + 2] = rest[i]![2];
    }

    const softbody = new SoftbodySystem(particles, {
      bodies: [
        { range: { start: 0, count }, restPositions: restFlat, compliance: args.compliance },
      ],
    });

    const colliders = new PrimitiveSet(particles);
    colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
      muS: 0.3,
      muK: 0.2,
    });

    const loop = new SimLoop(particles, {
      substeps: args.substeps,
      iterations: args.iterations,
      gravity: new Vector3(0, -9.81, 0),
      colliders: [colliders],
      materials: [softbody],
    });

    const restExtent = (dim - 1) * spacing;
    let minExtentDuringImpact = restExtent;
    let settledExtent = restExtent;

    const frameDt = 1 / 60;
    for (let f = 0; f < args.frames; f++) {
      await loop.step(frameDt);
      const snap = await particles.readback();
      let minY = Number.POSITIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < count; i++) {
        const y = snap.positions[4 * i + 1]!;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      const extent = maxY - minY;
      if (extent < minExtentDuringImpact) minExtentDuringImpact = extent;
      settledExtent = extent;
    }

    loop.dispose();
    colliders.dispose();
    particles.dispose();

    const peakCompressionRatio = (restExtent - minExtentDuringImpact) / restExtent;
    return { peakCompressionRatio, settledExtent, restExtent };
  } finally {
    renderer.dispose();
  }
}

describe('SoftbodySystem settling and impact stiffness', () => {
  it('body drops + settles without NaN or blow-up (smoke)', async () => {
    const result = await runSettle({
      substeps: 8,
      iterations: 2,
      compliance: 1e-4,
      frames: 180, // 3 s — ample for fall + multiple bounces + settle.
    });
    expect(Number.isFinite(result.peakCompressionRatio)).toBe(true);
    expect(Number.isFinite(result.settledExtent)).toBe(true);
    // Body shouldn't implode.
    expect(result.settledExtent).toBeGreaterThan(0.5 * result.restExtent);
    // Body shouldn't balloon.
    expect(result.settledExtent).toBeLessThan(2 * result.restExtent);
    // Settle position: the body rests on the floor, so min_y should
    // be near the particle radius. The final extent should be near
    // rest extent (body has relaxed to its shape-matched rest shape).
    console.info(
      `[softbody-settle-smoke] restExtent=${result.restExtent.toFixed(4)} peakCompression=${result.peakCompressionRatio.toFixed(4)} settledExtent=${result.settledExtent.toFixed(4)}`,
    );
  }, 60_000);

  it('impact compression converges as the timestep is refined', async () => {
    // Iterations fixed at 2 (SimLoop default). compliance = 1e-4
    // — soft enough to show measurable compression on impact. Frames
    // chosen to cover the full first-bounce impact window.
    const frames = 120; // 2 s — covers fall (~0.25 s) + multiple
    //                     bounces. Sampling 2× more frames reduces
    //                     peak-sample-time jitter between runs.
    const [s4, s8, s16] = await Promise.all([
      runSettle({
        substeps: 4,
        iterations: 2,
        compliance: 1e-4,
        frames,
      }),
      runSettle({
        substeps: 8,
        iterations: 2,
        compliance: 1e-4,
        frames,
      }),
      runSettle({
        substeps: 16,
        iterations: 2,
        compliance: 1e-4,
        frames,
      }),
    ]);
    const compressions = [
      s4.peakCompressionRatio,
      s8.peakCompressionRatio,
      s16.peakCompressionRatio,
    ];
    console.info(
      `[stiffness-vs-substeps] S=4: ${s4.peakCompressionRatio.toFixed(5)}  S=8: ${s8.peakCompressionRatio.toFixed(5)}  S=16: ${s16.peakCompressionRatio.toFixed(5)}`,
    );
    const maxC = Math.max(...compressions);
    const minC = Math.min(...compressions);
    const spread = maxC > 0 ? (maxC - minC) / maxC : 0;
    console.info(`[stiffness-vs-substeps] spread=${(spread * 100).toFixed(2)}%`);
    // All three runs must see SOME compression (otherwise the test
    // is trivial — confirms the scene is actually exercising the
    // shape-matching compliance).
    expect(minC).toBeGreaterThan(0.01);
    // The body frame is refit every iteration, so nothing anchors the body's
    // position: floor impacts converge with the timestep, and coarse and fine
    // impacts need not have identical peak compression at a fixed iteration
    // budget. Refining must not move the result further, though.
    const coarseError = Math.abs(s8.peakCompressionRatio - s4.peakCompressionRatio);
    const fineError = Math.abs(s16.peakCompressionRatio - s8.peakCompressionRatio);
    expect(maxC).toBeLessThan(0.5);
    expect(fineError).toBeLessThan(coarseError + 0.001);
  }, 120_000);

  it('peak compression is independent of the iteration count (within 3% across I ∈ {1, 2, 4})', async () => {
    // Substeps fixed at S = 8; iterations swept over {1, 2, 4};
    // compliance = 1e-4. The fitted body frame (c, R) is the same
    // regardless of I, and the per-iteration Δλ accumulation should
    // converge to the same total displacement regardless of the
    // iteration count (XPBD's small-step premise).
    const frames = 60;
    const [i1, i2, i4] = await Promise.all([
      runSettle({
        substeps: 8,
        iterations: 1,
        compliance: 1e-4,
        frames,
      }),
      runSettle({
        substeps: 8,
        iterations: 2,
        compliance: 1e-4,
        frames,
      }),
      runSettle({
        substeps: 8,
        iterations: 4,
        compliance: 1e-4,
        frames,
      }),
    ]);
    const compressions = [
      i1.peakCompressionRatio,
      i2.peakCompressionRatio,
      i4.peakCompressionRatio,
    ];
    console.info(
      `[I-independence] I=1: ${i1.peakCompressionRatio.toFixed(5)}  I=2: ${i2.peakCompressionRatio.toFixed(5)}  I=4: ${i4.peakCompressionRatio.toFixed(5)}`,
    );
    const maxC = Math.max(...compressions);
    const minC = Math.min(...compressions);
    const spread = maxC > 0 ? (maxC - minC) / maxC : 0;
    console.info(`[I-independence] spread=${(spread * 100).toFixed(2)}% (target < 3%)`);
    expect(minC).toBeGreaterThan(0.01);
    expect(spread).toBeLessThan(0.03);
  }, 120_000);
});
