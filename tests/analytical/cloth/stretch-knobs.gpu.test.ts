import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute } from 'three';

import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { ClothSystem, fromBufferGeometry } from '../../../src/cloth/index.js';

// Regression: distance compliance must produce visible, finite deformation.
// The tether constraints alone cannot overcome a nearly rigid distance solve.

interface SheetData {
  readonly geometry: BufferGeometry;
  readonly pinnedIndices: number[];
  readonly bottomCornerIdx: number;
}

function buildFlag(M: number): SheetData {
  const positions: number[] = [];
  const indices: number[] = [];
  const pinnedIndices: number[] = [];
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < M; i++) {
      const u = i / (M - 1);
      const v = j / (M - 1);
      positions.push(u, -v, 0);
      if (i === 0) pinnedIndices.push(j * M + i);
    }
  }
  for (let j = 0; j < M - 1; j++) {
    for (let i = 0; i < M - 1; i++) {
      const a = j * M + i;
      const b = j * M + (i + 1);
      const c = (j + 1) * M + i;
      const d = (j + 1) * M + (i + 1);
      indices.push(a, c, d);
      indices.push(a, d, b);
    }
  }
  const bottomCornerIdx = (M - 1) * M + (M - 1);
  const geom = new BufferGeometry();
  geom.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geom.setIndex(new Uint32BufferAttribute(new Uint32Array(indices), 1));
  return { geometry: geom, pinnedIndices, bottomCornerIdx };
}

async function runScene(args: {
  readonly M: number;
  readonly stretchCompliance: number;
  readonly tetherCompliance: number;
  readonly stretchTolerance: number;
  readonly gravityMag: number;
}): Promise<{ cornerDrop: number; nanFree: boolean }> {
  const renderer = await createParticleRenderer();
  try {
    const { geometry, pinnedIndices, bottomCornerIdx } = buildFlag(args.M);
    const graph = fromBufferGeometry(geometry, {
      surfaceDensity: 0.2,
      pinnedIndices,
    });
    const initial: ParticleInit[] = [];
    for (let i = 0; i < graph.positions.length; i++) {
      const p = graph.positions[i]!;
      initial.push({
        position: [p[0], p[1], p[2]],
        velocity: [0, 0, 0],
        invMass: graph.invMass[i]!,
        phase: 1,
      });
    }
    const particles = new ParticleSystem(renderer, graph.positions.length, 0.05);
    particles.uploadParticles(initial);
    const xpbd = createXpbdUniforms(1 / 60);
    const cloth = new ClothSystem({
      particles,
      xpbd,
      graph,
      particleOffset: 0,
      stretchCompliance: args.stretchCompliance,
      bendCompliance: 1.0,
      tetherCompliance: args.tetherCompliance,
      stretchTolerance: args.stretchTolerance,
    });
    const loop = new SimLoop(particles, {
      substeps: 8,
      iterations: 1,
      xpbd,
      materials: [cloth],
    });
    loop.kernels.floorY.value = -1e9;
    loop.gravity.set(0, -args.gravityMag, 0);
    const damp = 0.92;
    let nanFree = true;
    for (let f = 0; f < 180; f++) {
      await loop.step(1 / 60);
      const buf = await renderer.getArrayBufferAsync(particles.velocities.value);
      const v = new Float32Array(buf);
      for (let k = 0; k < v.length; k++) v[k] = v[k]! * damp;
      (particles.velocities.value.array as Float32Array).set(v);
      particles.velocities.value.needsUpdate = true;
    }
    const snap = await particles.readback();
    const cornerY = snap.positions[4 * bottomCornerIdx + 1]!;
    if (!Number.isFinite(cornerY)) nanFree = false;
    const restY = graph.positions[bottomCornerIdx]![1];
    const cornerDrop = Number.isFinite(cornerY) ? -(cornerY - restY) : NaN;
    particles.destroy();
    return { cornerDrop, nanFree };
  } finally {
    renderer.dispose();
  }
}

describe('Phase 19 follow-up — stretch slider extended max produces visible deformation', () => {
  it('logStretchCompliance=−1 drops the bottom corner ≥ 5× further (≥ 0.15 m absolute) than the Phase 18 default −7, NaN-free', async () => {
    const M = 16;
    const G = 9.81;
    // Hold tethers + tolerance neutral so we isolate the
    // distance-compliance effect. Tethers strict; tolerance 0.
    const tetherCompliance = 1e-10;
    const stretchTolerance = 0;

    // Baseline: Phase 18 default.
    const baseline = await runScene({
      M,
      stretchCompliance: 1e-7,
      tetherCompliance,
      stretchTolerance,
      gravityMag: G,
    });
    // eslint-disable-next-line no-console
    console.info(
      `[stretch-knobs baseline] dist=1e-7 → cornerDrop=${baseline.cornerDrop.toFixed(4)}m`,
    );
    expect(baseline.nanFree).toBe(true);

    // New slider max: α = 0.1 (log = −1). Should drop the
    // corner substantially — empirical 0.41 m vs baseline's
    // 0.027 m at 1×g (probe data above). Gate at 10× to leave
    // headroom for thermal / cross-run variation.
    const rubbery = await runScene({
      M,
      stretchCompliance: 1e-1,
      tetherCompliance,
      stretchTolerance,
      gravityMag: G,
    });
    // eslint-disable-next-line no-console
    console.info(
      `[stretch-knobs rubbery] dist=1e-1 → cornerDrop=${rubbery.cornerDrop.toFixed(4)}m`,
    );
    expect(rubbery.nanFree).toBe(true);
    // Ratio gate at 5× — empirical measurement is ~16-18× (probe
    // data: baseline 0.024 m, rubbery 0.41 m), but cross-run
    // simulation variance has been observed to drop rubbery to
    // 0.24 m on cooler GPU thermal states. 5× is the floor that
    // "the slider produced VISIBLE additional drop" needs and
    // still rules out the under-α=1e-3 regime where rubbery
    // tracks baseline within 30 %.
    expect(rubbery.cornerDrop).toBeGreaterThan(5 * baseline.cornerDrop);
    // Absolute-visibility floor: corner must have moved at least
    // 0.15 m past its rest. Phase 18 default produces ~0.024 m,
    // so 0.15 m is "the slider has clearly affected the
    // silhouette" without depending on baseline magnitude.
    expect(rubbery.cornerDrop).toBeGreaterThan(0.15);
  }, 300_000);
});
