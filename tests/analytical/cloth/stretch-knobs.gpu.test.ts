import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute, Vector3 } from 'three';

import {
  ClothSystem,
  ParticleSystem,
  SimLoop,
  createClothGraph,
  createParticleRenderer,
} from '../../../src/index.js';

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
    const graph = createClothGraph(geometry, {
      surfaceDensity: 0.2,
      pinnedIndices,
    });
    const particles = new ParticleSystem(renderer, graph.positions.length, 0.05);
    const cloth = new ClothSystem(particles, {
      graph,
      stretchCompliance: args.stretchCompliance,
      bendCompliance: 1.0,
      tetherCompliance: args.tetherCompliance,
      stretchTolerance: args.stretchTolerance,
    });
    const loop = new SimLoop(particles, {
      substeps: 8,
      iterations: 1,
      gravity: new Vector3(0, -args.gravityMag, 0),
      materials: [cloth],
    });
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
    loop.dispose();
    particles.dispose();
    return { cornerDrop, nanFree };
  } finally {
    renderer.dispose();
  }
}

describe('cloth stretch compliance produces visible deformation', () => {
  it('stretchCompliance 1e-1 drops the bottom corner ≥ 5× further (and ≥ 0.15 m) than the default 1e-7, NaN-free', async () => {
    const M = 16;
    const G = 9.81;
    // Hold tethers + tolerance neutral so we isolate the
    // distance-compliance effect. Tethers strict; tolerance 0.
    const tetherCompliance = 1e-10;
    const stretchTolerance = 0;

    // Baseline: the default stretch compliance.
    const baseline = await runScene({
      M,
      stretchCompliance: 1e-7,
      tetherCompliance,
      stretchTolerance,
      gravityMag: G,
    });
    console.info(
      `[stretch-knobs baseline] dist=1e-7 → cornerDrop=${baseline.cornerDrop.toFixed(4)}m`,
    );
    expect(baseline.nanFree).toBe(true);

    // Very soft: α = 0.1. Should drop the corner substantially —
    // empirically 0.41 m vs the baseline's 0.027 m at 1×g.
    const rubbery = await runScene({
      M,
      stretchCompliance: 1e-1,
      tetherCompliance,
      stretchTolerance,
      gravityMag: G,
    });
    console.info(
      `[stretch-knobs rubbery] dist=1e-1 → cornerDrop=${rubbery.cornerDrop.toFixed(4)}m`,
    );
    expect(rubbery.nanFree).toBe(true);
    // Ratio check at 5× — measured ~16-18× (baseline 0.024 m,
    // rubbery 0.41 m), but cross-run variance has been observed to
    // drop rubbery to 0.24 m. 5× is the floor for "the compliance
    // produced VISIBLE additional drop" and still rules out the
    // under-α=1e-3 regime where rubbery tracks baseline within 30 %.
    expect(rubbery.cornerDrop).toBeGreaterThan(5 * baseline.cornerDrop);
    // Absolute-visibility floor: corner must have moved at least
    // 0.15 m past its rest. The default compliance produces ~0.024 m,
    // so 0.15 m means "the compliance has clearly changed the
    // silhouette" without depending on the baseline's magnitude.
    expect(rubbery.cornerDrop).toBeGreaterThan(0.15);
  }, 300_000);
});
