import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute, Vector3 } from 'three';

import {
  ClothSystem,
  ParticleSystem,
  SimLoop,
  createClothGraph,
  createParticleRenderer,
} from '../../../src/index.js';

// Tethers (long-range attachments, Kim et al. 2012) prevent stretch.
//
// A 32×32 sheet pinned along its top edge under 100× gravity. With
// tethers active, max edge stretch stays < 5 % of rest length. Without
// tethers (distance constraints alone) it exceeds 20 % at this load,
// which documents the value of tethers.
//
// Why this matters: without tethers, stretch corrections propagate
// through the graph-colored distance solve only a few edges per
// iteration; a sheet hung from its two top corners shows ~13 % stretch
// at the corner-pin edges even at 1×g.
// Tethers propagate the entire stretch correction in a single pass per
// iteration (Kim et al. 2012, Algorithm 1: "Enforcing this simple
// constraint allows tensile pressure waves to propagate immediately from
// the source (attachment) to all the free particles in a single step").
// At 100×g this gap is dramatic: tethers cap the column stretch well
// under 5 %; without them the cloth either over-stretches or needs an
// order of magnitude more substeps to settle.
//
// Both cases run at the same S, I budget so the comparison is
// apples-to-apples.

interface SheetData {
  readonly geometry: BufferGeometry;
  readonly pinnedIndices: number[];
}

function buildTopPinnedSheet(M: number): SheetData {
  const positions: number[] = [];
  const indices: number[] = [];
  const pinnedIndices: number[] = [];
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < M; i++) {
      const u = i / (M - 1);
      const v = j / (M - 1);
      positions.push(u, -v, 0);
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
  // Pin the entire TOP row (j = 0). A single connected island (Kim
  // et al. 2012 §3.4): every top-row pinned vertex is graph-connected
  // to its left/right pinned neighbour through one cloth edge, so every
  // free particle ends up with exactly one tether.
  for (let i = 0; i < M; i++) pinnedIndices.push(i);

  const geom = new BufferGeometry();
  geom.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geom.setIndex(new Uint32BufferAttribute(new Uint32Array(indices), 1));
  return { geometry: geom, pinnedIndices };
}

async function runStretchScene(args: {
  readonly M: number;
  readonly tetherCompliance: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly frames: number;
  readonly gravity: number;
}): Promise<{
  readonly maxStretch: number;
  readonly nanFree: boolean;
  readonly nTethers: number;
}> {
  const renderer = await createParticleRenderer();
  try {
    const { geometry, pinnedIndices } = buildTopPinnedSheet(args.M);
    const graph = createClothGraph(geometry, {
      surfaceDensity: 0.2,
      pinnedIndices,
    });

    const particles = new ParticleSystem(renderer, graph.positions.length, 0.05);
    const cloth = new ClothSystem(particles, {
      graph,
      stretchCompliance: 1e-7,
      bendCompliance: 1.0, // effectively no bend stiffness — tether vs. distance is the focus
      tetherCompliance: args.tetherCompliance,
    });

    const loop = new SimLoop(particles, {
      substeps: args.substeps,
      iterations: args.iterations,
      gravity: new Vector3(0, args.gravity, 0),
      materials: [cloth],
    });

    const frameDt = 1 / 60;
    const damp = 0.9;
    let nanFree = true;
    for (let f = 0; f < args.frames; f++) {
      await loop.step(frameDt);
      const buf = await renderer.getArrayBufferAsync(particles.velocities.value);
      const v = new Float32Array(buf);
      for (let k = 0; k < v.length; k++) v[k] = v[k]! * damp;
      (particles.velocities.value.array as Float32Array).set(v);
      particles.velocities.value.needsUpdate = true;
    }

    const snap = await particles.readback();
    let maxStretch = 0;
    for (let e = 0; e < graph.distancePairs.length; e++) {
      const [i, j] = graph.distancePairs[e]!;
      const rest = graph.distanceRestLengths[e]!;
      const dx = snap.positions[4 * i + 0]! - snap.positions[4 * j + 0]!;
      const dy = snap.positions[4 * i + 1]! - snap.positions[4 * j + 1]!;
      const dz = snap.positions[4 * i + 2]! - snap.positions[4 * j + 2]!;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!Number.isFinite(len)) {
        nanFree = false;
        break;
      }
      const stretch = (len - rest) / rest;
      const abs = Math.abs(stretch);
      if (abs > maxStretch) maxStretch = abs;
    }

    loop.dispose();
    particles.dispose();
    return { maxStretch, nanFree, nTethers: cloth.tethers.length };
  } finally {
    renderer.dispose();
  }
}

describe('cloth tethers prevent stretch (Kim et al. 2012)', () => {
  it('32×32 sheet at 100× gravity — with tethers max stretch < 5%; without tethers exceeds 20%', async () => {
    const M = 32;
    const heavyG = -981; // 100× standard g
    const S = 8;
    const I = 4;
    const frames = 120;

    // With tethers (strict α = 0).
    const withTethers = await runStretchScene({
      M,
      tetherCompliance: 0,
      substeps: S,
      iterations: I,
      frames,
      gravity: heavyG,
    });
    console.info(
      `[tether-stretch with-tethers] M=${M} S=${S} I=${I} g=${heavyG} maxStretch=${(withTethers.maxStretch * 100).toFixed(2)}% NaN-free=${withTethers.nanFree} nTethers=${withTethers.nTethers}`,
    );
    expect(withTethers.nanFree).toBe(true);
    // Target: < 5 %.
    expect(withTethers.maxStretch).toBeLessThan(0.05);
    // Sanity: tethers were actually built (top row M=32 pinned →
    // M*(M-1) = 992 free particles → 992 tethers, one each).
    expect(withTethers.nTethers).toBe(M * (M - 1));

    // Without tethers (effectively disabled via large α — XPBD
    // compliance makes the tether constraint produce vanishing Δλ).
    // Target: stretch > 20 %.
    const withoutTethers = await runStretchScene({
      M,
      tetherCompliance: 1e10,
      substeps: S,
      iterations: I,
      frames,
      gravity: heavyG,
    });
    console.info(
      `[tether-stretch without-tethers] M=${M} S=${S} I=${I} g=${heavyG} maxStretch=${(withoutTethers.maxStretch * 100).toFixed(2)}% NaN-free=${withoutTethers.nanFree}`,
    );
    // The without-tethers run may saturate to NaN if the stretch
    // becomes catastrophic; we accept either explosion (NaN) or
    // measurable over-stretch (> 20%) as evidence that tethers
    // are the load-bearing constraint at 100×g. The point of the
    // check is "tethers matter", not "the unconstrained version
    // is well-behaved".
    const exceededLimit = !withoutTethers.nanFree || withoutTethers.maxStretch > 0.2;
    expect(exceededLimit).toBe(true);
  }, 240_000);
});
