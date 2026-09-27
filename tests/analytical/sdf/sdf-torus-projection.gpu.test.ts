import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SDFCollider,
  SimLoop,
  bakeMeshToSdf,
  createParticleRenderer,
  sampleSdf,
  type ParticleInit,
} from '../../../src/index.js';
import { makeLcg, makeTorus } from '../../_helpers/sdf-test-meshes.js';

// Projection correctness: place 1000 particles inside a baked mesh. After
// one solve, 100% are outside the SDF (`φ(x) > 0`).
//
// The mesh is a procedural torus rather than a scanned model. A torus has
// a genuine non-convex interior (the tube), so the projection test
// exercises the inside-to-outside sign flip the same way an arbitrary
// mesh would — not a convex primitive reduction.
//
// Test shape:
//   1. Generate a watertight torus mesh.
//   2. Bake at 64³.
//   3. Enumerate voxel centres with `φ < -0.5·tubeRadius` (deep interior,
//      well away from the surface) — candidate seed positions.
//   4. Seed 1000 particles at randomly-picked candidates.
//   5. Run `SimLoop.step` with gravity disabled — single step, multiple
//      inner iterations — so the projection is isolated from integration.
//   6. Read positions back and verify every particle has sampled
//      `φ(x) > 0`.

describe('SDFCollider: projection out of a baked torus', () => {
  it('1000 particles seeded inside a baked torus are all outside after one solve', async () => {
    const ringRadius = 0.5;
    const tubeRadius = 0.2;
    const resolution = 64;
    const padding = 0.1;
    const N = 1000;

    const sdf = bakeMeshToSdf(makeTorus(ringRadius, tubeRadius, 64, 32), {
      resolution,
      padding,
    });

    // Enumerate deep-interior voxel centres: `φ < -0.5·tubeRadius` keeps
    // every seed point well inside the tube.
    const deepThreshold = -0.5 * tubeRadius;
    const interior: Array<[number, number, number]> = [];
    const [resX, resY, resZ] = sdf.resolution;
    const [vx, vy, vz] = sdf.voxelSize;
    const { origin, data } = sdf;
    for (let k = 0; k < resZ; k++) {
      const z = origin[2] + (k + 0.5) * vz;
      for (let j = 0; j < resY; j++) {
        const y = origin[1] + (j + 0.5) * vy;
        const row = k * resX * resY + j * resX;
        for (let i = 0; i < resX; i++) {
          if (data[row + i]! < deepThreshold) {
            interior.push([origin[0] + (i + 0.5) * vx, y, z]);
          }
        }
      }
    }
    expect(interior.length).toBeGreaterThan(N);

    // Deterministic pick of N distinct candidates, scattered around the
    // ring.
    const rand = makeLcg(0xfeed);
    const pickedIndices = new Set<number>();
    while (pickedIndices.size < N) {
      pickedIndices.add(Math.floor(rand() * interior.length));
    }

    const particleRadius = 0.05;
    const initial: ParticleInit[] = [];
    for (const idx of pickedIndices) {
      const p = interior[idx]!;
      initial.push({
        position: [p[0], p[1], p[2]],
        velocity: [0, 0, 0],
        invMass: 1,
      });
    }

    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, N, particleRadius);
      particles.uploadParticles(initial);

      const collider = new SDFCollider(particles, sdf, {
        muS: 0.5,
        muK: 0.4,
      });

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 4,
        colliders: [collider],
      });
      // Isolate projection from integration: no gravity.
      loop.gravity.set(0, 0, 0);

      await loop.step(1 / 60);

      const snap = await particles.readback();
      let worstPhi = Infinity;
      let worstIndex = -1;
      let insideCount = 0;
      for (let p = 0; p < N; p++) {
        const x = snap.positions[p * 4 + 0]!;
        const y = snap.positions[p * 4 + 1]!;
        const z = snap.positions[p * 4 + 2]!;
        const phi = sampleSdf(sdf, x, y, z);
        if (phi < worstPhi) {
          worstPhi = phi;
          worstIndex = p;
        }
        if (phi < 0) insideCount++;
      }

      console.info(
        `[sdf-torus-projection] N=${N} worstPhi=${worstPhi.toFixed(
          5,
        )} (particle ${worstIndex}) insideCount=${insideCount}/${N} seedCandidates=${interior.length}`,
      );

      // Every particle must be outside the surface — a hard "100% outside",
      // tolerance 0. Any penetration here would indicate the projection
      // either under-shot (bad gradient) or failed to fire (SDF sampling
      // returning wrong value).
      expect(insideCount).toBe(0);
      expect(worstPhi).toBeGreaterThanOrEqual(0);

      loop.dispose();
      collider.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
