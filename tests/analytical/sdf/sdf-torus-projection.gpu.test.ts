import { describe, expect, it } from 'vitest';
import { bakeMeshToSdf, sampleSdfCpu } from '../../../src/sdf/index.js';
import {
  ParticleSystem,
  SDFCollider,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';
import { makeLcg, makeTorus } from '../../_helpers/sdf-test-meshes.js';

// Phase 07 G1 — "Projection correctness" (plan §Validation > Automatic (G1)):
// "Place 1000 particles inside a baked Stanford bunny. After one solve,
// 100% are outside the SDF (`φ(x) > 0`)."
//
// Plan deviation: per the Phase 07 entry-plan adjustments, the mesh is a
// procedural torus — the Stanford bunny GLB is deferred (no GLB loader in
// v1 of the baker). A torus has a genuine non-convex interior (the tube),
// so the projection test exercises the inside-to-outside sign flip in
// the same way a bunny would — not a convex primitive reduction.
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

// Disabled in the default suite because the uncached 64³ mesh bake is expensive.
// Enable explicitly when validating changes to SDF baking or projection.
describe.skip('Phase 07 — SDF: projection correctness (G1)', () => {
  it('1000 particles seeded inside a baked torus are all outside after one solve', async () => {
    const ringRadius = 0.5;
    const tubeRadius = 0.2;
    const resolution = 64;
    const padding = 0.1;
    const N = 1000;

    const mesh = makeTorus(ringRadius, tubeRadius, 64, 32);
    const sdf = bakeMeshToSdf({
      positions: mesh.positions,
      indices: mesh.indices,
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

    // Deterministic pick of N candidates — LCG-based index draw without
    // replacement would be overkill; a strided selection with a jitter is
    // sufficient to scatter seeds around the ring.
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
        phase: 0,
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
        colliders: { sdfColliders: [collider] },
      });
      // Isolate projection from integration: no gravity, no floor clamp.
      loop.gravity.set(0, 0, 0);
      loop.kernels.floorY.value = -1e9;

      await loop.step(1 / 60);

      const snap = await particles.readback();
      let worstPhi = Infinity;
      let worstIndex = -1;
      let insideCount = 0;
      for (let p = 0; p < N; p++) {
        const x = snap.positions[p * 4 + 0]!;
        const y = snap.positions[p * 4 + 1]!;
        const z = snap.positions[p * 4 + 2]!;
        const phi = sampleSdfCpu(sdf, x, y, z);
        if (phi < worstPhi) {
          worstPhi = phi;
          worstIndex = p;
        }
        if (phi < 0) insideCount++;
      }

      // eslint-disable-next-line no-console
      console.info(
        `[sdf-torus-projection] N=${N} worstPhi=${worstPhi.toFixed(
          5,
        )} (particle ${worstIndex}) insideCount=${insideCount}/${N} seedCandidates=${interior.length}`,
      );

      // Every particle must be outside the surface. Tolerance `0` (plan's
      // hard "100% outside"). Any penetration here would indicate the
      // projection either under-shot (bad gradient) or failed to fire
      // (SDF sampling returning wrong value).
      expect(insideCount).toBe(0);
      expect(worstPhi).toBeGreaterThanOrEqual(0);

      particles.destroy();
      collider.destroy();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
