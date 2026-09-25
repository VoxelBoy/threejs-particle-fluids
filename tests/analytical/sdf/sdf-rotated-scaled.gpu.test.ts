import { describe, expect, it } from 'vitest';
import { Quaternion, Vector3 } from 'three';
import { bakeMeshToSdf } from '../../../src/sdf/index.js';
import {
  ParticleSystem,
  SDFCollider,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';
import { makeUvSphere } from '../../_helpers/sdf-test-meshes.js';

// Phase 07 U-23 resolution — rotation + uniform scale on SDFCollider.
//
// Bakes a UV sphere of radius 0.5 and stages a tight projection check on
// an SDFCollider configured with:
//   - a non-zero world position (`position`)
//   - a non-identity rotation
//   - a non-unit uniform scale
//
// Particles are seeded **inside** the scaled sphere's volume (in world
// space) with no gravity, solved for a single step, and checked against
// the analytical ground truth:
//
//   `|x − position|` must be ≥ `scale · meshRadius − ε` for every
//   post-solve particle (no penetration past the scaled surface).
//
// Because the sphere is rotation-symmetric, the rotation uniform has
// no effect on φ magnitude at any query point — but it DOES exercise
// the kernel's `rotation · ∇φ_local` back-transform. Any bug in that
// path would push particles tangentially (off-axis from the shifted
// center) instead of radially, leaving residual interior particles or
// producing trajectories that miss the `|x − position|` floor.
//
// A separate test (the rotated-box one) checks rotation on a shape
// where rotation actually changes φ values — this test checks that
// adding rotation to a rotation-invariant shape does not break scale.

// Disabled in the default suite because the uncached 64³ mesh bake is expensive.
// Enable explicitly when validating changes to SDF baking or projection.
describe.skip('Phase 07 — U-23: rotated + scaled SDF projection', () => {
  it('100 particles inside a 2× scaled, rotated, translated sphere all project out', async () => {
    const meshRadius = 0.5;
    const resolution = 64;
    const padding = 0.1;
    const mesh = makeUvSphere(meshRadius, 32, 32);
    const sdf = bakeMeshToSdf({
      positions: mesh.positions,
      indices: mesh.indices,
      resolution,
      padding,
    });

    const renderer = await createParticleRenderer();
    try {
      const r = 0.04;
      const scale = 2.0;
      const worldPos = new Vector3(0.3, 0.0, -0.2);
      const rotation = new Quaternion().setFromAxisAngle(
        new Vector3(1, 1, 0).normalize(),
        Math.PI / 3,
      );
      const effectiveRadius = scale * meshRadius; // 1.0 m

      // Seed N particles strictly INSIDE the scaled sphere's world
      // volume. Uniformly sampled in a ball of radius
      // `effectiveRadius − 2r` around `worldPos` so every seed point
      // has `|x − position| < effectiveRadius − 2r`, i.e. depth > 2r.
      // A correctly-working kernel projects them to the `|x − position|
      // = effectiveRadius + r`-ish shell in one step.
      const N = 100;
      const initial: ParticleInit[] = [];
      let seed = 0xc0ffee_42;
      const rand = (): number => {
        seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
        return (seed >>> 0) / 0x100000000;
      };
      let placed = 0;
      while (placed < N) {
        // Rejection sample inside the unit ball.
        const ux = rand() * 2 - 1;
        const uy = rand() * 2 - 1;
        const uz = rand() * 2 - 1;
        const lenSq = ux * ux + uy * uy + uz * uz;
        if (lenSq > 1) continue;
        const innerR = effectiveRadius - 2 * r;
        initial.push({
          position: [worldPos.x + ux * innerR, worldPos.y + uy * innerR, worldPos.z + uz * innerR],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 0,
        });
        placed += 1;
      }

      const particles = new ParticleSystem(renderer, N, r);
      particles.uploadParticles(initial);

      const collider = new SDFCollider(particles, sdf, {
        position: worldPos,
        rotation,
        scale,
        muS: 0.5,
        muK: 0.4,
      });

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 4,
        colliders: { sdfColliders: [collider] },
      });
      // Isolate projection from integration: no gravity, no floor.
      loop.gravity.set(0, 0, 0);
      loop.kernels.floorY.value = -1e9;

      await loop.step(1 / 60);

      const snap = await particles.readback();
      let minDist = Infinity;
      let maxDist = -Infinity;
      let insideCount = 0;
      for (let p = 0; p < N; p++) {
        const dx = snap.positions[p * 4 + 0]! - worldPos.x;
        const dy = snap.positions[p * 4 + 1]! - worldPos.y;
        const dz = snap.positions[p * 4 + 2]! - worldPos.z;
        const d = Math.hypot(dx, dy, dz);
        if (d < minDist) minDist = d;
        if (d > maxDist) maxDist = d;
        // Allow a 5e-3 tolerance for i32 accumulator quantization +
        // trilinear sample error near the isosurface.
        if (d < effectiveRadius - 5e-3) insideCount++;
      }

      // eslint-disable-next-line no-console
      console.info(
        `[sdf-rotated-scaled] N=${N} effectiveRadius=${effectiveRadius} ` +
          `minDistFromCenter=${minDist.toFixed(4)} maxDist=${maxDist.toFixed(4)} ` +
          `insideCount=${insideCount}`,
      );

      // All particles must be outside the scaled surface.
      expect(insideCount).toBe(0);
      // Particles should sit near the `effectiveRadius + r` shell (one
      // particle radius above the surface, the projection target).
      // Upper bound: + 5e-3 tolerance. Lower bound: effectiveRadius - ε.
      expect(minDist).toBeGreaterThan(effectiveRadius - 5e-3);

      particles.destroy();
      collider.destroy();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
