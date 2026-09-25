import { describe, expect, it } from 'vitest';
import { bakeMeshToSdf, sampleSdfCpu } from '../../../src/sdf/index.js';
import { makeLcg, makeUvSphere } from '../../_helpers/sdf-test-meshes.js';

// Phase 07 G1 — "Sphere-bake accuracy" (plan §Validation > Automatic (G1)):
// "Bake a sphere mesh of radius 0.5 at resolution 64³. At 100 random
// interior points (inside the padding), sampled SDF must match analytic
// `|x| - 0.5` within **0.5 × voxelSize**."
//
// "Interior" is interpreted per the plan's shape ("inside the padding")
// as: inside the voxel grid but not within one voxel of the boundary
// layer (trilinear interpolation's validity range). Points are sampled
// uniformly in a box that is 3 voxels smaller on every side than the
// grid, which guarantees the trilinear interpolant is well-defined.
//
// Error budget:
//   - trilinear interpolant error on a smooth SDF: `≤ 0.125 · voxelSize²/r`
//     (Taylor remainder of a linear fit against a locally-smooth f). At
//     64³ over ~1.4 m span → voxelSize ≈ 0.022 m → interpolant error
//     ≤ ~5e-4 m, well under the gate.
//   - UV-sphere facet deviation from the ideal sphere: `r · (1 − cos
//     (π/stacks))` ≈ 2.4e-3 m at stacks=32.
//   - Half-float quantization in storage: tested implicitly by the CPU
//     path here (`sampleSdfCpu` reads raw f32 data, no half-float), so
//     this test isolates baker correctness from GPU upload precision.
// Sum of known error sources ≪ 0.5 · voxelSize gate.

// Disabled in the default suite because the uncached 64³ mesh bake is expensive.
// Enable explicitly when validating changes to SDF baking or projection.
describe.skip('Phase 07 — SDF: sphere bake accuracy (G1)', () => {
  it('baked 64³ SDF of a radius-0.5 UV sphere matches analytic `|x| - 0.5` within 0.5·voxelSize', () => {
    const radius = 0.5;
    const resolution = 64;
    const padding = 0.1;
    const mesh = makeUvSphere(radius, 32, 32);
    const sdf = bakeMeshToSdf({
      positions: mesh.positions,
      indices: mesh.indices,
      resolution,
      padding,
    });

    const [vx, vy, vz] = sdf.voxelSize;
    const tolerance = 0.5 * Math.max(vx, vy, vz);

    // Sample inside a box 3 voxels inset from each face so the 7-sample
    // trilinear stencil stays well inside the grid.
    const inset = 3;
    const [resX, resY, resZ] = sdf.resolution;
    const minX = sdf.origin[0] + inset * vx;
    const maxX = sdf.origin[0] + (resX - inset) * vx;
    const minY = sdf.origin[1] + inset * vy;
    const maxY = sdf.origin[1] + (resY - inset) * vy;
    const minZ = sdf.origin[2] + inset * vz;
    const maxZ = sdf.origin[2] + (resZ - inset) * vz;

    const rand = makeLcg(0xbaab);
    let worstAbsError = 0;
    let worstAt: [number, number, number] = [0, 0, 0];
    let worstSampled = 0;
    let worstExpected = 0;

    const sampleCount = 100;
    for (let i = 0; i < sampleCount; i++) {
      const x = minX + (maxX - minX) * rand();
      const y = minY + (maxY - minY) * rand();
      const z = minZ + (maxZ - minZ) * rand();
      const sampled = sampleSdfCpu(sdf, x, y, z);
      const expected = Math.hypot(x, y, z) - radius;
      const err = Math.abs(sampled - expected);
      if (err > worstAbsError) {
        worstAbsError = err;
        worstAt = [x, y, z];
        worstSampled = sampled;
        worstExpected = expected;
      }
    }

    // eslint-disable-next-line no-console
    console.info(
      `[sdf-sphere-bake-accuracy] voxelSize=${vx.toFixed(
        5,
      )} tolerance=${tolerance.toFixed(5)} worstAbsError=${worstAbsError.toFixed(
        5,
      )} at=(${worstAt[0].toFixed(3)}, ${worstAt[1].toFixed(3)}, ${worstAt[2].toFixed(
        3,
      )}) sampled=${worstSampled.toFixed(5)} expected=${worstExpected.toFixed(5)}`,
    );

    expect(worstAbsError).toBeLessThan(tolerance);
  });
});
