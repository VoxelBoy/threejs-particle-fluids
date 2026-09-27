import { describe, expect, it } from 'vitest';
import { bakeMeshToSdf, sampleSdfGradient } from '../../../src/index.js';
import { makeLcg, makeUvSphere } from '../../_helpers/sdf-test-meshes.js';

// Gradient sanity: across 1000 random points outside a 1-voxel boundary
// layer, `|∇φ| ∈ [0.8, 1.2]` (≈ 1 everywhere for a signed-distance
// function; the deviation is the sampling-artifact floor).
//
// Central-difference `∇φ` samples the SDF at `x ± voxelSize` along each
// axis; the gradient magnitude tells us how well the field satisfies the
// Eikonal property `|∇φ| = 1`. Deviation sources:
//   - Mesh facet normal vs ideal-sphere normal (small on a 32×32 UV sphere).
//   - Trilinear interpolant is piecewise linear — derivative is piecewise
//     constant, and at a voxel boundary the gradient can jump.
//   - The stencil halves a sliver of voxels near the medial axis where
//     multiple face contributions cancel.
// The [0.8, 1.2] window absorbs all three comfortably.

describe('SDF: gradient magnitude', () => {
  it('|∇φ| ∈ [0.8, 1.2] at 1000 random points outside the 1-voxel boundary layer', () => {
    const radius = 0.5;
    const resolution = 64;
    const padding = 0.1;
    const sdf = bakeMeshToSdf(makeUvSphere(radius, 32, 32), { resolution, padding });

    const [vx, vy, vz] = sdf.voxelSize;
    const [resX, resY, resZ] = sdf.resolution;
    // 2-voxel inset leaves the central-difference stencil (`x ± voxelSize`)
    // at least 1 voxel from the boundary on every axis.
    const inset = 2;
    const minX = sdf.origin[0] + inset * vx;
    const maxX = sdf.origin[0] + (resX - inset) * vx;
    const minY = sdf.origin[1] + inset * vy;
    const maxY = sdf.origin[1] + (resY - inset) * vy;
    const minZ = sdf.origin[2] + inset * vz;
    const maxZ = sdf.origin[2] + (resZ - inset) * vz;

    const rand = makeLcg(0xdada);
    let minMag = Infinity;
    let maxMag = -Infinity;
    let mean = 0;
    const sampleCount = 1000;

    for (let i = 0; i < sampleCount; i++) {
      const x = minX + (maxX - minX) * rand();
      const y = minY + (maxY - minY) * rand();
      const z = minZ + (maxZ - minZ) * rand();
      const g = sampleSdfGradient(sdf, x, y, z);
      const mag = Math.hypot(g[0], g[1], g[2]);
      if (mag < minMag) minMag = mag;
      if (mag > maxMag) maxMag = mag;
      mean += mag;
    }
    mean /= sampleCount;

    console.info(
      `[sdf-gradient-sanity] voxelSize=${vx.toFixed(5)} |∇φ| min=${minMag.toFixed(
        4,
      )} mean=${mean.toFixed(4)} max=${maxMag.toFixed(4)}`,
    );

    expect(minMag).toBeGreaterThan(0.8);
    expect(maxMag).toBeLessThan(1.2);
  }, 60_000);
});
