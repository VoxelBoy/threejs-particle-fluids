import { describe, expect, it } from 'vitest';
import {
  bakeMeshToSdf,
  decodeSdfBinary,
  encodeSdfBinary,
  sampleSdf,
  type SDFData,
} from '../../../src/index.js';
import { makeUvSphere } from '../../_helpers/sdf-test-meshes.js';

function field(resolution: [number, number, number]): SDFData {
  return {
    data: new Float32Array(resolution[0] * resolution[1] * resolution[2]),
    resolution,
    origin: [0, 0, 0],
    voxelSize: [1, 1, 1],
  };
}

describe('SDF: input validation and defaults', () => {
  it('sampleSdf rejects resolutions below 2', () => {
    expect(() => sampleSdf(field([1, 4, 4]), 0, 0, 0)).toThrow(
      'sampleSdf: resolution must be integers ≥ 2',
    );
    expect(sampleSdf(field([2, 2, 2]), 0, 0, 0)).toBe(0);
  });

  it('decodeSdfBinary rejects resolutions below 2 and non-positive voxel sizes', () => {
    const buffer = encodeSdfBinary(field([2, 2, 2]));
    new DataView(buffer).setUint32(8, 1, true);
    expect(() => decodeSdfBinary(buffer)).toThrow('decodeSdfBinary: resolution must be ≥ 2');
    const zeroVoxel = encodeSdfBinary({ ...field([2, 2, 2]), voxelSize: [1, 0, 1] });
    expect(() => decodeSdfBinary(zeroVoxel)).toThrow('decodeSdfBinary: voxelSize must be positive');
  });

  it('bakeMeshToSdf pads by two voxels by default', () => {
    const radius = 0.5;
    const resolution = 16;
    const sdf = bakeMeshToSdf(makeUvSphere(radius, 16, 16), { resolution });
    const voxel = sdf.voxelSize[0];
    // Bounds span 2·radius; two voxels of padding on each side fill the rest.
    expect(voxel * resolution).toBeCloseTo(2 * radius + 4 * voxel, 6);
    // Every edge voxel reads well outside the surface.
    let edgeMin = Infinity;
    for (let z = 0; z < resolution; z++)
      for (let y = 0; y < resolution; y++)
        for (let x = 0; x < resolution; x++) {
          const edge = [x, y, z].some((i) => i === 0 || i === resolution - 1);
          if (edge) edgeMin = Math.min(edgeMin, sdf.data[x + resolution * (y + resolution * z)]!);
        }
    expect(edgeMin).toBeGreaterThan(1.4 * voxel);
  });

  it('bakes a sphere whose meridians lie on the grid diagonals', () => {
    // Axis rays from voxel centers on the diagonal used to run along the
    // sphere's 45° meridian edges and count both neighbouring triangles.
    expect(() =>
      bakeMeshToSdf(makeUvSphere(0.5, 32, 32), { resolution: 32, padding: 0 }),
    ).not.toThrow();
  });
});
