import { describe, expect, it } from 'vitest';

import {
  decodeVoxelizeBinary,
  encodeVoxelizeBinary,
  voxelize,
  type TriangleMesh,
} from '../../../src/softbody/index.js';

// Phase 10 G1 — voxelizer surface classification.
//
// Plan test: voxelize a solid unit cube at particleRadius = 0.1.
// Expected outcome: exactly the outermost layer of voxels is flagged
// surface (interior voxels have all six face-axis neighbors occupied
// and so are not on the boundary); ratio must match the 6-neighbor
// analytic prediction within ±2 particles (allowing for axis-aligned-
// boundary voxels whose face-neighbor offsets leave the grid).
//
// Analytic prediction for a unit cube [-0.5, 0.5]³ voxelized at
// spacing = 2·particleRadius = 0.2:
//   Voxels per side: N = ceil(1.0 / 0.2) = 5
//   Total occupied:  5³ = 125
//   Interior (all 6 neighbors occupied): (N − 2)³ = 3³ = 27
//   Surface: 125 − 27 = 98

function unitCubeMesh(): TriangleMesh {
  // 8 corners at ±0.5, 12 triangles (2 per face × 6 faces).
  // eslint-disable-next-line prettier/prettier
  const vertices = new Float32Array([
    -0.5,
    -0.5,
    -0.5, // 0
    0.5,
    -0.5,
    -0.5, // 1
    -0.5,
    0.5,
    -0.5, // 2
    0.5,
    0.5,
    -0.5, // 3
    -0.5,
    -0.5,
    0.5, // 4
    0.5,
    -0.5,
    0.5, // 5
    -0.5,
    0.5,
    0.5, // 6
    0.5,
    0.5,
    0.5, // 7
  ]);
  // Outward-normal winding on every face.
  // eslint-disable-next-line prettier/prettier
  const indices = new Uint32Array([
    // -z face
    0, 2, 1, 1, 2, 3,
    // +z face
    4, 5, 6, 5, 7, 6,
    // -y face
    0, 1, 4, 1, 5, 4,
    // +y face
    2, 6, 3, 3, 6, 7,
    // -x face
    0, 4, 2, 2, 4, 6,
    // +x face
    1, 3, 5, 3, 7, 5,
  ]);
  return { vertices, indices };
}

describe('Phase 10 — voxelize (unit cube G1)', () => {
  it('solid unit cube at particleRadius = 0.1 produces 125 voxels, 98 surface, 27 interior', async () => {
    const mesh = unitCubeMesh();
    const result = voxelize(mesh, { particleRadius: 0.1 });

    // Total count: exactly 5³ = 125.
    expect(Math.abs(result.count - 125)).toBeLessThanOrEqual(2);
    // Surface count: 98 ± 2 per plan.
    expect(Math.abs(result.surfaceCount - 98)).toBeLessThanOrEqual(2);
    // Interior count: count − surfaceCount = 27 ± 2.
    const interiorCount = result.count - result.surfaceCount;
    expect(Math.abs(interiorCount - 27)).toBeLessThanOrEqual(2);
  });

  it('surfaceFlag matches the [surface..., interior...] ordering', () => {
    const mesh = unitCubeMesh();
    const result = voxelize(mesh, { particleRadius: 0.1 });
    // First `surfaceCount` entries flagged 1, rest flagged 0.
    for (let i = 0; i < result.surfaceCount; i++) {
      expect(result.surfaceFlag[i]).toBe(1);
    }
    for (let i = result.surfaceCount; i < result.count; i++) {
      expect(result.surfaceFlag[i]).toBe(0);
    }
  });

  it('interior voxels are strictly inside the bounding box', () => {
    // Interior voxel centres must lie strictly inside [-0.5+0.2, 0.5−0.2]
    // = [-0.3, 0.3] along every axis (one spacing inward from the
    // bounding box so every 6-neighbor cell fits in the grid + is
    // occupied).
    const mesh = unitCubeMesh();
    const result = voxelize(mesh, { particleRadius: 0.1 });
    const interiorStart = 3 * result.surfaceCount;
    const interiorCount = result.count - result.surfaceCount;
    for (let i = 0; i < interiorCount; i++) {
      const x = result.positions[interiorStart + 3 * i + 0]!;
      const y = result.positions[interiorStart + 3 * i + 1]!;
      const z = result.positions[interiorStart + 3 * i + 2]!;
      // Interior voxel centres should be at least spacing − ε inside
      // the outer bbox. For a 5×5×5 grid with spacing 0.2, interior
      // voxels are at ±0.2 and origin on each axis.
      expect(Math.abs(x)).toBeLessThan(0.3);
      expect(Math.abs(y)).toBeLessThan(0.3);
      expect(Math.abs(z)).toBeLessThan(0.3);
    }
  });

  it('surface voxels include all four corner slabs and stay within the cube bbox', () => {
    const mesh = unitCubeMesh();
    const result = voxelize(mesh, { particleRadius: 0.1 });
    let cornerCount = 0;
    for (let i = 0; i < result.surfaceCount; i++) {
      const x = result.positions[3 * i + 0]!;
      const y = result.positions[3 * i + 1]!;
      const z = result.positions[3 * i + 2]!;
      // All surface positions inside the bounding box with particle-
      // radius slop.
      expect(x).toBeGreaterThanOrEqual(-0.5);
      expect(x).toBeLessThanOrEqual(0.5);
      expect(y).toBeGreaterThanOrEqual(-0.5);
      expect(y).toBeLessThanOrEqual(0.5);
      expect(z).toBeGreaterThanOrEqual(-0.5);
      expect(z).toBeLessThanOrEqual(0.5);
      // Count the 8 corner voxels (centres at ±0.4 on every axis —
      // outermost voxels in each dimension).
      if (
        Math.abs(Math.abs(x) - 0.4) < 1e-5 &&
        Math.abs(Math.abs(y) - 0.4) < 1e-5 &&
        Math.abs(Math.abs(z) - 0.4) < 1e-5
      ) {
        cornerCount++;
      }
    }
    expect(cornerCount).toBe(8);
  });

  it('accepts a non-cubic rank-3 mesh (rectangular slab)', () => {
    // 1 × 2 × 3 slab at origin. Should produce a rank-3 cloud and not
    // throw. Grid: ceil(1/0.2) × ceil(2/0.2) × ceil(3/0.2) = 5 × 10 × 15
    // = 750 voxels (all occupied for a filled cuboid).
    // eslint-disable-next-line prettier/prettier
    const vertices = new Float32Array([
      -0.5, -1.0, -1.5, 0.5, -1.0, -1.5, -0.5, 1.0, -1.5, 0.5, 1.0, -1.5, -0.5, -1.0, 1.5, 0.5,
      -1.0, 1.5, -0.5, 1.0, 1.5, 0.5, 1.0, 1.5,
    ]);
    // eslint-disable-next-line prettier/prettier
    const indices = new Uint32Array([
      0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1,
      3, 5, 3, 7, 5,
    ]);
    const result = voxelize({ vertices, indices }, { particleRadius: 0.1 });
    expect(result.count).toBeGreaterThan(0);
    expect(result.surfaceCount).toBeGreaterThan(0);
    expect(result.surfaceCount).toBeLessThan(result.count);
  });

  it('rejects invalid particleRadius', () => {
    const mesh = unitCubeMesh();
    expect(() => voxelize(mesh, { particleRadius: 0 })).toThrow(/particleRadius/);
    expect(() => voxelize(mesh, { particleRadius: -0.1 })).toThrow(/particleRadius/);
    expect(() => voxelize(mesh, { particleRadius: NaN })).toThrow(/particleRadius/);
  });

  it('rejects malformed indices / vertices', () => {
    expect(() =>
      voxelize(
        {
          vertices: new Float32Array([0, 0]), // not a multiple of 3
          indices: new Uint32Array(),
        },
        { particleRadius: 0.1 },
      ),
    ).toThrow(/vertices/);
    expect(() =>
      voxelize(
        {
          vertices: new Float32Array([0, 0, 0]),
          indices: new Uint32Array([0, 1]), // not a multiple of 3
        },
        { particleRadius: 0.1 },
      ),
    ).toThrow(/indices/);
  });
});

describe('Phase 10 — voxelize binary encode/decode', () => {
  it('round-trips a voxelized cube through encode → decode', () => {
    const mesh = unitCubeMesh();
    const original = voxelize(mesh, { particleRadius: 0.1 });
    const buf = encodeVoxelizeBinary(original);
    const decoded = decodeVoxelizeBinary(buf);

    expect(decoded.count).toBe(original.count);
    expect(decoded.surfaceCount).toBe(original.surfaceCount);
    for (let i = 0; i < original.positions.length; i++) {
      expect(decoded.positions[i]).toBe(original.positions[i]);
    }
    for (let i = 0; i < original.surfaceFlag.length; i++) {
      expect(decoded.surfaceFlag[i]).toBe(original.surfaceFlag[i]);
    }
  });

  it('rejects a truncated buffer', () => {
    const mesh = unitCubeMesh();
    const original = voxelize(mesh, { particleRadius: 0.1 });
    const buf = encodeVoxelizeBinary(original);
    // Slice off the last 8 bytes — the surfaceFlag section is now short.
    const truncated = buf.slice(0, buf.byteLength - 8);
    expect(() => decodeVoxelizeBinary(truncated)).toThrow(/buffer size/);
  });

  it('rejects a too-small buffer', () => {
    const buf = new ArrayBuffer(4);
    expect(() => decodeVoxelizeBinary(buf)).toThrow(/too small/);
  });
});
