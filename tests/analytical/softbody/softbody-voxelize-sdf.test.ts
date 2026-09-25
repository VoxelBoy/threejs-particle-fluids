import { describe, expect, it } from 'vitest';

import {
  decodeVoxelizeBinary,
  encodeVoxelizeBinary,
  voxelize,
  type TriangleMesh,
} from '../../../src/softbody/index.js';

// Phase 15 G1 — voxelize bakeSdf.
//
// Macklin 2014 §5.1 sparse SDF: per particle, voxelize emits `(φ, ∇φ)` where
// φ is the signed distance to the closed mesh surface (≤ 0 inside, paper
// convention) and ∇φ is the unit outward normal at the closest projection.
// Computed by direct closest-point-on-mesh per particle (Ericson RTCD
// §5.1.5), no volumetric grid intermediate.

function unitCubeMesh(): TriangleMesh {
  // eslint-disable-next-line prettier/prettier
  const vertices = new Float32Array([
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
  ]);
  // Outward-normal winding on every face.
  // eslint-disable-next-line prettier/prettier
  const indices = new Uint32Array([
    0,
    2,
    1,
    1,
    2,
    3, // -z
    4,
    5,
    6,
    5,
    7,
    6, // +z
    0,
    1,
    4,
    1,
    5,
    4, // -y
    2,
    6,
    3,
    3,
    6,
    7, // +y
    0,
    4,
    2,
    2,
    4,
    6, // -x
    1,
    3,
    5,
    3,
    7,
    5, // +x
  ]);
  return { vertices, indices };
}

describe('Phase 15 — voxelize bakeSdf (cube G1)', () => {
  it('omits restSDF when bakeSdf is unset (Phase 10 back-compat)', () => {
    const mesh = unitCubeMesh();
    const result = voxelize(mesh, { particleRadius: 0.1 });
    expect(result.restSDF).toBeUndefined();
  });

  it('populates restSDF with paper-convention φ ≥ 0 inside and unit ∇φ', () => {
    // Convention: φ ≥ 0 inside the closed mesh, ∇φ points INWARD (toward
    // the body's interior, away from the closest surface point). Required
    // for paper §5.1 eqs. 17–20 to produce correct contact separation —
    // see voxelize.ts §"Gradient direction" derivation. The outward-∇φ
    // convention silently breaks rigid-rigid contact (cubes blow apart on
    // touch); the sign is locked in here.
    const mesh = unitCubeMesh();
    const result = voxelize(mesh, { particleRadius: 0.1, bakeSdf: true });
    expect(result.restSDF).toBeDefined();
    expect(result.restSDF!.length).toBe(4 * result.count);

    for (let i = 0; i < result.count; i++) {
      const phi = result.restSDF![4 * i + 0]!;
      const gx = result.restSDF![4 * i + 1]!;
      const gy = result.restSDF![4 * i + 2]!;
      const gz = result.restSDF![4 * i + 3]!;
      // φ ≥ 0 for every voxelized particle (all interior by construction).
      expect(phi).toBeGreaterThanOrEqual(-1e-6);
      // ∇φ unit-length within FP tolerance.
      const gMag = Math.sqrt(gx * gx + gy * gy + gz * gz);
      expect(Math.abs(gMag - 1)).toBeLessThan(1e-5);
    }
  });

  it('surface particles have |φ| ≤ particleRadius', () => {
    // Surface particle = voxel whose face neighbours include outside-of-grid
    // or unoccupied cells — its centre is one half-spacing from the nearest
    // mesh face, so |φ| should be at most ~particleRadius (one voxel
    // half-extent). Match the paper §5.1 boundary criterion |φ| < r.
    const mesh = unitCubeMesh();
    const r = 0.1;
    const result = voxelize(mesh, { particleRadius: r, bakeSdf: true });
    for (let i = 0; i < result.surfaceCount; i++) {
      const phi = result.restSDF![4 * i + 0]!;
      expect(Math.abs(phi)).toBeLessThanOrEqual(r + 1e-5);
    }
  });

  it('interior particles have |φ| > particleRadius', () => {
    // Interior particle = all six face neighbours occupied — its centre is
    // at least one full spacing from the surface, i.e. |φ| ≥ r.
    const mesh = unitCubeMesh();
    const r = 0.1;
    const result = voxelize(mesh, { particleRadius: r, bakeSdf: true });
    for (let i = result.surfaceCount; i < result.count; i++) {
      const phi = result.restSDF![4 * i + 0]!;
      expect(Math.abs(phi)).toBeGreaterThan(r - 1e-5);
    }
  });

  it('cube-centre particle has φ ≈ −0.5 (half edge length) and gradient on a face axis', () => {
    // The cube-centre voxel's nearest surface point is on whichever face is
    // closest; |φ| at the centre of a unit cube is 0.5 (half edge). Gradient
    // points toward that face — one component magnitude ≈ 1, the others ≈ 0.
    // Our 5×5×5 voxelization at r=0.1 places the centre voxel at the cube
    // origin (centre at (0,0,0) when N=5 is odd, spacing 0.2, origin offset
    // 0.5).
    const mesh = unitCubeMesh();
    const result = voxelize(mesh, { particleRadius: 0.1, bakeSdf: true });
    let centreIdx = -1;
    for (let i = 0; i < result.count; i++) {
      const x = result.positions[3 * i + 0]!;
      const y = result.positions[3 * i + 1]!;
      const z = result.positions[3 * i + 2]!;
      if (Math.abs(x) < 1e-6 && Math.abs(y) < 1e-6 && Math.abs(z) < 1e-6) {
        centreIdx = i;
        break;
      }
    }
    expect(centreIdx).toBeGreaterThanOrEqual(0);
    const phi = result.restSDF![4 * centreIdx + 0]!;
    // |φ| ≈ 0.5; sign is positive (interior, convention φ ≥ 0 inside).
    expect(Math.abs(phi - 0.5)).toBeLessThan(1e-5);
    // Gradient is on a face axis — one component ≈ ±1, others ≈ 0.
    const g = [
      result.restSDF![4 * centreIdx + 1]!,
      result.restSDF![4 * centreIdx + 2]!,
      result.restSDF![4 * centreIdx + 3]!,
    ];
    const dominant = Math.max(...g.map(Math.abs));
    expect(dominant).toBeGreaterThan(0.99);
    const others = g.filter((v) => Math.abs(v) !== dominant);
    for (const o of others) {
      expect(Math.abs(o)).toBeLessThan(0.01);
    }
  });

  it('gradient points inward — particle near the +x face has ∇φ ≈ (−1, 0, 0)', () => {
    // Paper §5.1 contact math (eqs. 17–20) requires `∇φ` to point INWARD
    // (toward the body's interior), away from the closest surface point.
    // A particle on the +x face centre (closest surface uniquely at
    // (+0.5, 0, 0)) must have `∇φ = (−1, 0, 0)` exactly.
    //
    // This locks in the sign: an outward-pointing gradient is the
    // failure mode that flips eq. 18 / eq. 19 separation direction —
    // bodies move TOWARD each other on contact, manifesting as bouncy
    // interpenetration / tunneling (rigid-pile demo "cubes blow apart"
    // symptom 2026-05-02; root cause derived from paper math, see
    // voxelize.ts §"Gradient direction" comment).
    //
    // Corner particles are equidistant from three faces and the closest-
    // point-on-triangle iteration arbitrarily picks one face's gradient,
    // so this test specifically targets a face-centre voxel where the
    // closest surface is unique.
    const mesh = unitCubeMesh();
    const result = voxelize(mesh, { particleRadius: 0.1, bakeSdf: true });
    let centreIdx = -1;
    for (let i = 0; i < result.count; i++) {
      const x = result.positions[3 * i + 0]!;
      const y = result.positions[3 * i + 1]!;
      const z = result.positions[3 * i + 2]!;
      if (Math.abs(x - 0.4) < 1e-5 && Math.abs(y) < 1e-5 && Math.abs(z) < 1e-5) {
        centreIdx = i;
        break;
      }
    }
    expect(centreIdx).toBeGreaterThanOrEqual(0);
    const gx = result.restSDF![4 * centreIdx + 1]!;
    const gy = result.restSDF![4 * centreIdx + 2]!;
    const gz = result.restSDF![4 * centreIdx + 3]!;
    expect(gx).toBeLessThan(-0.99); // pointing AWAY from +x face = inward (-x)
    expect(Math.abs(gy)).toBeLessThan(0.01);
    expect(Math.abs(gz)).toBeLessThan(0.01);
  });
});

describe('Phase 15 — voxelize binary codec (v2 with SDF)', () => {
  it('round-trips through v2 encode → decode with restSDF intact', () => {
    const mesh = unitCubeMesh();
    const original = voxelize(mesh, { particleRadius: 0.1, bakeSdf: true });
    const buf = encodeVoxelizeBinary(original);
    const decoded = decodeVoxelizeBinary(buf);

    expect(decoded.count).toBe(original.count);
    expect(decoded.surfaceCount).toBe(original.surfaceCount);
    expect(decoded.restSDF).toBeDefined();
    expect(decoded.restSDF!.length).toBe(original.restSDF!.length);
    for (let i = 0; i < original.restSDF!.length; i++) {
      expect(decoded.restSDF![i]).toBe(original.restSDF![i]);
    }
  });

  it('round-trips a non-SDF (v1-shape) result with restSDF undefined', () => {
    const mesh = unitCubeMesh();
    const original = voxelize(mesh, { particleRadius: 0.1 }); // no bakeSdf
    const buf = encodeVoxelizeBinary(original);
    const decoded = decodeVoxelizeBinary(buf);
    expect(decoded.restSDF).toBeUndefined();
  });
});
