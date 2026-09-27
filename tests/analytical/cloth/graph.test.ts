import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute } from 'three';

import { createClothGraph } from '../../../src/index.js';

// Build a 2-triangle quad sharing one diagonal edge — minimal mesh
// that has both distance constraints (5 edges) and exactly one
// bending tuple.
//
//   p0 ─── p1
//    \    /│
//     \  / │
//      \/  │
//      /\  │
//     /  \ │
//    /    \│
//   p2 ─── p3
//
// Two triangles: (p0, p2, p1), (p1, p2, p3). Shared edge: (p1, p2).
function buildMinimalQuad(): BufferGeometry {
  const geom = new BufferGeometry();
  // prettier-ignore
  geom.setAttribute(
    'position',
    new Float32BufferAttribute(
      [
        0, 1, 0, // p0
        1, 1, 0, // p1
        0, 0, 0, // p2
        1, 0, 0, // p3
      ],
      3,
    ),
  );
  geom.setIndex(new Uint32BufferAttribute([0, 2, 1, 1, 2, 3], 1));
  return geom;
}

describe('createClothGraph', () => {
  it('builds minimal quad: 4 particles, 5 distance edges, 1 bending tuple', () => {
    const geom = buildMinimalQuad();
    const graph = createClothGraph(geom, { surfaceDensity: 0.2 });

    expect(graph.positions.length).toBe(4);
    // Quad has 5 unique edges: 4 boundary + 1 diagonal (the shared edge).
    expect(graph.distancePairs.length).toBe(5);
    expect(graph.distanceRestLengths.length).toBe(5);
    // Exactly one shared edge → exactly one bending tuple.
    expect(graph.bendingTuples.length).toBe(1);
    expect(graph.bendingRestAngles.length).toBe(1);
    // Boundary edges = 4 (the four perimeter edges of the quad).
    expect(graph.diagnostics.boundaryEdges).toBe(4);
    expect(graph.diagnostics.nonManifoldEdgesSkipped).toBe(0);
    expect(graph.diagnostics.degenerateTrianglesSkipped).toBe(0);
  });

  it('flat planar quad has bending rest angle = 0 (Bridson §4 convention)', () => {
    const geom = buildMinimalQuad();
    const graph = createClothGraph(geom);
    expect(graph.bendingRestAngles[0]!).toBeCloseTo(0, 5);
  });

  it('90° folded triangle pair has bending rest angle = π/2', () => {
    // Two triangles sharing edge (p1, p2): triangle 1 in the xy plane,
    // triangle 2 in the plane spanned by edge p1-p2 + a vertical lift,
    // engineered so the triangle normals are exactly perpendicular.
    //
    //   Triangle 1: (0,0,0), (1,0,0), (0,1,0)  — far vertex (0,0,0)
    //   Triangle 2: (1,0,0), (1,0,1), (0,1,0)  — far vertex (1,0,1)
    //   Shared edge: from (1,0,0) to (0,1,0).
    //   Triangle 1 normal: along ±z.
    //   Triangle 2 normal: (p3-p2)×(p3-p1) = (1,-1,1)×(0,0,1) = (-1,-1,0).
    //   Dot: 0 → 90°.
    const geom = new BufferGeometry();
    // prettier-ignore
    geom.setAttribute(
      'position',
      new Float32BufferAttribute(
        [
          0, 0, 0, // 0 — far vertex of triangle 1
          1, 0, 0, // 1 — shared edge endpoint
          0, 1, 0, // 2 — shared edge endpoint
          1, 0, 1, // 3 — far vertex of triangle 2 (lifted in z)
        ],
        3,
      ),
    );
    geom.setIndex(new Uint32BufferAttribute([0, 1, 2, 1, 3, 2], 1));
    const graph = createClothGraph(geom);
    expect(graph.bendingTuples.length).toBe(1);
    // Signed atan2 — the sign depends on whether the fold is valley
    // or mountain relative to the shared-edge direction. |restAngle|
    // is what physically encodes "90° fold"; tests that pin a specific
    // sign would over-constrain the convention. Magnitude only.
    expect(Math.abs(graph.bendingRestAngles[0]!)).toBeCloseTo(Math.PI / 2, 4);
  });

  it('pinning sets invMass = 0 on pinned indices', () => {
    const geom = buildMinimalQuad();
    const graph = createClothGraph(geom, { pinnedIndices: [0, 1] });
    expect(graph.invMass[0]).toBe(0);
    expect(graph.invMass[1]).toBe(0);
    expect(graph.invMass[2]).toBeGreaterThan(0);
    expect(graph.invMass[3]).toBeGreaterThan(0);
  });

  it('welds coincident vertices', () => {
    const geom = new BufferGeometry();
    // Two triangles where vertex 0 and 4 occupy the same world position.
    // prettier-ignore
    geom.setAttribute(
      'position',
      new Float32BufferAttribute(
        [
          0, 0, 0, // 0
          1, 0, 0, // 1
          0, 1, 0, // 2
          1, 1, 0, // 3
          0, 0, 0, // 4 — duplicate of 0
          1, 0, 0, // 5 — duplicate of 1
        ],
        3,
      ),
    );
    geom.setIndex(new Uint32BufferAttribute([0, 1, 2, 4, 5, 3], 1));
    const graph = createClothGraph(geom);
    // 4 unique positions after dedup (0=4, 1=5, plus 2 and 3).
    expect(graph.positions.length).toBe(4);
  });

  it('rejects non-indexed geometry', () => {
    const geom = new BufferGeometry();
    geom.setAttribute('position', new Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
    expect(() => createClothGraph(geom)).toThrow(/must be indexed/);
  });

  it('inverse mass scales with surface density', () => {
    const geom = buildMinimalQuad();
    const dense = createClothGraph(geom, { surfaceDensity: 1.0 });
    const light = createClothGraph(geom, { surfaceDensity: 0.1 });
    // m_dense = 10 · m_light  ⇒  invMass_dense = 0.1 · invMass_light.
    for (let i = 0; i < 4; i++) {
      expect(dense.invMass[i]!).toBeCloseTo(0.1 * light.invMass[i]!, 6);
    }
  });
});
