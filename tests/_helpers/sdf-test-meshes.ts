/**
 * Procedural test meshes for Phase 07 SDF tests. Hand-rolled instead of
 * three's `SphereGeometry` / `TorusKnotGeometry` so the tests can run
 * under the `analytical` (node) vitest suite without importing three —
 * three's geometry builders reach for `window`/`document` on import on
 * some r184 code paths, and node-importing them is a rabbit hole the
 * test surface doesn't need to poke.
 *
 * Every returned mesh is watertight and manifold (the Phase 07 baker's
 * ray-cast sign test assumes this — see `src/sdf/bake.ts`).
 */

export interface TestMesh {
  readonly positions: Float32Array;
  readonly indices: Uint32Array;
}

/**
 * UV sphere — stacks × slices watertight with triangle fans at the poles.
 * For `stacks = 32, slices = 32` (1920 triangles, radius 0.5): the
 * maximum facet-to-ideal-sphere deviation is `r·(1 − cos(π/stacks))
 * ≈ 2.4e-3 m`, well under the Phase 07 plan's `0.5 · voxelSize` gate at
 * 64³ resolution (≈ 9e-3 m).
 */
export function makeUvSphere(radius: number, stacks: number, slices: number): TestMesh {
  if (stacks < 3 || slices < 3) {
    throw new Error(`makeUvSphere: stacks/slices must each be ≥ 3`);
  }
  const vertexCount = 2 + (stacks - 1) * slices; // 2 poles + (stacks-1) latitude rings.
  const positions = new Float32Array(vertexCount * 3);
  // North pole at index 0, south pole at index vertexCount - 1.
  positions[0] = 0;
  positions[1] = radius;
  positions[2] = 0;
  const southBase = (vertexCount - 1) * 3;
  positions[southBase + 0] = 0;
  positions[southBase + 1] = -radius;
  positions[southBase + 2] = 0;
  // Interior rings: stack s in [1, stacks-1], slice k in [0, slices-1].
  for (let s = 1; s < stacks; s++) {
    const phi = (Math.PI * s) / stacks; // [0, π]
    const y = Math.cos(phi) * radius;
    const ringR = Math.sin(phi) * radius;
    for (let k = 0; k < slices; k++) {
      const theta = (2 * Math.PI * k) / slices;
      const idx = 1 + (s - 1) * slices + k;
      positions[idx * 3 + 0] = Math.cos(theta) * ringR;
      positions[idx * 3 + 1] = y;
      positions[idx * 3 + 2] = Math.sin(theta) * ringR;
    }
  }

  const indices: number[] = [];
  const northIdx = 0;
  const southIdx = vertexCount - 1;
  const ringStart = (s: number) => 1 + (s - 1) * slices;
  // Top cap — triangle fan from north pole to ring 0.
  for (let k = 0; k < slices; k++) {
    const a = ringStart(1) + k;
    const b = ringStart(1) + ((k + 1) % slices);
    indices.push(northIdx, a, b);
  }
  // Middle quads — ring s to ring s+1.
  for (let s = 1; s < stacks - 1; s++) {
    const rs = ringStart(s);
    const rsNext = ringStart(s + 1);
    for (let k = 0; k < slices; k++) {
      const kNext = (k + 1) % slices;
      const a = rs + k;
      const b = rs + kNext;
      const c = rsNext + k;
      const d = rsNext + kNext;
      indices.push(a, c, d);
      indices.push(a, d, b);
    }
  }
  // Bottom cap — triangle fan from ring (stacks-2) to south pole.
  const rsLast = ringStart(stacks - 1);
  for (let k = 0; k < slices; k++) {
    const a = rsLast + k;
    const b = rsLast + ((k + 1) % slices);
    indices.push(a, southIdx, b);
  }
  return {
    positions,
    indices: new Uint32Array(indices),
  };
}

/**
 * Torus — tube of radius `tubeRadius` swept around a ring of radius
 * `ringRadius`. `ringSegs` divisions around the major circle, `tubeSegs`
 * around the minor. Watertight quad strip.
 *
 * Used as a stand-in for the Phase 07 plan's "baked Stanford bunny"
 * projection test: the interior of the tube is a genuine non-convex
 * inside region, so "place 1000 particles inside and solve out" has a
 * non-trivial sign-flip at the inner wall.
 */
export function makeTorus(
  ringRadius: number,
  tubeRadius: number,
  ringSegs: number,
  tubeSegs: number,
): TestMesh {
  if (ringSegs < 3 || tubeSegs < 3) {
    throw new Error('makeTorus: ringSegs/tubeSegs must each be ≥ 3');
  }
  const vertexCount = ringSegs * tubeSegs;
  const positions = new Float32Array(vertexCount * 3);
  for (let i = 0; i < ringSegs; i++) {
    const u = (2 * Math.PI * i) / ringSegs;
    const cu = Math.cos(u);
    const su = Math.sin(u);
    for (let j = 0; j < tubeSegs; j++) {
      const v = (2 * Math.PI * j) / tubeSegs;
      const cv = Math.cos(v);
      const sv = Math.sin(v);
      const r = ringRadius + tubeRadius * cv;
      const idx = (i * tubeSegs + j) * 3;
      positions[idx + 0] = r * cu;
      positions[idx + 1] = tubeRadius * sv;
      positions[idx + 2] = r * su;
    }
  }
  const indices: number[] = [];
  for (let i = 0; i < ringSegs; i++) {
    const iNext = (i + 1) % ringSegs;
    for (let j = 0; j < tubeSegs; j++) {
      const jNext = (j + 1) % tubeSegs;
      const a = i * tubeSegs + j;
      const b = i * tubeSegs + jNext;
      const c = iNext * tubeSegs + j;
      const d = iNext * tubeSegs + jNext;
      indices.push(a, c, d);
      indices.push(a, d, b);
    }
  }
  return { positions, indices: new Uint32Array(indices) };
}

/** Deterministic LCG — same constants as the Phase 06 tests. */
export function makeLcg(seed: number): () => number {
  let s = seed | 0;
  return (): number => {
    s = (Math.imul(s, 1664525) + 1013904223) | 0;
    return (s >>> 0) / 0x100000000;
  };
}
