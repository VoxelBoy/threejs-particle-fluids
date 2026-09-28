import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';

import { ParticleSystem, SoftbodySystem } from '../../../src/index.js';
import { bindSoftbodyMesh } from '../../../src/softbody/bindMesh.js';

/**
 * A one-body {@link SoftbodySystem} over the given rest positions. Binding is
 * CPU-only, so the particles never touch a GPU and the renderer is a stub.
 */
function makeSoftbody(
  particleRadius: number,
  restPositions: Float32Array,
  baseSlot = 0,
  edges?: Uint32Array,
): SoftbodySystem {
  const count = restPositions.length / 3;
  const particles = new ParticleSystem({} as WebGPURenderer, baseSlot + count, particleRadius);
  return new SoftbodySystem(particles, {
    bodies: [
      {
        range: { start: baseSlot, count },
        restPositions,
        compliance: 1e-6,
        ...(edges ? { edges } : {}),
      },
    ],
  });
}

/**
 * 3x3x3 voxel grid of particles spaced 2·r apart, centered at origin
 * (mesh-frame). Identical to the layout the voxelizer produces for a small
 * box, suitable for the bind tests.
 */
function buildSmallGrid(particleRadius: number): Float32Array {
  const spacing = 2 * particleRadius;
  const N = 3;
  const out = new Float32Array(3 * N * N * N);
  let k = 0;
  for (let ix = 0; ix < N; ix++) {
    for (let iy = 0; iy < N; iy++) {
      for (let iz = 0; iz < N; iz++) {
        out[3 * k + 0] = (ix - 1) * spacing;
        out[3 * k + 1] = (iy - 1) * spacing;
        out[3 * k + 2] = (iz - 1) * spacing;
        k++;
      }
    }
  }
  return out;
}

function meshFromVertices(verts: Float32Array): BufferGeometry {
  const geom = new BufferGeometry();
  geom.setAttribute('position', new Float32BufferAttribute(verts.slice(), 3));
  // Trivial fan-style index — enough for the geometry to be valid; the
  // bind step doesn't read indices, only positions.
  const tris = (verts.length / 3 - 2) * 3;
  if (tris > 0) {
    const idx = new Uint32Array(tris);
    for (let i = 0; i < tris / 3; i++) {
      idx[3 * i + 0] = 0;
      idx[3 * i + 1] = i + 1;
      idx[3 * i + 2] = i + 2;
    }
    geom.setIndex(new Uint32BufferAttribute(idx, 1));
  }
  return geom;
}

const PARTICLE_RADIUS = 0.05;

describe('bindSoftbodyMesh', () => {
  it('emits partition-of-unity weights (|Σw − 1| < 1e-6) for every vertex', () => {
    const rest = buildSmallGrid(PARTICLE_RADIUS);
    const softbody = makeSoftbody(PARTICLE_RADIUS, rest);

    // Mesh vertices at and near particle slots. The partition-of-unity
    // invariant is independent of vertex placement.
    const verts = new Float32Array([
      0,
      0,
      0,
      2 * PARTICLE_RADIUS,
      0,
      0,
      0,
      2 * PARTICLE_RADIUS,
      0,
      0,
      0,
      2 * PARTICLE_RADIUS,
      0.5 * PARTICLE_RADIUS,
      0,
      0,
      0,
      0.5 * PARTICLE_RADIUS,
      0.5 * PARTICLE_RADIUS,
    ]);
    const geom = meshFromVertices(verts);
    expect(bindSoftbodyMesh(geom, softbody, 0)).toBe(geom);

    // Four influences per vertex.
    const wts = geom.getAttribute('weights')!;
    expect(wts.itemSize).toBe(4);
    const weights = wts.array as Float32Array;
    for (let v = 0; v < verts.length / 3; v++) {
      const w =
        weights[4 * v + 0]! + weights[4 * v + 1]! + weights[4 * v + 2]! + weights[4 * v + 3]!;
      expect(Math.abs(w - 1)).toBeLessThan(1e-6);
    }
  });

  it('writes 4-wide influence + weight attributes with the expected shape', () => {
    const rest = buildSmallGrid(PARTICLE_RADIUS);
    const softbody = makeSoftbody(PARTICLE_RADIUS, rest, 17 /* baseSlot */);
    const verts = new Float32Array([0, 0, 0, 0.05, 0, 0, 0, 0.05, 0]);
    const geom = meshFromVertices(verts);
    bindSoftbodyMesh(geom, softbody, 0);

    const inf = geom.getAttribute('influences')!;
    const wts = geom.getAttribute('weights')!;
    expect(inf.itemSize).toBe(4);
    expect(wts.itemSize).toBe(4);
    expect(inf.array.length).toBe(4 * 3);
    expect(wts.array.length).toBe(4 * 3);

    // Indices must be in the body's global slot range.
    const arr = inf.array as Uint32Array;
    for (const idx of arr) {
      expect(idx).toBeGreaterThanOrEqual(17);
      expect(idx).toBeLessThan(17 + 27);
    }
  });

  it('excludes single-particle chain ends (degree < 3) from the nearest particles when edges are supplied', () => {
    // 3x3x3 voxel grid where every particle has 6-face degree ≥ 3 (the
    // densest interior particle has 6, corners have 3), plus a single
    // "ear-tip" particle off one face with no grid neighbor at the edge
    // spacing. Its degree is below 3, so it MUST be excluded from the
    // nearest-particle search even when a vertex sits right on it.
    const rGrid = buildSmallGrid(PARTICLE_RADIUS); // 27 particles, indices 0..26
    const earTip = [3 * PARTICLE_RADIUS, 0, 0];
    const positions = new Float32Array(rGrid.length + 3);
    positions.set(rGrid, 0);
    positions.set(earTip, rGrid.length);
    const earIndex = 27;

    // Edge graph — every (i,j) pair exactly 2·r apart.
    const all: number[][] = [];
    for (let i = 0; i < positions.length / 3; i++) {
      all.push([positions[3 * i + 0]!, positions[3 * i + 1]!, positions[3 * i + 2]!]);
    }
    const edges: number[] = [];
    for (let i = 0; i < all.length; i++) {
      for (let j = i + 1; j < all.length; j++) {
        const dx = all[i]![0]! - all[j]![0]!;
        const dy = all[i]![1]! - all[j]![1]!;
        const dz = all[i]![2]! - all[j]![2]!;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (Math.abs(d - 2 * PARTICLE_RADIUS) < 1e-6) edges.push(i, j);
      }
    }
    const softbody = makeSoftbody(PARTICLE_RADIUS, positions, 0, new Uint32Array(edges));

    // Vertex placed AT the ear-tip particle. Without chain exclusion the
    // nearest particle would be index 27 (distance 0). With exclusion the
    // binding must skip 27 entirely and use grid particles.
    const verts = new Float32Array([3 * PARTICLE_RADIUS, 0, 0]);
    const geom = meshFromVertices(verts);
    bindSoftbodyMesh(geom, softbody, 0);

    const inf = geom.getAttribute('influences')!.array as Uint32Array;
    for (let k = 0; k < 4; k++) expect(inf[k]).not.toBe(earIndex);
  });

  it("exposes the body's rest center, which skinning measures vertices from", () => {
    // Skinning places vertices relative to the body's rest center. Shift the
    // entire grid by (0.5, 0, 0) → restCenter ≈ (0.5, 0, 0).
    const rest = buildSmallGrid(PARTICLE_RADIUS);
    for (let i = 0; i < rest.length; i += 3) rest[i]! += 0.5;
    const softbody = makeSoftbody(PARTICLE_RADIUS, rest);
    const verts = new Float32Array([0.5, 0, 0]);
    const geom = meshFromVertices(verts);
    bindSoftbodyMesh(geom, softbody, 0);
    const cBar = softbody.bodies[0]!.restCenter;
    expect(cBar[0]).toBeCloseTo(0.5, 6);
    expect(cBar[1]).toBeCloseTo(0, 6);
    expect(cBar[2]).toBeCloseTo(0, 6);
  });

  it('reports errors under the SoftbodyMesh name', () => {
    const softbody = makeSoftbody(PARTICLE_RADIUS, buildSmallGrid(PARTICLE_RADIUS));
    const geom = meshFromVertices(new Float32Array([0, 0, 0]));
    expect(() => bindSoftbodyMesh(geom, softbody, 1)).toThrow(/^SoftbodyMesh: no body 1/);
    expect(() => bindSoftbodyMesh(new BufferGeometry(), softbody, 0)).toThrow(
      /^SoftbodyMesh: geometry has no position attribute/,
    );
  });
});
