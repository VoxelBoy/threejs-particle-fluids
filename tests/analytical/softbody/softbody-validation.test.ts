import { uniform } from 'three/tsl';
import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';

import {
  ParticleSystem,
  SoftbodySystem,
  type SoftbodyDef,
  type SolverContext,
} from '../../../src/index.js';

// Construction and build checks that need no GPU: the particles never touch
// the device, so the renderer is a stub.

/** 8 corners of a cube of side 0.1, one body starting at `start`. */
function cube(start = 0, edges?: Uint32Array): SoftbodyDef {
  const rest = new Float32Array(24);
  for (let i = 0; i < 8; i++) {
    rest[i * 3] = (i & 1) * 0.1;
    rest[i * 3 + 1] = ((i >> 1) & 1) * 0.1;
    rest[i * 3 + 2] = ((i >> 2) & 1) * 0.1;
  }
  return { range: { start, count: 8 }, restPositions: rest, ...(edges ? { edges } : {}) };
}

function particles(capacity = 8): ParticleSystem {
  return new ParticleSystem({} as WebGPURenderer, capacity, 0.05);
}

function build(system: SoftbodySystem, firstGroup: number): void {
  let next = firstGroup;
  system.build({
    particles: system.particles,
    dt: uniform(0.01),
    allocateCollisionGroup: () => next++,
  } as unknown as SolverContext);
}

describe('SoftbodySystem edge validation', () => {
  for (const shapeMatching of ['global', 'local'] as const) {
    it(`checks edges in the constructor under '${shapeMatching}'`, () => {
      const make = (edges: Uint32Array) => () =>
        new SoftbodySystem(particles(), { shapeMatching, bodies: [cube(0, edges)] });
      expect(make(new Uint32Array([0, 1, 2]))).toThrow(/body 0 has an odd number of edge indices/);
      expect(make(new Uint32Array([0, 8]))).toThrow(/body 0 has an invalid edge \(0, 8\)/);
      expect(make(new Uint32Array([3, 3]))).toThrow(/body 0 has an invalid edge \(3, 3\)/);
      expect(make(new Uint32Array([0, 1, 1, 3]))).not.toThrow();
    });
  }

  it("requires edges under 'local' when constructed, not only when built", () => {
    expect(
      () => new SoftbodySystem(particles(), { shapeMatching: 'local', bodies: [cube()] }),
    ).toThrow(/body 0 needs edges for local shape matching/);
    expect(() => new SoftbodySystem(particles(), { bodies: [cube()] })).not.toThrow();
  });
});

describe('SoftbodySystem empty bodies', () => {
  it('names a body with no particles', () => {
    expect(
      () =>
        new SoftbodySystem(particles(), { bodies: [cube(), { range: { start: 0, count: 0 } }] }),
    ).toThrow(/SoftbodySystem: body 1 has no particles/);
  });
});

describe('SoftbodySystem collision groups', () => {
  it('gives each body its own group when none was set', () => {
    const p = particles(16);
    build(new SoftbodySystem(p, { bodies: [cube(0), cube(8)] }), 5);
    const groups = p.collisionGroup.value.array as Uint32Array;
    expect(Array.from(groups)).toEqual([...Array(8).fill(5), ...Array(8).fill(6)]);
  });

  it("keeps a group the user gave a whole body, and still assigns the others'", () => {
    const p = particles(16);
    p.setCollisionGroup({ start: 0, count: 8 }, 3);
    build(new SoftbodySystem(p, { bodies: [cube(0), cube(8)] }), 4);
    const groups = p.collisionGroup.value.array as Uint32Array;
    expect(Array.from(groups)).toEqual([...Array(8).fill(3), ...Array(8).fill(4)]);
  });

  it('rejects a body whose particles have mixed groups', () => {
    const p = particles();
    p.setCollisionGroup({ start: 0, count: 4 }, 3);
    const system = new SoftbodySystem(p, { bodies: [cube()] });
    expect(() => build(system, 4)).toThrow(/body 0 has mixed collision groups/);
  });

  it('leaves groups alone with selfCollision', () => {
    const p = particles();
    p.setCollisionGroup({ start: 0, count: 4 }, 3);
    build(new SoftbodySystem(p, { bodies: [cube()], selfCollision: true }), 4);
    const groups = p.collisionGroup.value.array as Uint32Array;
    expect(Array.from(groups)).toEqual([3, 3, 3, 3, 0, 0, 0, 0]);
  });
});
