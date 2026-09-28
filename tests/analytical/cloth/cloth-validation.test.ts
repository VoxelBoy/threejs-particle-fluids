import { PlaneGeometry } from 'three';
import { uniform } from 'three/tsl';
import type { WebGPURenderer } from 'three/webgpu';
import { describe, expect, it } from 'vitest';

import {
  ClothSystem,
  ParticleSystem,
  createClothGraph,
  type ClothSystemOptions,
  type SolverContext,
} from '../../../src/index.js';

// Construction checks that need no GPU: the particles never touch the
// device, so the renderer is a stub.

const graph = createClothGraph(new PlaneGeometry(1, 1, 4, 4), { pinnedIndices: [0, 4] });

function particles(capacity = graph.positions.length): ParticleSystem {
  return new ParticleSystem({} as WebGPURenderer, capacity, 0.05);
}

function cloth(options: Omit<ClothSystemOptions, 'graph'> = {}): ClothSystem {
  return new ClothSystem(particles(), { graph, ...options });
}

describe('ClothSystem validation', () => {
  it('rejects negative or non-finite options with a ClothSystem prefix', () => {
    for (const key of [
      'stretchCompliance',
      'bendCompliance',
      'tetherCompliance',
      'stretchTolerance',
      'drag',
      'lift',
    ] as const) {
      expect(() => cloth({ [key]: -1 })).toThrow(`ClothSystem: ${key} must be ≥ 0, got -1`);
      expect(() => cloth({ [key]: NaN })).toThrow(`ClothSystem: ${key} must be ≥ 0`);
    }
    expect(() => cloth({ damping: 1.5 })).toThrow('ClothSystem: damping must be between 0 and 1');
  });

  it('validates the drag, lift, damping, and bendCompliance setters', () => {
    const c = cloth({ damping: 0.1 });
    expect(() => (c.drag = -0.1)).toThrow('ClothSystem: drag must be ≥ 0');
    expect(() => (c.lift = Infinity)).toThrow('ClothSystem: lift must be ≥ 0');
    expect(() => (c.damping = -0.1)).toThrow('ClothSystem: damping must be between 0 and 1');
    expect(() => (c.bendCompliance = -1)).toThrow('ClothSystem: bendCompliance must be ≥ 0');
    c.drag = 0.2;
    c.lift = 0;
    c.damping = 0.4;
    expect([c.drag, c.lift, c.damping]).toEqual([0.2, 0, 0.4]);
  });

  it('only lets damping change when the option enabled it', () => {
    const c = cloth();
    expect(c.damping).toBe(0);
    expect(() => (c.damping = 0.2)).toThrow(
      'ClothSystem: pass `damping` in the options to enable it before changing it',
    );
    const kernels = c.build({ particles: c.particles, dt: uniform(0.01) } as SolverContext);
    expect(kernels.postSolve).toEqual([]);
  });

  it('keeps collision groups set before construction', () => {
    const p = particles(graph.positions.length + 2);
    const range = { start: 2, count: graph.positions.length };
    p.setCollisionGroup(range, 3);
    new ClothSystem(p, { graph, offset: 2 });
    const groups = p.collisionGroup.value.array as Uint32Array;
    expect(Array.from(groups.subarray(2)).every((g) => g === 3)).toBe(true);
  });

  it('anchors tethers to pinned particle indices', () => {
    const c = cloth();
    expect(c.tethers.length).toBeGreaterThan(0);
    for (const t of c.tethers) {
      expect(graph.invMass[t.anchor]).toBe(0);
      expect(graph.invMass[t.particle]).toBeGreaterThan(0);
    }
  });
});
