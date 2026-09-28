import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute } from 'three';
import { uniform } from 'three/tsl';
import type { WebGPURenderer } from 'three/webgpu';

import {
  FrameStepper,
  HashGrid,
  ParticleSystem,
  colorConstraints,
  createDistanceConstraints,
  createParticleMesh,
  createSphKernelUniforms,
  toTriangleMesh,
} from '../../../src/index.js';

// CPU-side validation and bookkeeping in the core module. No GPU: the
// renderer is only stored by these constructors, never called.
const renderer = {} as WebGPURenderer;

describe('FrameStepper', () => {
  async function pump(stepper: FrameStepper, elapsedMs: number) {
    let runs = 0;
    const result = await stepper.pump(1000 + elapsedMs, async () => {
      runs++;
    });
    return { ...result, runs };
  }

  it.each([0.01, 1 / 90, 1 / 60, 1 / 30])(
    'runs the full cap on a truncated frame (fixedDt %f)',
    async (fixedDt) => {
      const stepper = new FrameStepper({ fixedDt, maxStepsPerFrame: 4 });
      await stepper.pump(1000, async () => {});
      const result = await pump(stepper, 1000);
      expect(result).toMatchObject({ steps: 4, runs: 4, truncated: true, remainderSeconds: 0 });
    },
  );

  it('runs exactly the cap without truncating when the elapsed time equals it', async () => {
    const stepper = new FrameStepper({ fixedDt: 0.01, maxStepsPerFrame: 4 });
    await stepper.pump(1000, async () => {});
    const result = await pump(stepper, 40);
    expect(result.steps).toBe(4);
    expect(result.truncated).toBe(false);
    expect(result.remainderSeconds).toBeLessThan(1e-9);
  });

  it('carries the remainder between frames', async () => {
    const stepper = new FrameStepper({ fixedDt: 0.01 });
    await stepper.pump(1000, async () => {});
    const first = await pump(stepper, 15);
    expect(first.steps).toBe(1);
    expect(first.remainderSeconds).toBeCloseTo(0.005, 9);
    const second = await stepper.pump(1020, async () => {});
    expect(second.steps).toBe(1);
  });
});

describe('ParticleSystem validation', () => {
  const particles = new ParticleSystem(renderer, 4, 0.1);

  it('checks the start of an empty upload', () => {
    expect(() => particles.uploadParticles([], 4)).not.toThrow();
    expect(() => particles.uploadParticles([], 5)).toThrow(
      'ParticleSystem.uploadParticles: invalid particle range',
    );
    expect(() => particles.uploadParticles([], -1)).toThrow('invalid particle range');
    expect(() => particles.uploadParticles([], 1.5)).toThrow('invalid particle range');
  });

  it('checks inverse masses and collision groups', () => {
    const all = { start: 0, count: 4 };
    expect(() => particles.setInvMass(all, -1)).toThrow('ParticleSystem.setInvMass: invMass');
    expect(() => particles.setInvMass(all, NaN)).toThrow('ParticleSystem.setInvMass: invMass');
    expect(() => particles.setInvMass(all, Infinity)).toThrow('invMass');
    expect(() => particles.setCollisionGroup(all, 1.5)).toThrow(
      'ParticleSystem.setCollisionGroup: collisionGroup',
    );
    expect(() => particles.setCollisionGroup(all, -1)).toThrow('collisionGroup');
    expect(() => particles.setCollisionGroup(all, 2 ** 32)).toThrow('collisionGroup');
    expect(() => particles.uploadParticles([{ position: [0, 0, 0], invMass: -2 }])).toThrow(
      'ParticleSystem.uploadParticles: invMass',
    );
    expect(() => particles.uploadParticles([{ position: [0, 0, 0], collisionGroup: 0.5 }])).toThrow(
      'ParticleSystem.uploadParticles: collisionGroup',
    );
    particles.setCollisionGroup(all, 2 ** 32 - 1);
    expect((particles.collisionGroup.value.array as Uint32Array)[0]).toBe(2 ** 32 - 1);
  });
});

describe('HashGrid construction', () => {
  it('rejects a hashTableSize that is not an integer power of two', () => {
    const particles = new ParticleSystem(renderer, 8, 0.1);
    expect(() => new HashGrid(particles, { cellSize: 0.2, hashTableSize: 2.5 })).toThrow(
      'HashGrid: hashTableSize must be a positive power of two',
    );
    expect(() => new HashGrid(particles, { cellSize: 0.2, hashTableSize: 12 })).toThrow(
      'power of two',
    );
  });

  it('caps the default table for capacities past 524,288 and names capacity in the error', () => {
    const particles = new ParticleSystem(renderer, 600_000, 0.01);
    const grid = new HashGrid(particles, { cellSize: 0.02 });
    expect(grid.hashTableSize).toBe(1 << 20);
    expect(() => new HashGrid(particles, { cellSize: 0.02, hashTableSize: 1 << 21 })).toThrow(
      /hashTableSize=2097152 \(capacity 600000\)/,
    );
  });

  it('moves the kernels’ origin through hashOrigin', () => {
    const particles = new ParticleSystem(renderer, 8, 0.1);
    const grid = new HashGrid(particles, { cellSize: 0.2 });
    grid.hashOrigin.set(100, 0, -5);
    expect(grid.hashOriginUniform.value.toArray()).toEqual([100, 0, -5]);
  });
});

describe('createSphKernelUniforms', () => {
  it('rederives the coefficients when h changes', () => {
    const u = createSphKernelUniforms(0.1);
    u.h.value = 0.2;
    expect(u.hSq.value).toBeCloseTo(0.04, 12);
    expect(u.poly6Coef.value).toBeCloseTo(315 / (64 * Math.PI * 0.2 ** 9), 6);
    expect(u.spikyCoef.value).toBeCloseTo(45 / (Math.PI * 0.2 ** 6), 6);
    expect(() => (u.h.value = 0)).toThrow('createSphKernelUniforms: h must be positive');
    expect(u.h.value).toBe(0.2);
  });
});

describe('constraint helpers', () => {
  it('colorConstraints checks the participant count', () => {
    expect(() =>
      colorConstraints({ arity: 2, nConstraints: 2, participantsPerConstraint: [0, 1, 2] }),
    ).toThrow('colorConstraints: participantsPerConstraint length 3');
  });

  it('createDistanceConstraints takes per-pair compliance as a typed array', () => {
    const particles = new ParticleSystem(renderer, 3, 0.1);
    const constraints = createDistanceConstraints({
      particles,
      pairs: [
        [0, 1],
        [1, 2],
      ],
      compliance: new Float32Array([1e-3, 2e-3]),
      restLength: new Float32Array([0.5, 0.5]),
      dt: uniform(1 / 240, 'float'),
    });
    expect(Array.from(constraints.compliance.value.array as Float32Array)).toEqual([
      Math.fround(1e-3),
      Math.fround(2e-3),
    ]);
  });
});

describe('toTriangleMesh', () => {
  it('rejects geometry without positions', () => {
    expect(() => toTriangleMesh(new BufferGeometry())).toThrow(
      'toTriangleMesh: geometry has no position attribute',
    );
  });

  it('validates geometry indices and copies TriangleMesh input', () => {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
    geometry.setIndex([0, 1, 5]);
    expect(() => toTriangleMesh(geometry)).toThrow('TriangleMesh: index 5 is out of range');

    const mesh = { vertices: new Float32Array(9), indices: new Uint32Array([0, 1, 2]) };
    const copy = toTriangleMesh(mesh);
    expect(copy.vertices).not.toBe(mesh.vertices);
    expect(copy.indices).toEqual(mesh.indices);
  });
});

describe('createParticleMesh', () => {
  it('validates the range', () => {
    const particles = new ParticleSystem(renderer, 4, 0.1);
    expect(() => createParticleMesh(particles, { range: { start: 2, count: 3 } })).toThrow(
      'createParticleMesh: invalid particle range',
    );
  });
});
