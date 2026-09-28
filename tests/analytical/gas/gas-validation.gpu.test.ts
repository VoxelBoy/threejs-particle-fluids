import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import { instancedArray } from 'three/tsl';
import {
  FluidSystem,
  GasSpriteRenderer,
  GasSystem,
  GasVolumeRenderer,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type SmokeTracers,
} from '../../../src/index.js';
import { fixedPointScale } from '../../../src/gas/heat.js';

// Option and setter validation, loop wiring checks, and disposal for the gas
// system and its renderers.

const source = { position: new Vector3(), radius: 0.1 };

async function setup() {
  const renderer = await createParticleRenderer();
  const particles = new ParticleSystem(renderer, 2, 0.01);
  particles.uploadParticles([{ position: [0, 0, 0] }, { position: [0.5, 0, 0] }]);
  const air = new FluidSystem(particles, { smoothingRadius: 0.05 });
  return { renderer, particles, air };
}

describe('GasSystem validation', () => {
  it('rejects buoyancy and cooling without heat sources, and bad setter values', async () => {
    const { renderer, air } = await setup();
    try {
      expect(() => new GasSystem(air, { capacity: 1, buoyancy: 2 })).toThrow(
        'GasSystem: buoyancy and cooling need heatSources',
      );
      expect(() => new GasSystem(air, { capacity: 1, cooling: 1, heatSources: [] })).toThrow(
        'GasSystem: buoyancy and cooling need heatSources',
      );
      const gas = new GasSystem(air, { capacity: 1, heatSources: [source] });
      expect(() => (gas.cooling = -1)).toThrow('GasSystem: cooling must be ≥ 0');
      expect(() => (gas.buoyancy = NaN)).toThrow('GasSystem: buoyancy must be finite');
      expect(gas.cooling).toBe(0.8);
      expect(gas.buoyancy).toBe(3);
      gas.cooling = 0.5;
      expect(gas.cooling).toBe(0.5);
    } finally {
      renderer.dispose();
    }
  });

  it('keeps the fixed-point temperature sum below 2³²', () => {
    expect(fixedPointScale(1000)).toBe(4096);
    expect(fixedPointScale(1_048_575)).toBe(4096);
    for (const count of [1_048_577, 5_000_000, 50_000_000]) {
      expect(fixedPointScale(count)).toBeLessThan(4096);
      expect(fixedPointScale(count) * count).toBeLessThanOrEqual(0xffffffff);
    }
  });
});

describe('GasSystem loop wiring', () => {
  it('can be rebuilt in a second loop, but not listed after its fluid', async () => {
    const { renderer, particles, air } = await setup();
    try {
      const gas = new GasSystem(air, { capacity: 4, heatSources: [source] });
      const first = new SimLoop(particles, { substeps: 1, materials: [gas, air] });
      await first.step(1 / 60);
      first.dispose();
      const second = new SimLoop(particles, { substeps: 1, materials: [gas, air] });
      await second.step(1 / 60);
      second.dispose();
      expect(() => new SimLoop(particles, { materials: [air, gas] })).toThrow(
        'GasSystem: list it before its FluidSystem in `materials`',
      );
    } finally {
      renderer.dispose();
    }
  });

  it('throws on the first step when its fluid is not in the loop', async () => {
    const { renderer, particles, air } = await setup();
    try {
      const gas = new GasSystem(air, { capacity: 4 });
      const loop = new SimLoop(particles, { materials: [gas] });
      await expect(loop.step(1 / 60)).rejects.toThrow(
        "GasSystem: its FluidSystem must be in the same SimLoop's `materials`",
      );
      // The fluid built by an earlier loop doesn't count either.
      new SimLoop(particles, { materials: [air] });
      const stale = new SimLoop(particles, { materials: [gas] });
      await expect(stale.step(1 / 60)).rejects.toThrow('same SimLoop');
    } finally {
      renderer.dispose();
    }
  });

  it('frees its storage buffers on dispose', async () => {
    const { renderer, particles, air } = await setup();
    try {
      const gas = new GasSystem(air, { capacity: 64, heatSources: [source] });
      const loop = new SimLoop(particles, { substeps: 1, materials: [gas, air] });
      gas.emit([0, 0, 0]);
      await loop.step(1 / 60);
      loop.dispose();
      const before = renderer.info.memory.storageAttributes;
      gas.dispose();
      // Five tracer buffers, the temperatures, and the temperature sum.
      expect(before - renderer.info.memory.storageAttributes).toBe(7);
      expect(() => gas.emit([0, 0, 0])).toThrow('GasSystem: already disposed');
      gas.dispose();
    } finally {
      renderer.dispose();
    }
  });
});

describe('gas renderer validation', () => {
  const tracers = (): SmokeTracers => ({
    capacity: 1,
    lifetime: 1,
    smokePositions: instancedArray(1, 'vec4'),
    smokeAge: instancedArray(1, 'float'),
    smokeAlive: instancedArray(1, 'uint'),
  });

  it('GasSpriteRenderer validates size in the constructor and setter', () => {
    expect(() => new GasSpriteRenderer(tracers(), { size: 0 })).toThrow(
      'GasSpriteRenderer: size must be positive',
    );
    const sprites = new GasSpriteRenderer(tracers());
    expect(() => (sprites.size = -1)).toThrow('GasSpriteRenderer: size must be positive');
    expect(() => (sprites.size = Infinity)).toThrow('GasSpriteRenderer: size must be positive');
    sprites.size = 0.2;
    expect(sprites.size).toBe(0.2);
    sprites.dispose();
  });

  it('GasVolumeRenderer names itself in every option error', async () => {
    const renderer = await createParticleRenderer();
    try {
      const make = (options: object) =>
        new GasVolumeRenderer(tracers(), {
          renderer,
          min: new Vector3(0, 0, 0),
          max: new Vector3(1, 1, 1),
          ...options,
        });
      expect(() => make({ resolution: [3, 8, 8] })).toThrow(
        'GasVolumeRenderer: resolution must be integers in 4–128',
      );
      expect(() => make({ steps: 20.5 })).toThrow(
        'GasVolumeRenderer: steps must be an integer in 8–128',
      );
      expect(() => make({ steps: 4 })).toThrow('GasVolumeRenderer: steps must be');
      expect(() => make({ max: new Vector3(1, 0, 1) })).toThrow(
        'GasVolumeRenderer: max must exceed min on every axis',
      );
      expect(() => make({ edgeFade: { sides: 0.6 } })).toThrow(
        'GasVolumeRenderer: edgeFade.sides must be in [0, 0.5]',
      );
      expect(() => make({ edgeFade: { top: -0.1 } })).toThrow(
        'GasVolumeRenderer: edgeFade.top must be in [0, 1]',
      );
      // Zero-width fades are skipped rather than compiled as smoothstep(0, 0, x).
      const unfaded = make({ edgeFade: { sides: 0, bottom: 0, top: 0 } });
      await unfaded.update();
      unfaded.dispose();
    } finally {
      renderer.dispose();
    }
  });
});
