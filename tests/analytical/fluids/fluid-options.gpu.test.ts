import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FluidSystem,
  ParticleSystem,
  SimLoop,
  ViscositySolver,
  createParticleRenderer,
} from '../../../src/index.js';

// Settings changed after construction are checked like the constructor's.

describe('FluidSystem and ViscositySolver settings', () => {
  it('reject invalid values after construction', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 100, 0.01);
      const fluid = new FluidSystem(particles, {
        range: { start: 0, count: 40 },
        viscosity: 0.01,
        vorticity: 0,
        surfaceTension: 0,
        adhesion: 0,
      });
      for (const key of ['viscosity', 'vorticity', 'surfaceTension', 'adhesion'] as const) {
        expect(() => (fluid[key] = NaN)).toThrow(`FluidSystem: ${key} must be finite`);
        fluid[key] = 0.2;
        expect(fluid[key]).toBeCloseTo(0.2);
      }

      const thick = new ViscositySolver(fluid, { viscosity: 1 });
      expect(() => (thick.viscosity = -1)).toThrow('ViscositySolver: viscosity must be ≥ 0');
      expect(() => (thick.viscosity = Infinity)).toThrow('ViscositySolver: viscosity must be ≥ 0');
      thick.viscosity = 20;
      expect(thick.viscosity).toBe(20);

      fluid.addBoundary({ start: 40, count: 30 });
      expect(() => fluid.addBoundary({ start: 60, count: 20 })).toThrow(
        'FluidSystem.addBoundary: boundaries cannot overlap each other',
      );
      fluid.addBoundary({ start: 70, count: 30 });
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('reports a truncated neighbor list', async () => {
    const renderer = await createParticleRenderer();
    try {
      // 5³ particles a quarter spacing apart: each has ~124 within h. One
      // substep, so the rebuild that is read sees them still packed.
      const packed = async (gap: number): Promise<boolean> => {
        const particles = new ParticleSystem(renderer, 125, 0.01);
        const points = Array.from({ length: 125 }, (_, i) => ({
          position: [i % 5, Math.floor(i / 5) % 5, Math.floor(i / 25)].map((k) => k * gap) as [
            number,
            number,
            number,
          ],
        }));
        particles.uploadParticles(points);
        const fluid = new FluidSystem(particles);
        await expect(fluid.readbackOverflow()).rejects.toThrow('add the fluid to a SimLoop first');
        const loop = new SimLoop(particles, {
          materials: [fluid],
          gravity: new Vector3(),
          substeps: 1,
        });
        await loop.step(1 / 240);
        const overflow = await fluid.readbackOverflow();
        loop.dispose();
        particles.dispose();
        return overflow;
      };
      expect(await packed(0.005)).toBe(true);
      expect(await packed(0.02)).toBe(false);
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
