import { describe, expect, it } from 'vitest';
import { Box3, Vector3 } from 'three';
import {
  FluidSystem,
  GasSystem,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
} from '../../../src/index.js';

// Heat sources and Boussinesq buoyancy.
//
// Four air particles, farther apart than the smoothing radius so the
// density solve leaves them alone; one sits inside a heat source. After
// one substep of length dt:
//
//   T_hot  = exp(−cooling · dt)           (set to 1, then cooled)
//   T̄      = T_hot / 4
//   Δv_y   = buoyancy · (T − T̄) · dt
//
// Lifting against the mean makes the vertical momentum change sum to zero.
// The mean is summed in fixed point (1/4096), hence the tolerances.

describe('gas heat sources', () => {
  it('heats air in the source, lifts it against the mean, and conserves momentum', async () => {
    const renderer = await createParticleRenderer();
    try {
      const h = 0.05;
      const particles = new ParticleSystem(renderer, 4, 0.01);
      particles.uploadParticles([
        { position: [0, 0, 0] },
        { position: [0.5, 0, 0] },
        { position: [0, 0.5, 0] },
        { position: [0, 0, 0.5] },
      ]);
      const air = new FluidSystem(particles, { smoothingRadius: h });
      const buoyancy = 3,
        cooling = 0.8;
      const gas = new GasSystem(air, {
        capacity: 1,
        heatSources: [{ position: new Vector3(0, 0, 0), radius: 0.1 }],
        buoyancy,
        cooling,
      });
      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        gravity: new Vector3(),
        materials: [gas, air],
      });
      const dt = 1 / 60;
      await loop.step(dt);

      const temperature = new Float32Array(
        await renderer.getArrayBufferAsync(gas.temperature!.value),
      );
      const velocities = new Float32Array(
        await renderer.getArrayBufferAsync(particles.velocities.value),
      );
      const hot = Math.exp(-cooling * dt);
      const mean = hot / 4;
      expect(temperature[0]).toBeCloseTo(hot, 5);
      for (const k of [1, 2, 3]) expect(temperature[k]).toBe(0);
      expect(velocities[1]).toBeCloseTo(buoyancy * (hot - mean) * dt, 4);
      expect(velocities[5]).toBeCloseTo(-buoyancy * mean * dt, 4);
      const net = [0, 1, 2, 3].reduce((sum, k) => sum + velocities[k * 4 + 1]!, 0);
      expect(Math.abs(net)).toBeLessThan(1e-4);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('retires tracers that leave the bounds', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 1, 0.01);
      particles.uploadParticles([{ position: [0, 0, 0] }]);
      const air = new FluidSystem(particles, { smoothingRadius: 0.05 });
      const gas = new GasSystem(air, {
        capacity: 2,
        lifetime: 100,
        bounds: new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1, 1)),
      });
      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        gravity: new Vector3(),
        materials: [gas, air],
      });
      expect(gas.emit([0, 0, 0])).toBe(true);
      expect(gas.emit([0, 2, 0])).toBe(true);
      await loop.step(1 / 60);

      const alive = new Uint32Array(await renderer.getArrayBufferAsync(gas.smokeAlive.value));
      expect([...alive]).toEqual([1, 0]);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });
});
