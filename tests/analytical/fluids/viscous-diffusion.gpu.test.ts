import { expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
} from '../../../src/core/index.js';
import { FluidSystem, ViscositySolver } from '../../../src/fluids/index.js';

it('strong viscosity dissipates shear, preserves uniform drift, and remains bounded', async () => {
  const renderer = await createParticleRenderer();
  const particles = new ParticleSystem(renderer, 27, 0.02);
  const grid = new HashGrid(particles, { cellSize: 0.08 });
  try {
    particles.uploadParticles(
      Array.from({ length: 27 }, (_, i) => ({
        position: [
          (i % 3) * 0.04,
          (Math.floor(i / 3) % 3) * 0.04 + 1,
          Math.floor(i / 9) * 0.04,
        ] as const,
        velocity: [(i % 3) - 1, 0.5, 0] as const,
        invMass: 1,
        phase: 0,
      })),
    );
    const xpbd = createXpbdUniforms(1 / 60);
    const fluid = new FluidSystem({
      particles,
      hashGrid: grid,
      xpbd,
      h: 0.08,
      particleSpacing: 0.04,
      restDensity: 1000,
      compliance: 1e-4,
      fluidParticles: { start: 0, count: 27 },
    });
    const loop = new SimLoop(particles, {
      substeps: 1,
      iterations: 1,
      xpbd,
      hashGrid: grid,
      materials: [fluid],
    });
    loop.gravity.set(0, 0, 0);
    await loop.step(1 / 60);
    const before = new Float32Array(await renderer.getArrayBufferAsync(particles.velocities.value));
    const viscosity = new ViscositySolver({ fluid, viscosity: 3, iterations: 16 });
    await renderer.computeAsync([...viscosity.postAdvectKernels]);
    const after = new Float32Array(await renderer.getArrayBufferAsync(particles.velocities.value));
    let beforeShear = 0,
      afterShear = 0;
    for (let i = 0; i < 27; i++) {
      beforeShear += before[i * 4]! ** 2;
      afterShear += after[i * 4]! ** 2;
      expect(after[i * 4 + 1]).toBeCloseTo(0.5, 3);
      expect(Number.isFinite(after[i * 4])).toBe(true);
      expect(Math.abs(after[i * 4]!)).toBeLessThanOrEqual(1.01);
    }
    expect(afterShear).toBeLessThan(beforeShear * 0.5);
  } finally {
    particles.destroy();
    grid.destroy();
    renderer.dispose();
  }
});
