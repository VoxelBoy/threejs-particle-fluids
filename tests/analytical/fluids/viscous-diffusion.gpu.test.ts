import { expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FluidSystem,
  ParticleSystem,
  SimLoop,
  ViscositySolver,
  createParticleRenderer,
} from '../../../src/index.js';

it('strong viscosity dissipates shear, preserves uniform drift, and remains bounded', async () => {
  const renderer = await createParticleRenderer();
  const particles = new ParticleSystem(renderer, 27, 0.02);
  try {
    // 3×3×3 block with a shear profile vx ∈ {−1, 0, 1} across x and a
    // uniform drift vy = 0.5.
    particles.uploadParticles(
      Array.from({ length: 27 }, (_, i) => ({
        position: [
          (i % 3) * 0.04,
          (Math.floor(i / 3) % 3) * 0.04 + 1,
          Math.floor(i / 9) * 0.04,
        ] as const,
        velocity: [(i % 3) - 1, 0.5, 0] as const,
      })),
    );
    const fluid = new FluidSystem(particles, {
      smoothingRadius: 0.08,
      particleSpacing: 0.04,
      restDensity: 1000,
      compliance: 1e-4,
    });
    const loop = new SimLoop(particles, {
      substeps: 1,
      iterations: 1,
      gravity: new Vector3(0, 0, 0),
      materials: [fluid],
    });
    await loop.step(1 / 60);
    const before = new Float32Array(await renderer.getArrayBufferAsync(particles.velocities.value));

    // Run one implicit viscosity pass on the velocities the step produced:
    // the kernels a loop runs after the fluid when the solver is listed
    // after it in `materials`, built against the fluid's neighbor list.
    const viscosity = new ViscositySolver(fluid, { viscosity: 3, iterations: 16 });
    await renderer.computeAsync([...(viscosity.build().postSolve ?? [])]);
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
    loop.dispose();
  } finally {
    particles.dispose();
    renderer.dispose();
  }
});
