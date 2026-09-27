import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FluidSystem,
  GasSystem,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
} from '../../../src/index.js';

// Passive advection (Macklin et al. 2014 §7.2.1, eq. 28).
//
// One kinematic fluid particle at the origin with velocity (1, 0, 0).
// One smoke tracer co-located at the origin. After one substep of
// smoke advection, the SPH-weighted velocity at the tracer's position
// must match the fluid particle's velocity:
//
//   v(x_s) = Σ_j v_j · W(x_s − x_j) / Σ_j W(x_s − x_j)
//
// With a single fluid neighbour the weighted mean collapses to v_j
// directly: `v_smoke = v_fluid · W / W = v_fluid`. Tolerance: 1%.
//
// The fluid particle is kinematic (invMass = 0) so predict + advect
// leave its velocity unchanged across the substep, isolating the SPH
// interpolation from the integration loop.

describe('passive advection (eq. 28)', () => {
  it('co-located smoke samples fluid velocity within 1% via SPH', async () => {
    const renderer = await createParticleRenderer();
    try {
      const h = 0.05;
      const r = 0.01;

      const particles = new ParticleSystem(renderer, 1, r);
      particles.uploadParticles([{ position: [0, 0, 0], velocity: [1, 0, 0] }]);
      const fluid = new FluidSystem(particles, { smoothingRadius: h });
      // FluidSystem assigns its particles the fluid's mass; pin the particle
      // afterwards so predict/advect skip it and v stays (1, 0, 0).
      particles.setInvMass(fluid.range, 0);

      const gas = new GasSystem(fluid, {
        capacity: 1,
        lifetime: 100, // far exceeds the test runtime so the age gate is dormant
      });

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        gravity: new Vector3(0, 0, 0),
        // The gas is listed before its fluid.
        materials: [gas, fluid],
      });

      // The tracer spawns on the GPU at the start of the next step and is
      // advected during that step.
      expect(gas.emit([0, 0, 0])).toBe(true);
      await loop.step(1 / 60);

      const smokeVel = new Float32Array(
        await renderer.getArrayBufferAsync(gas.smokeVelocities.value),
      );
      const vx = smokeVel[0]!;
      const vy = smokeVel[1]!;
      const vz = smokeVel[2]!;

      console.log(
        `[passive-advection] smokeVelocity = (${vx.toFixed(6)}, ${vy.toFixed(6)}, ${vz.toFixed(6)})`,
      );
      expect(Math.abs(vx - 1)).toBeLessThan(0.01);
      expect(Math.abs(vy)).toBeLessThan(0.01);
      expect(Math.abs(vz)).toBeLessThan(0.01);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });
});
