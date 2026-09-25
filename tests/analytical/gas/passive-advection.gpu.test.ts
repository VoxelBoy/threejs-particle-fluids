import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { GasSystem } from '../../../src/gas/index.js';

// Phase 16 G1 — Macklin 2014 §7.2.1 eq. 28 passive advection sanity.
//
// One kinematic fluid particle at the origin with velocity (1, 0, 0).
// One smoke particle co-located at the origin. After one substep of
// smoke advection, the SPH-weighted velocity at the smoke's position
// must approach the fluid particle's velocity:
//
//   v(x_s) = Σ_j v_j · W(x_s − x_j) / Σ_j W(x_s − x_j)
//
// With a single fluid neighbour at distance 0, the weighted mean
// collapses to v_j directly: `v_smoke = v_fluid · W(0) / W(0) = v_fluid`.
// Plan tolerance: 1%.
//
// The fluid particle is kinematic (invMass = 0) so predict + advect
// leave its velocity unchanged across the substep, isolating the SPH
// interpolation from the integration loop.

describe('Phase 16 — passive-advection correctness (eq. 28)', () => {
  it('co-located smoke samples fluid velocity within 1% via SPH', async () => {
    const renderer = await createParticleRenderer();
    try {
      const h = 0.05;
      const r = 0.01;

      // One kinematic fluid particle at origin.
      const initial: ParticleInit[] = [
        {
          position: [0, 0, 0],
          velocity: [1, 0, 0],
          invMass: 0, // kinematic — predict/advect skip; v stays (1,0,0).
          phase: 0,
        },
      ];

      const particles = new ParticleSystem(renderer, 1, r);
      particles.uploadParticles(initial);
      const hashGrid = new HashGrid(particles, { cellSize: h });
      const xpbd = createXpbdUniforms(1 / 60);

      const gas = new GasSystem({
        capacity: 1,
        fluidParticles: particles,
        fluidRange: { start: 0, count: 1 },
        hashGrid,
        h,
        xpbd,
        lifetime: 100, // far exceed test runtime so age gate is dormant
      });
      gas.emit([0, 0, 0], [0, 0, 0], 1);

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        xpbd,
        hashGrid,
        materials: [gas],
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, 0, 0);

      await loop.step(1 / 60);

      const smokeVel = new Float32Array(
        await renderer.getArrayBufferAsync(gas.smokeVelocities.value),
      );
      const vx = smokeVel[0];
      const vy = smokeVel[1];
      const vz = smokeVel[2];

      // eslint-disable-next-line no-console
      console.log(
        `[passive-advection] smokeVelocity = (${vx!.toFixed(6)}, ${vy!.toFixed(6)}, ${vz!.toFixed(6)})`,
      );
      expect(Math.abs(vx! - 1)).toBeLessThan(0.01);
      expect(Math.abs(vy!)).toBeLessThan(0.01);
      expect(Math.abs(vz!)).toBeLessThan(0.01);
    } finally {
      renderer.dispose();
    }
  });
});
