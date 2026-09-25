import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';

// Phase 08 G1 — mass conservation.
//
// Particle count is fixed by `ParticleSystem.capacity`; the interesting
// test is "no particles escape the sealed box over 10 s of sim time".
// Escapes would indicate either (a) integration NaN followed by the
// floor-Y clamp pulling particles to `-∞`, (b) XPBD step over-correction
// tunnelling past an analytic plane, or (c) the solver producing
// velocities that exceed the plane-collider's capture response within
// one substep.
//
// Scene matches the hydrostatic / incompressibility tests: 768
// particles in a 0.3 × 0.3 m open-top tank, 10 s settle under gravity.

const TANK_HALF = 0.15; // m — plane colliders at ±0.15 in x and z.
const CEILING = 10; // m — "sky"; any y > 10 m is considered escape.

describe('Phase 08 — fluid mass conservation', () => {
  it('no particles escape the sealed tank over 10 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const spacing = 0.025;
      const h = 0.05;
      const r = spacing * 0.5;
      const restDensity = 1000;

      const nx = 8;
      const ny = 12;
      const nz = 8;
      const count = nx * ny * nz;

      const initial: ParticleInit[] = [];
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          for (let i = 0; i < nx; i++) {
            initial.push({
              position: [
                -((nx * spacing) / 2) + spacing * 0.5 + i * spacing,
                spacing * 0.5 + j * spacing,
                -((nz * spacing) / 2) + spacing * 0.5 + k * spacing,
              ],
              velocity: [0, 0, 0],
              invMass: 1,
              phase: 0,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, count, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles, { capacity: 5 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
      colliders.addPlane(new Vector3(1, 0, 0), new Vector3(-TANK_HALF, 0, 0));
      colliders.addPlane(new Vector3(-1, 0, 0), new Vector3(TANK_HALF, 0, 0));
      colliders.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -TANK_HALF));
      colliders.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, TANK_HALF));
      colliders.upload();

      const hashGrid = new HashGrid(particles, { cellSize: h });
      const xpbd = createXpbdUniforms(1 / 60);
      const fluid = new FluidSystem({
        particles,
        hashGrid,
        xpbd,
        restDensity,
        h,
        particleSpacing: spacing,
        compliance: 1e-4,
        fluidParticles: { start: 0, count },
        vorticity: { strength: 0 },
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        xpbd,
        hashGrid,
        colliders: { colliders },
        materials: [fluid],
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.81, 0);

      // 600 frames × 1/60 s = 10 s.
      for (let n = 0; n < 600; n++) await loop.step(1 / 60);

      const snap = await particles.readback();
      let escaped = 0;
      let nan = 0;
      let outsideX = 0;
      let outsideZ = 0;
      let below = 0;
      let above = 0;
      for (let i = 0; i < count; i++) {
        const x = snap.positions[4 * i + 0]!;
        const y = snap.positions[4 * i + 1]!;
        const z = snap.positions[4 * i + 2]!;
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          nan++;
          continue;
        }
        // Small tolerance for position-level solve residual: the plane
        // collider gates on `φ − r < 0`, so particles may sit up to `r`
        // past the plane surface in a single-step residual. Give 2·r of
        // slack for iteration lag.
        const tol = 2 * r;
        let thisEscaped = false;
        if (x < -TANK_HALF - tol || x > TANK_HALF + tol) {
          outsideX++;
          thisEscaped = true;
        }
        if (z < -TANK_HALF - tol || z > TANK_HALF + tol) {
          outsideZ++;
          thisEscaped = true;
        }
        if (y < -tol) {
          below++;
          thisEscaped = true;
        }
        if (y > CEILING) {
          above++;
          thisEscaped = true;
        }
        if (thisEscaped) escaped++;
      }

      // eslint-disable-next-line no-console
      console.info(
        `[mass-conservation] count=${count} nan=${nan} escaped=${escaped} outsideX=${outsideX} outsideZ=${outsideZ} below=${below} above=${above}`,
      );

      expect(nan).toBe(0);
      expect(escaped).toBe(0);

      particles.destroy();
      colliders.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 300_000);
});
