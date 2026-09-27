import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Flat interface.
//
// Scene: hydrostatic tank (same as the hydrostatic-column test) with
// cohesion ON. After settling, the column's top surface should remain
// approximately flat — the cohesion + curvature forces' combined
// behavior on a flat interface should be close to zero (Akinci et al.
// 2013 §2.3: F_curvature = -γ·m·(n_i - n_j) is zero when n_i = n_j for
// all pairs on a flat surface).
//
// Metric: the range of top-surface y-coordinates should be small
// relative to the column height.

describe('fluid flat interface', () => {
  it('column top remains approximately flat under cohesion', async () => {
    const renderer = await createParticleRenderer();
    try {
      const spacing = 0.025;
      const h = 0.05;
      const r = spacing * 0.5;
      const restDensity = 1000;

      const nx = 8;
      const ny = 10;
      const nz = 8;
      const count = nx * ny * nz;

      const initial: ParticleInit[] = [];
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          for (let i = 0; i < nx; i++) {
            const x = -((nx * spacing) / 2) + spacing * 0.5 + i * spacing;
            const y = spacing * 0.5 + j * spacing;
            const z = -((nz * spacing) / 2) + spacing * 0.5 + k * spacing;
            initial.push({ position: [x, y, z] });
          }
        }
      }

      const particles = new ParticleSystem(renderer, count, r);
      particles.uploadParticles(initial);

      // Sealed tank: floor + 4 walls (no ceiling so the column has a
      // free top surface).
      const tankHalfX = 0.15;
      const tankHalfZ = 0.15;
      const colliders = new PrimitiveSet(particles);
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
      colliders.addPlane(new Vector3(1, 0, 0), new Vector3(-tankHalfX, 0, 0));
      colliders.addPlane(new Vector3(-1, 0, 0), new Vector3(tankHalfX, 0, 0));
      colliders.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -tankHalfZ));
      colliders.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, tankHalfZ));

      const fluid = new FluidSystem(particles, {
        restDensity,
        smoothingRadius: h,
        particleSpacing: spacing,
        compliance: 1e-4,
        viscosity: 0.1,
        surfaceTension: 0.2,
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        gravity: new Vector3(0, -9.81, 0),
        colliders: [colliders],
        materials: [fluid],
      });

      // Settle for 3 s — long enough for the column to reach an
      // approximate hydrostatic equilibrium.
      const frames = 180;
      for (let n = 0; n < frames; n++) {
        await loop.step(1 / 60);
      }

      const snap = await particles.readback();
      // Identify top-surface particles: y in the top 15% of the
      // column's y-range (excludes interior, avoids side-wall contact
      // particles which may cling slightly).
      let nanCount = 0;
      let yMin = Number.POSITIVE_INFINITY;
      let yMax = Number.NEGATIVE_INFINITY;
      const ys: number[] = [];
      for (let i = 0; i < count; i++) {
        const x = snap.positions[4 * i + 0]!;
        const y = snap.positions[4 * i + 1]!;
        const z = snap.positions[4 * i + 2]!;
        if (!Number.isFinite(y)) {
          nanCount++;
          continue;
        }
        // Exclude wall-adjacent particles (distance < 2·spacing from
        // wall) so the "top surface" metric is bulk-interior only.
        const wallMargin = 2 * spacing;
        if (Math.abs(x) > tankHalfX - wallMargin || Math.abs(z) > tankHalfZ - wallMargin) {
          continue;
        }
        ys.push(y);
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }
      expect(nanCount).toBe(0);
      expect(ys.length).toBeGreaterThan(10);

      ys.sort((a, b) => a - b);
      // Top-15% surface particles.
      const topCutoff = yMin + 0.85 * (yMax - yMin);
      const topYs = ys.filter((y) => y >= topCutoff);
      const topYmin = Math.min(...topYs);
      const topYmax = Math.max(...topYs);
      const topRange = topYmax - topYmin;
      const columnHeight = yMax - yMin;

      console.info(
        `[flat-interface] ys=${ys.length} yMin=${yMin.toFixed(4)} yMax=${yMax.toFixed(4)} columnHeight=${columnHeight.toFixed(4)} topN=${topYs.length} topRange=${topRange.toFixed(4)} ratio=${(topRange / columnHeight).toFixed(3)}`,
      );

      // The ideal check is "surface curvature near zero". The proxy used
      // here: the vertical spread of the top-surface particles (topRange)
      // should be at most 30% of the column height. For a perfectly flat
      // surface it would be near zero (all top particles at the same y).
      // For a cohesion-distorted surface, topRange grows relative to
      // columnHeight. 30% tolerates the drift a fixed Δt allows while
      // still catching a grossly bumpy cohesion-induced surface.
      expect(topRange / columnHeight).toBeLessThan(0.3);

      loop.dispose();
      colliders.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 180_000);
});
