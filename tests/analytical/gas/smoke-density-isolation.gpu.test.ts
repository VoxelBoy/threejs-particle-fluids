import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FluidSystem,
  GasSystem,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type Material,
  type ParticleInit,
} from '../../../src/index.js';

// Smoke does not interact with fluid density.
//
// The architectural guarantee: smoke tracers live in gas-owned buffers,
// never in the shared `ParticleSystem`. The hash grid hashes every slot
// in `ParticleSystem` up to its capacity — tracers aren't there to be
// hashed, so they cannot appear as a fluid neighbour during the density
// walk, the boundary-volume walk, the contact pipeline, or any
// fluid-kernel pair iteration.
//
// This test verifies that guarantee with **paired runs**: run an
// identical fluid scene twice, once with a `GasSystem` and 100 smoke
// tracers alongside, once with no gas at all. The fluid density buffers
// from the two runs must match within FP noise. If the smoke ever leaks
// into a fluid walk, the density would diverge.

describe('smoke does not interact with fluid density', () => {
  it('fluid density buffer matches between with-smoke and without-smoke runs', async () => {
    const renderer = await createParticleRenderer();
    try {
      const restDensity = 1000;
      const spacing = 0.025;
      const h = 0.05;
      const r = spacing * 0.5;
      const dt = 1 / 240;
      const substeps = 4;
      const iterations = 2;
      const numFrames = 5;

      // 4×4×4 fluid cube centred at origin — same setup for both runs.
      const side = 4;
      const fluidCount = side * side * side;
      const buildInitial = (): ParticleInit[] => {
        const out: ParticleInit[] = [];
        const offset = -((side - 1) * spacing) / 2;
        for (let z = 0; z < side; z++) {
          for (let y = 0; y < side; y++) {
            for (let x = 0; x < side; x++) {
              out.push({
                position: [offset + x * spacing, offset + y * spacing, offset + z * spacing],
              });
            }
          }
        }
        return out;
      };

      const runScene = async (withSmoke: boolean): Promise<Float32Array> => {
        const particles = new ParticleSystem(renderer, fluidCount, r);
        particles.uploadParticles(buildInitial());

        const fluid = new FluidSystem(particles, {
          restDensity,
          smoothingRadius: h,
          particleSpacing: spacing,
          compliance: 1e-4,
        });

        const materials: Material[] = [fluid];
        if (withSmoke) {
          const gas = new GasSystem(fluid, { capacity: 100, lifetime: 100 });
          // Distribute 100 smoke tracers inside the fluid cube.
          const halton = (i: number, base: number): number => {
            let f = 1;
            let v = 0;
            let n = i;
            while (n > 0) {
              f /= base;
              v += f * (n % base);
              n = Math.floor(n / base);
            }
            return v;
          };
          const half = ((side - 1) * spacing) / 2;
          for (let i = 0; i < 100; i++) {
            const px = (halton(i + 1, 2) * 2 - 1) * half * 0.8;
            const py = (halton(i + 1, 3) * 2 - 1) * half * 0.8;
            const pz = (halton(i + 1, 5) * 2 - 1) * half * 0.8;
            gas.emit([px, py, pz]);
          }
          // The gas is listed before its fluid.
          materials.unshift(gas);
        }

        const loop = new SimLoop(particles, {
          substeps,
          iterations,
          gravity: new Vector3(0, -9.81, 0),
          materials,
        });

        for (let f = 0; f < numFrames; f++) {
          await loop.step(dt);
        }

        const density = new Float32Array(await renderer.getArrayBufferAsync(fluid.density.value));
        loop.dispose();
        particles.dispose();
        return density;
      };

      const densityWithoutSmoke = await runScene(false);
      const densityWithSmoke = await runScene(true);

      let maxAbsDiff = 0;
      for (let i = 0; i < fluidCount; i++) {
        const a = densityWithoutSmoke[i] ?? 0;
        const b = densityWithSmoke[i] ?? 0;
        const d = Math.abs(a - b);
        if (d > maxAbsDiff) maxAbsDiff = d;
      }

      console.log(
        `[smoke-density-isolation] frames=${numFrames} fluidCount=${fluidCount} ` +
          `maxAbsDiff(density)=${maxAbsDiff.toExponential(3)} kg/m³`,
      );

      // Tolerance 1e-5 ("differs by < 1e-5"). Hash-grid scatter is
      // bit-deterministic in single-pass; per-pair walks are reductions
      // over a fixed neighbour set, so identical inputs produce identical
      // outputs even across SimLoop construction boundaries. Any
      // non-zero result here proves smoke leaked into a fluid walk.
      expect(maxAbsDiff).toBeLessThan(1e-5);
    } finally {
      renderer.dispose();
    }
  });
});
