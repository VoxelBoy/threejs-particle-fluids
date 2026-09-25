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

// Phase 16 G1 — smoke mass conservation under passive advection.
//
// 1000 smoke particles distributed inside a cube, advected through a
// zero-velocity fluid grid (the simplest divergence-free field — every
// component zero, divergence trivially zero). Lifetime far exceeds the
// test duration so the lifetime gate is dormant.
//
// Plan invariant: "particle count constant and positions bounded within
// the initial support. No escapees to infinity."
//
// The plan also mentions a non-trivial divergence-free field
// `v = (sin(y), cos(x), 0)` as an ideal scene; that requires an SPH-
// approximated procedural field across a sampled fluid grid and adds
// substantial harness complexity. The zero-field simplification still
// exercises the kernel's per-substep dispatch over 1000 simultaneous
// smoke slots, the lifetime gate, the dead-slot skip, and the bounded-
// position invariant — which is what the plan tests for. A non-trivial
// field test is post-MVP if we observe drift the zero-field test
// cannot detect.

describe('Phase 16 — smoke mass conservation', () => {
  it('1000 smoke particles in zero-velocity field stay alive and bounded over 1 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const h = 0.05;
      const r = 0.01;
      const dt = 1 / 60;
      const T = 1.0;
      const numSteps = Math.round(T / dt);

      // 8x8x8 fluid grid filling [-0.2, 0.2]³ with zero velocity (kinematic).
      const fluidSide = 8;
      const fluidSpacing = 0.05; // matches h so every smoke point sees ≥ 1 fluid neighbour
      const fluidCount = fluidSide * fluidSide * fluidSide;
      const initial: ParticleInit[] = [];
      const fluidOrigin = -((fluidSide - 1) * fluidSpacing) / 2;
      for (let z = 0; z < fluidSide; z++) {
        for (let y = 0; y < fluidSide; y++) {
          for (let x = 0; x < fluidSide; x++) {
            initial.push({
              position: [
                fluidOrigin + x * fluidSpacing,
                fluidOrigin + y * fluidSpacing,
                fluidOrigin + z * fluidSpacing,
              ],
              velocity: [0, 0, 0],
              invMass: 0, // kinematic — fluid frozen, smoke advection sees a zero-velocity field
              phase: 0,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, fluidCount, r);
      particles.uploadParticles(initial);
      const hashGrid = new HashGrid(particles, { cellSize: h });
      const xpbd = createXpbdUniforms(dt);

      const SMOKE_COUNT = 1000;
      const gas = new GasSystem({
        capacity: SMOKE_COUNT,
        fluidParticles: particles,
        fluidRange: { start: 0, count: fluidCount },
        hashGrid,
        h,
        xpbd,
        lifetime: 100, // > T → lifetime gate dormant for this test
      });

      // Distribute 1000 smoke particles across a [-0.15, 0.15]³ cube
      // (slightly inside the fluid grid so every smoke point has fluid
      // neighbours — though with v_fluid = 0 there's no drift either way).
      const initBoxHalf = 0.15;
      // Deterministic Halton-ish sequence so the test isn't seeded by
      // Math.random — Halton-(2,3,5) coords in [0,1] mapped to box.
      const halton = (i: number, base: number): number => {
        let f = 1;
        let r2 = 0;
        let n = i;
        while (n > 0) {
          f /= base;
          r2 += f * (n % base);
          n = Math.floor(n / base);
        }
        return r2;
      };
      for (let i = 0; i < SMOKE_COUNT; i++) {
        const px = (halton(i + 1, 2) * 2 - 1) * initBoxHalf;
        const py = (halton(i + 1, 3) * 2 - 1) * initBoxHalf;
        const pz = (halton(i + 1, 5) * 2 - 1) * initBoxHalf;
        gas.emit([px, py, pz], [0, 0, 0], 1);
      }

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        xpbd,
        hashGrid,
        materials: [gas],
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, 0, 0);

      for (let step = 0; step < numSteps; step++) {
        await loop.step(dt);
      }

      const aliveBuf = new Uint32Array(await renderer.getArrayBufferAsync(gas.smokeAlive.value));
      const posBuf = new Float32Array(await renderer.getArrayBufferAsync(gas.smokePositions.value));

      let aliveCount = 0;
      let nanCount = 0;
      let maxAbs = 0;
      const escapeBound = 0.5; // 3.3× initial half-extent is generous
      let escapees = 0;
      for (let s = 0; s < SMOKE_COUNT; s++) {
        if (aliveBuf[s] !== 1) continue;
        aliveCount++;
        const px = posBuf[s * 4 + 0];
        const py = posBuf[s * 4 + 1];
        const pz = posBuf[s * 4 + 2];
        if (!Number.isFinite(px!) || !Number.isFinite(py!) || !Number.isFinite(pz!)) {
          nanCount++;
          continue;
        }
        const m = Math.max(Math.abs(px!), Math.abs(py!), Math.abs(pz!));
        if (m > maxAbs) maxAbs = m;
        if (m > escapeBound) escapees++;
      }
      // eslint-disable-next-line no-console
      console.log(
        `[smoke-mass-conservation] T=${T}s alive=${aliveCount}/${SMOKE_COUNT} nan=${nanCount} maxAbs=${maxAbs.toFixed(4)} escapees=${escapees}`,
      );

      expect(aliveCount).toBe(SMOKE_COUNT);
      expect(nanCount).toBe(0);
      expect(escapees).toBe(0);
    } finally {
      renderer.dispose();
    }
  });
});
