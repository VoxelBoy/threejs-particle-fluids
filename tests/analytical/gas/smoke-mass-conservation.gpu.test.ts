import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FluidSystem,
  GasSystem,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Smoke mass conservation under passive advection.
//
// 1000 smoke tracers distributed inside a cube, advected through a
// zero-velocity fluid grid (the simplest divergence-free field — every
// component zero, divergence trivially zero). Lifetime far exceeds the
// test duration so the lifetime gate is dormant.
//
// Invariant: tracer count stays constant and positions stay bounded
// within the initial support. No escapees to infinity.
//
// A non-trivial divergence-free field such as `v = (sin(y), cos(x), 0)`
// would need an SPH-approximated procedural field across a sampled fluid
// grid. The zero-field version still exercises the kernel's per-substep
// dispatch over 1000 simultaneous tracer slots, the lifetime gate, the
// dead-slot skip, and the bounded-position invariant.

describe('smoke mass conservation', () => {
  it('1000 smoke particles in zero-velocity field stay alive and bounded over 1 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const h = 0.05;
      const r = 0.01;
      const dt = 1 / 60;
      const T = 1.0;
      const numSteps = Math.round(T / dt);

      // 8x8x8 fluid grid filling [-0.175, 0.175]³ with zero velocity (kinematic).
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
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, fluidCount, r);
      particles.uploadParticles(initial);
      const fluid = new FluidSystem(particles, {
        smoothingRadius: h,
        particleSpacing: fluidSpacing,
      });
      // Kinematic: FluidSystem assigns its particles the fluid's mass, so
      // pin them afterwards. The fluid stays frozen and smoke advection
      // sees a zero-velocity field.
      particles.setInvMass(fluid.range, 0);

      const SMOKE_COUNT = 1000;
      const gas = new GasSystem(fluid, {
        capacity: SMOKE_COUNT,
        lifetime: 100, // > T → lifetime gate dormant for this test
      });

      // Distribute 1000 smoke tracers across a [-0.15, 0.15]³ cube
      // (slightly inside the fluid grid so every smoke point has fluid
      // neighbours — though with v_fluid = 0 there's no drift either way).
      const initBoxHalf = 0.15;
      // Deterministic Halton-(2,3,5) coords in [0,1] mapped to the box,
      // so the test isn't seeded by Math.random.
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
      let emitted = 0;
      for (let i = 0; i < SMOKE_COUNT; i++) {
        const px = (halton(i + 1, 2) * 2 - 1) * initBoxHalf;
        const py = (halton(i + 1, 3) * 2 - 1) * initBoxHalf;
        const pz = (halton(i + 1, 5) * 2 - 1) * initBoxHalf;
        if (gas.emit([px, py, pz])) emitted++;
      }
      expect(emitted).toBe(SMOKE_COUNT);

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        gravity: new Vector3(0, 0, 0),
        // The gas is listed before its fluid.
        materials: [gas, fluid],
      });

      // Tracers spawn at the start of the first step.
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
        const px = posBuf[s * 4 + 0]!;
        const py = posBuf[s * 4 + 1]!;
        const pz = posBuf[s * 4 + 2]!;
        if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) {
          nanCount++;
          continue;
        }
        const m = Math.max(Math.abs(px), Math.abs(py), Math.abs(pz));
        if (m > maxAbs) maxAbs = m;
        if (m > escapeBound) escapees++;
      }
      console.log(
        `[smoke-mass-conservation] T=${T}s alive=${aliveCount}/${SMOKE_COUNT} (CPU count ${gas.aliveCount}) nan=${nanCount} maxAbs=${maxAbs.toFixed(4)} escapees=${escapees}`,
      );

      expect(aliveCount).toBe(SMOKE_COUNT);
      expect(nanCount).toBe(0);
      expect(escapees).toBe(0);

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });
});
