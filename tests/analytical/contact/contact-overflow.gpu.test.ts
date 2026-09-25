import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 05 G3 — contact buffer overflow diagnostic (BLOCKING, plan §Exit).
//
// Plan §Validation: "Contact buffer overflow diagnostic. Force maxContacts
// = small; verify the overflow path logs a warning and does not crash."
//
// Our overflow path (see `generate.ts`): `atomicAdd(counter, 1)` keeps
// counting past `maxContacts`, but the `pairs` write is gated by
// `If(slot < maxContacts)`. CPU code reads `counter` after a step; any value
// above `maxContacts` means the buffer overflowed and contacts were dropped.
// This is not logged from inside the kernel (no WebGPU-compute logging
// facility); `SimLoop` consumers are expected to readback `counter` when
// they suspect overflow and surface the diagnostic themselves.
//
// The contract this test pins down:
//   1. Pipeline does not crash when overflow occurs.
//   2. The atomic counter reports a value `>= maxContacts` when the emitter
//      would have emitted more than the cap.
//   3. The subsequent coloring + solve pipeline still produces a physically
//      valid state (the capped `maxContacts` pairs are solved; the dropped
//      ones are missed this substep, caught next substep when the hash grid
//      re-emits — paper §9 expanded radius gives the slack).

describe('Phase 05 — contact: overflow diagnostic', () => {
  it('counter exceeds maxContacts under saturation; pipeline does not crash', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const twoR = 2 * r;

      // Densely packed 5×5×5 = 125 particles in an `r`-spaced lattice so
      // every particle touches many neighbors. Expected pair count is
      // O(125 · ~8 / 2) ~ 500; we cap at `maxContacts = 16` to force a
      // heavy overflow on every substep.
      const spacing = twoR * 1.01;
      const initial: ParticleInit[] = [];
      for (let ix = 0; ix < 5; ix++) {
        for (let iy = 0; iy < 5; iy++) {
          for (let iz = 0; iz < 5; iz++) {
            initial.push({
              position: [(ix - 2) * spacing, 0.5 + iy * spacing, (iz - 2) * spacing],
              velocity: [0, 0, 0],
              invMass: 1,
              phase: 0,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const hashGrid = new HashGrid(particles, {
        cellSize: twoR * 1.1,
      });

      const artificialCap = 16;
      const loop = new SimLoop(particles, {
        substeps: 2,
        iterations: 2,
        contact: {
          hashGrid,
          maxContacts: artificialCap,
          friction: { muS: 0.4, muK: 0.3 },
        },
      });
      loop.gravity.set(0, -9.81, 0);

      // One frame is enough; the initial lattice is saturated at `t=0`.
      await loop.step(1 / 60);
      const counter = await loop.contacts!.readbackCount();

      // eslint-disable-next-line no-console
      console.info(
        `[contact-overflow] maxContacts=${artificialCap} ` +
          `finalCounter=${counter} overflowed=${counter > artificialCap}`,
      );

      // (1) No crash — reaching this assertion means `step` completed.
      // (2) Counter exceeded the cap.
      expect(counter).toBeGreaterThan(artificialCap);

      // Run a few more frames to verify the pipeline doesn't crash under
      // sustained overflow.
      for (let n = 0; n < 10; n++) await loop.step(1 / 60);

      // (3) Sanity — final positions are finite (no NaN/Inf from a
      // cascading coloring failure).
      const snap = await particles.readback();
      for (let i = 0; i < initial.length; i++) {
        const x = snap.positions[i * 4]!;
        const y = snap.positions[i * 4 + 1]!;
        const z = snap.positions[i * 4 + 2]!;
        expect(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)).toBe(true);
      }

      particles.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
