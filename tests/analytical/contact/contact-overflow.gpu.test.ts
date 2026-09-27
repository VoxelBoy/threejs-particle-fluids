import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Contact buffer overflow: forcing a tiny `maxContacts` must be detectable
// and must not crash the pipeline.
//
// The overflow path (see `generate.ts`): `atomicAdd(counter, 1)` keeps
// counting past `maxContacts`, but the record write is gated by
// `If(slot < maxContacts)`. After a step, a counter above `maxContacts`
// means the buffer overflowed and contacts were dropped. Kernels cannot log,
// so callers read the counter back (`ContactBuffer.readbackCount`) when they
// suspect overflow and surface the diagnostic themselves.
//
// The contract this test pins down:
//   1. The pipeline does not crash when overflow occurs.
//   2. The counter reports a value `> maxContacts` when the generator would
//      have emitted more than the cap.
//   3. The solve still produces a valid (finite) state: the first
//      `maxContacts` pairs are solved, and the dropped ones are missed this
//      substep but can be caught in a later one, since candidates are
//      gathered within an expanded radius (Macklin 2014 §9).

describe('contact: buffer overflow', () => {
  it('counter exceeds maxContacts under saturation; pipeline does not crash', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const twoR = 2 * r;

      // Densely packed 5×5×5 = 125 particles, almost touching, so every
      // particle is a contact candidate for several neighbors. The expected
      // pair count is in the hundreds; capping at `maxContacts = 16` forces
      // a heavy overflow on every substep.
      const spacing = twoR * 1.01;
      const initial: ParticleInit[] = [];
      for (let ix = 0; ix < 5; ix++) {
        for (let iy = 0; iy < 5; iy++) {
          for (let iz = 0; iz < 5; iz++) {
            initial.push({
              position: [(ix - 2) * spacing, 0.5 + iy * spacing, (iz - 2) * spacing],
              velocity: [0, 0, 0],
              invMass: 1,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const artificialCap = 16;
      const loop = new SimLoop(particles, {
        substeps: 2,
        iterations: 2,
        contact: { maxContacts: artificialCap, muS: 0.4, muK: 0.3 },
      });
      loop.gravity.set(0, -9.81, 0);

      // One frame is enough; the initial lattice is saturated at t = 0.
      await loop.step(1 / 60);
      const counter = await loop.contacts!.readbackCount();

      console.info(
        `[contact-overflow] maxContacts=${artificialCap} ` +
          `finalCounter=${counter} overflowed=${counter > artificialCap}`,
      );

      // (1) No crash — reaching this assertion means `step` completed.
      // (2) The counter exceeded the cap.
      expect(counter).toBeGreaterThan(artificialCap);

      // Run a few more frames: sustained overflow must not crash either.
      for (let n = 0; n < 10; n++) await loop.step(1 / 60);

      // (3) Final positions are finite (no NaN/Inf from the dropped pairs).
      const snap = await particles.readback();
      for (let i = 0; i < initial.length; i++) {
        const x = snap.positions[i * 4]!;
        const y = snap.positions[i * 4 + 1]!;
        const z = snap.positions[i * 4 + 2]!;
        expect(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)).toBe(true);
      }

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
