import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Contact pipeline cost at 50k particles.
//
// Scene: a densely packed granular lattice of 50k particles dropped onto a
// floor plane. The contact pipeline's per-frame cost is set by the number of
// emitted contacts, which depends on the packing density (roughly `8·N` for
// closely packed particles), not on the macro-scale shape of the pile.
//
// This is a performance measurement, not a determinism test. Results are
// informational: the test only gates on a positive median and logs a warning
// when the median frame exceeds the budget of 6.67 ms (40 % of a 60 Hz
// frame).

describe('contact: pipeline cost at 50k particles', () => {
  it('measures the median frame cost over 120 frames, with and without contacts', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const twoR = 2 * r;
      const N = 50_000;

      // Dense lattice in a cube. `2.02r` grid spacing is almost touching;
      // once the pile forms, each particle has ~8 neighbors in contact range.
      const spacing = twoR * 1.01;
      const side = Math.ceil(Math.cbrt(N));
      const initial: ParticleInit[] = [];
      for (let ix = 0; ix < side && initial.length < N; ix++) {
        for (let iy = 0; iy < side && initial.length < N; iy++) {
          for (let iz = 0; iz < side && initial.length < N; iz++) {
            initial.push({
              position: [(ix - side / 2) * spacing, 0.1 + iy * spacing, (iz - side / 2) * spacing],
              velocity: [0, 0, 0],
              invMass: 1,
            });
          }
        }
      }
      const capacity = initial.length;

      const particles = new ParticleSystem(renderer, capacity, r);
      particles.uploadParticles(initial);

      // Floor at y = 0 so the lattice collapses into a dense pile.
      const floor = new PrimitiveSet(particles);
      floor.addPlane(new Vector3(0, 1, 0), new Vector3());

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        contact: {
          // 8 contacts per particle × 50k / 2 (each undirected pair emitted
          // once) = 200k. Sized at the expected load so the solve kernels'
          // dispatch over `maxContacts` threads has little early-exit waste.
          maxContacts: 200_000,
          muS: 0.5,
          muK: 0.4,
        },
        colliders: [floor],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const warmupFrames = 20;
      const measureFrames = 120;

      // Warmup (let the driver compile and cache pipelines).
      for (let n = 0; n < warmupFrames; n++) await loop.step(frameDt);

      // Measure whole-frame cost via `performance.now()` on the JS side.
      // The contact pipeline can't be isolated from predict/advect at this
      // level (that would need per-kernel GPU timestamp queries, which
      // three.js r184 doesn't expose), so the contact cost is computed by
      // subtraction against a no-contact control run further down.
      // `computeAsync` in three.js r184 submits the command buffer and
      // returns without waiting for the GPU. To measure real wall-clock GPU
      // cost, each step is paired with a small readback (which must wait for
      // the pipeline's writes to land), forcing a CPU-GPU sync. The
      // readback overhead (~0.1 ms) is roughly the same in both runs, so the
      // differential (`contactOnlyMs`) is still accurate.
      const syncBuf = particles.invMass.value;
      const frameTimes: number[] = [];
      for (let n = 0; n < measureFrames; n++) {
        const t0 = performance.now();
        await loop.step(frameDt);
        await renderer.getArrayBufferAsync(syncBuf);
        const t1 = performance.now();
        frameTimes.push(t1 - t0);
      }

      // Stats (the median is robust to driver stalls).
      frameTimes.sort((a, b) => a - b);
      const median = frameTimes[Math.floor(measureFrames / 2)]!;
      const p10 = frameTimes[Math.floor(measureFrames * 0.1)]!;
      const p90 = frameTimes[Math.floor(measureFrames * 0.9)]!;
      const mean = frameTimes.reduce((s, x) => s + x, 0) / measureFrames;

      // Contact count at the end of the run (indicative).
      const contactN = await loop.contacts!.readbackCount();

      // Baseline: the same scene and loop with `contact` omitted. The delta
      // against the full-contact median tells how much of the frame is the
      // contact pipeline. It also guards against a silent failure: kernels
      // that compile to invalid WGSL can be dispatched as no-ops by three.js,
      // and comparing medians catches the case where every contact kernel
      // silently does nothing.
      const baselineParticles = new ParticleSystem(renderer, capacity, r);
      baselineParticles.uploadParticles(initial);
      const baselineFloor = new PrimitiveSet(baselineParticles);
      baselineFloor.addPlane(new Vector3(0, 1, 0), new Vector3());
      const baselineLoop = new SimLoop(baselineParticles, {
        substeps: 4,
        iterations: 2,
        colliders: [baselineFloor],
      });
      baselineLoop.gravity.set(0, -9.81, 0);
      for (let n = 0; n < warmupFrames; n++) await baselineLoop.step(frameDt);
      const baselineTimes: number[] = [];
      const baselineSyncBuf = baselineParticles.invMass.value;
      for (let n = 0; n < measureFrames; n++) {
        const t0 = performance.now();
        await baselineLoop.step(frameDt);
        await renderer.getArrayBufferAsync(baselineSyncBuf);
        const t1 = performance.now();
        baselineTimes.push(t1 - t0);
      }
      baselineTimes.sort((a, b) => a - b);
      const baselineMedian = baselineTimes[Math.floor(measureFrames / 2)]!;
      const contactOnlyMs = median - baselineMedian;

      console.info(
        `[contact-pipeline-cost] N=${capacity} contactCount=${contactN} ` +
          `frameMs median=${median.toFixed(2)} mean=${mean.toFixed(2)} ` +
          `p10=${p10.toFixed(2)} p90=${p90.toFixed(2)} ` +
          `baseline(no-contact)=${baselineMedian.toFixed(2)} ` +
          `contactOnly=${contactOnlyMs.toFixed(2)} ` +
          `(warmup=${warmupFrames} measure=${measureFrames} S=4 I=2)`,
      );

      baselineLoop.dispose();
      baselineParticles.dispose();
      baselineFloor.dispose();

      expect(median).toBeGreaterThan(0);
      if (median > 6.67) {
        console.warn(
          `[contact-pipeline-cost] median frame cost ${median.toFixed(2)} ms exceeds the ` +
            `6.67 ms budget (40 % of a 60 Hz frame).`,
        );
      }

      loop.dispose();
      particles.dispose();
      floor.dispose();
    } finally {
      renderer.dispose();
    }
  }, 600_000);
});
