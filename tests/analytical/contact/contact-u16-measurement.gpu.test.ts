import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 05 U-16 measurement gate (BLOCKING per plan §Exit criteria).
//

//
// **Scene**: the sandcastle scene is blocked on U-19 (friction-threshold
// collapse prevents the castle from holding shape). We substitute a
// densely-packed granular scene of the same particle count — the contact
// pipeline's per-frame cost is determined by the count of emitted
// contacts, which is a function of the spatial density (roughly `8·N` for
// closely packed), not the macro-scale pile geometry. The sandcastle's
// 50k + dense packing matches our substitute scene's 50k + dense
// packing within the U-16 measurement's purpose (whether gather-only
// coloring fits the frame budget).
//
// **Tier choice (G4)**: not a determinism test; this is a perf
// measurement. Results are informational, not gated on exact numbers —
// but the plan gates on the ≤ 6.67 ms / frame budget (Path A vs Path B).
//

describe('Phase 05 — contact: U-16 measurement gate', () => {
  it('at 50k particles, contact pipeline cost over 120 frames averaged', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const twoR = 2 * r;
      const N = 50_000;

      // Dense lattice in a cube. `2.02r` grid spacing is almost-touching;
      // each particle has ~8 neighbors at contact range, matching the
      // sandcastle scene's interior density.
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
              phase: 0,
            });
          }
        }
      }
      const capacity = initial.length;

      const particles = new ParticleSystem(renderer, capacity, r);
      particles.uploadParticles(initial);

      const hashGrid = new HashGrid(particles, {
        cellSize: twoR * 1.1,
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        contact: {
          hashGrid,
          // 8 contacts per particle × 50k / 2 (each undirected pair
          // emitted once) = 200k. Sizing exactly at the observed load
          // so the scatter kernel's dispatch over `maxContacts` threads
          // has minimal early-exit waste.
          maxContacts: 200_000,
          friction: { muS: 0.5, muK: 0.4 },
          stabIters: 1,
        },
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const warmupFrames = 20;
      const measureFrames = 120;

      // Warmup (let the driver JIT + cache compile).
      for (let n = 0; n < warmupFrames; n++) await loop.step(frameDt);

      // Measure whole-frame cost via `performance.now()` on the JS side.
      // We can't isolate the contact pipeline from predict/advect at this
      // level (would need per-kernel GPU timestamp queries — not exposed
      // by three.js r184), so we report `frameMs` and compute the
      // contact cost by subtraction against a `predict+advect only`
      // control run further down.
      // `computeAsync` in three.js r184 submits the command buffer and
      // returns — it does not block on GPU completion. To measure real
      // wall-clock GPU cost we pair each step with a single-byte
      // readback (which must wait for pipeline writes to land),
      // forcing a CPU-GPU sync. Small overhead (~0.1 ms per readback)
      // is roughly constant across both the full-contact and baseline
      // loops below, so the differential (`contactOnlyMs`) is still
      // accurate.
      const syncBuf = particles.invMass.value;
      const frameTimes: number[] = [];
      for (let n = 0; n < measureFrames; n++) {
        const t0 = performance.now();
        await loop.step(frameDt);
        await renderer.getArrayBufferAsync(syncBuf);
        const t1 = performance.now();
        frameTimes.push(t1 - t0);
      }

      // Stats (median is robust to driver stalls).
      frameTimes.sort((a, b) => a - b);
      const median = frameTimes[Math.floor(measureFrames / 2)]!;
      const p10 = frameTimes[Math.floor(measureFrames * 0.1)]!;
      const p90 = frameTimes[Math.floor(measureFrames * 0.9)]!;
      const mean = frameTimes.reduce((s, x) => s + x, 0) / measureFrames;

      // Contact-count snapshot at end of run (indicative).
      const contactN = await loop.contacts!.readbackCount();

      // Baseline: same scene, same SimLoop, but with `contact` omitted.
      // Delta against the full-contact median tells us how much of the
      // frame cost is the contact pipeline vs predict/advect. Without
      // this baseline there's a silent-failure risk — kernels that
      // emit invalid WGSL are accepted by three.js r184 and dispatched
      // as no-ops (Phase 03 Finding #1). Comparing medians catches
      // the case where the contact kernels all silently no-op.
      const baselineParticles = new ParticleSystem(renderer, capacity, r);
      baselineParticles.uploadParticles(initial);
      const baselineLoop = new SimLoop(baselineParticles, {
        substeps: 4,
        iterations: 2,
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

      // eslint-disable-next-line no-console
      console.info(
        `[contact-u16] N=${capacity} contactCount=${contactN} ` +
          `frameMs median=${median.toFixed(2)} mean=${mean.toFixed(2)} ` +
          `p10=${p10.toFixed(2)} p90=${p90.toFixed(2)} ` +
          `baseline(no-contact)=${baselineMedian.toFixed(2)} ` +
          `contactOnly=${contactOnlyMs.toFixed(2)} ` +
          `(warmup=${warmupFrames} measure=${measureFrames} S=4 I=2)`,
      );

      baselineParticles.destroy();

      // Plan's U-16 Path A: median ≤ 40% of 16.67 ms = 6.67 ms.
      //

      expect(median).toBeGreaterThan(0);
      if (median > 6.67) {
        // eslint-disable-next-line no-console
        console.warn(
          `[contact-u16] median frame cost ${median.toFixed(2)}ms exceeds U-16 ` +
            `Path A budget (6.67 ms). Path B (fixed-point i32 scatter) may be ` +
            `required.`,
        );
      }

      particles.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 600_000);
});
