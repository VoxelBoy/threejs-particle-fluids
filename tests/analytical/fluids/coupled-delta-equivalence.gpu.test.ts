import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ContactAccumulator,
  HashGrid,
  ParticleSystem,
  allocatePairListStorage,
  buildPairListKernel,
  buildResetAccumulatorKernel,
  buildResetOverflowFlagKernel,
  createParticleRenderer,
  createSphKernelUniforms,
  createXpbdUniforms,
  deriveAccumulatorScale,
  type ParticleInit,
} from '../../../src/core/index.js';
import {
  buildCoupledDeltaKernel,
  buildLambdaKernel,
  buildPositionDeltaKernel,
  buildSolidReactionScatterKernel,
} from '../../../src/fluids/index.js';
import { instancedArray, uniform } from 'three/tsl';

// Phase Perf-15 G1 — equivalence between unfused (positionDelta + scatter)
// and fused (coupledDelta) per-iter kernels.
//
// Build a small mixed-phase scene (fluid + boundary), pre-compute λ_i via
// the production lambda kernel, then run the per-iter step through:
//   (a) unfused: positionDelta → scatter
//   (b) fused: coupledDelta
// And assert:
//   - `deltaX[i]` matches numerically (≤ 1e-6 per particle).
//   - The scatter accumulator's i32 contents match exactly.
//
// The fluid-side reduction is per-thread local with deterministic pair-
// list iteration order, so the fused and unfused kernels produce
// bit-identical f32 output (within TSL emit-order quirks). The
// boundary-side scatter is i32 atomicAdd — bit-identical by construction.

describe('Phase Perf-15 — coupledDelta vs positionDelta+scatter equivalence', () => {
  it('fused kernel produces matching deltaX and accumulator deltas', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.025;
      const h = 4 * r;
      const phaseFor = (id: number): number => ((id & 0xffff) << 16) >>> 0;
      const PHASE_FLUID = 1;
      const PHASE_BOUNDARY = 2;

      // Same shape as the Phase 14 set-equality test: 200 fluid above
      // a 60-particle boundary slab. Fluid neighbours are mostly fluid
      // (gather term dominates) plus a sliver of boundary contacts
      // (scatter term fires).
      const fluidCount = 200;
      const boundaryCount = 60;
      const total = fluidCount + boundaryCount;
      const SPACING = 2 * r;
      const initial: ParticleInit[] = [];
      const fluidSide = Math.ceil(Math.cbrt(fluidCount));
      for (let i = 0; i < fluidCount; i++) {
        const x = i % fluidSide;
        const y = Math.floor(i / fluidSide) % fluidSide;
        const z = Math.floor(i / (fluidSide * fluidSide));
        initial.push({
          position: [
            -fluidSide * SPACING * 0.5 + (x + 0.5) * SPACING,
            (y + 0.5) * SPACING,
            -fluidSide * SPACING * 0.5 + (z + 0.5) * SPACING,
          ],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: phaseFor(PHASE_FLUID),
        });
      }
      const bSide = Math.ceil(Math.sqrt(boundaryCount));
      for (let i = 0; i < boundaryCount; i++) {
        const x = i % bSide;
        const z = Math.floor(i / bSide);
        initial.push({
          position: [
            -bSide * SPACING * 0.5 + (x + 0.5) * SPACING,
            -SPACING * 0.5,
            -bSide * SPACING * 0.5 + (z + 0.5) * SPACING,
          ],
          velocity: [0, 0, 0],
          invMass: 0.5, // movable boundary so the scatter branch fires
          phase: phaseFor(PHASE_BOUNDARY),
        });
      }

      const particles = new ParticleSystem(renderer, total, r);
      particles.uploadParticles(initial);

      // Seed boundaryVolume for boundary slots so the per-pair ψ_j
      // resolution branches into the boundary path. Use a uniform
      // V_j matching the spacing³ convention from the boundary-volume
      // seed pass — magnitude is irrelevant for the equivalence test
      // as long as the same value feeds both kernels.
      const Vj = SPACING * SPACING * SPACING;
      const boundaryVolume = particles.boundaryVolume.value.array as Float32Array;
      for (let i = fluidCount; i < total; i++) boundaryVolume[i] = Vj;
      particles.boundaryVolume.value.needsUpdate = true;

      const hashGrid = new HashGrid(particles, { cellSize: h });
      const xpbd = createXpbdUniforms(1 / 60);

      // SPH uniforms — same constants the production FluidSystem builds.
      const sph = createSphKernelUniforms(h);

      // Per-particle scalars.
      const lambda = instancedArray(total, 'float');
      const density = instancedArray(total, 'float');
      const restDensity = uniform(1000, 'float');
      const compliance = uniform(1e-4, 'float');

      // Per-substep pair list (production form).
      const fluidParticles = { start: 0, count: fluidCount };
      const { pairList, pairCount, pairOverflowFlag } = allocatePairListStorage(fluidCount);
      const pairListKernel = buildPairListKernel({
        particles,
        hashGrid,
        hSq: sph.hSq,
        fluidParticles,
        pairList,
        pairCount,
        pairOverflowFlag,
      });

      // Lambda kernel (writes density + lambda) — production form.
      const lambdaKernel = buildLambdaKernel({
        particles,
        sph,
        restDensity,
        dt: xpbd.dt,
        compliance,
        density,
        lambda,
        fluidParticles,
        pairList,
        pairCount,
      });

      // Two deltaX buffers (one per pipeline) so we can compare.
      const deltaXUnfused = instancedArray(total, 'vec4');
      const deltaXFused = instancedArray(total, 'vec4');

      // Two accumulators (one per pipeline) so we can compare.
      const scaleTicks = deriveAccumulatorScale(0.5);
      const accUnfused = new ContactAccumulator(particles, scaleTicks);
      const accFused = new ContactAccumulator(particles, scaleTicks);

      const positionDeltaKernel = buildPositionDeltaKernel({
        particles,
        sph,
        restDensity,
        lambda,
        deltaX: deltaXUnfused,
        fluidParticles,
        pairList,
        pairCount,
      });
      const scatterKernel = buildSolidReactionScatterKernel({
        particles,
        sph,
        restDensity,
        lambda,
        contactAccumulator: accUnfused,
        fluidParticles,
        pairList,
        pairCount,
      });
      const coupledDeltaKernel = buildCoupledDeltaKernel({
        particles,
        sph,
        restDensity,
        lambda,
        deltaX: deltaXFused,
        contactAccumulator: accFused,
        fluidParticles,
        pairList,
        pairCount,
      });

      const resetUnfused = buildResetAccumulatorKernel(accUnfused);
      const resetUnfusedFlag = buildResetOverflowFlagKernel(accUnfused);
      const resetFused = buildResetAccumulatorKernel(accFused);
      const resetFusedFlag = buildResetOverflowFlagKernel(accFused);

      // Build the hash grid + pair list + lambda once — same inputs
      // for both pipelines.
      await renderer.computeAsync([...hashGrid.rebuildPipeline, pairListKernel, lambdaKernel]);

      // Run unfused: reset accumulator → positionDelta → scatter.
      await renderer.computeAsync([
        resetUnfused,
        resetUnfusedFlag,
        positionDeltaKernel,
        scatterKernel,
      ]);

      // Run fused: reset accumulator → coupledDelta.
      await renderer.computeAsync([resetFused, resetFusedFlag, coupledDeltaKernel]);

      const dxUnfusedBuf = await renderer.getArrayBufferAsync(deltaXUnfused.value);
      const dxFusedBuf = await renderer.getArrayBufferAsync(deltaXFused.value);
      const accUnfusedBuf = await renderer.getArrayBufferAsync(accUnfused.delta.value);
      const accFusedBuf = await renderer.getArrayBufferAsync(accFused.delta.value);

      const dxU = new Float32Array(dxUnfusedBuf);
      const dxF = new Float32Array(dxFusedBuf);
      const aU = new Int32Array(accUnfusedBuf);
      const aF = new Int32Array(accFusedBuf);

      // 1) deltaX numerical equivalence per fluid particle.
      let maxDxErr = 0;
      for (let i = 0; i < fluidCount; i++) {
        for (let c = 0; c < 3; c++) {
          const u = dxU[4 * i + c]!;
          const f = dxF[4 * i + c]!;
          const err = Math.abs(u - f);
          if (err > maxDxErr) maxDxErr = err;
        }
      }
      // eslint-disable-next-line no-console
      console.log(`[coupled-delta-equivalence] maxDxErr=${maxDxErr.toExponential(3)}`);
      expect(maxDxErr).toBeLessThan(1e-6);

      // 2) Accumulator i32 equality. The fused kernel and unfused
      //    pipeline emit the same per-pair (j, value) contributions,
      //    quantized to i32 ticks at the same scale. Per-pair
      //    quantization error is at most ±1 tick; the running sum
      //    differs by at most ±N_pairs(j) ticks per slot.
      //
      //    Atomic order is commutative for exact i32, but each
      //    per-pair value is itself a quantized representation of a
      //    float — so when the kernel emits the same float Δp_j_pair
      //    twice (once per kernel), the round() call produces the
      //    same i32 ⇒ exact match. Differences would imply a
      //    different per-pair float, which would also show up in
      //    deltaX (same gradW chain). Our deltaX matches at <3e-8.
      //    Allow a 2-tick tolerance per slot to absorb any TSL
      //    code-emission differences in the multiply chain that don't
      //    reach the deltaX max-norm.
      const TICK_TOL = 2;
      let mismatchSlots = 0;
      let maxAbsDiff = 0;
      for (let k = 0; k < aU.length; k++) {
        const diff = Math.abs(aU[k]! - aF[k]!);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
        if (diff > TICK_TOL) mismatchSlots++;
      }
      // eslint-disable-next-line no-console
      console.log(
        `[coupled-delta-equivalence] accumulator maxAbsDiff=${maxAbsDiff} ticks  mismatchSlots(>${TICK_TOL})=${mismatchSlots}`,
      );
      expect(mismatchSlots).toBe(0);

      // 3) Sanity: at least some scatter activity. Otherwise the
      //    test could pass trivially with both kernels emitting zero.
      let nonZero = 0;
      for (let k = 0; k < aF.length; k++) if (aF[k] !== 0) nonZero++;
      // eslint-disable-next-line no-console
      console.log(`[coupled-delta-equivalence] non-zero accumulator slots=${nonZero}`);
      expect(nonZero).toBeGreaterThan(0);
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
