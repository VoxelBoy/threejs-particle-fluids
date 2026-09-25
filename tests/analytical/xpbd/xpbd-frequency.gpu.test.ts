import { describe, expect, it } from 'vitest';
import { Fn, instancedArray, uniform, uint } from 'three/tsl';
import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createDistanceConstraints,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 04 G1 — XPBD frequency test (BLOCKING, per plan §Exit criteria).
//
// Two particles of mass 1 connected by a distance constraint with compliance
// α. Particle 0 is pinned (invMass=0), particle 1 is free. Release particle
// 1 offset from rest. For the undamped linear spring limit of the distance
// constraint near equilibrium, the oscillation angular frequency is
//   ω = √(k/m) with k = 1/α
// (Macklin 2016 §6.1 "Spring", which uses the identical spring model at
// α = 0.001 to validate XPBD against the analytic solution — see Figure 2).
//
// The paper's Figure 2 shows XPBD "closely reproduces the analytic result
// regardless of time step and iteration count" for this scenario, so if our
// implementation is correct the frequency error should be small — the plan
// specifies ≤ 2% across at least three compliance values.
//
// We measure frequency by zero-crossing detection on the y-position
// trajectory. Each compliance value uses a per-α timestep sized to give
// roughly 20 samples per period (Nyquist + margin) and 10 periods of data.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

interface FreqCase {
  readonly alpha: number;
  readonly expectedOmega: number;
  readonly expectedPeriod: number;
  readonly frameDt: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly nFrames: number;
}

function buildCase(alpha: number): FreqCase {
  const omega = Math.sqrt(1 / alpha);
  const period = (2 * Math.PI) / omega;
  // ~20 samples per period, 10 periods of data.
  const frameDt = period / 20;
  const nFrames = Math.ceil(10 / (frameDt / period));
  // Substep dt ≤ T/80 gives us Nyquist × 40 resolution inside XPBD's implicit
  // step, plenty for a stiff spring. Iterations bumped to 4 at the stiffest
  // compliance to keep the Gauss-Seidel residual under control.
  const substeps = 4;
  const iterations = alpha <= 1e-5 ? 8 : 2;
  return {
    alpha,
    expectedOmega: omega,
    expectedPeriod: period,
    frameDt,
    substeps,
    iterations,
    nFrames,
  };
}

async function measureFrequency(c: FreqCase): Promise<{
  measuredPeriod: number;
  relError: number;
}> {
  const renderer = await createParticleRenderer();
  try {
    const restLength = 1.0;
    const offset = 0.05; // small linear-regime offset

    const particles = new ParticleSystem(renderer, 2, 0.02);
    const initial: ParticleInit[] = [
      // Pinned anchor
      {
        position: [0, 0, 0],
        velocity: [0, 0, 0],
        invMass: 0,
        phase: 0,
      },
      // Free mass, offset along +y from rest length
      {
        position: [0, -(restLength + offset), 0],
        velocity: [0, 0, 0],
        invMass: 1,
        phase: 0,
      },
    ];
    particles.uploadParticles(initial);

    const xpbd = createXpbdUniforms(c.frameDt / c.substeps);
    const dist = createDistanceConstraints({
      particles,
      pairs: [[0, 1]],
      compliance: c.alpha,
      restLength: [restLength],
      xpbd,
    });

    const loop = new SimLoop(particles, {
      xpbd,
      constraints: [dist],
      substeps: c.substeps,
      iterations: c.iterations,
    });
    // Disable floor clamp; the pair oscillates around y = -restLength.
    loop.kernels.floorY.value = -1e9;
    // Disable gravity so the oscillation is purely about rest length. The
    // plan mentions gravity, but a gravity-free test isolates frequency from
    // the equilibrium shift `Δy = m·g/k` and gives a cleaner measurement.
    // See discussion in Macklin 2016 §6.1 (no gravity in the reference
    // harmonic oscillator test).
    loop.gravity.set(0, 0, 0);

    // Trajectory recorder — one-thread kernel writes particle 1's y into
    // `traj[frame]`. Saves `nFrames` per-frame readbacks in exchange for a
    // single bulk readback at the end.
    const traj = instancedArray(c.nFrames, 'float');
    const frameIdx = uniform(uint(0));
    const recordKernel = Fn(() => {
      const y: Any = particles.positions.element(1).y;
      traj.element(frameIdx as Any).assign(y);
    })().compute(1);

    for (let n = 0; n < c.nFrames; n++) {
      await loop.step(c.frameDt);
      frameIdx.value = n;
      await renderer.computeAsync(recordKernel);
    }

    const raw = new Float32Array(await renderer.getArrayBufferAsync(traj.value));
    particles.destroy();

    // Convert absolute y into displacement from equilibrium (y_eq ≈ -rest).
    // The initial-release state has displacement = -offset (below rest).
    const disp = new Float64Array(c.nFrames);
    for (let i = 0; i < c.nFrames; i++) disp[i] = raw[i]! + restLength;

    // Zero-crossing detection with linear interpolation for sub-frame
    // precision. Record crossing times for the negative-to-positive direction
    // only (half the sample rate is fine for period estimation).
    const crossings: number[] = [];
    for (let i = 1; i < c.nFrames; i++) {
      const a = disp[i - 1]!;
      const b = disp[i]!;
      if (a <= 0 && b > 0) {
        const frac = a === b ? 0 : -a / (b - a);
        const t = (i - 1 + frac) * c.frameDt;
        crossings.push(t);
      }
    }
    expect(
      crossings.length,
      `α=${c.alpha}: expected ≥ 3 zero crossings, got ${crossings.length}`,
    ).toBeGreaterThanOrEqual(3);

    // Period = mean gap between consecutive same-direction crossings.
    const gaps: number[] = [];
    for (let i = 1; i < crossings.length; i++) {
      gaps.push(crossings[i]! - crossings[i - 1]!);
    }
    const measured = gaps.reduce((s, x) => s + x, 0) / gaps.length;
    const rel = Math.abs(measured - c.expectedPeriod) / c.expectedPeriod;
    return { measuredPeriod: measured, relError: rel };
  } finally {
    renderer.dispose();
  }
}

describe('Phase 04 — XPBD: harmonic oscillator frequency', () => {
  it.each([{ alpha: 1e-2 }, { alpha: 1e-4 }, { alpha: 1e-6 }])(
    'α = $alpha: measured period matches √(α) within 2%',
    async ({ alpha }) => {
      const c = buildCase(alpha);
      const { measuredPeriod, relError } = await measureFrequency(c);
      // eslint-disable-next-line no-console
      console.info(
        `[xpbd-freq] α=${alpha} T_expected=${c.expectedPeriod.toExponential(3)} ` +
          `T_measured=${measuredPeriod.toExponential(3)} |err|=${(relError * 100).toFixed(2)}% ` +
          `(S=${c.substeps} I=${c.iterations} frames=${c.nFrames})`,
      );
      expect(relError).toBeLessThan(0.02);
    },
    120_000,
  );
});
