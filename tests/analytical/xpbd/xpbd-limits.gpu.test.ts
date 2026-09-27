import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  constraintKernels,
  createDistanceConstraints,
  createParticleRenderer,
  type Material,
  type ParticleInit,
} from '../../../src/index.js';

// Rigid and soft compliance limits.
//
// Rigid (α = 0): under gravity, a one-pinned / one-free pair connected by a
// rigid distance constraint must preserve distance across 1000 frames.
// Tolerance: |length − rest| < 1e-4 at any sampled point during the run. In
// XPBD α = 0 reduces eq. (18) to the classic PBD scaling factor `s_j`
// (Macklin 2016 §4.1: "in the case of α_j = 0 it corresponds exactly to the
// scaling factor s_j in the original PBD algorithm (2)"), so this exercises
// PBD's own iteration-count-stiff regime — a few iterations are enough.
//
// Soft (α → ∞): the constraint is effectively absent and free-fall matches
// unconstrained motion. With α = 1e12 the per-step correction is of order
// C/(α̃+w) ≈ C·dt²/α ≈ 3e-16, well below f32 precision at particle-1's
// falling magnitudes. We run an instance without any constraint alongside
// and compare per frame.

/** A material that solves one distance constraint between particles 0 and 1. */
function distanceMaterial(particles: ParticleSystem, compliance: number, rest: number): Material {
  return {
    build: ({ dt }) =>
      constraintKernels([
        createDistanceConstraints({
          particles,
          pairs: [[0, 1]],
          compliance,
          restLength: [rest],
          dt,
        }),
      ]),
  };
}

async function runRigid(): Promise<{ maxLengthDrift: number }> {
  const renderer = await createParticleRenderer();
  try {
    const L0 = 1.0;
    const particles = new ParticleSystem(renderer, 2, 0.02);
    const data: ParticleInit[] = [
      { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 0 },
      { position: [0, -L0, 0], velocity: [0, 0, 0], invMass: 1 },
    ];
    particles.uploadParticles(data);

    const loop = new SimLoop(particles, {
      materials: [distanceMaterial(particles, 0, L0)],
      substeps: 4,
      iterations: 4,
    });

    let maxDrift = 0;
    const dt = 1 / 60;
    // Sample every 25 frames to keep readback overhead manageable.
    for (let n = 0; n < 1000; n++) {
      await loop.step(dt);
      if (n % 25 === 0 || n === 999) {
        const snap = await particles.readback();
        const dx = snap.positions[4]! - snap.positions[0]!;
        const dy = snap.positions[5]! - snap.positions[1]!;
        const dz = snap.positions[6]! - snap.positions[2]!;
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const drift = Math.abs(len - L0);
        if (drift > maxDrift) maxDrift = drift;
      }
    }

    loop.dispose();
    particles.dispose();
    return { maxLengthDrift: maxDrift };
  } finally {
    renderer.dispose();
  }
}

async function runSoft(): Promise<{ maxDeviation: number }> {
  const renderer = await createParticleRenderer();
  try {
    const L0 = 1.0;

    // Two independent systems: one constrained (α = 1e12), one bare. Both
    // share identical seeds and identical integrator settings.
    const constrained = new ParticleSystem(renderer, 2, 0.02);
    const bare = new ParticleSystem(renderer, 2, 0.02);
    const initial: ParticleInit[] = [
      { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 0 },
      { position: [0, -L0, 0], velocity: [0, 0, 0], invMass: 1 },
    ];
    constrained.uploadParticles(initial);
    bare.uploadParticles(initial);

    const loopC = new SimLoop(constrained, {
      materials: [distanceMaterial(constrained, 1e12, L0)],
      substeps: 1,
      iterations: 2,
    });
    const loopB = new SimLoop(bare, { substeps: 1 });

    const dt = 1 / 60;
    let maxDev = 0;
    for (let n = 0; n < 120; n++) {
      await loopC.step(dt);
      await loopB.step(dt);
      if (n % 20 === 0 || n === 119) {
        const [snapC, snapB] = await Promise.all([constrained.readback(), bare.readback()]);
        for (let k = 0; k < 3; k++) {
          const d = Math.abs(snapC.positions[4 + k]! - snapB.positions[4 + k]!);
          if (d > maxDev) maxDev = d;
        }
      }
    }

    loopC.dispose();
    loopB.dispose();
    constrained.dispose();
    bare.dispose();
    return { maxDeviation: maxDev };
  } finally {
    renderer.dispose();
  }
}

describe('XPBD: rigid limit (α = 0)', () => {
  it('distance drift < 1e-4 over 1000 frames under gravity', async () => {
    const { maxLengthDrift } = await runRigid();
    console.info(`[xpbd-rigid] max |len − L₀| = ${maxLengthDrift.toExponential(3)}`);
    expect(maxLengthDrift).toBeLessThan(1e-4);
  }, 120_000);
});

describe('XPBD: soft limit (α → ∞)', () => {
  it('particle 1 trajectory matches no-constraint free fall within 1e-6', async () => {
    const { maxDeviation } = await runSoft();
    console.info(
      `[xpbd-soft] max |x_constrained − x_unconstrained| = ${maxDeviation.toExponential(3)}`,
    );
    expect(maxDeviation).toBeLessThan(1e-6);
  }, 120_000);
});
