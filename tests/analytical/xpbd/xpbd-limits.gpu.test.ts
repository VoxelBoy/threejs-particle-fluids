import { describe, expect, it } from 'vitest';
import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createDistanceConstraints,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 04 G1 — rigid and soft compliance limits (BLOCKING).
//
// Rigid (α = 0): under gravity, a one-pinned / one-free pair connected by a
// rigid distance constraint must preserve distance across 1000 frames. Plan
// tolerance: |length − rest| < 1e-4 at any point during the run. In XPBD
// α = 0 reduces eq. (18) to the classic PBD scaling factor `s_j` (Macklin
// 2016 §4.1: "in the case of α_j = 0 it corresponds exactly to the scaling
// factor s_j in the original PBD algorithm (2)"), so this exercises PBD's
// own iteration-count-stiff regime — a few iterations are enough.
//
// Soft (α → ∞): the constraint is effectively absent and free-fall matches
// unconstrained motion. With α = 1e12 the per-step correction is of order
// C/(α̃+w) ≈ C·dt²/α ≈ 3e-16, well below f32 precision at particle-1's
// falling magnitudes. We run an instance without any constraint alongside
// and compare per-frame.

async function runRigid(): Promise<{ maxLengthDrift: number }> {
  const renderer = await createParticleRenderer();
  try {
    const L0 = 1.0;
    const particles = new ParticleSystem(renderer, 2, 0.02);
    const data: ParticleInit[] = [
      { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 0, phase: 0 },
      { position: [0, -L0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
    ];
    particles.uploadParticles(data);

    const xpbd = createXpbdUniforms(1 / 60);
    const dist = createDistanceConstraints({
      particles,
      pairs: [[0, 1]],
      compliance: 0,
      restLength: [L0],
      xpbd,
    });
    const loop = new SimLoop(particles, {
      xpbd,
      constraints: [dist],
      substeps: 4,
      iterations: 4,
    });
    loop.kernels.floorY.value = -1e9;

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

    particles.destroy();
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
      { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 0, phase: 0 },
      { position: [0, -L0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
    ];
    constrained.uploadParticles(initial);
    bare.uploadParticles(initial);

    const xpbdC = createXpbdUniforms(1 / 60);
    const dist = createDistanceConstraints({
      particles: constrained,
      pairs: [[0, 1]],
      compliance: 1e12,
      restLength: [L0],
      xpbd: xpbdC,
    });
    const loopC = new SimLoop(constrained, {
      xpbd: xpbdC,
      constraints: [dist],
      substeps: 1,
      iterations: 2,
    });
    const loopB = new SimLoop(bare, { substeps: 1 });
    loopC.kernels.floorY.value = -1e9;
    loopB.kernels.floorY.value = -1e9;

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

    constrained.destroy();
    bare.destroy();
    return { maxDeviation: maxDev };
  } finally {
    renderer.dispose();
  }
}

describe('Phase 04 — XPBD: rigid limit (α = 0)', () => {
  it('distance drift < 1e-4 over 1000 frames under gravity', async () => {
    const { maxLengthDrift } = await runRigid();
    // eslint-disable-next-line no-console
    console.info(`[xpbd-rigid] max |len − L₀| = ${maxLengthDrift.toExponential(3)}`);
    expect(maxLengthDrift).toBeLessThan(1e-4);
  }, 120_000);
});

describe('Phase 04 — XPBD: soft limit (α → ∞)', () => {
  it('particle 1 trajectory matches no-constraint free fall within 1e-6', async () => {
    const { maxDeviation } = await runSoft();
    // eslint-disable-next-line no-console
    console.info(
      `[xpbd-soft] max |x_constrained − x_unconstrained| = ${maxDeviation.toExponential(3)}`,
    );
    expect(maxDeviation).toBeLessThan(1e-6);
  }, 120_000);
});
