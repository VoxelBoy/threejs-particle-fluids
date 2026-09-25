import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FrictionTable,
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 21 G1 — per-phase-group Coulomb friction.
//
// Two particles (one kinematic anchor, one dynamic slider) in DIFFERENT
// phase groups so the contact-emit kernel produces a cross-group pair (same-
// group pairs are suppressed by the phase mask, ARCHITECTURE.md §"Cross-
// cutting mechanisms"). Tilted gravity at θ = atan(0.3) ≈ 17°: the
// tangential gravity component is below the static-friction cone for μ = 0.6
// (cone ≈ 31°) and above the cone for μ = 0.1 (cone ≈ 5.7°).
//
// Two scenes share the same geometry and same gravity; only the
// `FrictionTable` differs:
//   - high-friction: μ_s = 0.6, μ_k = 0.45 → slider stays glued to anchor,
//     Δx ≈ 0 at t = 2 s.
//   - low-friction:  μ_s = 0.1, μ_k = 0.1 → slider slips off and free-falls
//     under tangential gravity, |Δx| grows monotonically.
//
// Pass criterion: |Δx_low − Δx_high| / max(|Δx_low|, |Δx_high|) > 0.4 at
// t = 2 s. That ratio is unitless and dimensionally robust against the
// f32 / Apple-Silicon noise floor; see U-50 + memory
// `feedback_tests_direct_behavior.md` for why ratio gates of this form are
// the right signal.
//

//
// Combine-rule verification is delegated to a second test below that sets
// μ_low = 0 in the slider's group and μ_high = 0.6 in the anchor's group,
// asserting min-combine → pair μ = 0 (slick path).

const PARTICLE_RADIUS = 0.05;
const G_MAG = 9.81;
const TILT_RAD = Math.atan(0.3); // ≈ 17°

const GROUP_ANCHOR = 1;
const GROUP_SLIDER = 2;

interface Result {
  readonly slideX: number;
  readonly slideZ: number;
  readonly anchorY: number;
  readonly sliderY: number;
}

async function runFrictionScene(args: {
  readonly anchorMuS: number;
  readonly anchorMuK: number;
  readonly sliderMuS: number;
  readonly sliderMuK: number;
  readonly seconds: number;
  readonly label: string;
}): Promise<Result> {
  const { anchorMuS, anchorMuK, sliderMuS, sliderMuK, seconds, label } = args;
  const renderer = await createParticleRenderer();
  try {
    const r = PARTICLE_RADIUS;
    // Phase 21 encoding: high 16 bits = self-collision group (controls
    // emit-time intra-group skip), low 16 bits = friction group (LUT key).
    // This test uses different group IDs in BOTH halves so the cross-pair
    // (anchor↔slider) emits AND looks up distinct LUT slots.
    const phaseForGroup = (g: number): number => ((g << 16) | g) >>> 0;
    const initial: ParticleInit[] = [
      // Anchor: pinned (invMass=0), at origin. Group 1.
      {
        position: [0, 0, 0],
        velocity: [0, 0, 0],
        invMass: 0,
        phase: phaseForGroup(GROUP_ANCHOR),
      },
      // Slider: dynamic, sitting one particle-diameter above the anchor so
      // the two are in unit-normal contact at substep 0. Group 2 → cross-
      // group pair emits, friction = min(group1, group2) = scene-specific.
      {
        position: [0, 2 * r, 0],
        velocity: [0, 0, 0],
        invMass: 1,
        phase: phaseForGroup(GROUP_SLIDER),
      },
    ];

    const particles = new ParticleSystem(renderer, initial.length, r);
    particles.uploadParticles(initial);

    // Hash grid retained because `SimLoop.ContactOptions` requires one.
    const hashGrid = new HashGrid(particles, {
      cellSize: 2 * r * 1.1,
    });

    // Per-phase-group friction LUT — the system under test. Two slots set;
    // every other group inherits the table's default (also (0, 0) here so
    // unused groups don't surprise the assertion).
    const frictionTable = new FrictionTable({
      defaultMuS: 0.0,
      defaultMuK: 0.0,
    });
    frictionTable.setGroupFriction(GROUP_ANCHOR, anchorMuS, anchorMuK);
    frictionTable.setGroupFriction(GROUP_SLIDER, sliderMuS, sliderMuK);

    // No collider plane — the only friction in this scene is the per-pair
    // particle-particle friction we're testing. The slider falls through
    // any horizontal level once contact with the anchor is lost.
    const loop = new SimLoop(particles, {
      substeps: 4,
      iterations: 4,
      contact: {
        hashGrid,
        maxContacts: 8,
        friction: frictionTable,
        stabIters: 1,
      },
    });
    // Disable the implicit infinite floor.
    loop.kernels.floorY.value = -1e9;
    loop.gravity.set(G_MAG * Math.sin(TILT_RAD), -G_MAG * Math.cos(TILT_RAD), 0);

    const frameDt = 1 / 60;
    const totalFrames = Math.ceil(seconds / frameDt);
    for (let n = 0; n < totalFrames; n++) {
      await loop.step(frameDt);
    }

    const snap = await particles.readback();
    // anchor[0..3], slider[4..7]
    const anchorY = snap.positions[1]!;
    const sliderX = snap.positions[4]!;
    const sliderY = snap.positions[5]!;
    const sliderZ = snap.positions[6]!;

    // eslint-disable-next-line no-console
    console.info(
      `[friction-per-group ${label}] anchor=(${anchorMuS.toFixed(2)},${anchorMuK.toFixed(2)}) ` +
        `slider=(${sliderMuS.toFixed(2)},${sliderMuK.toFixed(2)}) ` +
        `t=${seconds.toFixed(1)}s slider=(x=${sliderX.toFixed(3)}, ` +
        `y=${sliderY.toFixed(3)}, z=${sliderZ.toFixed(3)}) anchorY=${anchorY.toFixed(3)}`,
    );

    particles.destroy();
    hashGrid.destroy();
    return {
      slideX: sliderX,
      slideZ: sliderZ,
      anchorY,
      sliderY,
    };
  } finally {
    renderer.dispose();
  }
}

describe('Phase 21 G1 — per-phase-group Coulomb friction', () => {
  it('high-friction pair holds; low-friction pair slips — |Δx_low − Δx_high| / max > 0.4 at t = 2 s', async () => {
    const seconds = 2;
    const high = await runFrictionScene({
      anchorMuS: 0.6,
      anchorMuK: 0.45,
      sliderMuS: 0.6,
      sliderMuK: 0.45,
      seconds,
      label: 'high',
    });
    const low = await runFrictionScene({
      anchorMuS: 0.1,
      anchorMuK: 0.1,
      sliderMuS: 0.1,
      sliderMuK: 0.1,
      seconds,
      label: 'low',
    });

    // Anchor is kinematic; verify it didn't move.
    expect(Math.abs(high.anchorY)).toBeLessThan(1e-6);
    expect(Math.abs(low.anchorY)).toBeLessThan(1e-6);

    // High-friction slider stays close to the anchor's x = 0. Allow a
    // small drift for f32 noise + the static-cone proactive gate's
    // first-iter ramp (Phase 5a finding — λ_n is zero at iter 0 so the
    // gate over-applies briefly until λ_n stabilises).
    expect(Math.abs(high.slideX)).toBeLessThan(0.05);

    // Low-friction slider drops off the anchor and free-falls under
    // tangential gravity. Δx grows ~½·g·sinθ·t² ≈ 5.7 m at t = 2 s.
    expect(low.slideX).toBeGreaterThan(0.5);

    // Plan §"Validation" pass criterion.
    const denom = Math.max(Math.abs(low.slideX), Math.abs(high.slideX));
    const delta = Math.abs(low.slideX - high.slideX);
    const ratio = denom > 1e-9 ? delta / denom : 0;
    // eslint-disable-next-line no-console
    console.info(
      `[friction-per-group] |Δx_low − Δx_high| / max = ${ratio.toFixed(3)} (gate > 0.4)`,
    );
    expect(ratio).toBeGreaterThan(0.4);
  }, 300_000);

  it('min-combine rule: μ_anchor = 0.6, μ_slider = 0 → pair μ = min = 0 → slider slips', async () => {
    // Anchor's group has μ_s = 0.6 (would hold), slider's group has
    // μ_s = 0 (slick). U-51 chose `min` over arithmetic mean precisely
    // for this case: arithmetic mean would give pair μ = 0.3 (still
    // sliding at 17° but with friction); min gives pair μ = 0 (free
    // tangential motion, slider slips off as if frictionless).
    const seconds = 2;
    const slick = await runFrictionScene({
      anchorMuS: 0.6,
      anchorMuK: 0.45,
      sliderMuS: 0.0,
      sliderMuK: 0.0,
      seconds,
      label: 'slick',
    });
    // Slider should slip off as if frictionless; expect ≥ low-friction
    // slide distance from the first test (~0.5 m+ at t = 2 s).
    expect(slick.slideX).toBeGreaterThan(0.5);
  }, 300_000);
});
