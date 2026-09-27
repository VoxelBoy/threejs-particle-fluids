import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Two balls on an inclined plane: static and kinetic friction.
//
// Two dynamic particles rest on a horizontal analytic plane, with gravity
// tilted to angle θ (mathematically equivalent to tilting the plane). A flat
// plane makes the slide distance a quantitative Newtonian check:
//   At θ = atan(μ_s) − 3°:   neither slides within 5 s (|Δx| < 1e-3 m).
//   At θ = atan(μ_s) + 3°:   both slide by Δx(t) = ½·a·t² with
//                            a = g · (sin θ − μ_k · cos θ).
//
// Paper references:
//   - Macklin 2020 §3.5 position-level static-friction gate
//     (`λ_t < μ_s · λ_n`) — holds the below-cone case.
//   - Macklin 2020 §3.6 velocity-level kinetic-friction clamp — caps the
//     above-cone slide acceleration to `g·sin θ − μ_k · g · cos θ`, which
//     is the Newtonian prediction.

const PARTICLE_RADIUS = 0.05;
const MU_S = 0.6;
const MU_K = 0.5;
const G_MAG = 9.81;

async function runIncline(args: {
  readonly thetaRad: number;
  readonly seconds: number;
  readonly label: string;
}): Promise<{
  readonly leftDelta: number;
  readonly rightDelta: number;
  readonly maxDelta: number;
}> {
  const { thetaRad, seconds, label } = args;
  const renderer = await createParticleRenderer();
  try {
    const r = PARTICLE_RADIUS;
    const initial: ParticleInit[] = [
      { position: [-0.2, r, 0], velocity: [0, 0, 0], invMass: 1 },
      { position: [+0.2, r, 0], velocity: [0, 0, 0], invMass: 1 },
    ];
    const [leftIdx, rightIdx] = [0, 1];

    const particles = new ParticleSystem(renderer, initial.length, r);
    particles.uploadParticles(initial);

    const colliders = new PrimitiveSet(particles);
    colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
      muS: MU_S,
      muK: MU_K,
    });

    const loop = new SimLoop(particles, {
      substeps: 4,
      iterations: 4,
      contact: { maxContacts: 8, muS: MU_S, muK: MU_K },
      colliders: [colliders],
    });
    loop.gravity.set(G_MAG * Math.sin(thetaRad), -G_MAG * Math.cos(thetaRad), 0);

    const x0Left = initial[leftIdx]!.position[0];
    const x0Right = initial[rightIdx]!.position[0];

    const frameDt = 1 / 60;
    const totalFrames = Math.ceil(seconds / frameDt);
    for (let n = 0; n < totalFrames; n++) {
      await loop.step(frameDt);
    }

    const snap = await particles.readback();
    const xLeft = snap.positions[leftIdx * 4]!;
    const xRight = snap.positions[rightIdx * 4]!;
    const leftDelta = xLeft - x0Left;
    const rightDelta = xRight - x0Right;
    const maxDelta = Math.max(Math.abs(leftDelta), Math.abs(rightDelta));

    console.info(
      `[two-ball-${label}] θ=${((thetaRad * 180) / Math.PI).toFixed(2)}° ` +
        `t=${seconds.toFixed(1)}s leftΔx=${leftDelta.toFixed(5)} ` +
        `rightΔx=${rightDelta.toFixed(5)} maxΔ=${maxDelta.toFixed(5)}`,
    );

    loop.dispose();
    particles.dispose();
    colliders.dispose();
    return { leftDelta, rightDelta, maxDelta };
  } finally {
    renderer.dispose();
  }
}

describe('contact: two balls on an inclined plane', () => {
  // Static-friction regime: below the cone, both particles stick.
  it('static (θ = atan(μ_s) − 3°): neither particle slides', async () => {
    const theta = Math.atan(MU_S) - (3 * Math.PI) / 180;
    const { maxDelta } = await runIncline({
      thetaRad: theta,
      seconds: 5,
      label: 'static',
    });
    expect(maxDelta).toBeLessThan(1e-3);
  }, 300_000);

  // Kinetic-friction regime: above the cone, both particles slide at the
  // Newtonian rate:
  //   a = g · (sin θ − μ_k · cos θ)
  //   Δx(t) = ½ · a · t²
  // At θ = atan(μ_s) + 3° = 33.96°, μ_k = 0.5, g = 9.81:
  //   a ≈ 9.81 · (sin 33.96° − 0.5 · cos 33.96°) ≈ 9.81 · (0.559 − 0.414) ≈ 1.42 m/s²
  //   Δx(2s) ≈ 0.5 · 1.42 · 4 ≈ 2.85 m
  it('kinetic (θ = atan(μ_s) + 3°): slide matches Newtonian ½·a·t² within 15%', async () => {
    const theta = Math.atan(MU_S) + (3 * Math.PI) / 180;
    const seconds = 2;
    const { leftDelta, rightDelta } = await runIncline({
      thetaRad: theta,
      seconds,
      label: 'kinetic',
    });
    const aNewton = G_MAG * (Math.sin(theta) - MU_K * Math.cos(theta));
    const dxExpected = 0.5 * aNewton * seconds * seconds;
    console.info(
      `[two-ball-kinetic] Newtonian a=${aNewton.toFixed(3)} m/s² Δx(${seconds}s)=${dxExpected.toFixed(3)} m`,
    );
    // Both particles slide at the same rate (identical setup). Allow 15%
    // slack for friction-threshold edge effects in the first few substeps,
    // while λ_n is still ramping up.
    expect(leftDelta).toBeGreaterThan(dxExpected * 0.85);
    expect(leftDelta).toBeLessThan(dxExpected * 1.15);
    expect(rightDelta).toBeGreaterThan(dxExpected * 0.85);
    expect(rightDelta).toBeLessThan(dxExpected * 1.15);
  }, 300_000);
});
