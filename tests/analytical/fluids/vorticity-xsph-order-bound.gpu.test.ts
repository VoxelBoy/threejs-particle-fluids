import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';

// Phase Perf-06 G1 — vorticity+xsph substep-internal-ordering bound.
//
// Macklin 2013 §5 prescribes vorticity-confinement-then-XSPH within a
// substep. Phase Perf-06 fuses the two corrections' walks and tails so
// XSPH reads the pre-confinement velocity field instead of the post-
// confinement field. The plan's §Design analysis bounds the resulting
// per-substep divergence at `5 · ε · c · dt`; this test pins that bound
// to a measurable scene.
//
// One leg uses the un-fused paper-faithful dispatch order (Pass 1 →
// Pass 2 → Pass 3 → xsph compute → xsph apply), selected via the
// FluidSystem private `__forceUnfusedPostAdvect: true` option. The
// other leg uses the production fused chain (fused walk → Pass 2 →
// fused tail). The bound formula `5 · ε · c · dt` per substep is
// derived in the plan §"Substep-internal ordering" from the linear
// vorticity-XSPH coupling: ε≈0.05–0.1, c≈0.01, the two corrections add
// O(10⁻⁴–10⁻³) m/s per substep on water-scale velocities, and whether
// XSPH smooths the pre- or post-confinement field changes the result
// by a quantity proportional to ε·c.

const SPACING = 0.025;
const H = 0.05;
const REST_DENSITY = 1000;
const VORTICITY_STRENGTH = 0.1;
const XSPH_C = 0.01;
const SUBSTEPS = 2;
const FRAMES = 5;
const FRAME_DT = 1 / 60;
const SUBSTEP_DT = FRAME_DT / SUBSTEPS;
const TOTAL_SUBSTEPS = SUBSTEPS * FRAMES;
// `5 · ε · c · dt` per substep, accumulated over `TOTAL_SUBSTEPS` —
// generous one-OOM headroom over the analytic linear-coupling estimate
// (per the plan's §"Validation" paragraph).
const PER_SUBSTEP_BOUND = 5 * VORTICITY_STRENGTH * XSPH_C * SUBSTEP_DT;
const VELOCITY_BOUND = PER_SUBSTEP_BOUND * TOTAL_SUBSTEPS;
const KINETIC_ENERGY_REL_BOUND = 0.01;

// 10×10×10 cube of fluid particles in free-fall. No colliders → no
// contact-pipeline non-determinism. No surface tension / adhesion → no
// per-pair atomic scatter. Vorticity + XSPH both enabled at paper-
// default strengths so both halves of the post-advect block contribute.
function buildInitialCube(): readonly ParticleInit[] {
  const NX = 10;
  const NY = 10;
  const NZ = 10;
  const initial: ParticleInit[] = [];
  for (let j = 0; j < NY; j++) {
    for (let k = 0; k < NZ; k++) {
      for (let i = 0; i < NX; i++) {
        initial.push({
          position: [
            -((NX * SPACING) / 2) + SPACING * 0.5 + i * SPACING,
            SPACING * 0.5 + j * SPACING,
            -((NZ * SPACING) / 2) + SPACING * 0.5 + k * SPACING,
          ],
          velocity: [0, 0, 0],
          invMass: 1, // overwritten by FluidSystem at construction
          phase: 0,
        });
      }
    }
  }
  return initial;
}

interface ScenarioResult {
  readonly velocities: Float32Array; // length 4·count
  readonly count: number;
  readonly invMass: number;
}

async function runScenario(opts: { readonly forceUnfused: boolean }): Promise<ScenarioResult> {
  const renderer = await createParticleRenderer();
  try {
    const r = SPACING * 0.5;
    const initial = buildInitialCube();
    const count = initial.length;

    const particles = new ParticleSystem(renderer, count, r);
    particles.uploadParticles(initial);
    const hashGrid = new HashGrid(particles, { cellSize: H });
    const xpbd = createXpbdUniforms(SUBSTEP_DT);
    const fluid = new FluidSystem({
      particles,
      hashGrid,
      xpbd,
      restDensity: REST_DENSITY,
      h: H,
      particleSpacing: SPACING,
      compliance: 1e-4,
      fluidParticles: { start: 0, count },
      vorticity: { strength: VORTICITY_STRENGTH },
      xsph: { c: XSPH_C },
      __forceUnfusedPostAdvect: opts.forceUnfused,
    });

    const loop = new SimLoop(particles, {
      substeps: SUBSTEPS,
      iterations: 2,
      xpbd,
      hashGrid,
      materials: [fluid],
    });
    // Cube falls in vacuum — gravity is the only force driving motion;
    // vorticity + xsph corrections sit on top. Floor disabled so the
    // velocities at the end of the run are entirely the result of the
    // post-advect block, not collision response.
    loop.gravity.set(0, -9.81, 0);
    loop.kernels.floorY.value = -1e9;

    for (let f = 0; f < FRAMES; f++) {
      await loop.step(FRAME_DT);
    }

    const snap = await particles.readback();
    const out = new Float32Array(snap.velocities.length);
    out.set(snap.velocities);
    const invMass = (particles.invMass.value as { array: Float32Array }).array[0]!;

    particles.destroy();
    hashGrid.destroy();

    return { velocities: out, count, invMass };
  } finally {
    renderer.dispose();
  }
}

describe('Phase Perf-06 — vorticity+xsph substep-internal-ordering bound', () => {
  it('un-fused vs fused dispatch order: max|Δv| ≤ 5·ε·c·dt per substep', async () => {
    const a = await runScenario({ forceUnfused: true });
    const b = await runScenario({ forceUnfused: false });

    expect(b.count).toBe(a.count);
    expect(b.velocities.length).toBe(a.velocities.length);

    let maxAbsDelta = 0;
    let totalKineticEnergyA = 0;
    let totalKineticEnergyB = 0;
    for (let i = 0; i < a.count; i++) {
      const ax = a.velocities[4 * i + 0]!;
      const ay = a.velocities[4 * i + 1]!;
      const az = a.velocities[4 * i + 2]!;
      const bx = b.velocities[4 * i + 0]!;
      const by = b.velocities[4 * i + 1]!;
      const bz = b.velocities[4 * i + 2]!;
      const dx = Math.abs(ax - bx);
      const dy = Math.abs(ay - by);
      const dz = Math.abs(az - bz);
      const d = Math.max(dx, dy, dz);
      if (d > maxAbsDelta) maxAbsDelta = d;
      totalKineticEnergyA += 0.5 * (ax * ax + ay * ay + az * az);
      totalKineticEnergyB += 0.5 * (bx * bx + by * by + bz * bz);
    }
    // KE units omit mass (cancels in the ratio); particles share invMass.
    const keRelDelta =
      Math.abs(totalKineticEnergyA - totalKineticEnergyB) / Math.max(totalKineticEnergyA, 1e-30);

    // eslint-disable-next-line no-console
    console.info(
      `[order-bound] count=${a.count} totalSubsteps=${TOTAL_SUBSTEPS} ` +
        `maxAbsDelta=${maxAbsDelta.toExponential(3)} ` +
        `velocityBound=${VELOCITY_BOUND.toExponential(3)} ` +
        `keRelDelta=${keRelDelta.toExponential(3)}`,
    );

    // Per-particle velocity bound — `5·ε·c·dt` per substep accumulated
    // over the run. Generous one-OOM headroom over the analytic
    // linear-coupling estimate.
    expect(maxAbsDelta).toBeLessThanOrEqual(VELOCITY_BOUND);
    // Total kinetic-energy relative bound — generous 1% per the plan.
    expect(keRelDelta).toBeLessThanOrEqual(KINETIC_ENERGY_REL_BOUND);
  }, 180_000);
});
