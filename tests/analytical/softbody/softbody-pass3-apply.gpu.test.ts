import { describe, expect, it } from 'vitest';

import {
  ParticleSystem,
  createParticleRenderer,
  createXpbdUniforms,
} from '../../../src/core/index.js';
import { SoftbodySystem } from '../../../src/softbody/index.js';

// Phase 10 G1 — Pass 3 (Δx apply) correctness.
//
// Pass 3 computes goal_i = R · r_i + c per iter and applies the XPBD
// per-component λ update (Macklin 2016 eq. 18 with identity Jacobian).
// Tests here cover:
//   (1) at rest configuration, Δx = 0 exactly;
//   (2) rigid translation matches goal → Δx = 0;
//   (3) rigid rotation matches goal → Δx = 0;
//   (4) single-particle perturbation at zero compliance collapses back
//       in one iter (high stiffness = goal-snap);
//   (5) high compliance barely corrects (bounded Δx);
//   (6) iterated solve converges to shape-matched configuration.
//
// dt is fixed at 1/240 s across tests — Pass 3's α̃ = α / dt² and the
// compliance values in each test are chosen so the rigid-like (α = 1e-12)
// and soft (α = 1e-4) regimes exercise distinct behaviors.

const DT = 1 / 240;
const XPBD = createXpbdUniforms(DT);

function unitCubeRest(): [number, number, number][] {
  // 8-corner unit cube around origin (rest COM = 0 simplifies reasoning).
  return [
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
  ];
}

function matVec(
  m: readonly number[],
  v: readonly [number, number, number],
): [number, number, number] {
  return [
    m[0]! * v[0] + m[1]! * v[1] + m[2]! * v[2],
    m[3]! * v[0] + m[4]! * v[1] + m[5]! * v[2],
    m[6]! * v[0] + m[7]! * v[1] + m[8]! * v[2],
  ];
}
function rotZ(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  // eslint-disable-next-line prettier/prettier
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}

interface RunArgs {
  readonly rest: readonly [number, number, number][];
  readonly initialPredicted: readonly [number, number, number][];
  readonly matchCompliance: number;
  readonly invMass?: number;
  readonly iters?: number;
}

interface RunResult {
  readonly finalPredicted: [number, number, number][];
  readonly bodyCenter: [number, number, number];
  readonly bodyRotation: number[]; // row-major 9-element
}

/**
 * Helper: run {@link SoftbodySystem}'s preIter + perIter kernels the
 * specified number of times and read back the particle state.
 */
async function runPass3(args: RunArgs): Promise<RunResult> {
  const { rest, initialPredicted, matchCompliance } = args;
  const invMass = args.invMass ?? 1;
  const iters = args.iters ?? 1;
  if (rest.length !== initialPredicted.length) {
    throw new Error('runPass3: rest and initialPredicted length mismatch');
  }
  const n = rest.length;
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, n, 0.05);
    particles.uploadParticles(
      rest.map((p) => ({
        position: [p[0], p[1], p[2]] as [number, number, number],
        velocity: [0, 0, 0] as [number, number, number],
        invMass,
        phase: 1,
      })),
    );
    // Override predictedPositions.
    const pp = particles.predictedPositions.value.array as Float32Array;
    for (let i = 0; i < n; i++) {
      const q = initialPredicted[i]!;
      pp[4 * i + 0] = q[0];
      pp[4 * i + 1] = q[1];
      pp[4 * i + 2] = q[2];
      pp[4 * i + 3] = 0;
    }
    particles.predictedPositions.value.needsUpdate = true;

    const restFlat = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const p = rest[i]!;
      restFlat[3 * i + 0] = p[0];
      restFlat[3 * i + 1] = p[1];
      restFlat[3 * i + 2] = p[2];
    }

    const softbody = new SoftbodySystem({
      particles,
      xpbd: XPBD,
      bodies: [
        {
          particleRange: { start: 0, count: n },
          restPositions: restFlat,
          surfaceFlag: new Uint8Array(n).fill(1),
          phaseId: 1,
          matchCompliance,
        },
      ],
    });

    // One substep: preIterKernels (reset λ + Pass 1 + Pass 2), then
    // `iters` iterations of perIterKernels (Pass 3).
    const pipeline = [
      ...softbody.preIterKernels,
      ...Array.from({ length: iters }).flatMap(() => [...softbody.perIterKernels]),
    ];
    await renderer.computeAsync(pipeline);

    const finalPred = new Float32Array(
      await renderer.getArrayBufferAsync(particles.predictedPositions.value),
    );
    const centerBuf = new Float32Array(
      await renderer.getArrayBufferAsync(softbody.bodyCenters.value),
    );
    const rotBuf = new Float32Array(
      await renderer.getArrayBufferAsync(softbody.bodyRotations.value),
    );
    const finalPredicted: [number, number, number][] = [];
    for (let i = 0; i < n; i++) {
      finalPredicted.push([finalPred[4 * i + 0]!, finalPred[4 * i + 1]!, finalPred[4 * i + 2]!]);
    }
    const bodyCenter: [number, number, number] = [centerBuf[0]!, centerBuf[1]!, centerBuf[2]!];
    const bodyRotation: number[] = [
      rotBuf[0]!,
      rotBuf[1]!,
      rotBuf[2]!, // eslint-disable-line prettier/prettier
      rotBuf[4]!,
      rotBuf[5]!,
      rotBuf[6]!, // eslint-disable-line prettier/prettier
      rotBuf[8]!,
      rotBuf[9]!,
      rotBuf[10]!, // eslint-disable-line prettier/prettier
    ];
    particles.destroy();
    return { finalPredicted, bodyCenter, bodyRotation };
  } finally {
    renderer.dispose();
  }
}

function maxPerParticleDiff(
  a: readonly (readonly [number, number, number])[],
  b: readonly (readonly [number, number, number])[],
): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    const dx = a[i]![0] - b[i]![0];
    const dy = a[i]![1] - b[i]![1];
    const dz = a[i]![2] - b[i]![2];
    d = Math.max(d, Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  return d;
}

describe('Phase 10 — SoftbodySystem Pass 3 (Δx apply)', () => {
  it('rest configuration: Δx = 0 exactly', async () => {
    const rest = unitCubeRest();
    const { finalPredicted } = await runPass3({
      rest,
      initialPredicted: rest,
      matchCompliance: 1e-12,
    });
    expect(maxPerParticleDiff(finalPredicted, rest)).toBeLessThan(1e-5);
  });

  it('rigidly translated body: Δx = 0 (near-rigid compliance)', async () => {
    const rest = unitCubeRest();
    const t: [number, number, number] = [3, -1, 2];
    const translated = rest.map(
      (r) => [r[0] + t[0], r[1] + t[1], r[2] + t[2]] as [number, number, number],
    );
    const { finalPredicted } = await runPass3({
      rest,
      initialPredicted: translated,
      matchCompliance: 1e-12,
    });
    // All particles stay at their translated positions — goal matches x*.
    expect(maxPerParticleDiff(finalPredicted, translated)).toBeLessThan(1e-4);
  });

  it('rigidly rotated body about rest COM: Δx = 0', async () => {
    const rest = unitCubeRest();
    const Rknown = rotZ(Math.PI / 4);
    const rotated = rest.map((r) => matVec(Rknown, r));
    const { finalPredicted, bodyRotation } = await runPass3({
      rest,
      initialPredicted: rotated,
      matchCompliance: 1e-12,
    });
    // Pass 2 should recover R_known; Pass 3 then has goal = x* so no
    // correction.
    expect(maxPerParticleDiff(finalPredicted, rotated)).toBeLessThan(1e-4);
    // Sanity: the recovered rotation matches the applied rotation.
    for (let k = 0; k < 9; k++) {
      expect(Math.abs(bodyRotation[k]! - Rknown[k]!)).toBeLessThan(1e-3);
    }
  });

  it('one perturbed particle at near-zero compliance: single iter snaps to the shape-matched goal', async () => {
    // Near-rigid compliance → one iter should drive every particle
    // essentially to its shape-matched goal (Macklin 2016 eq. 18 with
    // identity Jacobian converges in one iter when α → 0). The final
    // residual against the goal — NOT against rest — is the correct
    // convergence metric, because shape matching absorbs rigid motion
    // but leaves a per-particle residual for single-particle
    // perturbations (the best-fit (R, c) are biased by the perturbation).
    const rest = unitCubeRest();
    const initialPredicted = rest.map((r) => [r[0], r[1], r[2]] as [number, number, number]);
    initialPredicted[0] = [rest[0]![0] + 0.5, rest[0]![1], rest[0]![2]];
    const { finalPredicted, bodyCenter, bodyRotation } = await runPass3({
      rest,
      initialPredicted,
      matchCompliance: 1e-12,
      iters: 1,
    });
    // Compute shape-matched goals CPU-side: goal_i = R · r_i + c.
    // Rest COM for this unit cube is origin, so r_i = rest[i].
    const goals = rest.map((r) => {
      const rotated = matVec(bodyRotation, r);
      return [
        rotated[0] + bodyCenter[0],
        rotated[1] + bodyCenter[1],
        rotated[2] + bodyCenter[2],
      ] as [number, number, number];
    });
    expect(maxPerParticleDiff(finalPredicted, goals)).toBeLessThan(1e-4);
  });

  it('one perturbed particle at high compliance: single iter barely moves it', async () => {
    // Soft compliance — α = 1e-2, α̃ = 5760. w = 1.
    //   Δλ = −C / (w + α̃) = −C / 5761 → very small Δx.
    // After one iter the particle should still be near its initial
    // (perturbed) position.
    const rest = unitCubeRest();
    const initialPredicted = rest.map((r) => [r[0], r[1], r[2]] as [number, number, number]);
    initialPredicted[0] = [rest[0]![0] + 0.5, rest[0]![1], rest[0]![2]];
    const { finalPredicted } = await runPass3({
      rest,
      initialPredicted,
      matchCompliance: 1e-2,
      iters: 1,
    });
    // The soft-body correction in one iter is at most a few percent
    // of the perturbation.
    const delta0After = Math.abs(finalPredicted[0]![0] - initialPredicted[0]![0]);
    expect(delta0After).toBeLessThan(0.05);
  });

  it('multi-iter solve at zero compliance converges to the shape-matched goal', async () => {
    // Ten iters, α → 0. XPBD with identity Jacobian is already at
    // fixed point after iter 1; further iters should leave the
    // per-particle residual against goal at f32 noise.
    const rest = unitCubeRest();
    const initialPredicted = rest.map((r) => [r[0], r[1], r[2]] as [number, number, number]);
    initialPredicted[0] = [rest[0]![0] + 0.5, rest[0]![1], rest[0]![2]];

    const { finalPredicted, bodyCenter, bodyRotation } = await runPass3({
      rest,
      initialPredicted,
      matchCompliance: 1e-12,
      iters: 10,
    });

    // Residual against the shape-matched goal: after convergence
    // |x* − goal| should be at f32 noise level.
    const goals = rest.map((r) => {
      const rotated = matVec(bodyRotation, r);
      return [
        rotated[0] + bodyCenter[0],
        rotated[1] + bodyCenter[1],
        rotated[2] + bodyCenter[2],
      ] as [number, number, number];
    });
    expect(maxPerParticleDiff(finalPredicted, goals)).toBeLessThan(1e-5);
  });

  it('kinematic particle (invMass = 0) is not corrected', async () => {
    // XPBD Δx = w · Δλ with w = 0 gives Δx = 0. Verify a kinematic
    // pinned body does not drift under shape matching.
    const rest = unitCubeRest();
    const translated = rest.map((r) => [r[0] + 1, r[1], r[2]] as [number, number, number]);
    const { finalPredicted } = await runPass3({
      rest,
      initialPredicted: translated,
      matchCompliance: 1e-12,
      invMass: 0,
    });
    // Even at near-zero compliance, w = 0 means Δx = 0 per particle.
    expect(maxPerParticleDiff(finalPredicted, translated)).toBeLessThan(1e-5);
  });

  it('per-substep λ reset: λ is written during an iter and zeroed at the next substep start', async () => {
    // Correctness check on λ reset. Perturb a single particle (rigid
    // motion leaves C = 0 → zero λ, so we need a non-rigid config to
    // exercise the λ write). After an iter, some λ slots hold non-zero
    // Δλ; after the next substep's preIterKernels fire, λ is all zero.
    const rest = unitCubeRest();
    const initialPredicted = rest.map((r) => [r[0], r[1], r[2]] as [number, number, number]);
    initialPredicted[0] = [rest[0]![0] + 0.3, rest[0]![1], rest[0]![2]];

    // Run two back-to-back single-iter substeps. The λ reset at the
    // start of each preIter should make substep 2 begin from its
    // (already-corrected) predictedPositions as if it were a fresh
    // substep — no accumulated λ carry.
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, rest.length, 0.05);
      particles.uploadParticles(
        rest.map((p) => ({
          position: [p[0], p[1], p[2]] as [number, number, number],
          velocity: [0, 0, 0] as [number, number, number],
          invMass: 1,
          phase: 1,
        })),
      );
      const pp = particles.predictedPositions.value.array as Float32Array;
      for (let i = 0; i < rest.length; i++) {
        const q = initialPredicted[i]!;
        pp[4 * i + 0] = q[0];
        pp[4 * i + 1] = q[1];
        pp[4 * i + 2] = q[2];
        pp[4 * i + 3] = 0;
      }
      particles.predictedPositions.value.needsUpdate = true;

      const restFlat = new Float32Array(rest.length * 3);
      for (let i = 0; i < rest.length; i++) {
        const p = rest[i]!;
        restFlat[3 * i + 0] = p[0];
        restFlat[3 * i + 1] = p[1];
        restFlat[3 * i + 2] = p[2];
      }
      const softbody = new SoftbodySystem({
        particles,
        xpbd: XPBD,
        bodies: [
          {
            particleRange: { start: 0, count: rest.length },
            restPositions: restFlat,
            surfaceFlag: new Uint8Array(rest.length).fill(1),
            phaseId: 1,
            matchCompliance: 1e-6,
          },
        ],
      });

      // Substep 1
      await renderer.computeAsync([...softbody.preIterKernels, ...softbody.perIterKernels]);
      // Read lambda mid-way
      const lambdaMid = new Float32Array(await renderer.getArrayBufferAsync(softbody.lambda.value));
      // Find a softbody slot with non-zero lambda (confirms Pass 3 did write)
      let anyNonZero = false;
      for (let i = 0; i < rest.length * 4; i++) {
        if (Math.abs(lambdaMid[i]!) > 1e-10) {
          anyNonZero = true;
          break;
        }
      }
      expect(anyNonZero).toBe(true);

      // Substep 2 — triggers lambda reset at start, should zero lambda
      await renderer.computeAsync([...softbody.preIterKernels]);
      const lambdaAfterReset = new Float32Array(
        await renderer.getArrayBufferAsync(softbody.lambda.value),
      );
      for (let i = 0; i < rest.length * 4; i++) {
        expect(lambdaAfterReset[i]).toBe(0);
      }
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });
});
