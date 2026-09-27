import { describe, expect, it } from 'vitest';
import { instancedArray, uniform } from 'three/tsl';

import {
  ParticleSystem,
  SoftbodySystem,
  createParticleRenderer,
  type SolverContext,
} from '../../../src/index.js';
import {
  buildCenterOfMassKernel,
  buildMomentAndPolarDecompKernel,
  buildResetLambdaKernel,
  buildShapeMatchDeltaApplyKernel,
} from '../../../src/softbody/shapeMatch.js';

// Global shape matching: the per-particle Δx correction.
//
// Every iteration the solver computes goal_i = R · r_i + c and applies the
// XPBD per-component λ update (Macklin et al. 2016 eq. 18 with an identity
// Jacobian). Tests here cover:
//   (1) at the rest configuration, Δx = 0 exactly;
//   (2) rigid translation matches goal → Δx = 0;
//   (3) rigid rotation matches goal → Δx = 0;
//   (4) single-particle perturbation at zero compliance collapses back
//       in one iteration (high stiffness = goal-snap);
//   (5) high compliance barely corrects (bounded Δx);
//   (6) iterated solve converges to the shape-matched configuration;
//   (7) kinematic particles are never corrected;
//   (8) λ accumulates within a substep and is reset at the next one.
//
// dt is fixed at 1/240 s across tests: the correction uses α̃ = α / dt², and
// the compliance values in each test are chosen so the rigid-like
// (α = 1e-12) and soft (α = 1e-2) regimes exercise distinct behaviors.

const DT = 1 / 240;

/** The solver state SimLoop hands a material, so its kernels can run without a loop. */
function solverContext(particles: ParticleSystem, dt: number): SolverContext {
  let group = 0;
  return {
    particles,
    dt: uniform(dt, 'float'),
    get hashGrid(): never {
      throw new Error('soft bodies do not use the neighbor grid');
    },
    allocateCollisionGroup: () => ++group,
  };
}

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
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}

/** Upload `rest` as committed positions, then overwrite the predicted positions. */
function uploadDeformed(
  particles: ParticleSystem,
  rest: readonly [number, number, number][],
  predicted: readonly [number, number, number][],
  invMass: number,
): void {
  if (rest.length !== predicted.length) {
    throw new Error('uploadDeformed: rest and predicted length mismatch');
  }
  particles.uploadParticles(rest.map((p) => ({ position: p, invMass })));
  const pp = particles.predictedPositions.value.array as Float32Array;
  predicted.forEach((q, i) => pp.set([q[0], q[1], q[2], 0], 4 * i));
  particles.predictedPositions.value.needsUpdate = true;
}

function flatten(points: readonly (readonly [number, number, number])[]): Float32Array {
  const out = new Float32Array(points.length * 3);
  points.forEach((p, i) => out.set(p, i * 3));
  return out;
}

interface RunArgs {
  readonly rest: readonly [number, number, number][];
  readonly initialPredicted: readonly [number, number, number][];
  readonly compliance: number;
  readonly invMass?: number;
  readonly iters?: number;
}

interface RunResult {
  readonly finalPredicted: [number, number, number][];
  readonly bodyCenter: [number, number, number];
  readonly bodyRotation: number[]; // row-major 9-element
}

/**
 * Run one substep of the soft body's kernels, without a SimLoop: its
 * pre-solve kernels, then `iters` solver iterations. Reads back the
 * predicted positions and the body frame fitted in the last iteration.
 */
async function runPass3(args: RunArgs): Promise<RunResult> {
  const { rest, initialPredicted, compliance } = args;
  const iters = args.iters ?? 1;
  const n = rest.length;
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, n, 0.05);
    uploadDeformed(particles, rest, initialPredicted, args.invMass ?? 1);

    const softbody = new SoftbodySystem(particles, {
      bodies: [{ range: { start: 0, count: n }, restPositions: flatten(rest), compliance }],
    });
    const kernels = softbody.build(solverContext(particles, DT));
    const pipeline = [
      ...(kernels.preSolve ?? []),
      ...Array.from({ length: iters }).flatMap(() => [...(kernels.solve ?? [])]),
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
      rotBuf[2]!,
      rotBuf[4]!,
      rotBuf[5]!,
      rotBuf[6]!,
      rotBuf[8]!,
      rotBuf[9]!,
      rotBuf[10]!,
    ];
    particles.dispose();
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

describe('SoftbodySystem shape-matching correction (Δx apply)', () => {
  it('rest configuration: Δx = 0 exactly', async () => {
    const rest = unitCubeRest();
    const { finalPredicted } = await runPass3({
      rest,
      initialPredicted: rest,
      compliance: 1e-12,
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
      compliance: 1e-12,
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
      compliance: 1e-12,
    });
    // The fit should recover R_known, so goal = x* and there is no
    // correction.
    expect(maxPerParticleDiff(finalPredicted, rotated)).toBeLessThan(1e-4);
    // Sanity: the recovered rotation matches the applied rotation.
    for (let k = 0; k < 9; k++) {
      expect(Math.abs(bodyRotation[k]! - Rknown[k]!)).toBeLessThan(1e-3);
    }
  });

  it('one perturbed particle at near-zero compliance: single iteration snaps to the shape-matched goal', async () => {
    // Near-rigid compliance → one iteration should drive every particle
    // essentially to its shape-matched goal (Macklin et al. 2016 eq. 18 with
    // identity Jacobian converges in one iteration when α → 0). The final
    // residual against the goal — NOT against rest — is the correct
    // convergence metric, because shape matching absorbs rigid motion but
    // leaves a per-particle residual for single-particle perturbations (the
    // best-fit (R, c) are biased by the perturbation).
    const rest = unitCubeRest();
    const initialPredicted = rest.map((r) => [r[0], r[1], r[2]] as [number, number, number]);
    initialPredicted[0] = [rest[0]![0] + 0.5, rest[0]![1], rest[0]![2]];
    const { finalPredicted, bodyCenter, bodyRotation } = await runPass3({
      rest,
      initialPredicted,
      compliance: 1e-12,
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

  it('one perturbed particle at high compliance: single iteration barely moves it', async () => {
    // Soft compliance — α = 1e-2, α̃ = α / dt² = 576, w = 1.
    //   Δλ = −C / (w + α̃) = −C / 577 → very small Δx.
    // After one iteration the particle should still be near its initial
    // (perturbed) position.
    const rest = unitCubeRest();
    const initialPredicted = rest.map((r) => [r[0], r[1], r[2]] as [number, number, number]);
    initialPredicted[0] = [rest[0]![0] + 0.5, rest[0]![1], rest[0]![2]];
    const { finalPredicted } = await runPass3({
      rest,
      initialPredicted,
      compliance: 1e-2,
      iters: 1,
    });
    // The soft-body correction in one iteration is at most a few percent of
    // the perturbation.
    const delta0After = Math.abs(finalPredicted[0]![0] - initialPredicted[0]![0]);
    expect(delta0After).toBeLessThan(0.05);
  });

  it('multi-iteration solve at zero compliance converges to the shape-matched goal', async () => {
    // Ten iterations, α → 0. XPBD with identity Jacobian is already at its
    // fixed point after the first iteration; further iterations should leave
    // the per-particle residual against the goal at f32 noise.
    const rest = unitCubeRest();
    const initialPredicted = rest.map((r) => [r[0], r[1], r[2]] as [number, number, number]);
    initialPredicted[0] = [rest[0]![0] + 0.5, rest[0]![1], rest[0]![2]];

    const { finalPredicted, bodyCenter, bodyRotation } = await runPass3({
      rest,
      initialPredicted,
      compliance: 1e-12,
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
    // XPBD Δx = w · Δλ with w = 0 gives Δx = 0. Verify a kinematic body does
    // not drift under shape matching.
    const rest = unitCubeRest();
    const translated = rest.map((r) => [r[0] + 1, r[1], r[2]] as [number, number, number]);
    const { finalPredicted } = await runPass3({
      rest,
      initialPredicted: translated,
      compliance: 1e-12,
      invMass: 0,
    });
    // Even at near-zero compliance, w = 0 means Δx = 0 per particle.
    expect(maxPerParticleDiff(finalPredicted, translated)).toBeLessThan(1e-5);
  });

  it('per-substep λ reset: λ is written during an iteration and zeroed at the next substep start', async () => {
    // Correctness check on the λ reset. Perturb a single particle (rigid
    // motion leaves C = 0 → zero λ, so a non-rigid configuration is needed
    // to exercise the λ write). After an iteration, some λ slots hold a
    // non-zero Δλ; after the next substep's reset kernel, λ is all zero.
    //
    // λ is private to SoftbodySystem, so this test assembles the same
    // kernels from their builders around a λ buffer it can read.
    const rest = unitCubeRest();
    const n = rest.length;
    const initialPredicted = rest.map((r) => [r[0], r[1], r[2]] as [number, number, number]);
    initialPredicted[0] = [rest[0]![0] + 0.3, rest[0]![1], rest[0]![2]];

    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, n, 0.05);
      uploadDeformed(particles, rest, initialPredicted, 1);

      // The unit cube's rest centroid is the origin, so its rest positions
      // are already the centered rest offsets.
      const restOffsets = instancedArray(n, 'vec4');
      rest.forEach((r, i) =>
        (restOffsets.value.array as Float32Array).set([r[0], r[1], r[2], 0], 4 * i),
      );
      restOffsets.value.needsUpdate = true;
      const lambda = instancedArray(n, 'vec4');
      const bodyCenters = instancedArray(1, 'vec4');
      const bodyRotations = instancedArray(3, 'vec4');
      const weights = instancedArray(new Float32Array(n).fill(1), 'float');
      const shared = {
        particles,
        bodyStart: instancedArray(new Uint32Array([0]), 'uint'),
        bodyCount: instancedArray(new Uint32Array([n]), 'uint'),
        numBodies: 1,
      };
      const resetLambda = buildResetLambdaKernel({ particles, lambda });
      const iteration = [
        buildCenterOfMassKernel({ ...shared, weights, bodyCenters }),
        buildMomentAndPolarDecompKernel({ ...shared, weights, restOffsets, bodyRotations }),
        buildShapeMatchDeltaApplyKernel({
          ...shared,
          restOffsets,
          bodyCenters,
          bodyRotations,
          bodyCompliance: instancedArray(new Float32Array([1e-6]), 'float'),
          lambda,
          dt: uniform(DT, 'float'),
        }),
      ];

      // Substep 1: reset, then one iteration.
      await renderer.computeAsync([resetLambda, ...iteration]);
      const lambdaMid = new Float32Array(await renderer.getArrayBufferAsync(lambda.value));
      // Find a slot with non-zero λ (confirms the correction wrote it).
      let anyNonZero = false;
      for (let i = 0; i < n * 4; i++) {
        if (Math.abs(lambdaMid[i]!) > 1e-10) {
          anyNonZero = true;
          break;
        }
      }
      expect(anyNonZero).toBe(true);

      // Substep 2 starts with the reset, which must zero λ.
      await renderer.computeAsync([resetLambda]);
      const lambdaAfterReset = new Float32Array(await renderer.getArrayBufferAsync(lambda.value));
      for (let i = 0; i < n * 4; i++) {
        expect(lambdaAfterReset[i]).toBe(0);
      }
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });
});
