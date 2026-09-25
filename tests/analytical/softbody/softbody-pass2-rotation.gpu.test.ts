import { describe, expect, it } from 'vitest';

import {
  ParticleSystem,
  createParticleRenderer,
  createXpbdUniforms,
} from '../../../src/core/index.js';
import { SoftbodySystem } from '../../../src/softbody/index.js';

const XPBD_FOR_TESTS = createXpbdUniforms(1 / 60);

// Phase 10 G1 — Pass 2 (moment + polar decomposition) integration.
//
// Upload a body with a known rest configuration, then set its
// predictedPositions to `R_known · r_i + c` for a known rotation R_known
// and translation c. Dispatch Pass 1 + Pass 2 and verify:
//   - bodyRotations[b] recovers R_known within f32 tolerance
//   - the recovered rotation is orthogonal (R · R^T ≈ I)
//
// This is the integration path Pass 3 (Δx apply) will consume:
// preIterKernels pre-compute (c, R) from x*_i; Pass 3 reads them and
// computes goal_i = R · r_i + c on the fly.
//
// Tolerance: 1e-3 (bounded-max-error tier 2 per ARCH §Guardrails G4
// for a 9-component f32 reduction over N particles plus a polar
// decomposition with 24 Jacobi rotations).

const TOLERANCE = 1e-3;

function rotZ(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  // eslint-disable-next-line prettier/prettier
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}
function rotY(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  // eslint-disable-next-line prettier/prettier
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}
function rotX(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  // eslint-disable-next-line prettier/prettier
  return [1, 0, 0, 0, c, -s, 0, s, c];
}
function matMul(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array(9).fill(0) as number[];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[3 * i + k]! * b[3 * k + j]!;
      out[3 * i + j] = s;
    }
  }
  return out;
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

function unitCube(): [number, number, number][] {
  // 8-corner unit cube centred on origin (rest COM = 0 → pre-centering
  // is a no-op, simplifies reasoning about what Pass 2 reconstructs).
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

async function runPass2(
  restPositions: readonly [number, number, number][],
  predictedPositions: readonly [number, number, number][],
): Promise<{ R: number[]; c: [number, number, number] }> {
  if (restPositions.length !== predictedPositions.length) {
    throw new Error('runPass2: rest and predicted length mismatch');
  }
  const n = restPositions.length;
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, n, 0.05);

    // Upload: committed = rest, but then overwrite predictedPositions
    // with the deformed config below (uploadParticles writes both).
    const initData = restPositions.map((p, _i) => ({
      position: [p[0], p[1], p[2]] as [number, number, number],
      velocity: [0, 0, 0] as [number, number, number],
      invMass: 1,
      phase: 1,
    }));
    particles.uploadParticles(initData);

    // Overwrite predictedPositions with the post-deformation configuration.
    const pp = particles.predictedPositions.value.array as Float32Array;
    for (let i = 0; i < n; i++) {
      const q = predictedPositions[i]!;
      pp[4 * i + 0] = q[0];
      pp[4 * i + 1] = q[1];
      pp[4 * i + 2] = q[2];
      pp[4 * i + 3] = 0;
    }
    particles.predictedPositions.value.needsUpdate = true;

    const rest = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const p = restPositions[i]!;
      rest[3 * i + 0] = p[0];
      rest[3 * i + 1] = p[1];
      rest[3 * i + 2] = p[2];
    }

    const softbody = new SoftbodySystem({
      particles,
      xpbd: XPBD_FOR_TESTS,
      bodies: [
        {
          particleRange: { start: 0, count: n },
          restPositions: rest,
          surfaceFlag: new Uint8Array(n).fill(1),
          phaseId: 1,
          matchCompliance: 1e-6,
        },
      ],
    });

    // Dispatch preIterKernels (Pass 1 + Pass 2).
    await renderer.computeAsync([...softbody.preIterKernels]);

    const rotBuf = new Float32Array(
      await renderer.getArrayBufferAsync(softbody.bodyRotations.value),
    );
    const centerBuf = new Float32Array(
      await renderer.getArrayBufferAsync(softbody.bodyCenters.value),
    );
    // R is row-major in three contiguous vec4 slots.
    const R: number[] = [
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
    const c: [number, number, number] = [centerBuf[0]!, centerBuf[1]!, centerBuf[2]!];
    particles.destroy();
    return { R, c };
  } finally {
    renderer.dispose();
  }
}

function expectMat3Close(
  actual: readonly number[],
  expected: readonly number[],
  tol = TOLERANCE,
): void {
  for (let k = 0; k < 9; k++) {
    const label = `[${Math.floor(k / 3)}][${k % 3}]`;
    expect(
      Math.abs(actual[k]! - expected[k]!),
      `R${label}: got ${actual[k]!.toExponential(4)} expected ${expected[k]!.toExponential(4)}`,
    ).toBeLessThan(tol);
  }
}

describe('Phase 10 — SoftbodySystem Pass 2 (moment + polar decomp)', () => {
  it('no deformation → R = identity', async () => {
    const rest = unitCube();
    const { R } = await runPass2(rest, rest);
    // eslint-disable-next-line prettier/prettier
    expectMat3Close(R, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('rotation around z by π/4 → recovers R_z', async () => {
    const theta = Math.PI / 4;
    const Rknown = rotZ(theta);
    const rest = unitCube();
    const deformed = rest.map((r) => matVec(Rknown, r));
    const { R } = await runPass2(rest, deformed);
    expectMat3Close(R, Rknown);
  });

  it('rotation around y by π/3 → recovers R_y', async () => {
    const theta = Math.PI / 3;
    const Rknown = rotY(theta);
    const rest = unitCube();
    const deformed = rest.map((r) => matVec(Rknown, r));
    const { R } = await runPass2(rest, deformed);
    expectMat3Close(R, Rknown);
  });

  it('composed rotation → recovers the composition', async () => {
    const Rknown = matMul(matMul(rotZ(0.3), rotY(-0.7)), rotX(0.5));
    const rest = unitCube();
    const deformed = rest.map((r) => matVec(Rknown, r));
    const { R } = await runPass2(rest, deformed);
    expectMat3Close(R, Rknown);
  });

  it('rotation + translation → recovers rotation; c matches translation', async () => {
    // Shape-matching decomposes the deformation into (R, c) pair.
    // Rest COM for unit cube is origin, so after R·r + t the current
    // COM is t.
    const theta = Math.PI / 6;
    const Rknown = rotX(theta);
    const t: [number, number, number] = [3, -1, 2];
    const rest = unitCube();
    const deformed = rest.map((r) => {
      const rotated = matVec(Rknown, r);
      return [rotated[0] + t[0], rotated[1] + t[1], rotated[2] + t[2]] as [number, number, number];
    });
    const { R, c } = await runPass2(rest, deformed);
    expectMat3Close(R, Rknown);
    // c should equal the translation (rest COM = 0).
    expect(Math.abs(c[0] - t[0])).toBeLessThan(TOLERANCE);
    expect(Math.abs(c[1] - t[1])).toBeLessThan(TOLERANCE);
    expect(Math.abs(c[2] - t[2])).toBeLessThan(TOLERANCE);
  });

  it('recovered R is a valid orthogonal rotation', async () => {
    // Exercises the "proper rotation" check — det(R) ≈ 1 and R·R^T ≈ I.
    // Uses a rotation-only deformation so shape matching has the
    // simplest possible target.
    const theta = 0.9;
    const Rknown = rotZ(theta);
    const rest = unitCube();
    const deformed = rest.map((r) => matVec(Rknown, r));
    const { R } = await runPass2(rest, deformed);

    // Orthogonality: R · R^T ≈ I
    const Rt = [R[0]!, R[3]!, R[6]!, R[1]!, R[4]!, R[7]!, R[2]!, R[5]!, R[8]!];
    const RRt = matMul(R, Rt);
    // eslint-disable-next-line prettier/prettier
    expectMat3Close(RRt, [1, 0, 0, 0, 1, 0, 0, 0, 1], 1e-3);

    // Proper rotation: det = +1
    const det =
      R[0]! * (R[4]! * R[8]! - R[5]! * R[7]!) -
      R[1]! * (R[3]! * R[8]! - R[5]! * R[6]!) +
      R[2]! * (R[3]! * R[7]! - R[4]! * R[6]!);
    expect(Math.abs(det - 1.0)).toBeLessThan(1e-3);
  });

  it('multiple bodies with different rotations each produce their own R', async () => {
    // Two bodies in one ParticleSystem — Pass 2 is dispatched per-
    // workgroup-per-body so each body gets its own (c, R) without
    // cross-talk.
    const theta0 = Math.PI / 4;
    const theta1 = -Math.PI / 6;
    const Rknown0 = rotZ(theta0);
    const Rknown1 = rotY(theta1);
    const rest0 = unitCube();
    const rest1 = unitCube();
    // Body 0 at slots [0, 8); Body 1 at slots [8, 16).
    const def0 = rest0.map((r) => matVec(Rknown0, r));
    const def1 = rest1.map((r) => matVec(Rknown1, r));

    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 16, 0.05);
      const init: {
        position: [number, number, number];
        velocity: [number, number, number];
        invMass: number;
        phase: number;
      }[] = [];
      for (let i = 0; i < 8; i++) {
        init.push({
          position: [rest0[i]![0], rest0[i]![1], rest0[i]![2]],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 1,
        });
      }
      for (let i = 0; i < 8; i++) {
        init.push({
          position: [rest1[i]![0], rest1[i]![1], rest1[i]![2]],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 2,
        });
      }
      particles.uploadParticles(init);

      const pp = particles.predictedPositions.value.array as Float32Array;
      for (let i = 0; i < 8; i++) {
        pp[4 * i + 0] = def0[i]![0];
        pp[4 * i + 1] = def0[i]![1];
        pp[4 * i + 2] = def0[i]![2];
      }
      for (let i = 0; i < 8; i++) {
        pp[4 * (i + 8) + 0] = def1[i]![0];
        pp[4 * (i + 8) + 1] = def1[i]![1];
        pp[4 * (i + 8) + 2] = def1[i]![2];
      }
      particles.predictedPositions.value.needsUpdate = true;

      const restFlat0 = new Float32Array(24);
      const restFlat1 = new Float32Array(24);
      for (let i = 0; i < 8; i++) {
        restFlat0[3 * i + 0] = rest0[i]![0];
        restFlat0[3 * i + 1] = rest0[i]![1];
        restFlat0[3 * i + 2] = rest0[i]![2];
        restFlat1[3 * i + 0] = rest1[i]![0];
        restFlat1[3 * i + 1] = rest1[i]![1];
        restFlat1[3 * i + 2] = rest1[i]![2];
      }

      const softbody = new SoftbodySystem({
        particles,
        xpbd: XPBD_FOR_TESTS,
        bodies: [
          {
            particleRange: { start: 0, count: 8 },
            restPositions: restFlat0,
            surfaceFlag: new Uint8Array(8).fill(1),
            phaseId: 1,
            matchCompliance: 1e-6,
          },
          {
            particleRange: { start: 8, count: 8 },
            restPositions: restFlat1,
            surfaceFlag: new Uint8Array(8).fill(1),
            phaseId: 2,
            matchCompliance: 1e-6,
          },
        ],
      });

      await renderer.computeAsync([...softbody.preIterKernels]);
      const rotBuf = new Float32Array(
        await renderer.getArrayBufferAsync(softbody.bodyRotations.value),
      );
      // R0 at slots 0..2, R1 at slots 3..5.
      const R0: number[] = [
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
      const R1: number[] = [
        rotBuf[12]!,
        rotBuf[13]!,
        rotBuf[14]!, // eslint-disable-line prettier/prettier
        rotBuf[16]!,
        rotBuf[17]!,
        rotBuf[18]!, // eslint-disable-line prettier/prettier
        rotBuf[20]!,
        rotBuf[21]!,
        rotBuf[22]!, // eslint-disable-line prettier/prettier
      ];
      expectMat3Close(R0, Rknown0);
      expectMat3Close(R1, Rknown1);

      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });
});
