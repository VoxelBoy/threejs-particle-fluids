import { describe, expect, it } from 'vitest';
import { uniform } from 'three/tsl';

import {
  ParticleSystem,
  SoftbodySystem,
  createParticleRenderer,
  type SolverContext,
} from '../../../src/index.js';

// Global shape matching: body rotation from the moment matrix and its polar
// decomposition (Müller et al. 2005 §3.3).
//
// Upload a body with a known rest configuration, then set its
// predictedPositions to `R_known · r_i + c` for a known rotation R_known and
// translation c. Run one solver iteration of the soft body's kernels and
// verify:
//   - bodyRotations[b] recovers R_known within f32 tolerance
//   - the recovered rotation is orthogonal (R · R^T ≈ I)
//
// The shape-matching correction reads this fitted frame (c, R) and moves
// each particle toward goal_i = R · r_i + c.
//
// Tolerance: 1e-3, the bounded error of a 9-component f32 reduction over N
// particles plus a polar decomposition with 24 Jacobi rotations.

const TOLERANCE = 1e-3;

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

function rotZ(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}
function rotY(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}
function rotX(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
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
  // 8-corner unit cube centred on origin (rest COM = 0, so pre-centering
  // is a no-op, which simplifies reasoning about the fitted frame).
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

function flatten(points: readonly (readonly [number, number, number])[]): Float32Array {
  const out = new Float32Array(points.length * 3);
  points.forEach((p, i) => out.set(p, i * 3));
  return out;
}

/** Row-major 3×3 of body `b` from `bodyRotations` (three vec4 rows per body). */
function readRotation(rotBuf: Float32Array, b: number): number[] {
  const base = b * 12;
  return [
    rotBuf[base + 0]!,
    rotBuf[base + 1]!,
    rotBuf[base + 2]!,
    rotBuf[base + 4]!,
    rotBuf[base + 5]!,
    rotBuf[base + 6]!,
    rotBuf[base + 8]!,
    rotBuf[base + 9]!,
    rotBuf[base + 10]!,
  ];
}

/**
 * Upload `rest` shapes (one body each, back to back), overwrite the predicted
 * positions with `predicted`, and run one solver iteration of the soft body.
 */
async function fitFrames(
  bodies: readonly {
    readonly rest: readonly [number, number, number][];
    readonly predicted: readonly [number, number, number][];
  }[],
): Promise<{ rotations: number[][]; centers: [number, number, number][] }> {
  const n = bodies.reduce((sum, body) => sum + body.rest.length, 0);
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, n, 0.05);

    // Committed positions = rest; uploadParticles writes predicted too, so
    // overwrite predictedPositions with the deformed configuration below.
    particles.uploadParticles(bodies.flatMap((body) => body.rest.map((p) => ({ position: p }))));
    const pp = particles.predictedPositions.value.array as Float32Array;
    let start = 0;
    const defs = bodies.map((body) => {
      if (body.rest.length !== body.predicted.length) {
        throw new Error('fitFrames: rest and predicted length mismatch');
      }
      body.predicted.forEach((q, i) => pp.set([q[0], q[1], q[2], 0], 4 * (start + i)));
      const def = {
        range: { start, count: body.rest.length },
        restPositions: flatten(body.rest),
        compliance: 1e-6,
      };
      start += body.rest.length;
      return def;
    });
    particles.predictedPositions.value.needsUpdate = true;

    const softbody = new SoftbodySystem(particles, { bodies: defs });
    const kernels = softbody.build(solverContext(particles, 1 / 60));
    await renderer.computeAsync([...(kernels.preSolve ?? []), ...(kernels.solve ?? [])]);

    const rotBuf = new Float32Array(
      await renderer.getArrayBufferAsync(softbody.bodyRotations.value),
    );
    const centerBuf = new Float32Array(
      await renderer.getArrayBufferAsync(softbody.bodyCenters.value),
    );
    particles.dispose();
    return {
      rotations: bodies.map((_, b) => readRotation(rotBuf, b)),
      centers: bodies.map(
        (_, b) =>
          [centerBuf[4 * b]!, centerBuf[4 * b + 1]!, centerBuf[4 * b + 2]!] as [
            number,
            number,
            number,
          ],
      ),
    };
  } finally {
    renderer.dispose();
  }
}

async function runPass2(
  rest: readonly [number, number, number][],
  predicted: readonly [number, number, number][],
): Promise<{ R: number[]; c: [number, number, number] }> {
  const { rotations, centers } = await fitFrames([{ rest, predicted }]);
  return { R: rotations[0]!, c: centers[0]! };
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

describe('SoftbodySystem body rotation (moment matrix + polar decomposition)', () => {
  it('no deformation → R = identity', async () => {
    const rest = unitCube();
    const { R } = await runPass2(rest, rest);
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
    // Shape matching decomposes the deformation into an (R, c) pair.
    // The rest COM of the unit cube is the origin, so after R·r + t the
    // current COM is t.
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
    // det(R) ≈ 1 and R·R^T ≈ I. Uses a rotation-only deformation so shape
    // matching has the simplest possible target.
    const theta = 0.9;
    const Rknown = rotZ(theta);
    const rest = unitCube();
    const deformed = rest.map((r) => matVec(Rknown, r));
    const { R } = await runPass2(rest, deformed);

    // Orthogonality: R · R^T ≈ I
    const Rt = [R[0]!, R[3]!, R[6]!, R[1]!, R[4]!, R[7]!, R[2]!, R[5]!, R[8]!];
    const RRt = matMul(R, Rt);
    expectMat3Close(RRt, [1, 0, 0, 0, 1, 0, 0, 0, 1], 1e-3);

    // Proper rotation: det = +1
    const det =
      R[0]! * (R[4]! * R[8]! - R[5]! * R[7]!) -
      R[1]! * (R[3]! * R[8]! - R[5]! * R[6]!) +
      R[2]! * (R[3]! * R[7]! - R[4]! * R[6]!);
    expect(Math.abs(det - 1.0)).toBeLessThan(1e-3);
  });

  it('multiple bodies with different rotations each produce their own R', async () => {
    // Two bodies in one ParticleSystem. The fit is dispatched one workgroup
    // per body, so each body gets its own (c, R) without cross-talk.
    // Body 0 at slots [0, 8); body 1 at slots [8, 16).
    const Rknown0 = rotZ(Math.PI / 4);
    const Rknown1 = rotY(-Math.PI / 6);
    const rest = unitCube();
    const { rotations } = await fitFrames([
      { rest, predicted: rest.map((r) => matVec(Rknown0, r)) },
      { rest, predicted: rest.map((r) => matVec(Rknown1, r)) },
    ]);
    expectMat3Close(rotations[0]!, Rknown0);
    expectMat3Close(rotations[1]!, Rknown1);
  });
});
