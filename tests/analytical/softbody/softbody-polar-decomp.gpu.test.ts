import { describe, expect, it } from 'vitest';
import { Fn, instancedArray } from 'three/tsl';

import { createParticleRenderer } from '../../../src/core/index.js';
import { emitPolarDecomposition, type Mat3Nodes } from '../../../src/softbody/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

// Phase 10 G1 — emitPolarDecomposition unit tests.
//
// Construct A = R_known · S_known and verify the polar decomposition
// recovers R_known. Covers:
//   - identity (R = I, S = I)
//   - pure rotations around each axis (R = R_known, S = I)
//   - pure stretches on the diagonal (R = I, S = diag(s1, s2, s3))
//   - rotation + non-uniform stretch (R = R_known, S non-trivial)
//
// Tolerance: the polar decomposition is a finite sequence of elementary
// arithmetic ops on f32 registers + 24 guarded Jacobi rotations. Per-
// element error stays well under 1e-5 for the configurations tested
// here. Use 1e-4 as the assertion threshold to leave headroom.
const TOLERANCE = 1e-4;

async function runPolarDecomposition(a: readonly number[]): Promise<number[]> {
  if (a.length !== 9) {
    throw new Error(`runPolarDecomposition: A must have 9 elements, got ${a.length}`);
  }
  const renderer = await createParticleRenderer();
  try {
    const aBuf = instancedArray(9, 'float');
    const rBuf = instancedArray(9, 'float');
    const aArr = aBuf.value.array as Float32Array;
    for (let k = 0; k < 9; k++) aArr[k] = a[k]!;
    aBuf.value.needsUpdate = true;

    const kernel = Fn(() => {
      // Snapshot each A entry into a local var so downstream reads hit
      // registers, not storage. Nine-use fan-out from the polar decomp
      // would otherwise become nine storage reads.
      const aNodes: Mat3Nodes = {
        m00: aBuf.element(0).toVar(),
        m01: aBuf.element(1).toVar(),
        m02: aBuf.element(2).toVar(),
        m10: aBuf.element(3).toVar(),
        m11: aBuf.element(4).toVar(),
        m12: aBuf.element(5).toVar(),
        m20: aBuf.element(6).toVar(),
        m21: aBuf.element(7).toVar(),
        m22: aBuf.element(8).toVar(),
      };
      const R: Mat3Nodes = emitPolarDecomposition(aNodes);
      rBuf.element(0).assign(R.m00);
      rBuf.element(1).assign(R.m01);
      rBuf.element(2).assign(R.m02);
      rBuf.element(3).assign(R.m10);
      rBuf.element(4).assign(R.m11);
      rBuf.element(5).assign(R.m12);
      rBuf.element(6).assign(R.m20);
      rBuf.element(7).assign(R.m21);
      rBuf.element(8).assign(R.m22);
    })().compute(1);

    await renderer.computeAsync([kernel]);
    const out = new Float32Array(await renderer.getArrayBufferAsync(rBuf.value));
    return Array.from(out.slice(0, 9));
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

// --- Mat3 helpers (row-major, flat 9-element arrays) ---
function matMul(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array(9).fill(0) as number[];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        s += a[3 * i + k]! * b[3 * k + j]!;
      }
      out[3 * i + j] = s;
    }
  }
  return out;
}

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
function diag(sx: number, sy: number, sz: number): number[] {
  // eslint-disable-next-line prettier/prettier
  return [sx, 0, 0, 0, sy, 0, 0, 0, sz];
}
const IDENTITY: number[] = diag(1, 1, 1);

describe('Phase 10 — emitPolarDecomposition', () => {
  it('identity A yields identity R', async () => {
    const R = await runPolarDecomposition(IDENTITY);
    expectMat3Close(R, IDENTITY);
  });

  it('pure rotation around z recovers the rotation', async () => {
    const theta = Math.PI / 4;
    const A = rotZ(theta);
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, A);
  });

  it('pure rotation around y recovers the rotation', async () => {
    const theta = Math.PI / 3;
    const A = rotY(theta);
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, A);
  });

  it('pure rotation around x recovers the rotation', async () => {
    const theta = -Math.PI / 6;
    const A = rotX(theta);
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, A);
  });

  it('composed rotation recovers the composition', async () => {
    const A = matMul(matMul(rotZ(0.3), rotY(-0.7)), rotX(0.5));
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, A);
  });

  it('pure uniform stretch yields identity R', async () => {
    const A = diag(2.5, 2.5, 2.5);
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, IDENTITY);
  });

  it('pure non-uniform diagonal stretch yields identity R', async () => {
    const A = diag(3.0, 1.5, 0.8);
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, IDENTITY);
  });

  it('rotation then non-uniform stretch recovers the rotation', async () => {
    const Rknown = rotZ(Math.PI / 6);
    const S = diag(2.0, 1.0, 0.5);
    const A = matMul(Rknown, S);
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, Rknown);
  });

  it('rotation-plus-shear produces a valid rotation matrix', async () => {
    // Compose a random rotation with a symmetric stretch; R recovers
    // the rotation. Shear (non-diagonal S) is the stress case for polar
    // decomposition — S is symmetric PSD by construction here.
    const Rknown = matMul(rotX(0.4), rotZ(-0.3));
    // Symmetric PSD S: diag + small symmetric off-diagonals.
    // S[0][1] = S[1][0] = 0.2; rest diag.
    // eslint-disable-next-line prettier/prettier
    const S = [1.5, 0.2, 0, 0.2, 1.2, 0, 0, 0, 1.0];
    const A = matMul(Rknown, S);
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, Rknown);

    // Additional structural check: R should be orthogonal (R · R^T ≈ I)
    // and proper (det ≈ +1).
    const Rt = [R[0]!, R[3]!, R[6]!, R[1]!, R[4]!, R[7]!, R[2]!, R[5]!, R[8]!];
    const RRt = matMul(R, Rt);
    expectMat3Close(RRt, IDENTITY, 1e-4);
    const det =
      R[0]! * (R[4]! * R[8]! - R[5]! * R[7]!) -
      R[1]! * (R[3]! * R[8]! - R[5]! * R[6]!) +
      R[2]! * (R[3]! * R[7]! - R[4]! * R[6]!);
    expect(Math.abs(det - 1.0)).toBeLessThan(1e-4);
  });

  it('small rotation (near-identity) stays well-conditioned', async () => {
    const theta = 1e-3;
    const A = rotZ(theta);
    const R = await runPolarDecomposition(A);
    expectMat3Close(R, A, 1e-5);
  });
});
