import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';

import {
  ParticleSystem,
  SimLoop,
  SoftbodySystem,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// SoftbodySystem rigid-body dynamics through a full SimLoop.
//
// With `compliance = 1e-12` the shape-matching constraint is at its
// stiffness ceiling: every iteration snaps particles to the body's rigid
// goal. Under a uniform linear impulse the body must translate without
// deformation; under an angular impulse it must rotate without shear.

/**
 * Unit cube (edge length 1) with 8 particles at its corners. Centered
 * at origin so rest COM = 0.
 */
function unitCubeParticles(): [number, number, number][] {
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

function mat3Mul(a: readonly number[], b: readonly number[]): number[] {
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

describe('SoftbodySystem rigid-body dynamics', () => {
  it('near-rigid body under uniform impulse translates without deformation', async () => {
    const renderer = await createParticleRenderer();
    try {
      const rest = unitCubeParticles();
      const n = rest.length;
      const r = 0.05;

      const v0: [number, number, number] = [2, 0, 0];
      const initial: ParticleInit[] = rest.map((p) => ({
        position: [p[0], p[1], p[2]],
        velocity: [v0[0], v0[1], v0[2]],
      }));

      const particles = new ParticleSystem(renderer, n, r);
      particles.uploadParticles(initial);

      const restFlat = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        restFlat[3 * i + 0] = rest[i]![0];
        restFlat[3 * i + 1] = rest[i]![1];
        restFlat[3 * i + 2] = rest[i]![2];
      }

      const softbody = new SoftbodySystem(particles, {
        bodies: [{ range: { start: 0, count: n }, restPositions: restFlat, compliance: 1e-12 }],
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        gravity: new Vector3(0, 0, 0), // no gravity — isolate shape matching.
        materials: [softbody],
      });

      const frameDt = 1 / 60;
      const frames = 300; // 5 s of simulated time.

      for (let f = 0; f < frames; f++) {
        await loop.step(frameDt);
      }

      const snap = await particles.readback();

      // "Translate without deformation" is an intrinsic invariant: particle
      // positions relative to the body COM must match the rest
      // configuration. This is independent of how the body's COM has
      // integrated (COM-integration error over 1200 substeps at f32
      // precision is its own phenomenon, ~1e-4 m at v=2 m/s × 5 s; distinct
      // from shape deformation).
      let comX = 0,
        comY = 0,
        comZ = 0;
      for (let i = 0; i < n; i++) {
        comX += snap.positions[4 * i + 0]!;
        comY += snap.positions[4 * i + 1]!;
        comZ += snap.positions[4 * i + 2]!;
      }
      const invN = 1 / n;
      comX *= invN;
      comY *= invN;
      comZ *= invN;

      // Rest COM is origin for the unit cube (centered rest).
      let maxRelativeDeviation = 0;
      for (let i = 0; i < n; i++) {
        const relX = snap.positions[4 * i + 0]! - comX;
        const relY = snap.positions[4 * i + 1]! - comY;
        const relZ = snap.positions[4 * i + 2]! - comZ;
        const dx = relX - rest[i]![0];
        const dy = relY - rest[i]![1];
        const dz = relZ - rest[i]![2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d > maxRelativeDeviation) maxRelativeDeviation = d;
      }

      // Secondary sanity: COM should be near rest_COM + v·T.
      const T = frames * frameDt;
      const expectedComX = 0 + v0[0] * T;
      const expectedComY = 0 + v0[1] * T;
      const expectedComZ = 0 + v0[2] * T;
      const comDriftX = Math.abs(comX - expectedComX);
      const comDriftY = Math.abs(comY - expectedComY);
      const comDriftZ = Math.abs(comZ - expectedComZ);
      const comDrift = Math.sqrt(
        comDriftX * comDriftX + comDriftY * comDriftY + comDriftZ * comDriftZ,
      );

      console.info(
        `[rigid-translation] T=${T.toFixed(2)}s v0=(${v0.join(',')}) maxRelativeDeviation=${maxRelativeDeviation.toExponential(3)} m  comDrift=${comDrift.toExponential(3)} m`,
      );

      // Max per-particle deviation from rigid translation < 1e-4 m, measured
      // on the intrinsic rigidity metric (relative positions vs rest). COM
      // drift is a separate integrator precision concern tracked by the
      // secondary log line.
      expect(maxRelativeDeviation).toBeLessThan(1e-4);
      // COM drift at v=2, T=5s, 1200 substeps with f32 A_pq reduction
      // is ~4e-4 m empirically. This is not a deformation — the body
      // stays rigid — but rather cumulative integrator precision that
      // scales with translation × substep count. Loose bound here; the
      // 1e-4 tolerance is for the rigidity check, not the extrinsic
      // position check.
      expect(comDrift).toBeLessThan(5e-3);

      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('near-rigid body under angular impulse rotates without shear', async () => {
    const renderer = await createParticleRenderer();
    try {
      const rest = unitCubeParticles();
      const n = rest.length;
      const r = 0.05;

      // Angular velocity around z axis. Choose a modest ω so integrator
      // drift over the test duration stays small — this test validates
      // rigidity (no shear), not rotational fidelity.
      const omega = 2.0; // rad/s

      const initial: ParticleInit[] = rest.map((p) => ({
        position: [p[0], p[1], p[2]],
        // v = ω × r where ω = (0, 0, ω_z), r = (p.x, p.y, p.z)
        // => v = (-ω_z · p.y, ω_z · p.x, 0)
        velocity: [-omega * p[1], omega * p[0], 0],
      }));

      const particles = new ParticleSystem(renderer, n, r);
      particles.uploadParticles(initial);

      const restFlat = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        restFlat[3 * i + 0] = rest[i]![0];
        restFlat[3 * i + 1] = rest[i]![1];
        restFlat[3 * i + 2] = rest[i]![2];
      }

      const softbody = new SoftbodySystem(particles, {
        bodies: [{ range: { start: 0, count: n }, restPositions: restFlat, compliance: 1e-12 }],
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        gravity: new Vector3(0, 0, 0),
        materials: [softbody],
      });

      const frameDt = 1 / 60;
      const frames = 60; // 1 s — enough to complete roughly 1/π rotations at ω=2.

      // Pairwise rest distances for shear check.
      // Unit cube edges are length 1, face diagonals √2, body diagonals √3.
      const restDist = (i: number, j: number): number => {
        const dx = rest[i]![0] - rest[j]![0];
        const dy = rest[i]![1] - rest[j]![1];
        const dz = rest[i]![2] - rest[j]![2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
      };

      // Track bodyRotation orthogonality + determinant across frames.
      let maxOrthoError = 0;
      let maxDetError = 0;
      let maxEdgeDistError = 0;

      for (let f = 0; f < frames; f++) {
        await loop.step(frameDt);

        // Readback bodyRotation and verify orthogonality.
        const rotBuf = new Float32Array(
          await renderer.getArrayBufferAsync(softbody.bodyRotations.value),
        );
        // Each rotation is stored as three vec4 columns; take their xyz.
        const R = [0, 1, 2, 4, 5, 6, 8, 9, 10].map((k) => rotBuf[k]!);
        // R · R^T − I
        const Rt = [0, 3, 6, 1, 4, 7, 2, 5, 8].map((k) => R[k]!);
        const RRt = mat3Mul(R, Rt);
        for (let k = 0; k < 9; k++) {
          const target = k === 0 || k === 4 || k === 8 ? 1 : 0;
          maxOrthoError = Math.max(maxOrthoError, Math.abs(RRt[k]! - target));
        }
        const det =
          R[0]! * (R[4]! * R[8]! - R[5]! * R[7]!) -
          R[1]! * (R[3]! * R[8]! - R[5]! * R[6]!) +
          R[2]! * (R[3]! * R[7]! - R[4]! * R[6]!);
        maxDetError = Math.max(maxDetError, Math.abs(det - 1.0));

        // Pairwise distance check on a few edges — body diagonals are
        // the most sensitive to shear.
        const snap = await particles.readback();
        const dist = (i: number, j: number): number => {
          const dx = snap.positions[4 * i + 0]! - snap.positions[4 * j + 0]!;
          const dy = snap.positions[4 * i + 1]! - snap.positions[4 * j + 1]!;
          const dz = snap.positions[4 * i + 2]! - snap.positions[4 * j + 2]!;
          return Math.sqrt(dx * dx + dy * dy + dz * dz);
        };
        for (const [i, j] of [
          [0, 1], // edge
          [0, 2], // edge
          [0, 4], // edge
          [0, 7], // body diagonal
          [3, 4], // body diagonal
        ] as const) {
          const err = Math.abs(dist(i, j) - restDist(i, j));
          if (err > maxEdgeDistError) maxEdgeDistError = err;
        }
      }

      console.info(
        `[rigid-rotation] frames=${frames} ω=${omega} rad/s maxOrthoError=${maxOrthoError.toExponential(3)} maxDetError=${maxDetError.toExponential(3)} maxEdgeDistError=${maxEdgeDistError.toExponential(3)}`,
      );

      // det(R) ≈ +1, |R − orthogonal| < 1e-5.
      expect(maxOrthoError).toBeLessThan(1e-5);
      expect(maxDetError).toBeLessThan(1e-5);
      // Body rotates without shear — pairwise distances stay at rest
      // values. Integrator drift accumulates some residual; 1e-3 m on
      // a unit cube (0.1% of the shortest edge) is a comfortable bound.
      expect(maxEdgeDistError).toBeLessThan(1e-3);

      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
