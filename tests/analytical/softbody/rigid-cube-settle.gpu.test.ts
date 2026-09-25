import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';

import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import {
  RigidBodySystem,
  voxelize,
  type RigidBodyDef,
  type TriangleMesh,
} from '../../../src/softbody/index.js';

// Phase 15a G1 — rigid cube falls onto a plane and settles flat.
//
// Validates the unified contact pipeline end-to-end on a rigid body:
//   - Softbody's `contactGeometryExtension` produces correct (n, d) for
//     rigid-rigid pairs (paper §5.1 eqs. 17–20). N/A here — only one
//     body — but the rigid-vs-plane path goes through PrimitiveSet
//     colliders.
//   - The shared `emitContactSolveCorrection` helper consumes the
//     rigid-flagged participant's `contactInvMass` correctly through
//     the per-substep copy + softbody §5.2 override (when enabled).
//   - Friction (Macklin 2020 §3.5/§3.6) damps the cube to rest without
//     rotation, even though there is no rigid-specific friction code.
//
// Settled state for a unit cube (edge E = 1, particleRadius r = 0.05,
// spacingFactor = 1) on a plane at y = 0:
//   - Body COM at y ≈ E/2 = 0.5 (bottom-particle outer sphere flush
//     with plane → bottom-particle center at y = r → COM at center of
//     a vertically-symmetric particle cloud → y = E/2).
//   - All particle velocities near zero (no residual motion).
//   - Body rotation matrix near identity (cube settled on a face, not
//     an edge or corner).

function unitCubeMesh(): TriangleMesh {
  // eslint-disable-next-line prettier/prettier
  const vertices = new Float32Array([
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
  ]);
  // eslint-disable-next-line prettier/prettier
  const indices = new Uint32Array([
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3,
    5, 3, 7, 5,
  ]);
  return { vertices, indices };
}

describe('Phase 15a — rigid cube settle on plane (G1)', () => {
  it('voxelized rigid cube dropped on plane settles flat at y ≈ E/2 with no residual rotation', async () => {
    const renderer = await createParticleRenderer();
    try {
      const edgeLength = 1.0;
      const particleRadius = 0.05;
      const spacingFactor = 1.0; // commensurate, symmetric voxelization
      const dropHeight = 0.5; // start with COM at 1.0, falls 0.5 m to settle.

      const mesh = unitCubeMesh();
      const scaled: TriangleMesh = {
        vertices: mesh.vertices.map((v) => v * edgeLength),
        indices: mesh.indices,
      };
      const cube = voxelize(scaled, {
        particleRadius,
        bakeSdf: true,
        spacingFactor,
      });
      const n = cube.count;

      // Body-local Y extent, used to place the cube so its bottom
      // particle's outer sphere is at world y = `dropHeight`.
      let minLocalY = Infinity;
      for (let i = 0; i < n; i++) {
        const ly = cube.positions[3 * i + 1]!;
        if (ly < minLocalY) minLocalY = ly;
      }
      const cy = dropHeight + particleRadius - minLocalY;

      const initial: ParticleInit[] = [];
      const restFlat = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const lx = cube.positions[3 * i + 0]!;
        const ly = cube.positions[3 * i + 1]!;
        const lz = cube.positions[3 * i + 2]!;
        restFlat[3 * i + 0] = lx;
        restFlat[3 * i + 1] = ly;
        restFlat[3 * i + 2] = lz;
        initial.push({
          position: [lx, ly + cy, lz],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: (1 & 0xffff) << 16,
        });
      }

      const particles = new ParticleSystem(renderer, n, particleRadius);
      particles.uploadParticles(initial);

      const xpbd = createXpbdUniforms(1 / 60);

      const hashGrid = new HashGrid(particles, {
        cellSize: particleRadius * 2.1,
      });

      const bodies: RigidBodyDef[] = [
        {
          particleRange: { start: 0, count: n },
          restPositions: restFlat,
          restSDF: cube.restSDF!,
          phaseId: 1,
          compliance: 0,
        },
      ];

      const rigid = new RigidBodySystem({
        particles,
        xpbd,
        bodies,
      });

      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.6,
        muK: 0.4,
      });
      colliders.upload();

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        xpbd,
        hashGrid,
        colliders: { colliders },
        materials: [rigid],
        contact: {
          hashGrid,
          maxContacts: Math.max(8192, n * 6),
          friction: { muS: 0.6, muK: 0.4 },
          stabIters: 1,
        },
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.8, 0);

      const frameDt = 1 / 60;
      const frames = 240; // 4 s — more than enough to settle a 0.5 m drop.
      for (let f = 0; f < frames; f++) {
        await loop.step(frameDt);
      }

      const snap = await particles.readback();
      let comX = 0,
        comY = 0,
        comZ = 0;
      let maxSpeed = 0;
      for (let i = 0; i < n; i++) {
        comX += snap.positions[4 * i + 0]!;
        comY += snap.positions[4 * i + 1]!;
        comZ += snap.positions[4 * i + 2]!;
        const vx = snap.velocities[4 * i + 0]!;
        const vy = snap.velocities[4 * i + 1]!;
        const vz = snap.velocities[4 * i + 2]!;
        const sp = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (sp > maxSpeed) maxSpeed = sp;
      }
      const invN = 1 / n;
      comX *= invN;
      comY *= invN;
      comZ *= invN;

      // Body rotation readback — identity within tolerance means the
      // cube settled flat on a face, not tipped onto an edge/corner.
      const rotBuf = new Float32Array(
        await renderer.getArrayBufferAsync(rigid.bodyRotations.value),
      );
      const R = [
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
      let maxRotDeviation = 0;
      const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
      for (let k = 0; k < 9; k++) {
        const e = Math.abs(R[k]! - identity[k]!);
        if (e > maxRotDeviation) maxRotDeviation = e;
      }

      // eslint-disable-next-line no-console
      console.info(
        `[rigid-cube-settle] T=${(frames * frameDt).toFixed(2)}s comY=${comY.toFixed(4)} maxSpeed=${maxSpeed.toExponential(3)} m/s maxRotDev=${maxRotDeviation.toExponential(3)}`,
      );

      // COM at E/2 ± 5%. Tolerance accounts for the §5.2-style settling
      // residual at compliance=0 (the body sinks slightly into the
      // contact band before stabilizing) and any voxelizer asymmetry.
      const expectedComY = edgeLength * 0.5;
      expect(Math.abs(comY - expectedComY)).toBeLessThan(0.05);

      // X / Z stay near origin — gravity is purely vertical, friction
      // damps any spurious lateral drift.
      expect(Math.abs(comX)).toBeLessThan(0.01);
      expect(Math.abs(comZ)).toBeLessThan(0.01);

      // Settled — friction has bled off all kinetic energy.
      expect(maxSpeed).toBeLessThan(0.05);

      // Body rotation matrix close to identity — cube did not tip.
      expect(maxRotDeviation).toBeLessThan(0.05);

      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
