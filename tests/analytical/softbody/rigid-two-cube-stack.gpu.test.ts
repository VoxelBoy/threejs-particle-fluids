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

// Phase 15a G1 — two rigid cubes stacked on a plane.
//
// This is the minimal scene that exercises the full unified pipeline:
//   - Rigid-vs-rigid contact via softbody's `contactGeometryExtension`
//     (paper §5.1 SDF-based normal selection).
//   - Rigid-vs-plane contact via the spherical default (a non-rigid
//     pair flavour that flows through the same `emitContactSolveCorrection`
//     helper).
//   - Stiff-stack mass scaling (paper §5.2 eq. 21) writing
//     `particles.contactInvMass` from softbody's preIter, after the
//     SimLoop-dispatched per-substep copy.
//   - Friction (Macklin 2020 §3.5/§3.6) damping the upper cube's
//     residual lateral motion onto its lower neighbour.
//
// The plan-specified tolerance ("upper cube ≤ 5 % drift after 2 s")
// is the COM-vs-target horizontal drift bound. Vertical bound is
// asymmetric: the upper cube must NOT sink through the lower one
// (that would be the parallel-pipeline failure mode the refactor
// fixes).

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

describe('Phase 15a — two rigid cubes stacked on plane (G1)', () => {
  it('upper cube settles directly on lower cube without lateral drift', async () => {
    const renderer = await createParticleRenderer();
    try {
      const edgeLength = 1.0;
      const particleRadius = 0.05;
      const spacingFactor = 1.0;
      const dropHeight = 0.0;

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
      const perBodyCount = cube.count;
      const totalCount = perBodyCount * 2;

      // Stack-spacing derivation matches rigid-pile.ts demo: surface-
      // particle centers between adjacent cubes land at exactly 2r.
      let minLocalY = Infinity;
      let maxLocalY = -Infinity;
      for (let i = 0; i < perBodyCount; i++) {
        const ly = cube.positions[3 * i + 1]!;
        if (ly < minLocalY) minLocalY = ly;
        if (ly > maxLocalY) maxLocalY = ly;
      }
      const stackSpacing = maxLocalY - minLocalY + 2 * particleRadius;

      const initial: ParticleInit[] = [];
      const restPerBody: Float32Array[] = [];
      for (let b = 0; b < 2; b++) {
        const cy = dropHeight + particleRadius - minLocalY + b * stackSpacing;
        const rest = new Float32Array(3 * perBodyCount);
        for (let i = 0; i < perBodyCount; i++) {
          const lx = cube.positions[3 * i + 0]!;
          const ly = cube.positions[3 * i + 1]!;
          const lz = cube.positions[3 * i + 2]!;
          rest[3 * i + 0] = lx;
          rest[3 * i + 1] = ly;
          rest[3 * i + 2] = lz;
          initial.push({
            position: [lx, ly + cy, lz],
            velocity: [0, 0, 0],
            invMass: 1,
            phase: ((b + 1) & 0xffff) << 16,
          });
        }
        restPerBody.push(rest);
      }

      const particles = new ParticleSystem(renderer, totalCount, particleRadius);
      particles.uploadParticles(initial);

      const xpbd = createXpbdUniforms(1 / 60);
      const hashGrid = new HashGrid(particles, {
        cellSize: particleRadius * 2.1,
      });

      const bodies: RigidBodyDef[] = [
        {
          particleRange: { start: 0, count: perBodyCount },
          restPositions: restPerBody[0]!,
          restSDF: cube.restSDF!,
          phaseId: 1,
          compliance: 0,
        },
        {
          particleRange: { start: perBodyCount, count: perBodyCount },
          restPositions: restPerBody[1]!,
          restSDF: cube.restSDF!,
          phaseId: 2,
          compliance: 0,
        },
      ];

      const rigid = new RigidBodySystem({
        particles,
        xpbd,
        bodies,
        stackStabilization: { k: 3, groundY: 0 },
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
          maxContacts: Math.max(8192, totalCount * 6),
          friction: { muS: 0.6, muK: 0.4 },
          stabIters: 1,
        },
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.8, 0);

      const frameDt = 1 / 60;
      const frames = 180; // 3 s — settles within ~1 s, then 2 s of rest.
      for (let f = 0; f < frames; f++) {
        await loop.step(frameDt);
      }

      const snap = await particles.readback();

      const com = (start: number, count: number): [number, number, number] => {
        let cx = 0,
          cy = 0,
          cz = 0;
        for (let i = 0; i < count; i++) {
          cx += snap.positions[4 * (start + i) + 0]!;
          cy += snap.positions[4 * (start + i) + 1]!;
          cz += snap.positions[4 * (start + i) + 2]!;
        }
        const inv = 1 / count;
        return [cx * inv, cy * inv, cz * inv];
      };

      const [, lowerComY] = com(0, perBodyCount);
      const [upperComX, upperComY, upperComZ] = com(perBodyCount, perBodyCount);

      // Settled state: lower cube COM ≈ E/2; upper cube COM ≈ E/2 +
      // stackSpacing. Both within 5% of edge length.
      const tolerance = edgeLength * 0.05;

      // eslint-disable-next-line no-console
      console.info(
        `[rigid-two-cube-stack] T=${(frames * frameDt).toFixed(2)}s lowerComY=${lowerComY.toFixed(4)} upperCom=(${upperComX.toFixed(4)}, ${upperComY.toFixed(4)}, ${upperComZ.toFixed(4)}) stackSpacing=${stackSpacing.toFixed(4)}`,
      );

      const expectedLowerY = edgeLength * 0.5;
      const expectedUpperY = edgeLength * 0.5 + stackSpacing;

      expect(Math.abs(lowerComY - expectedLowerY)).toBeLessThan(tolerance);
      expect(Math.abs(upperComY - expectedUpperY)).toBeLessThan(tolerance);

      // Lateral drift bound — gravity is purely vertical, the small
      // §5.2-style oscillation should not produce horizontal motion.
      expect(Math.abs(upperComX)).toBeLessThan(tolerance);
      expect(Math.abs(upperComZ)).toBeLessThan(tolerance);

      // Sanity: upper cube did not sink into lower cube.
      expect(upperComY).toBeGreaterThan(lowerComY + edgeLength * 0.5);

      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
