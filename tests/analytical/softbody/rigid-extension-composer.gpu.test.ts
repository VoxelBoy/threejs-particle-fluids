import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';

import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ContactGeometryExtension,
  type Material,
  type ParticleInit,
} from '../../../src/core/index.js';
import {
  RigidBodySystem,
  voxelize,
  type RigidBodyDef,
  type TriangleMesh,
} from '../../../src/softbody/index.js';

// Phase 15a — forward-compatibility smoke test for the contact-geometry
// extension protocol. Registers a stub `Material` whose
// `contactGeometryExtension` returns an extension that never claims
// pairs (its `emit` leaves `outHandled = false` for every input). With
// the stub registered BEFORE the rigid material, the composer must
// still let the rigid extension claim rigid-rigid pairs — otherwise
// adding a no-op extension would break rigid contacts.
//
// Protects the protocol from drift: once a future cloth or other
// material starts shipping its own extension, this test will catch
// any composer change that accidentally consumes pairs the prior
// extension didn't claim.

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

/**
 * Stub material whose extension never claims a pair. Used to verify
 * that the composer correctly falls through a non-claiming extension
 * to the next one (or to the spherical default).
 */
const noopMaterial: Material = {
  contactGeometryExtension(): ContactGeometryExtension {
    return {
      emit() {
        // Intentionally empty — leaves `outHandled = false`.
      },
    };
  },
};

describe('Phase 15a — extension composer forward-compat (G1)', () => {
  it('stub extension registered before rigid: rigid pairs still resolve correctly', async () => {
    const renderer = await createParticleRenderer();
    try {
      const edgeLength = 1.0;
      const particleRadius = 0.05;
      const spacingFactor = 1.0;

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
        const cy = particleRadius - minLocalY + b * stackSpacing;
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

      // Critical: stub registered FIRST. The composer should let it
      // pass on every pair (its emit never sets outHandled), and the
      // rigid extension that follows then claims the rigid-rigid pairs.
      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        xpbd,
        hashGrid,
        colliders: { colliders },
        materials: [noopMaterial, rigid],
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
      const frames = 180;
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
      const [, upperComY] = com(perBodyCount, perBodyCount);

      // eslint-disable-next-line no-console
      console.info(
        `[rigid-extension-composer] lowerComY=${lowerComY.toFixed(4)} upperComY=${upperComY.toFixed(4)}`,
      );

      // Same expectations as the two-cube-stack test: stub extension
      // must NOT have changed the outcome. If the composer were broken
      // and the stub somehow claimed pairs (writing zero/identity-like
      // (n, d)), the upper cube would either sink into the lower one
      // or lift off — both outside the 5% tolerance.
      const tolerance = edgeLength * 0.05;
      expect(Math.abs(lowerComY - edgeLength * 0.5)).toBeLessThan(tolerance);
      expect(Math.abs(upperComY - (edgeLength * 0.5 + stackSpacing))).toBeLessThan(tolerance);
      expect(upperComY).toBeGreaterThan(lowerComY + edgeLength * 0.5);

      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
