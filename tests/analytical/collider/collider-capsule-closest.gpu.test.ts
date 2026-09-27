import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Capsule closest point: for 100 random points around a fixed capsule, the
// direction the solve pushes each particle must match the analytical
// closest-point direction.
//
// Strategy: place each test particle at a position `p` whose CPU-computed
// projection onto the capsule gives a known outward normal `n_expected`.
// After one gravity-free step of the collider solve, the particle should
// have been pushed along `n_expected`; the observed correction direction
// `(x_after − x_before).normalized()` is compared against it.

function closestPointOnSegment(p: Vector3, a: Vector3, b: Vector3): { c: Vector3; t: number } {
  const ab = new Vector3().subVectors(b, a);
  const ap = new Vector3().subVectors(p, a);
  const abLenSq = Math.max(ab.dot(ab), 1e-18);
  let t = ap.dot(ab) / abLenSq;
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  const c = new Vector3().copy(a).addScaledVector(ab, t);
  return { c, t };
}

describe('collider: capsule closest-point direction', () => {
  it('100 random test points: solve-kernel normal matches analytical closest-point within 1e-3', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;
      const N = 100;
      const capsuleA = new Vector3(-0.4, 0, 0);
      const capsuleB = new Vector3(0.4, 0.2, 0.1);
      const capsuleR = 0.15;

      // Deterministic pseudo-random.
      let seed = 0xbeefface;
      const rand = (): number => {
        seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
        return ((seed >>> 0) / 0x100000000) * 2 - 1;
      };

      // CPU-compute expected normals and initial penetrating positions.
      const expectedNormals: Vector3[] = [];
      const initial: ParticleInit[] = [];
      for (let i = 0; i < N; i++) {
        // Generate a random unit direction.
        let nx: number, ny: number, nz: number, len: number;
        do {
          nx = rand();
          ny = rand();
          nz = rand();
          len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        } while (len < 1e-3);
        nx /= len;
        ny /= len;
        nz /= len;

        // Pick a random point along the capsule axis, then offset it along
        // the random unit direction.
        const t = (rand() + 1) * 0.5; // [0, 1]
        const baseX = capsuleA.x + (capsuleB.x - capsuleA.x) * t;
        const baseY = capsuleA.y + (capsuleB.y - capsuleA.y) * t;
        const baseZ = capsuleA.z + (capsuleB.z - capsuleA.z) * t;
        // The particle's surface sits just inside the capsule surface, so
        // the solve fires and projects it outward along the expected normal.
        const offset = capsuleR - r + 0.01; // offset from axis to particle centre
        const px = baseX + nx * offset;
        const py = baseY + ny * offset;
        const pz = baseZ + nz * offset;

        // Expected normal: (particle − closest point on segment), normalized.
        // The construction direction (nx, ny, nz) is not the closest-point
        // direction in general (it has a component along the axis, and near
        // the ends the projection clamps onto an end cap), so recompute it
        // with the same closest-point routine the kernel uses.
        const { c } = closestPointOnSegment(new Vector3(px, py, pz), capsuleA, capsuleB);
        const nxE = px - c.x;
        const nyE = py - c.y;
        const nzE = pz - c.z;
        const lenE = Math.sqrt(nxE * nxE + nyE * nyE + nzE * nzE);
        expectedNormals.push(new Vector3(nxE / lenE, nyE / lenE, nzE / lenE));

        initial.push({ position: [px, py, pz], velocity: [0, 0, 0], invMass: 1 });
      }

      const particles = new ParticleSystem(renderer, N, r);
      particles.uploadParticles(initial);

      const colliders = new PrimitiveSet(particles);
      colliders.addCapsule(capsuleA, capsuleB, capsuleR, {
        muS: 0.0,
        muK: 0.0, // no friction — isolate the normal projection.
      });

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        colliders: [colliders],
      });
      loop.gravity.set(0, 0, 0); // no gravity — isolate the projection.

      await loop.step(1 / 60);
      const snap = await particles.readback();

      let maxAngleErr = 0;
      let correctedCount = 0;
      for (let i = 0; i < N; i++) {
        const p0 = initial[i]!.position;
        const p1x = snap.positions[i * 4 + 0]!;
        const p1y = snap.positions[i * 4 + 1]!;
        const p1z = snap.positions[i * 4 + 2]!;
        const dx = p1x - p0[0];
        const dy = p1y - p0[1];
        const dz = p1z - p0[2];
        const dLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
        // Skip non-correcting particles (numerical chance of d ≈ 0).
        if (dLen < 1e-6) continue;
        correctedCount++;
        const nObs = new Vector3(dx / dLen, dy / dLen, dz / dLen);
        const nExp = expectedNormals[i]!;
        // Angular error via the dot product.
        const dot = Math.min(1, Math.max(-1, nObs.dot(nExp)));
        const angleErr = Math.acos(dot);
        if (angleErr > maxAngleErr) maxAngleErr = angleErr;
      }

      console.info(
        `[collider-capsule-closest] N=${N} corrected=${correctedCount} maxAngleErr=${maxAngleErr.toExponential(3)} rad`,
      );

      // A perfect kernel gives zero angular error. The path is CPU → GPU f32
      // → CPU f32, so a single correction should be accurate to ≲ 1e-5 rad
      // in the common case; 1e-3 absorbs the corner case where a particle
      // sits exactly on the end-cap / cylinder transition (sub-ULP
      // instability in `clamp(t, 0, 1)`).
      expect(correctedCount).toBeGreaterThanOrEqual(Math.floor(N * 0.9));
      expect(maxAngleErr).toBeLessThan(1e-3);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
