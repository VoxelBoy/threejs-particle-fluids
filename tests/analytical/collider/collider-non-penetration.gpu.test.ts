import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 06 G3 — non-penetration invariant (plan §Validation > Automatic (G3)):
// "No particle ends a frame with φ(x) < -r - ε (penetrating) for any collider."
//
// Scene: three particles hitting three different colliders simultaneously
// — a plane, a sphere, and a box. Check the invariant on every frame over
// 2 s of simulation. If any collider's SDF under-projects on any frame we
// expect a frame-level violation.

function sdfPlane(x: Vector3, normal: Vector3, point: Vector3): number {
  return normal.dot(new Vector3().subVectors(x, point));
}

function sdfSphere(x: Vector3, center: Vector3, R: number, invert: boolean): number {
  const d = x.distanceTo(center) - R;
  return invert ? -d : d;
}

function sdfBoxAA(x: Vector3, center: Vector3, he: Vector3, invert: boolean): number {
  const q = new Vector3(
    Math.abs(x.x - center.x) - he.x,
    Math.abs(x.y - center.y) - he.y,
    Math.abs(x.z - center.z) - he.z,
  );
  const qPos = new Vector3(Math.max(q.x, 0), Math.max(q.y, 0), Math.max(q.z, 0));
  const outside = qPos.length();
  const inside = Math.min(Math.max(q.x, Math.max(q.y, q.z)), 0);
  const d = outside + inside;
  return invert ? -d : d;
}

describe('Phase 06 — collider: non-penetration invariant (G3)', () => {
  it('no particle ever has φ < -r - ε for any collider across 2 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;

      // Scene: 3 particles dropped above 3 different colliders.
      //   particle 0 drops onto a plane at y=0
      //   particle 1 drops onto a sphere (solid, not inverted) at y=-0.5
      //   particle 2 drops into an inverted box around origin
      const initial: ParticleInit[] = [
        { position: [-1.0, 0.5, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [0, 0.0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [1.0, 0.0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ];

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const planeN = new Vector3(0, 1, 0);
      const planeP = new Vector3(-1.0, 0, 0);
      const sphereC = new Vector3(0, -0.5, 0);
      const sphereR = 0.3;
      const boxC = new Vector3(1.0, 0.0, 0);
      const boxHE = new Vector3(0.3, 0.3, 0.3);

      const colliders = new PrimitiveSet(particles, { capacity: 3 });
      colliders.addPlane(planeN, planeP, { muS: 0.3, muK: 0.2 });
      colliders.addSphere(sphereC, sphereR, { muS: 0.3, muK: 0.2 });
      colliders.addBox(boxC, boxHE, { invert: true, muS: 0.3, muK: 0.2 });
      colliders.upload();

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        colliders: { colliders },
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const totalFrames = 120; // 2 s

      const eps = 5e-3; // per-frame overshoot tolerance (stabilization)
      let worstPhi = Infinity;
      let worstCollider = -1;
      let worstParticle = -1;
      let worstFrame = -1;

      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
        const snap = await particles.readback();
        // For each particle, check each collider's SDF evaluated on CPU
        // and verify `phi(x) ≥ -r - ε` — the particle's surface has not
        // penetrated past the collider surface.
        for (let pi = 0; pi < initial.length; pi++) {
          const x = new Vector3(
            snap.positions[pi * 4 + 0]!,
            snap.positions[pi * 4 + 1]!,
            snap.positions[pi * 4 + 2]!,
          );
          const phiPlane = sdfPlane(x, planeN, planeP);
          const phiSphere = sdfSphere(x, sphereC, sphereR, false);
          const phiBox = sdfBoxAA(x, boxC, boxHE, true);
          const phis = [phiPlane, phiSphere, phiBox];
          for (let ci = 0; ci < phis.length; ci++) {
            const phi = phis[ci]!;
            if (phi < worstPhi) {
              worstPhi = phi;
              worstCollider = ci;
              worstParticle = pi;
              worstFrame = n;
            }
          }
        }
      }

      // eslint-disable-next-line no-console
      console.info(
        `[collider-non-penetration] worstPhi=${worstPhi.toFixed(5)} ` +
          `(particle=${worstParticle}, collider=${worstCollider}, frame=${worstFrame}) ` +
          `bound=-r-ε=${(-r - eps).toFixed(5)}`,
      );

      // Invariant: `phi ≥ -r - ε`. Particle can be at most barely inside
      // the collider by up to `r` (its surface touches) plus a small
      // stabilization overshoot.
      expect(worstPhi).toBeGreaterThan(-r - eps);

      particles.destroy();
      colliders.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
