import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Non-penetration invariant: no particle ends a frame with φ(x) < −r − ε
// for any collider.
//
// Scene: three particles and three primitives in one PrimitiveSet — a plane
// at y = 0, a solid sphere below it, and an inverted box (a container)
// around (1, 0, 0). Every primitive acts on every particle, so on the first
// substep the container pulls particles 0 and 1 inside it (a correction of
// up to ~1.8 m), and all three then settle on the plane inside the box. The
// invariant is checked for every particle against every primitive on every
// frame over 2 s; if any SDF under-projects on any frame, that frame
// violates it.

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

describe('collider: non-penetration invariant', () => {
  it('no particle ever has φ < -r - ε for any collider across 2 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;

      // Particle 0 starts above the plane, particle 1 on the plane above the
      // sphere, and particle 2 inside the inverted box.
      const initial: ParticleInit[] = [
        { position: [-1.0, 0.5, 0], velocity: [0, 0, 0], invMass: 1 },
        { position: [0, 0.0, 0], velocity: [0, 0, 0], invMass: 1 },
        { position: [1.0, 0.0, 0], velocity: [0, 0, 0], invMass: 1 },
      ];

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const planeN = new Vector3(0, 1, 0);
      const planeP = new Vector3(-1.0, 0, 0);
      const sphereC = new Vector3(0, -0.5, 0);
      const sphereR = 0.3;
      const boxC = new Vector3(1.0, 0.0, 0);
      const boxHE = new Vector3(0.3, 0.3, 0.3);

      const colliders = new PrimitiveSet(particles);
      colliders.addPlane(planeN, planeP, { muS: 0.3, muK: 0.2 });
      colliders.addSphere(sphereC, sphereR, { muS: 0.3, muK: 0.2 });
      colliders.addBox(boxC, boxHE, { invert: true, muS: 0.3, muK: 0.2 });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        colliders: [colliders],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const totalFrames = 120; // 2 s

      const eps = 5e-3; // per-frame overshoot tolerance
      let worstPhi = Infinity;
      let worstCollider = -1;
      let worstParticle = -1;
      let worstFrame = -1;

      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
        const snap = await particles.readback();
        // For each particle, evaluate each collider's SDF on the CPU and
        // check `phi(x) ≥ -r - ε` — the particle's surface has not
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

      console.info(
        `[collider-non-penetration] worstPhi=${worstPhi.toFixed(5)} ` +
          `(particle=${worstParticle}, collider=${worstCollider}, frame=${worstFrame}) ` +
          `bound=-r-ε=${(-r - eps).toFixed(5)}`,
      );

      // Invariant: `phi ≥ -r - ε`. A particle can be inside a collider by
      // at most `r` (its surface touches) plus a small overshoot.
      expect(worstPhi).toBeGreaterThan(-r - eps);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
