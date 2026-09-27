import { describe, expect, it } from 'vitest';
import { Quaternion, Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Oriented boxes in a PrimitiveSet.
//
// Mirrors `collider-non-penetration` with three rotated boxes: two solid
// slabs tilted 30° about Z and 45° about X, and an inverted box (a
// container) rotated 36° about Y. Every frame, the oriented-box signed
// distance of every particle is evaluated on the CPU, and the
// non-penetration invariant `φ(x) ≥ −r − ε` must hold across 2 s.
//
// Every primitive acts on every particle, so on the first substep the
// rotated container pulls all three particles inside it, where they then
// settle. That exercises the rotated SDF and the rotation of its gradient
// back to world space: if either is wrong, particles are pushed along the
// box's local axes instead of its rotated walls.

/**
 * CPU oriented-box SDF. Rotates the query point into the box's local frame
 * with the conjugate quaternion, then applies the standard AABB SDF.
 * `invert = true` negates the result, matching the kernel's `FLAG_INVERT`.
 */
function sdfOrientedBox(
  x: Vector3,
  center: Vector3,
  halfExtents: Vector3,
  rotation: Quaternion,
  invert: boolean,
): number {
  const invRot = rotation.clone().invert();
  const local = x.clone().sub(center).applyQuaternion(invRot);
  const q = new Vector3(
    Math.abs(local.x) - halfExtents.x,
    Math.abs(local.y) - halfExtents.y,
    Math.abs(local.z) - halfExtents.z,
  );
  const qPos = new Vector3(Math.max(q.x, 0), Math.max(q.y, 0), Math.max(q.z, 0));
  const outside = qPos.length();
  const inside = Math.min(Math.max(q.x, Math.max(q.y, q.z)), 0);
  const d = outside + inside;
  return invert ? -d : d;
}

describe('collider: oriented-box non-penetration', () => {
  it('three particles drop onto 3 rotated boxes; φ ≥ -r - ε across 2 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;

      // Particles 0 and 1 start above the tilted slabs, particle 2 inside the
      // rotated container.
      const initial: ParticleInit[] = [
        { position: [-1.0, 0.8, 0], velocity: [0, 0, 0], invMass: 1 },
        { position: [0, 0.8, 0], velocity: [0, 0, 0], invMass: 1 },
        { position: [1.0, 0.0, 0], velocity: [0, 0, 0], invMass: 1 },
      ];

      const particles = new ParticleSystem(renderer, initial.length, r);
      particles.uploadParticles(initial);

      const boxAC = new Vector3(-1.0, 0.0, 0);
      const boxAHE = new Vector3(0.4, 0.05, 0.4);
      const boxARot = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 6);

      const boxBC = new Vector3(0.0, 0.0, 0);
      const boxBHE = new Vector3(0.4, 0.05, 0.4);
      const boxBRot = new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), Math.PI / 4);

      const boxCC = new Vector3(1.0, 0.0, 0);
      const boxCHE = new Vector3(0.3, 0.3, 0.3);
      const boxCRot = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 5);

      const colliders = new PrimitiveSet(particles);
      colliders.addBox(boxAC, boxAHE, {
        rotation: boxARot,
        muS: 0.3,
        muK: 0.2,
      });
      colliders.addBox(boxBC, boxBHE, {
        rotation: boxBRot,
        muS: 0.3,
        muK: 0.2,
      });
      colliders.addBox(boxCC, boxCHE, {
        invert: true,
        rotation: boxCRot,
        muS: 0.3,
        muK: 0.2,
      });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 4,
        colliders: [colliders],
      });
      loop.gravity.set(0, -9.81, 0);

      const frameDt = 1 / 60;
      const totalFrames = 120; // 2 s
      const eps = 5e-3;
      let worstPhi = Infinity;
      let worstCollider = -1;
      let worstParticle = -1;
      let worstFrame = -1;

      for (let n = 0; n < totalFrames; n++) {
        await loop.step(frameDt);
        const snap = await particles.readback();
        for (let pi = 0; pi < initial.length; pi++) {
          const x = new Vector3(
            snap.positions[pi * 4 + 0]!,
            snap.positions[pi * 4 + 1]!,
            snap.positions[pi * 4 + 2]!,
          );
          const phiA = sdfOrientedBox(x, boxAC, boxAHE, boxARot, false);
          const phiB = sdfOrientedBox(x, boxBC, boxBHE, boxBRot, false);
          const phiC = sdfOrientedBox(x, boxCC, boxCHE, boxCRot, true);
          const phis = [phiA, phiB, phiC];
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
        `[collider-rotated-box-non-penetration] worstPhi=${worstPhi.toFixed(5)} ` +
          `(particle=${worstParticle}, collider=${worstCollider}, frame=${worstFrame}) ` +
          `bound=-r-ε=${(-r - eps).toFixed(5)}`,
      );

      expect(worstPhi).toBeGreaterThan(-r - eps);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
