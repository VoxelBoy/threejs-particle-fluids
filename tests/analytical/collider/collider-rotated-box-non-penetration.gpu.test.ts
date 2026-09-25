import { describe, expect, it } from 'vitest';
import { Quaternion, Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 07 U-21 resolution — oriented boxes in PrimitiveSet.
//
// Mirrors the Phase 06 `collider-non-penetration` test shape but gives
// each of three boxes a non-identity world rotation. Each frame we CPU-
// evaluate the OBB signed-distance to every particle's position and
// confirm the non-penetration invariant `φ(x) ≥ -r - ε` holds across a
// 2 s simulation.
//
// If the kernel's rotation plumbing is wrong — identity quaternion
// leaked into the box, or gradient not rotated back to world — the
// particles will either drift through the rotated top face or be
// pushed off along the box's local axes rather than its rotated
// surface normal. Either failure mode would trip the invariant.

/**
 * CPU oriented-box SDF. Rotates the query point into the box's local
 * frame via the conjugate quaternion, then applies the standard AABB
 * SDF. `invert = true` negates the return — matches the kernel's
 * `FLAG_INVERT` behaviour.
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

describe('Phase 07 — U-21: oriented-box non-penetration', () => {
  it('three particles drop onto 3 rotated boxes; φ ≥ -r - ε across 2 s', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.05;

      // Scene: three particles, three rotated boxes.
      //   particle 0 drops onto a box tilted 30° around Z
      //   particle 1 drops onto a box tilted 45° around X
      //   particle 2 drops into an inverted-box container rotated around Y
      const initial: ParticleInit[] = [
        { position: [-1.0, 0.8, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [0, 0.8, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [1.0, 0.0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
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

      const colliders = new PrimitiveSet(particles, { capacity: 3 });
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

      // eslint-disable-next-line no-console
      console.info(
        `[collider-rotated-box-non-penetration] worstPhi=${worstPhi.toFixed(5)} ` +
          `(particle=${worstParticle}, collider=${worstCollider}, frame=${worstFrame}) ` +
          `bound=-r-ε=${(-r - eps).toFixed(5)}`,
      );

      expect(worstPhi).toBeGreaterThan(-r - eps);

      particles.destroy();
      colliders.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
