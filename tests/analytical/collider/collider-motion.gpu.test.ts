import { describe, expect, it } from 'vitest';
import { Object3D, Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
} from '../../../src/index.js';

// How primitive motion reaches particles: a conveyor's `velocity` is surface
// velocity only (the shape stays put within a frame), and an attached shape
// that turns carries particles with its surface.

describe('collider: primitive motion', () => {
  it('does not sweep a static primitive that has a conveyor velocity', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.02;
      const particles = new ParticleSystem(renderer, 1, r);
      // Just beside the box's −x face; a sweep would drag the box over it.
      const x0 = -0.5 - r - 0.01;
      particles.uploadParticles([{ position: [x0, 0, 0], velocity: [0, 0, 0], invMass: 1 }]);
      const colliders = new PrimitiveSet(particles);
      colliders.addBox(new Vector3(), new Vector3(0.5, 0.1, 0.5), {
        muS: 0,
        muK: 0,
        velocity: new Vector3(20, 0, 0),
      });
      const loop = new SimLoop(particles, { substeps: 4, iterations: 2, colliders: [colliders] });
      loop.gravity.set(0, 0, 0);
      for (let n = 0; n < 10; n++) await loop.step(1 / 60);
      const { positions } = await particles.readback();
      console.info(`[collider-motion] conveyor: x=${positions[0]} (start ${x0})`);
      expect(Math.abs(positions[0]! - x0)).toBeLessThan(1e-5);
      expect(Math.abs(positions[1]!)).toBeLessThan(1e-5);
      loop.dispose();
      colliders.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('carries a particle on an attached turntable', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.02;
      const omega = 2; // rad/s about +y
      const particles = new ParticleSystem(renderer, 1, r);
      particles.uploadParticles([{ position: [0.5, 0.1 + r, 0], velocity: [0, 0, 0], invMass: 1 }]);
      const colliders = new PrimitiveSet(particles);
      const turntable = new Object3D();
      turntable.updateMatrixWorld(true);
      colliders.attach(
        colliders.addBox(new Vector3(), new Vector3(1, 0.1, 1), { muS: 1, muK: 1 }),
        turntable,
      );
      const loop = new SimLoop(particles, { substeps: 4, iterations: 2, colliders: [colliders] });
      loop.gravity.set(0, -9.81, 0);
      const dt = 1 / 60;
      for (let n = 0; n < 30; n++) {
        turntable.rotation.y += omega * dt;
        turntable.updateMatrixWorld(true);
        await loop.step(dt);
      }
      const { positions, velocities } = await particles.readback();
      const radius = Math.hypot(positions[0]!, positions[2]!);
      const speed = Math.hypot(velocities[0]!, velocities[2]!);
      // Surface velocity ω × r is tangential: perpendicular to the radius.
      const radial = (velocities[0]! * positions[0]! + velocities[2]! * positions[2]!) / radius;
      console.info(
        `[collider-motion] turntable: radius=${radius.toFixed(4)} speed=${speed.toFixed(4)} (expected ≈ ${(omega * radius).toFixed(4)}) radial=${radial.toFixed(4)}`,
      );
      expect(speed).toBeGreaterThan(0.85 * omega * radius);
      expect(speed).toBeLessThan(1.15 * omega * radius);
      expect(Math.abs(radial)).toBeLessThan(0.15 * speed);
      expect(radius).toBeGreaterThan(0.45);
      expect(radius).toBeLessThan(0.6);
      loop.dispose();
      colliders.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
