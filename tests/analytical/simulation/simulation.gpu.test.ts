import { describe, expect, it, vi } from 'vitest';
import {
  Box3,
  BoxGeometry,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  SphereGeometry,
  Vector3,
} from 'three';
import { Simulation, createParticleRenderer } from '../../../src/index.js';

// The high-level Simulation: it sizes particles from a budget or a radius,
// lays out every material, couples them, and fails early with plain messages.

const tank = () => new Box3(new Vector3(-0.3, 0, -0.2), new Vector3(0.3, 0.6, 0.2));

describe('Simulation', () => {
  it('fills a fluid box at the given particle size', async () => {
    const renderer = await createParticleRenderer();
    try {
      const sim = new Simulation({
        renderer,
        scene: new Scene(),
        camera: new PerspectiveCamera(),
        container: tank(),
        particleRadius: 0.01,
      });
      // Clipped one radius inside the walls and filled on a 2 cm grid: 15 × 15 × 19.
      const water = sim.addFluid({
        box: new Box3(new Vector3(-0.3, 0, -0.2), new Vector3(0.005, 0.305, 0.2)),
      });
      await sim.step(1 / 60);
      expect(sim.particleCount).toBe(15 * 15 * 19);
      expect(water.particleCount).toBe(15 * 15 * 19);
      expect(sim.particleRadius).toBe(0.01);
      sim.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('spends close to, and never more than, a particle budget', async () => {
    const renderer = await createParticleRenderer();
    try {
      const scene = new Scene();
      const ball = new Mesh(new SphereGeometry(0.08, 24, 16), new MeshStandardMaterial());
      ball.position.set(0, 0.45, 0);
      scene.add(ball);
      const sim = new Simulation({
        renderer,
        scene,
        camera: new PerspectiveCamera(),
        container: tank(),
        particles: 6000,
      });
      const water = sim.addFluid({
        box: new Box3(new Vector3(-0.3, 0, -0.2), new Vector3(0, 0.3, 0.2)),
      });
      const body = sim.addSoftbody({ mesh: ball });
      sim.addSphere({ radius: 0.05, center: new Vector3(-0.15, 0.1, 0) });
      expect(sim.particleRadius).toBe(0);
      await sim.start();
      console.log(
        `[simulation] ${sim.particleCount} particles, radius ${sim.particleRadius}, body ${body.particleCount}`,
      );
      expect(sim.particleCount).toBeLessThanOrEqual(6000);
      expect(sim.particleCount).toBeGreaterThan(6000 * 0.95);
      expect(water.particleCount + body.particleCount).toBe(sim.particleCount);
      sim.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('warns when a soft body gets too few particles to keep its shape', async () => {
    const renderer = await createParticleRenderer();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const scene = new Scene();
      const pebble = new Mesh(new SphereGeometry(0.04, 16, 12), new MeshStandardMaterial());
      pebble.name = 'pebble';
      pebble.position.set(0, 0.45, 0);
      scene.add(pebble);
      const sim = new Simulation({
        renderer,
        scene,
        camera: new PerspectiveCamera(),
        container: tank(),
        particles: 2000,
      });
      sim.addFluid({ box: tank() });
      sim.addSoftbody({ mesh: pebble });
      await sim.start();
      expect(warn).toHaveBeenCalledWith(
        expect.stringMatching(/soft body "pebble" has \d+ particles/),
      );
      sim.dispose();
    } finally {
      warn.mockRestore();
      renderer.dispose();
    }
  });

  it('floats a light body above a heavy one', async () => {
    const renderer = await createParticleRenderer();
    try {
      const scene = new Scene();
      const light = new Mesh(new SphereGeometry(0.06, 24, 16), new MeshStandardMaterial());
      light.position.set(-0.12, 0.4, 0);
      const heavy = new Mesh(new BoxGeometry(0.1, 0.1, 0.1), new MeshStandardMaterial());
      heavy.position.set(0.12, 0.4, 0);
      scene.add(light, heavy);
      const sim = new Simulation({
        renderer,
        scene,
        camera: new PerspectiveCamera(),
        container: tank(),
        particleRadius: 0.009,
      });
      sim.addFluid({ box: new Box3(new Vector3(-0.3, 0, -0.2), new Vector3(0.3, 0.25, 0.2)) });
      const ball = sim.addSoftbody({ mesh: light, density: 300 });
      const block = sim.addSoftbody({ mesh: heavy, density: 3000, softness: 0.1 });
      for (let i = 0; i < 150; i++) await sim.step(1 / 60);

      const snapshot = await sim.particleSystem.readback();
      const meanY = (start: number, count: number) => {
        let sum = 0;
        for (let i = start; i < start + count; i++) sum += snapshot.positions[i * 4 + 1]!;
        return sum / count;
      };
      const ballRange = ball.softbodySystem.particleRange(ball.bodyIndex);
      const blockRange = block.softbodySystem.particleRange(block.bodyIndex);
      const ballY = meanY(ballRange.start, ballRange.count);
      const blockY = meanY(blockRange.start, blockRange.count);
      console.log(
        `[simulation] light body at ${ballY.toFixed(3)} m, heavy at ${blockY.toFixed(3)} m`,
      );
      expect(ballY).toBeGreaterThan(blockY + 0.08);
      expect(light.visible).toBe(false);
      sim.dispose();
      expect(light.visible).toBe(true);
    } finally {
      renderer.dispose();
    }
  });

  it('explains what went wrong', async () => {
    const renderer = await createParticleRenderer();
    try {
      const base = {
        renderer,
        scene: new Scene(),
        camera: new PerspectiveCamera(),
        particleRadius: 0.02,
      };
      expect(() => new Simulation({ ...base, particleRadius: 0 })).toThrow(/particleRadius/);
      expect(() => new Simulation({ ...base, particleRadius: undefined, particles: 2.5 })).toThrow(
        /particles must be a positive integer/,
      );
      // @ts-expect-error: exactly one of the two is allowed.
      expect(() => new Simulation({ ...base, particles: 5000 })).toThrow(/exactly one/);
      expect(() => new Simulation(base).addSmoke()).toThrow(/container/);
      expect(() => new Simulation(base).addFluid({})).toThrow(/box.*mesh/);
      await expect(new Simulation(base).step(1 / 60)).rejects.toThrow(/add a fluid/);

      const sim = new Simulation({ ...base, container: tank() });
      const water = sim.addFluid({ box: tank() });
      expect(() => water.surface).toThrow(/first step/);
      await sim.start();
      expect(() => sim.addFloor()).toThrow(/before the first step/);
      water.viscosity = 0.05;
      expect(water.fluidSystem.viscosity).toBeCloseTo(0.05);
      sim.dispose();
    } finally {
      renderer.dispose();
    }
  });
  it('keeps its settings working before and after the start', async () => {
    const renderer = await createParticleRenderer();
    try {
      const base = {
        renderer,
        scene: new Scene(),
        camera: new PerspectiveCamera(),
        particleRadius: 0.02,
      };

      // Mixing smoke with other materials fails at the add call that causes it.
      const smoky = new Simulation({ ...base, container: tank() });
      smoky.addSmoke();
      expect(() => smoky.addFluid({ box: tank() })).toThrow(/gas and liquid/);
      expect(() => smoky.addSmoke()).toThrow(/one smoke source/);
      const wet = new Simulation({ ...base, container: tank() });
      wet.addFluid({ box: tank() });
      expect(() => wet.addSmoke()).toThrow(/gas and liquid/);

      const sim = new Simulation({ ...base, container: tank() });
      const thick = sim.addFluid({
        box: new Box3(new Vector3(-0.3, 0, -0.2), new Vector3(0, 0.3, 0.2)),
        thickness: 0,
      });
      const thin = sim.addFluid({
        box: new Box3(new Vector3(0, 0, -0.2), new Vector3(0.3, 0.3, 0.2)),
      });
      const gravity = sim.gravity;

      // Overlapping steps share one run.
      const first = sim.step(1 / 60);
      expect(sim.step(1 / 60)).toBe(first);
      await first;

      expect(sim.gravity).toBe(gravity);
      gravity.set(0, -2, 0);
      await sim.step(1 / 60);
      expect(sim.loop.gravity.y).toBeCloseTo(-2);

      thick.thickness = 20;
      expect(() => (thin.thickness = 20)).toThrow(/even as 0/);
      sim.dispose();
    } finally {
      renderer.dispose();
    }
  });
});
