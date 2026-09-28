import { describe, expect, it } from 'vitest';
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

// The high-level Simulation: it sizes particles from a budget, lays out
// every material, couples them, and fails early with plain messages.

const tank = () => new Box3(new Vector3(-0.3, 0, -0.2), new Vector3(0.3, 0.6, 0.2));

describe('Simulation', () => {
  it('spends about the requested particle budget on a fluid', async () => {
    const renderer = await createParticleRenderer();
    try {
      const sim = new Simulation({
        renderer,
        scene: new Scene(),
        camera: new PerspectiveCamera(),
        container: tank(),
        particles: 6000,
      });
      sim.addFluid({ box: new Box3(new Vector3(-0.3, 0, -0.2), new Vector3(0, 0.3, 0.2)) });
      await sim.step(1 / 60);
      console.log(`[simulation] ${sim.particleCount} particles, radius ${sim.particleSize}`);
      expect(sim.particleCount).toBeGreaterThan(6000 * 0.8);
      expect(sim.particleCount).toBeLessThan(6000 * 1.2);
      sim.dispose();
    } finally {
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
        particles: 8000,
      });
      sim.addFluid({ box: new Box3(new Vector3(-0.3, 0, -0.2), new Vector3(0.3, 0.25, 0.2)) });
      const ball = sim.addSoftbody({ mesh: light, density: 300 });
      const block = sim.addSoftbody({ mesh: heavy, density: 3000, softness: 0.1 });
      for (let i = 0; i < 150; i++) await sim.step(1 / 60);

      const snapshot = await sim.particles.readback();
      const meanY = (start: number, count: number) => {
        let sum = 0;
        for (let i = start; i < start + count; i++) sum += snapshot.positions[i * 4 + 1]!;
        return sum / count;
      };
      const ballRange = ball.mesh.softbody.particleRange(ball.mesh.bodyIndex);
      const blockRange = block.mesh.softbody.particleRange(block.mesh.bodyIndex);
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
      const base = { renderer, scene: new Scene(), camera: new PerspectiveCamera() };
      expect(() => new Simulation(base).addSmoke()).toThrow(/container/);
      expect(() => new Simulation(base).addFluid({})).toThrow(/box.*mesh/);
      await expect(new Simulation(base).step(1 / 60)).rejects.toThrow(/add a fluid/);

      const sim = new Simulation({ ...base, container: tank(), particles: 2000 });
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
});
