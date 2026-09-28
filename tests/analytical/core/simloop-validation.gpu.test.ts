import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import type { WebGPURenderer } from 'three/webgpu';

import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type Material,
  type ParticleInit,
} from '../../../src/index.js';

// SimLoop's argument checks, its dispose contract, its overflow report, and
// GPU buffer release by ParticleSystem and HashGrid.

async function withRenderer(run: (renderer: WebGPURenderer) => Promise<void>): Promise<void> {
  const renderer = await createParticleRenderer();
  try {
    await run(renderer);
  } finally {
    renderer.dispose();
  }
}

function storageBuffers(renderer: WebGPURenderer): number {
  return (renderer.info.memory as unknown as { storageAttributes: number }).storageAttributes;
}

describe('SimLoop validation', () => {
  it('rejects a step length that is not positive and finite', async () => {
    await withRenderer(async (renderer) => {
      const particles = new ParticleSystem(renderer, 1, 0.05);
      particles.uploadParticles([{ position: [0, 0, 0] }]);
      const loop = new SimLoop(particles);
      for (const dt of [0, -1 / 60, NaN, Infinity]) {
        await expect(loop.step(dt)).rejects.toThrow('SimLoop: step dt must be a positive finite');
      }
      loop.dispose();
      particles.dispose();
    });
  }, 60_000);

  it('throws after dispose', async () => {
    await withRenderer(async (renderer) => {
      const particles = new ParticleSystem(renderer, 1, 0.05);
      particles.uploadParticles([{ position: [0, 0, 0] }]);
      const loop = new SimLoop(particles, { contact: true });
      await loop.step(1 / 60);
      loop.dispose();
      await expect(loop.step(1 / 60)).rejects.toThrow('SimLoop: the loop has been disposed');
      expect(() => (loop.substeps = 2)).toThrow('SimLoop: the loop has been disposed');
      await expect(loop.readbackOverflow()).rejects.toThrow('disposed');
      expect(() => loop.hashGrid!.rebuildPipeline).toThrow('HashGrid has been disposed');
      particles.dispose();
    });
  }, 60_000);

  it('checks materials’ particles and neighborRadius', async () => {
    await withRenderer(async (renderer) => {
      const particles = new ParticleSystem(renderer, 1, 0.05);
      const other = new ParticleSystem(renderer, 1, 0.05);
      const foreign: Material = { particles: other, build: () => ({}) };
      expect(() => new SimLoop(particles, { materials: [foreign] })).toThrow(
        'SimLoop: every material must be built for the same ParticleSystem',
      );
      for (const neighborRadius of [NaN, -0.1, Infinity]) {
        expect(
          () => new SimLoop(particles, { materials: [{ neighborRadius, build: () => ({}) }] }),
        ).toThrow("SimLoop: a material's neighborRadius must be finite and ≥ 0");
      }
      expect(() => new SimLoop(particles, { contact: { muS: -1 } })).toThrow(
        'SimLoop: friction coefficients must be non-negative',
      );
    });
  }, 60_000);

  it('lets materials destructure a missing hashGrid, and throws only on use', async () => {
    await withRenderer(async (renderer) => {
      const particles = new ParticleSystem(renderer, 1, 0.05);
      const unused: Material = {
        build: ({ hashGrid }) => {
          void hashGrid;
          return {};
        },
      };
      expect(() => new SimLoop(particles, { materials: [unused] })).not.toThrow();
      const used: Material = {
        build: ({ hashGrid }) => {
          void hashGrid.cellSizeUniform;
          return {};
        },
      };
      expect(() => new SimLoop(particles, { materials: [used] })).toThrow(
        'SimLoop: a material used the neighbor grid without declaring a neighborRadius',
      );
    });
  }, 60_000);
});

describe('SimLoop.readbackOverflow', () => {
  it('is all false for a quiet scene', async () => {
    await withRenderer(async (renderer) => {
      const particles = new ParticleSystem(renderer, 2, 0.05);
      particles.uploadParticles([{ position: [0, 0.5, 0] }, { position: [0.5, 0.5, 0] }]);
      const floor = new PrimitiveSet(particles);
      floor.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
      const loop = new SimLoop(particles, { colliders: [floor], contact: true });
      await loop.step(1 / 60);
      expect(await loop.readbackOverflow()).toEqual({
        positions: false,
        velocities: false,
        grid: false,
        contacts: false,
      });
      loop.dispose();
      floor.dispose();
      particles.dispose();
    });
  }, 60_000);

  it('keeps a saturated correction flagged through the rest of the step', async () => {
    await withRenderer(async (renderer) => {
      // 20 m below the floor: the first substep's push-out is past the 10 m
      // position headroom; later substeps have nothing left to correct.
      const particles = new ParticleSystem(renderer, 1, 0.05);
      particles.uploadParticles([{ position: [0, -20, 0] }]);
      const floor = new PrimitiveSet(particles);
      floor.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
      const loop = new SimLoop(particles, { colliders: [floor], substeps: 4 });
      await loop.step(1 / 60);
      expect((await loop.readbackOverflow()).positions).toBe(true);
      loop.dispose();
      floor.dispose();
      particles.dispose();
    });
  }, 60_000);

  // Also pins that a one-record contact buffer compiles.
  it('reports dropped contacts and grid overflow, and hashOrigin recenters the grid', async () => {
    await withRenderer(async (renderer) => {
      const r = 0.05;
      // Three overlapping particles 1 km out: past ±512 cells of the default origin.
      const cluster: ParticleInit[] = [
        { position: [1000, 0, 0] },
        { position: [1000 + r, 0, 0] },
        { position: [1000, r, 0] },
      ];
      const particles = new ParticleSystem(renderer, cluster.length, r);
      particles.uploadParticles(cluster);

      const far = new SimLoop(particles, {
        contact: { maxContacts: 1 },
        gravity: new Vector3(0, 0, 0),
      });
      await far.step(1 / 60);
      const overflow = await far.readbackOverflow();
      expect(overflow.contacts).toBe(true);
      expect(overflow.grid).toBe(true);
      far.dispose();

      const centered = new SimLoop(particles, {
        contact: true,
        gravity: new Vector3(0, 0, 0),
        hashOrigin: new Vector3(1000, 0, 0),
      });
      await centered.step(1 / 60);
      expect(await centered.readbackOverflow()).toMatchObject({ grid: false, contacts: false });
      centered.dispose();
      particles.dispose();
    });
  }, 60_000);
});

describe('GPU buffer release', () => {
  it('ParticleSystem and HashGrid free their storage buffers on dispose', async () => {
    await withRenderer(async (renderer) => {
      const particles = new ParticleSystem(renderer, 4, 0.05);
      particles.uploadParticles([{ position: [0, 0, 0] }]);
      const before = storageBuffers(renderer);
      const grid = new HashGrid(particles, { cellSize: 0.1 });
      await grid.rebuild();
      const withGrid = storageBuffers(renderer);
      expect(withGrid).toBeGreaterThan(before);

      grid.dispose();
      expect(storageBuffers(renderer)).toBeLessThan(withGrid);
      expect(() => grid.rebuildPipeline).toThrow('HashGrid has been disposed');
      await expect(grid.rebuild()).rejects.toThrow('HashGrid has been disposed');

      const withParticles = storageBuffers(renderer);
      particles.dispose();
      expect(storageBuffers(renderer)).toBeLessThan(withParticles);
      expect(particles.disposed).toBe(true);
      await expect(particles.readback()).rejects.toThrow('ParticleSystem has been disposed');
    });
  }, 60_000);
});
