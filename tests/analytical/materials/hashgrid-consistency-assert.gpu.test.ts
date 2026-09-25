import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
} from '../../../src/core/index.js';

// Phase 07b G1 — hashGrid consistency assert.
//
// `SimLoopOptions.hashGrid` is the canonical source; `ContactOptions.hashGrid`
// stays required for Phase 05/06 back-compat. If both are set they MUST
// reference the same `HashGrid` instance. Pins the diagnostic so a later
// refactor cannot silently drop the assert and allow two-grid misconfiguration
// (which would produce stale neighbor data for whichever consumer reads the
// "wrong" grid).
//
// The test requires a live renderer because `SimLoop` construction allocates
// GPU storage buffers via `buildIntegrationKernels`. Using the `.gpu.test.ts`
// suffix to route it through `npm run test:gpu`; no actual dispatch happens.

describe('Phase 07b — materials: hashGrid consistency assert', () => {
  it('throws when options.hashGrid and options.contact.hashGrid are different instances', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 4, 0.05);
      particles.uploadParticles([
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ]);
      const gridA = new HashGrid(particles, { cellSize: 0.1 });
      const gridB = new HashGrid(particles, { cellSize: 0.1 });

      expect(() => {
        new SimLoop(particles, {
          hashGrid: gridA,
          contact: {
            hashGrid: gridB,
            maxContacts: 16,
          },
        });
      }).toThrowError(/hashGrid.*same HashGrid instance/);

      gridA.destroy();
      gridB.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 30_000);

  it('accepts the same HashGrid instance in both fields', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 4, 0.05);
      particles.uploadParticles([
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ]);
      const grid = new HashGrid(particles, { cellSize: 0.1 });

      expect(() => {
        new SimLoop(particles, {
          hashGrid: grid,
          contact: {
            hashGrid: grid,
            maxContacts: 16,
          },
        });
      }).not.toThrow();

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 30_000);

  it('throws when hashGrid is constructed against a different ParticleSystem', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particlesA = new ParticleSystem(renderer, 4, 0.05);
      const particlesB = new ParticleSystem(renderer, 4, 0.05);
      particlesA.uploadParticles([
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ]);
      const gridForB = new HashGrid(particlesB, { cellSize: 0.1 });

      expect(() => {
        new SimLoop(particlesA, {
          hashGrid: gridForB,
        });
      }).toThrowError(/hashGrid.*same ParticleSystem/);

      gridForB.destroy();
      particlesA.destroy();
      particlesB.destroy();
    } finally {
      renderer.dispose();
    }
  }, 30_000);
});
