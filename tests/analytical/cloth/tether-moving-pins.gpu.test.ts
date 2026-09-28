import { describe, expect, it } from 'vitest';
import { PlaneGeometry, Vector3 } from 'three';

import {
  ClothSystem,
  ParticleSystem,
  SimLoop,
  createClothGraph,
  createParticleRenderer,
} from '../../../src/index.js';

// Tethers hang from pinned particles, not from where the pins started.
//
// Setup: a sheet pinned along its top row, in zero gravity, with stretch
// and bending effectively disabled so the tethers are the only constraint.
// Before the first step every pinned particle is moved 1.5 m along +x,
// far beyond any tether radius. Each free particle must end up inside the
// sphere of radius `restRadius` around its anchor's new position. Tethers
// anchored at the pins' rest positions would leave the free particles
// where they are, outside those spheres.

describe('cloth tethers follow moved pins', () => {
  it('free particles end within restRadius of the moved anchors', async () => {
    const segments = 7;
    const renderer = await createParticleRenderer();
    try {
      const geometry = new PlaneGeometry(1, 1, segments, segments);
      const graph = createClothGraph(geometry, {
        pinnedIndices: Array.from({ length: segments + 1 }, (_, i) => i),
      });
      const particles = new ParticleSystem(renderer, graph.positions.length, 0.02);
      const cloth = new ClothSystem(particles, {
        graph,
        stretchCompliance: 1e10,
        bendCompliance: 1e10,
        drag: 0,
        lift: 0,
      });
      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        gravity: new Vector3(0, 0, 0),
        materials: [cloth],
      });

      const shift = 1.5;
      const positions = particles.positions.value.array as Float32Array;
      const predicted = particles.predictedPositions.value.array as Float32Array;
      for (let i = 0; i < graph.positions.length; i++) {
        if (graph.invMass[i] !== 0) continue;
        positions[i * 4]! += shift;
        predicted[i * 4]! += shift;
      }
      particles.positions.value.needsUpdate = true;
      particles.predictedPositions.value.needsUpdate = true;

      for (let f = 0; f < 5; f++) await loop.step(1 / 60);

      const snap = await particles.readback();
      let maxOvershoot = -Infinity;
      for (const t of cloth.tethers) {
        const anchor = graph.positions[t.anchor]!;
        const dx = snap.positions[t.particle * 4]! - (anchor[0] + shift);
        const dy = snap.positions[t.particle * 4 + 1]! - anchor[1];
        const dz = snap.positions[t.particle * 4 + 2]! - anchor[2];
        maxOvershoot = Math.max(maxOvershoot, Math.hypot(dx, dy, dz) - t.restRadius);
      }
      expect(cloth.tethers.length).toBe(segments * (segments + 1));
      expect(maxOvershoot).toBeLessThan(1e-3);
      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
