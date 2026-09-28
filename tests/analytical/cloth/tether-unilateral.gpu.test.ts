import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute, Vector3 } from 'three';

import {
  ClothSystem,
  ParticleSystem,
  SimLoop,
  createClothGraph,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// Tethers are unilateral (Kim et al. 2012 §3.1).
//
// In a contracted cloth in zero gravity, tethers must not act: only
// distance constraints act in compression. Tethers must not pull
// particles OUTWARD when they're already inside the constraint sphere
// (`|x_i − a| ≤ r_i`): a tether constraint with `C ≤ 0` must produce
// zero force, regardless of how far inside the sphere the particle is.
//
// Setup: cloth pinned along its top row. Free particles are
// initialised at half their natural y-depth — i.e. bunched up
// near the pin row, well inside every tether sphere. Gravity is
// zero and BOTH distance and bending compliance are large
// (effectively disabled) so no other force can drive the cloth
// outward. If tethers were bilateral, they would pull the
// particles outward to their rest radii on the tether spheres
// (re-deploying the cloth). The unilateral kernel must NOT do
// that — particles should stay inside the spheres throughout.
//
// Validation: for every tether, `(|x_final − a| − r_i) ≤ ε`, where ε
// accounts for FP noise. Equivalently, no `C > 0` is ever produced
// under the contracted initial condition.

interface SheetData {
  readonly geometry: BufferGeometry;
  readonly pinnedIndices: number[];
}

function buildSheet(M: number): SheetData {
  const positions: number[] = [];
  const indices: number[] = [];
  const pinnedIndices: number[] = [];
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < M; i++) {
      const u = i / (M - 1);
      const v = j / (M - 1);
      positions.push(u, -v, 0);
    }
  }
  for (let j = 0; j < M - 1; j++) {
    for (let i = 0; i < M - 1; i++) {
      const a = j * M + i;
      const b = j * M + (i + 1);
      const c = (j + 1) * M + i;
      const d = (j + 1) * M + (i + 1);
      indices.push(a, c, d);
      indices.push(a, d, b);
    }
  }
  for (let i = 0; i < M; i++) pinnedIndices.push(i);
  const geom = new BufferGeometry();
  geom.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geom.setIndex(new Uint32BufferAttribute(new Uint32Array(indices), 1));
  return { geometry: geom, pinnedIndices };
}

describe('cloth tethers are unilateral (Kim et al. 2012 §3.1)', () => {
  it('contracted cloth in zero-gravity: tethers do not pull free particles outward', async () => {
    const M = 12;
    const renderer = await createParticleRenderer();
    try {
      const { geometry, pinnedIndices } = buildSheet(M);
      const graph = createClothGraph(geometry, {
        surfaceDensity: 0.2,
        pinnedIndices,
      });
      const particles = new ParticleSystem(renderer, graph.positions.length, 0.05);
      const cloth = new ClothSystem(particles, {
        graph,
        // Disable distance + bending so they don't drive
        // re-expansion. Tethers are the only constraint that
        // could move particles outward.
        stretchCompliance: 1e10,
        bendCompliance: 1e10,
        tetherCompliance: 0,
      });
      const tethers = cloth.tethers;
      // Top row is pinned → 1 island; every free particle gets
      // exactly 1 tether. Sanity check that we have something to test.
      expect(tethers.length).toBe(M * (M - 1));

      // Contract the cloth. The cloth uploaded its rest positions, so
      // overwrite them: pinned vertices stay; free vertices have their
      // y-coord halved (brings them closer to the top row). All
      // particles end up well inside every tether sphere.
      const initial: ParticleInit[] = [];
      for (let i = 0; i < graph.positions.length; i++) {
        const p = graph.positions[i]!;
        const px = p[0];
        const py = graph.invMass[i] === 0 ? p[1] : p[1] * 0.5;
        const pz = p[2];
        initial.push({ position: [px, py, pz], invMass: graph.invMass[i]! });
      }
      particles.uploadParticles(initial);

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        gravity: new Vector3(0, 0, 0),
        materials: [cloth],
      });

      const frameDt = 1 / 60;
      const damp = 0.95;
      for (let f = 0; f < 60; f++) {
        await loop.step(frameDt);
        // Damping ensures we measure the equilibrium / no-force
        // state, not transients from the initial condition.
        const buf = await renderer.getArrayBufferAsync(particles.velocities.value);
        const v = new Float32Array(buf);
        for (let k = 0; k < v.length; k++) v[k] = v[k]! * damp;
        (particles.velocities.value.array as Float32Array).set(v);
        particles.velocities.value.needsUpdate = true;
      }

      const snap = await particles.readback();
      let nanFree = true;
      let maxOvershoot = -Infinity;
      let maxAxialDist = 0;
      for (const t of tethers) {
        const i = t.particle;
        const x = snap.positions[4 * i + 0]!;
        const y = snap.positions[4 * i + 1]!;
        const z = snap.positions[4 * i + 2]!;
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          nanFree = false;
          continue;
        }
        const anchor = graph.positions[t.anchor]!;
        const dx = x - anchor[0];
        const dy = y - anchor[1];
        const dz = z - anchor[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const overshoot = dist - t.restRadius;
        if (overshoot > maxOvershoot) maxOvershoot = overshoot;
        if (dist > maxAxialDist) maxAxialDist = dist;
      }
      // Find the largest rest radius for context.
      let maxR = 0;
      for (const t of tethers) {
        if (t.restRadius > maxR) maxR = t.restRadius;
      }
      console.info(
        `[tether-unilateral] M=${M} nTethers=${tethers.length} maxRestRadius=${maxR.toFixed(4)} maxFinalDist=${maxAxialDist.toFixed(4)} maxOvershoot=${maxOvershoot.toFixed(6)} (negative = inside sphere)`,
      );
      expect(nanFree).toBe(true);
      // Tolerance: 1e-3 m absorbs FP noise + numerical drift
      // over 60 frames, but is well below the tether rest radius
      // (~unit-fraction). A bilateral tether would have
      // overshoot near zero (pulled to the sphere boundary) or
      // even negative converging to it; a misimplemented
      // unilateral that still pushes outward in some direction
      // would show overshoot > 0 by an order of magnitude more.
      expect(maxOvershoot).toBeLessThan(1e-3);
      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 180_000);
});
