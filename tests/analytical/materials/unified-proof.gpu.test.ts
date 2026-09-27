import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  SoftbodySystem,
  createParticleRenderer,
  voxelize,
  type ParticleInit,
  type TriangleMesh,
} from '../../../src/index.js';

/*
 * Fluid, soft body, and stiff bodies coupled in one simulation.
 *
 * Runs a single simulation of a mixed scene — a fluid pool, one soft cube,
 * and three stiff (rigid-like) cubes of different mass densities — in one
 * ParticleSystem and one SimLoop, and checks four properties in one run:
 *
 *   (1) Solid-vs-fluid non-penetration. < 0.5 % of fluid particles
 *       penetrate any body (defined here as a fluid-to-nearest-body-
 *       particle distance below the particle radius — contacts act at 2·r,
 *       so anything below r has crossed the body surface).
 *
 *   (2) Solid-vs-solid non-penetration. The max penetration depth across
 *       every pair of particles from different stiff bodies is < 5 % of
 *       the particle radius.
 *
 *   (3) Buoyancy ordering. COM_y(heavy) < COM_y(neutral) < COM_y(light),
 *       with the cube COMs separated by at least 1 mm.
 *
 *   (4) Mass conservation. The particle count is fixed (the scene neither
 *       emits nor drains); the check is that every particle's position is
 *       still finite.
 *
 * The stiff bodies are global shape-matching soft bodies with zero
 * compliance, and every body's surface is registered as a dynamic fluid
 * boundary. The scene is smaller than the demo's so it fits the GPU-test
 * budget; what matters is that fluid and bodies couple through the shared
 * particle core only.
 */

function unitCubeMesh(): TriangleMesh {
  const vertices = new Float32Array([
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
  ]);
  const indices = new Uint32Array([
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3,
    5, 3, 7, 5,
  ]);
  return { vertices, indices };
}

function scaledCube(edge: number): TriangleMesh {
  const m = unitCubeMesh();
  return {
    vertices: m.vertices.map((v) => v * edge),
    indices: m.indices,
  };
}

interface BodyRange {
  readonly start: number;
  readonly count: number;
}

describe('materials: fluid, soft body and stiff bodies in one simulation', () => {
  it('fluid + soft body + 3 stiff bodies couple via core only — non-penetration + buoyancy ordering', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.025;
      const spacing = 2 * r;
      const h = 4 * r;
      const restDensity = 1000;

      // Tank: 0.7 × 0.7 footprint, water level 0.2 m. Wide enough to keep
      // the three stiff cubes laterally apart at frame 0 — at an edge of
      // 0.15 m the cubes have a 0.075 m half-extent, and centers at
      // ±0.22 / 0 leave a ~7 cm gap between neighbors, comfortably above
      // the post-impact sloshing range.
      const tankHalf = 0.35;
      const waterLevel = 0.2;
      const FLOOR_Y = 0.0;

      // Bodies: smaller than the demo's, for the test budget. One
      // voxelization (3×3×3 particles) serves every body.
      const bodyEdge = 3 * spacing; // 0.15 m
      const vox = voxelize(scaledCube(bodyEdge), { particleRadius: r });

      // Fluid pack below the water line, so bodies dropped on top displace
      // into a settled column.
      const fluidPositions: [number, number, number][] = [];
      const margin = 1.25 * r;
      for (let y = FLOOR_Y + margin; y <= waterLevel - margin; y += spacing) {
        for (let x = -tankHalf + margin; x <= tankHalf - margin; x += spacing) {
          for (let z = -tankHalf + margin; z <= tankHalf - margin; z += spacing) {
            fluidPositions.push([x, y, z]);
          }
        }
      }
      const fluidCount = fluidPositions.length;
      const bodyCount = vox.count;

      // Body layout: soft body at (0, drop, +0.18); three stiff bodies in a
      // row at (−0.22, drop, −0.12), (0, drop, −0.12), (+0.22, drop, −0.12).
      // The lateral separation exceeds the body edge, so bodies start apart.
      const dropHeight = 0.4;
      const yDrop = waterLevel + dropHeight;
      const softOffset: readonly [number, number, number] = [0, yDrop, 0.18];
      const stiffOffsets: readonly (readonly [number, number, number])[] = [
        [-0.22, yDrop, -0.12], // light
        [0, yDrop, -0.12], // neutral
        [+0.22, yDrop, -0.12], // heavy
      ];
      const stiffInvMasses: readonly number[] = [200, 70, 25];

      const totalCount = fluidCount + bodyCount + 3 * bodyCount;
      const initial: ParticleInit[] = new Array(totalCount);

      for (let i = 0; i < fluidCount; i++) {
        const p = fluidPositions[i]!;
        initial[i] = { position: [p[0], p[1], p[2]], velocity: [0, 0, 0], invMass: 1 };
      }
      const softStart = fluidCount;
      for (let i = 0; i < bodyCount; i++) {
        initial[softStart + i] = {
          position: [
            vox.positions[3 * i + 0]! + softOffset[0],
            vox.positions[3 * i + 1]! + softOffset[1],
            vox.positions[3 * i + 2]! + softOffset[2],
          ],
          velocity: [0, 0, 0],
          invMass: 80,
        };
      }
      const stiffRanges: BodyRange[] = [];
      let cursor = softStart + bodyCount;
      for (let b = 0; b < 3; b++) {
        const off = stiffOffsets[b]!;
        const start = cursor;
        stiffRanges.push({ start, count: bodyCount });
        for (let i = 0; i < bodyCount; i++) {
          initial[start + i] = {
            position: [
              vox.positions[3 * i + 0]! + off[0],
              vox.positions[3 * i + 1]! + off[1],
              vox.positions[3 * i + 2]! + off[2],
            ],
            velocity: [0, 0, 0],
            invMass: stiffInvMasses[b]!,
          };
        }
        cursor += bodyCount;
      }

      const particles = new ParticleSystem(renderer, totalCount, r);
      particles.uploadParticles(initial);

      const fluid = new FluidSystem(particles, {
        range: { start: 0, count: fluidCount },
        restDensity,
        particleSpacing: spacing,
        smoothingRadius: h,
        compliance: 1e-4,
        viscosity: 0.1,
      });

      // Rest shapes default to the uploaded positions. Each body gets its
      // own collision group when the loop is built.
      const softbody = new SoftbodySystem(particles, {
        bodies: [
          {
            range: { start: softStart, count: bodyCount },
            surfaceCount: vox.surfaceCount,
            compliance: 1e-7,
          },
        ],
      });
      // Rigid-like bodies: global shape matching with zero compliance.
      const stiff = new SoftbodySystem(particles, {
        bodies: stiffRanges.map((range) => ({
          range,
          surfaceCount: vox.surfaceCount,
          compliance: 0,
        })),
      });

      // Tank: floor + 4 wall planes. Planes are simpler than the box
      // colliders the demo uses; the test only needs everything contained.
      const colliders = new PrimitiveSet(particles);
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, FLOOR_Y, 0), {
        muS: 0.5,
        muK: 0.35,
      });
      colliders.addPlane(new Vector3(1, 0, 0), new Vector3(-tankHalf, 0, 0), {
        muS: 0.5,
        muK: 0.35,
      });
      colliders.addPlane(new Vector3(-1, 0, 0), new Vector3(tankHalf, 0, 0), {
        muS: 0.5,
        muK: 0.35,
      });
      colliders.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -tankHalf), {
        muS: 0.5,
        muK: 0.35,
      });
      colliders.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, tankHalf), {
        muS: 0.5,
        muK: 0.35,
      });

      // One dynamic boundary range per body. Must happen before the SimLoop
      // is created.
      fluid.addBoundary(softbody.surfaceRange(0));
      for (let b = 0; b < stiffRanges.length; b++) {
        fluid.addBoundary(stiff.surfaceRange(b));
      }

      const loop = new SimLoop(particles, {
        substeps: 3,
        iterations: 2,
        contact: {
          maxContacts: Math.max(8192, totalCount * 6),
          muS: 0.5,
          muK: 0.35,
        },
        colliders: [colliders],
        materials: [fluid, softbody, stiff],
      });
      loop.gravity.set(0, -9.81, 0);

      // 4 s is enough for the stiff cubes to settle at their buoyancy-
      // determined heights, and keeps the test inside the GPU-test budget.
      const totalSeconds = 4.0;
      const dt = 1 / 60;
      const frames = Math.ceil(totalSeconds / dt);
      for (let f = 0; f < frames; f++) {
        await loop.step(dt);
      }

      const snap = await particles.readback();

      // ----------- Mass conservation ------------
      // The count is fixed (no emit/drain), so the actual check is that
      // every slot holds finite coordinates (no NaNs from a divide-by-zero
      // or a singular shape-matching matrix).
      for (let i = 0; i < totalCount; i++) {
        const x = snap.positions[4 * i + 0]!;
        const y = snap.positions[4 * i + 1]!;
        const z = snap.positions[4 * i + 2]!;
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          throw new Error(`non-finite position at slot ${i}: (${x}, ${y}, ${z})`);
        }
      }

      // ----------- Buoyancy ordering ------------
      const stiffComY: number[] = [];
      for (const range of stiffRanges) {
        let sumY = 0;
        for (let i = 0; i < range.count; i++) {
          sumY += snap.positions[4 * (range.start + i) + 1]!;
        }
        stiffComY.push(sumY / range.count);
      }
      const [comLight, comNeutral, comHeavy] = stiffComY as [number, number, number];

      console.info(
        `[unified-proof] T=${totalSeconds.toFixed(1)}s stiff COM_y: light=${comLight.toFixed(4)} neutral=${comNeutral.toFixed(4)} heavy=${comHeavy.toFixed(4)}`,
      );

      // Light > Neutral > Heavy with a 1 mm margin.
      const margin1mm = 0.001;
      expect(comLight - comNeutral).toBeGreaterThan(margin1mm);
      expect(comNeutral - comHeavy).toBeGreaterThan(margin1mm);

      // ----------- Solid-vs-solid non-penetration ------------
      // For every pair of stiff bodies, find the minimum particle-to-
      // particle distance. Penetration depth = max(0, 2r − min_dist); the
      // target is < 5 % of r = 0.00125 m at r = 0.025.
      let maxStiffPenetration = 0;
      for (let a = 0; a < stiffRanges.length; a++) {
        for (let b = a + 1; b < stiffRanges.length; b++) {
          const ra = stiffRanges[a]!;
          const rb = stiffRanges[b]!;
          let minDist = Infinity;
          for (let i = 0; i < ra.count; i++) {
            const ix = snap.positions[4 * (ra.start + i) + 0]!;
            const iy = snap.positions[4 * (ra.start + i) + 1]!;
            const iz = snap.positions[4 * (ra.start + i) + 2]!;
            for (let j = 0; j < rb.count; j++) {
              const jx = snap.positions[4 * (rb.start + j) + 0]!;
              const jy = snap.positions[4 * (rb.start + j) + 1]!;
              const jz = snap.positions[4 * (rb.start + j) + 2]!;
              const d = Math.hypot(ix - jx, iy - jy, iz - jz);
              if (d < minDist) minDist = d;
            }
          }
          const pen = Math.max(0, 2 * r - minDist);
          if (pen > maxStiffPenetration) maxStiffPenetration = pen;
        }
      }
      console.info(
        `[unified-proof] max stiff-stiff penetration = ${(maxStiffPenetration * 1000).toFixed(3)} mm (target < ${(0.05 * r * 1000).toFixed(3)} mm = 5% of r)`,
      );
      expect(maxStiffPenetration).toBeLessThan(0.05 * r);

      // ----------- Solid-vs-fluid non-penetration ------------
      // For each fluid particle, look for a body particle closer than r;
      // such a fluid particle has crossed the body surface (contacts act at
      // 2·r). Count the violations. Cost: O(fluid · body particles) pair
      // tests, well under a second in JS.
      const allBodyStart = softStart;
      const allBodyEnd = totalCount;
      let penetrators = 0;
      for (let f = 0; f < fluidCount; f++) {
        const fx = snap.positions[4 * f + 0]!;
        const fy = snap.positions[4 * f + 1]!;
        const fz = snap.positions[4 * f + 2]!;
        for (let b = allBodyStart; b < allBodyEnd; b++) {
          const dx = fx - snap.positions[4 * b + 0]!;
          const dy = fy - snap.positions[4 * b + 1]!;
          const dz = fz - snap.positions[4 * b + 2]!;
          if (dx * dx + dy * dy + dz * dz < r * r) {
            penetrators += 1;
            break;
          }
        }
      }
      const penFraction = penetrators / fluidCount;
      console.info(
        `[unified-proof] fluid penetrators: ${penetrators} / ${fluidCount} (${(penFraction * 100).toFixed(3)}%; target < 0.5%)`,
      );
      expect(penFraction).toBeLessThan(0.005);

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 240_000);
});
