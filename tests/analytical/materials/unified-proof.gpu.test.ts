import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { FluidSystem } from '../../../src/fluids/index.js';
import {
  RigidBodySystem,
  SoftbodySystem,
  voxelize,
  type RigidBodyDef,
  type TriangleMesh,
} from '../../../src/softbody/index.js';

/*
 * Phase 17 — unified-architecture proof (G1 / blocking).
 *
 * Runs a single simulation of the unified scene (fluid pool + 1
 * softbody cube + 3 rigid cubes with different mass densities) and
 * asserts four blocking exit-criterion gates in one run:
 *
 *   (1) Solid-vs-fluid non-penetration. < 0.5 % of fluid particles
 *       penetrate any body (defined here as fluid-to-nearest-body-
 *       particle distance below particleRadius — the contact pipeline
 *       fires at 2·r, so anything below r has crossed the body
 *       surface past the contact gate).
 *
 *   (2) Solid-vs-solid non-penetration. Max penetration depth across
 *       every cross-rigid-body particle pair < 5 % of particleRadius.
 *
 *   (3) Buoyancy monotonicity. COM_y(heavy) < COM_y(neutral) <
 *       COM_y(light) with strictly non-overlapping cube COM bands
 *       (margin 1 mm — same scale used in Phase 11
 *       fluid-solid-mass-ratio).
 *
 *   (4) Mass conservation. Particle count constant (fluid + body
 *       counts equal start counts at end of sim). Implicit since the
 *       scene neither emits nor drains.
 *
 *
 * Scene dimensions are smaller than the demo's (tankHalf 0.2 instead
 * of 0.5, single softbody instead of three) so the test fits inside
 * the GPU-test budget. Architectural claim is invariant under scene
 * size — what matters is that the THREE constraint families couple
 * through core only.
 */

function unitCubeMesh(): TriangleMesh {
  // eslint-disable-next-line prettier/prettier
  const vertices = new Float32Array([
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
  ]);
  // eslint-disable-next-line prettier/prettier
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
  readonly surfaceCount: number;
}

describe('Phase 17 — unified-architecture proof (G1)', () => {
  it('fluid + softbody + 3 rigids couple via core only — non-penetration + buoyancy monotonicity', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.025;
      const spacing = 2 * r;
      const h = 4 * r;
      const restDensity = 1000;

      // Tank: 0.7 × 0.7 footprint, water level 0.2 m. Wider than the
      // demo to keep the three rigid cubes laterally non-overlapping
      // at frame 0 — at rigidEdge = 0.15 m the cubes occupy 0.075 m
      // half-extent, and centers at ±0.22 / 0 leave ~7 cm gap
      // between adjacent cubes, comfortably above the post-impact
      // sloshing range.
      const tankHalf = 0.35;
      const waterLevel = 0.2;
      const FLOOR_Y = 0.0;

      // Bodies: smaller than the demo for test budget.
      const softEdge = 3 * spacing; // 0.15 m — about 3×3×3 voxels
      const rigidEdge = 3 * spacing;

      const softVox = voxelize(scaledCube(softEdge), {
        particleRadius: r,
        spacingFactor: 1.0,
      });
      const rigidVox = voxelize(scaledCube(rigidEdge), {
        particleRadius: r,
        bakeSdf: true,
        spacingFactor: 1.0,
      });

      // Fluid pack — 8×8×8 = 512 particles, comfortably below the
      // tank surface so bodies dropped on top displace into a settled
      // column.
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

      const softCount = softVox.count;
      const rigidCount = rigidVox.count;

      // Body grid layout: softbody at (0, drop, +0.12); 3 rigids in a
      // row at (-0.12, drop, -0.08), (0, drop, -0.08), (+0.12, drop, -0.08).
      // Lateral separation > rigidEdge so bodies start non-touching.
      const dropHeight = 0.4;
      const yDrop = waterLevel + dropHeight;
      const softOffset: readonly [number, number, number] = [0, yDrop, 0.18];
      const rigidOffsets: readonly (readonly [number, number, number])[] = [
        [-0.22, yDrop, -0.12], // light
        [0, yDrop, -0.12], // neutral
        [+0.22, yDrop, -0.12], // heavy
      ];
      const rigidInvMasses: readonly number[] = [200, 70, 25];

      const totalCount = fluidCount + softCount + 3 * rigidCount;
      const initial: ParticleInit[] = new Array(totalCount);

      for (let i = 0; i < fluidCount; i++) {
        const p = fluidPositions[i]!;
        initial[i] = {
          position: [p[0], p[1], p[2]],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: (1 << 16) >>> 0,
        };
      }
      const softStart = fluidCount;
      for (let i = 0; i < softCount; i++) {
        initial[softStart + i] = {
          position: [
            softVox.positions[3 * i + 0]! + softOffset[0],
            softVox.positions[3 * i + 1]! + softOffset[1],
            softVox.positions[3 * i + 2]! + softOffset[2],
          ],
          velocity: [0, 0, 0],
          invMass: 80,
          phase: (2 << 16) >>> 0,
        };
      }
      const rigidRanges: BodyRange[] = [];
      let cursor = softStart + softCount;
      for (let b = 0; b < 3; b++) {
        const off = rigidOffsets[b]!;
        const start = cursor;
        rigidRanges.push({
          start,
          count: rigidCount,
          surfaceCount: rigidVox.surfaceCount,
        });
        for (let i = 0; i < rigidCount; i++) {
          initial[start + i] = {
            position: [
              rigidVox.positions[3 * i + 0]! + off[0],
              rigidVox.positions[3 * i + 1]! + off[1],
              rigidVox.positions[3 * i + 2]! + off[2],
            ],
            velocity: [0, 0, 0],
            invMass: rigidInvMasses[b]!,
            phase: ((b + 3) << 16) >>> 0,
          };
        }
        cursor += rigidCount;
      }

      const particles = new ParticleSystem(renderer, totalCount, r);
      particles.uploadParticles(initial);

      const xpbd = createXpbdUniforms(1 / 60);
      const hashGrid = new HashGrid(particles, { cellSize: h });

      const fluid = new FluidSystem({
        particles,
        hashGrid,
        xpbd,
        restDensity,
        h,
        particleSpacing: spacing,
        compliance: 1e-4,
        fluidParticles: { start: 0, count: fluidCount },
        vorticity: { strength: 0 },
        xsph: { c: 0.1 },
        surfaceTension: 0,
        adhesion: 0,
      });

      const restSoftFlat = new Float32Array(softVox.positions);
      const softbody = new SoftbodySystem({
        particles,
        xpbd,
        bodies: [
          {
            particleRange: { start: softStart, count: softCount },
            restPositions: restSoftFlat,
            surfaceFlag: softVox.surfaceFlag.slice(),
            phaseId: 2,
            matchCompliance: 1e-7,
            edges: softVox.edges,
          },
        ],
      });

      const restRigidFlat = new Float32Array(rigidVox.positions);
      const rigidBodies: RigidBodyDef[] = rigidRanges.map((rng, b) => ({
        particleRange: { start: rng.start, count: rng.count },
        restPositions: restRigidFlat.slice(),
        restSDF: rigidVox.restSDF!.slice(),
        phaseId: b + 3,
        compliance: 0,
      }));
      const rigid = new RigidBodySystem({
        particles,
        xpbd,
        bodies: rigidBodies,
      });

      // Tank: floor + 4 wall planes (analytic). Wall planes simpler
      // than the box colliders the demo uses; for the test we just
      // need fluid + bodies to stay contained.
      const colliders = new PrimitiveSet(particles, { capacity: 5 });
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
      colliders.upload();

      // Multi-range registration — per-body, all dynamic. MUST run
      // before SimLoop construction.
      await fluid.registerBoundaryParticles(softbody.surfaceRange(0));
      for (const rng of rigidRanges) {
        await fluid.registerBoundaryParticles({
          start: rng.start,
          count: rng.surfaceCount,
        });
      }

      const loop = new SimLoop(particles, {
        substeps: 3,
        iterations: 2,
        xpbd,
        hashGrid,
        contact: {
          hashGrid,
          maxContacts: Math.max(8192, totalCount * 6),
          friction: { muS: 0.5, muK: 0.35 },
          stabIters: 1,
        },
        colliders: { colliders },
        materials: [fluid, softbody, rigid],
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.81, 0);

      // Run sim — 4 s is enough for rigid cubes to settle to their
      // buoyancy-determined positions; the test keeps inside the
      // 3 minute GPU-test budget.
      const totalSeconds = 4.0;
      const dt = 1 / 60;
      const frames = Math.ceil(totalSeconds / dt);
      for (let f = 0; f < frames; f++) {
        await loop.step(dt);
      }

      const snap = await particles.readback();

      // ----------- Mass conservation ------------
      // Trivially true (count is fixed, no emit/drain). The actual
      // check: every slot in the particle buffer holds finite
      // coordinates (no NaNs from a kernel divide-by-zero or runaway
      // shape-match singular A_pq).
      for (let i = 0; i < totalCount; i++) {
        const x = snap.positions[4 * i + 0]!;
        const y = snap.positions[4 * i + 1]!;
        const z = snap.positions[4 * i + 2]!;
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          throw new Error(`non-finite position at slot ${i}: (${x}, ${y}, ${z})`);
        }
      }

      // ----------- Buoyancy monotonicity ------------
      const rigidComY: number[] = [];
      for (const rng of rigidRanges) {
        let sumY = 0;
        for (let i = 0; i < rng.count; i++) {
          sumY += snap.positions[4 * (rng.start + i) + 1]!;
        }
        rigidComY.push(sumY / rng.count);
      }
      const [comLight, comNeutral, comHeavy] = rigidComY as [number, number, number];

      // eslint-disable-next-line no-console
      console.info(
        `[unified-proof] T=${totalSeconds.toFixed(1)}s rigid COM_y: light=${comLight.toFixed(4)} neutral=${comNeutral.toFixed(4)} heavy=${comHeavy.toFixed(4)}`,
      );

      // Light > Neutral > Heavy with 1 mm margin.
      const margin1mm = 0.001;
      expect(comLight - comNeutral).toBeGreaterThan(margin1mm);
      expect(comNeutral - comHeavy).toBeGreaterThan(margin1mm);

      // ----------- Solid-vs-solid non-penetration ------------
      // For every rigid-rigid pair, find the minimum particle-to-
      // particle distance. Penetration depth = max(0, 2r − min_dist).
      // Plan target: penetration < 5 % of r = 0.00125 m at r = 0.025.
      let maxRigidPenetration = 0;
      for (let a = 0; a < rigidRanges.length; a++) {
        for (let b = a + 1; b < rigidRanges.length; b++) {
          const ra = rigidRanges[a]!;
          const rb = rigidRanges[b]!;
          let minDist = Infinity;
          for (let i = 0; i < ra.count; i++) {
            const ix = snap.positions[4 * (ra.start + i) + 0]!;
            const iy = snap.positions[4 * (ra.start + i) + 1]!;
            const iz = snap.positions[4 * (ra.start + i) + 2]!;
            for (let j = 0; j < rb.count; j++) {
              const jx = snap.positions[4 * (rb.start + j) + 0]!;
              const jy = snap.positions[4 * (rb.start + j) + 1]!;
              const jz = snap.positions[4 * (rb.start + j) + 2]!;
              const dx = ix - jx;
              const dy = iy - jy;
              const dz = iz - jz;
              const d = Math.hypot(dx, dy, dz);
              if (d < minDist) minDist = d;
            }
          }
          const pen = Math.max(0, 2 * r - minDist);
          if (pen > maxRigidPenetration) maxRigidPenetration = pen;
        }
      }
      // eslint-disable-next-line no-console
      console.info(
        `[unified-proof] max rigid-rigid penetration = ${(maxRigidPenetration * 1000).toFixed(3)} mm (target < ${(0.05 * r * 1000).toFixed(3)} mm = 5% of r)`,
      );
      expect(maxRigidPenetration).toBeLessThan(0.05 * r);

      // ----------- Solid-vs-fluid non-penetration ------------
      // For each fluid particle, find the closest body particle. If
      // dist < r, the fluid has penetrated past the contact-pair
      // gate (which fires at 2·r). Count the violations.
      // Cost: O(fluidCount · totalBodyCount) ≈ 512 · (64 + 3·64) ≈
      // 130k pair tests — runs in well under a second on JS.
      const allBodyStart = softStart;
      const allBodyEnd = totalCount;
      let penetrators = 0;
      for (let f = 0; f < fluidCount; f++) {
        const fx = snap.positions[4 * f + 0]!;
        const fy = snap.positions[4 * f + 1]!;
        const fz = snap.positions[4 * f + 2]!;
        for (let b = allBodyStart; b < allBodyEnd; b++) {
          const bx = snap.positions[4 * b + 0]!;
          const by = snap.positions[4 * b + 1]!;
          const bz = snap.positions[4 * b + 2]!;
          const dx = fx - bx;
          const dy = fy - by;
          const dz = fz - bz;
          if (dx * dx + dy * dy + dz * dz < r * r) {
            penetrators += 1;
            break;
          }
        }
      }
      const penFraction = penetrators / fluidCount;
      // eslint-disable-next-line no-console
      console.info(
        `[unified-proof] fluid penetrators: ${penetrators} / ${fluidCount} (${(penFraction * 100).toFixed(3)}%; target < 0.5%)`,
      );
      expect(penFraction).toBeLessThan(0.005);

      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    } finally {
      renderer.dispose();
    }
  }, 240_000);
});
