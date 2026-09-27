import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute, Vector3 } from 'three';
import {
  ClothSystem,
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createClothGraph,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

/*
 * Fluid and cloth coupled in one simulation.
 *
 * One scene, one `ParticleSystem`, one `SimLoop`, two materials —
 * `[fluid, cloth]`. Coupling goes through the same boundary-particle
 * pipeline (Akinci 2012 §2, §2.2) soft bodies use: the cloth's particle
 * range is registered as a dynamic fluid boundary.
 *
 * Checks, in one run:
 *
 *   (1) Fluid ↔ cloth non-penetration. < 0.5 % of fluid particles end up
 *       within `r` of any cloth particle (contacts act at 2·r, so anything
 *       below `r` has crossed the cloth's contact surface).
 *
 *   (2) Mass conservation. Every position is finite (no NaN from a
 *       boundary-volume blow-up, density divergence at the cloth boundary,
 *       or a constraint solve going singular).
 *
 *   (3) The pinned corners stay where they were pinned.
 *
 * The scene is smaller than the demo's, for the test budget:
 *
 *   - Tank: half-extent 0.18 m, water level 0.12 m, floor at y = 0. Fluid
 *     packed at 2·r spacing inside the tank below the water line —
 *     6×6×2 = 72 fluid particles at r = 0.025.
 *   - Cloth: 8×8 quads (81 vertices), 0.20 × 0.20 m, pinned at the two TOP
 *     CORNERS (a curtain; each corner anchors its own long-range
 *     attachments, Kim 2012 §3.4). The pin row is 0.20 m above the water,
 *     so the bottom half of the curtain submerges into the pool over the
 *     run.
 *   - Materials: `[fluid, cloth]`, S = 4, I = 2, dt = 1/60 s.
 *   - One boundary registration: the entire cloth range, dynamic (moving,
 *     deforming boundary, Akinci 2012 §2.2).
 *
 * Run for 3 s (180 frames): enough for the curtain to settle onto the water
 * and, at the explicit cloth mass used here, start sinking its lower edge.
 * The fluid must not pass through the cloth even as the cloth deforms.
 */

interface ClothMeshGeom {
  readonly geometry: BufferGeometry;
  readonly pinnedIndices: number[];
  readonly nParticles: number;
  readonly nx: number;
  readonly ny: number;
}

function buildCurtainGeom(args: {
  readonly width: number;
  readonly height: number;
  readonly widthSegments: number;
  readonly heightSegments: number;
  readonly originY: number;
}): ClothMeshGeom {
  const { width, height, widthSegments, heightSegments, originY } = args;
  const nx = widthSegments + 1;
  const ny = heightSegments + 1;
  const positions = new Float32Array(nx * ny * 3);
  const indices: number[] = [];

  for (let j = 0; j < ny; j++) {
    const v = j / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const u = i / (nx - 1);
      const idx = j * nx + i;
      positions[3 * idx + 0] = (u - 0.5) * width;
      positions[3 * idx + 1] = originY - v * height;
      positions[3 * idx + 2] = 0;
    }
  }
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i;
      const b = j * nx + (i + 1);
      const c = (j + 1) * nx + i;
      const d = (j + 1) * nx + (i + 1);
      indices.push(a, c, d);
      indices.push(a, d, b);
    }
  }
  const pinnedIndices = [0, nx - 1]; // top-left, top-right corners

  const geom = new BufferGeometry();
  geom.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geom.setIndex(new Uint32BufferAttribute(new Uint32Array(indices), 1));

  return { geometry: geom, pinnedIndices, nParticles: nx * ny, nx, ny };
}

describe('materials: fluid and cloth in one simulation', () => {
  it('fluid + cloth couple via core only — non-penetration + mass conservation', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.025;
      const spacing = 2 * r;
      const h = 4 * r;
      const restDensity = 1000;

      const tankHalf = 0.18;
      const waterLevel = 0.12;
      const FLOOR_Y = 0.0;
      const margin = 1.25 * r;

      // ---- Fluid pack ----
      const fluidPositions: [number, number, number][] = [];
      for (let y = FLOOR_Y + margin; y <= waterLevel - margin; y += spacing) {
        for (let x = -tankHalf + margin; x <= tankHalf - margin; x += spacing) {
          for (let z = -tankHalf + margin; z <= tankHalf - margin; z += spacing) {
            fluidPositions.push([x, y, z]);
          }
        }
      }
      const fluidCount = fluidPositions.length;

      // ---- Cloth: 8×8 curtain pinned at its two top corners. The pin row
      //      is 0.20 m above the water and the cloth is 0.20 m long, so its
      //      bottom edge starts at the water line and submerges within the
      //      first second. ----
      const clothWidth = 0.2;
      const clothHeight = 0.2;
      const widthSegments = 8;
      const heightSegments = 8;
      const pinRowY = waterLevel + 0.2;
      const {
        geometry,
        pinnedIndices,
        nParticles: clothCount,
        nx,
      } = buildCurtainGeom({
        width: clothWidth,
        height: clothHeight,
        widthSegments,
        heightSegments,
        originY: pinRowY,
      });
      const graph = createClothGraph(geometry, {
        surfaceDensity: 0.2,
        pinnedIndices,
      });

      const totalCount = fluidCount + clothCount;

      // Fluid slots [0, fluidCount). The cloth writes its own slots
      // [fluidCount, fluidCount + clothCount) when it is created. Upload
      // before creating the fluid, which then sets its particles' mass.
      const fluidInit: ParticleInit[] = fluidPositions.map((p) => ({
        position: [p[0], p[1], p[2]],
        velocity: [0, 0, 0],
        invMass: 1,
      }));
      const particles = new ParticleSystem(renderer, totalCount, r);
      particles.uploadParticles(fluidInit);

      const fluid = new FluidSystem(particles, {
        range: { start: 0, count: fluidCount },
        restDensity,
        particleSpacing: spacing,
        smoothingRadius: h,
        compliance: 1e-4,
        viscosity: 0.1,
        adhesion: 0.5,
      });

      const cloth = new ClothSystem(particles, {
        graph,
        offset: fluidCount,
        stretchCompliance: 1e-7,
        bendCompliance: 1e-5,
        tetherCompliance: 1e-10,
        stretchTolerance: 0,
      });
      // Override the surface-density-derived masses with one explicit value
      // (invMass 25: 40 g per particle, against 125 g per fluid particle),
      // so the curtain settles slowly without ripping or shooting through
      // the surface. Pinned vertices keep invMass = 0.
      const clothInvMass = 25;
      particles.setInvMass(cloth.range, clothInvMass);
      for (const pin of pinnedIndices) {
        particles.setInvMass({ start: cloth.range.start + pin, count: 1 }, 0);
      }
      // Neighboring cloth particles sit closer than 2r, so the cloth must
      // not collide with itself: give its particles a shared group.
      particles.setCollisionGroup(cloth.range, 1);

      // ---- Tank colliders: floor + 4 wall planes. ----
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

      // ---- The coupling: register the cloth range as a dynamic boundary
      //      (Akinci 2012 §2, §2.2), the same call soft bodies use. Must
      //      happen before the SimLoop is created. ----
      fluid.addBoundary(cloth.range, { dynamic: true });

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        contact: {
          maxContacts: Math.max(8192, totalCount * 6),
          muS: 0.5,
          muK: 0.35,
        },
        colliders: [colliders],
        materials: [fluid, cloth],
      });
      loop.gravity.set(0, -9.81, 0);

      // 3 s is enough for the curtain to settle onto the water and start
      // submerging.
      const totalSeconds = 3.0;
      const dt = 1 / 60;
      const frames = Math.ceil(totalSeconds / dt);
      for (let f = 0; f < frames; f++) {
        await loop.step(dt);
      }

      const snap = await particles.readback();

      // ---- Mass conservation: NaN-free, finite positions.
      for (let i = 0; i < totalCount; i++) {
        const x = snap.positions[4 * i + 0]!;
        const y = snap.positions[4 * i + 1]!;
        const z = snap.positions[4 * i + 2]!;
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          throw new Error(`non-finite position at slot ${i}: (${x}, ${y}, ${z})`);
        }
      }

      // ---- Fluid ↔ cloth non-penetration.
      // For each fluid particle, look for a cloth particle closer than r;
      // such a fluid particle has crossed the cloth (contacts act at 2·r).
      // Count the violations.
      const clothStart = cloth.range.start;
      const clothEnd = clothStart + cloth.range.count;
      let penetrators = 0;
      for (let f = 0; f < fluidCount; f++) {
        const fx = snap.positions[4 * f + 0]!;
        const fy = snap.positions[4 * f + 1]!;
        const fz = snap.positions[4 * f + 2]!;
        for (let c = clothStart; c < clothEnd; c++) {
          const dx = fx - snap.positions[4 * c + 0]!;
          const dy = fy - snap.positions[4 * c + 1]!;
          const dz = fz - snap.positions[4 * c + 2]!;
          if (dx * dx + dy * dy + dz * dz < r * r) {
            penetrators += 1;
            break;
          }
        }
      }
      const penFraction = penetrators / fluidCount;
      console.info(
        `[wet-cloth] T=${totalSeconds.toFixed(1)}s fluid=${fluidCount} cloth=${clothCount} penetrators=${penetrators} (${(penFraction * 100).toFixed(3)}%; target < 0.5%)`,
      );
      expect(penFraction).toBeLessThan(0.005);

      // ---- Pinned corners: the top-left and top-right cloth vertices must
      //      still be at their pin positions. Drift here means a pin
      //      (invMass = 0) was overridden somewhere — a cloth regression
      //      rather than a coupling failure, but catching it keeps the test
      //      from passing on a half-broken setup.
      const pinIndices = [0, nx - 1];
      for (const pi of pinIndices) {
        const slot = clothStart + pi;
        const py = snap.positions[4 * slot + 1]!;
        if (Math.abs(py - pinRowY) > 1e-3) {
          throw new Error(`pin slot ${slot} drifted from y=${pinRowY} to y=${py}`);
        }
      }

      loop.dispose();
      particles.dispose();
      colliders.dispose();
    } finally {
      renderer.dispose();
    }
  }, 180_000);
});
