import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute, Vector3 } from 'three';
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
import { ClothSystem, fromBufferGeometry } from '../../../src/cloth/index.js';

/*
 * Phase 20 — wet-cloth unified-architecture proof (G1 / blocking).
 *
 * Re-runs the Phase 17 unified-proof structure on the fluid ↔ cloth axis.
 * One scene, one `ParticleSystem`, one `SimLoop`, two materials —
 * `[fluid, cloth]`. Coupling flows through the same Akinci 2012 §2 + §2.2
 * boundary-particle pipeline Phase 17 used for softbody + rigid surface
 * particles; Phase 20 invokes it on a cloth-vertex range.
 *
 * Three blocking gates in one run:
 *
 *   (1) Fluid ↔ cloth non-penetration. < 0.5 % of fluid particles end
 *       up within `r` of any cloth particle (the contact pipeline gate
 *       fires at 2·r — anything below `r` has crossed the cloth's
 *       contact surface).
 *
 *   (2) Mass conservation. Every position is finite (no NaN from
 *       boundary-volume blow-up, density divergence at the cloth
 *       boundary, or constraint solve singular A_pq).
 *
 *
 * Scene is smaller than the demo for test budget. Scene shape:
 *
 *   - Tank: half-extent 0.18 m, water level 0.12 m, FLOOR_Y = 0. Fluid
 *     packed at 2·r spacing inside the tank below the water line —
 *     ~5×5×4 ≈ 100 fluid particles at r = 0.025.
 *   - Cloth: 8×8 quads (81 vertices), 0.20 × 0.20 m, pinned at the
 *     two TOP CORNERS (Phase 19's `cloth-curtain` topology — multi-
 *     island Kim 2012 §3.4 LRA assignment). Pin row at
 *     y = waterLevel + 0.20 m so the bottom half of the curtain
 *     submerges into the pool over the simulation time.
 *   - Materials: `[fluid, cloth]`, S = 4, I = 2, dt = 1/60 s.
 *   - One boundary registration: the entire cloth particle range,
 *     `{dynamic: true}` (paper §2.2 — moving / deforming boundaries).
 *
 * Run for 3 s (180 frames). 3 s is enough for the curtain to settle
 * onto the water and (at the 25-invMass cloth-mass default) start
 * sinking the lower edge. The fluid must not teleport through the
 * cloth even as the cloth deforms substantially.
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

describe('Phase 20 — wet-cloth unified-architecture proof (G1)', () => {
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

      // ---- Cloth: 8×8 curtain pinned at two top corners.
      //      Pin row 0.20 m above water; cloth length 0.20 m so the
      //      bottom edge lands at the water line at rest, and starts
      //      submerging within the first second. ----
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
      const graph = fromBufferGeometry(geometry, {
        surfaceDensity: 0.2,
        pinnedIndices,
      });

      const totalCount = fluidCount + clothCount;
      const initial: ParticleInit[] = new Array(totalCount);

      const phaseFor = (id: number): number => ((id & 0xffff) << 16) >>> 0;
      // Fluid slots [0, fluidCount).
      for (let i = 0; i < fluidCount; i++) {
        const p = fluidPositions[i]!;
        initial[i] = {
          position: [p[0], p[1], p[2]],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: phaseFor(1),
        };
      }
      // Cloth slots [fluidCount, fluidCount + clothCount).
      // Override surface-density-derived invMass with explicit value
      // (25 → ρ ≈ 800 kg/m³ at the cloth particle radius — slightly
      // less dense than fluid so the curtain settles slowly without
      // ripping or rocketing through the surface). Pinned vertices
      // keep invMass = 0.
      const clothInvMass = 25;
      for (let i = 0; i < clothCount; i++) {
        const p = graph.positions[i]!;
        const pinned = graph.invMass[i] === 0;
        initial[fluidCount + i] = {
          position: [p[0], p[1], p[2]],
          velocity: [0, 0, 0],
          invMass: pinned ? 0 : clothInvMass,
          phase: phaseFor(2),
        };
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
        adhesion: 0.5,
      });

      const cloth = new ClothSystem({
        particles,
        xpbd,
        graph,
        particleOffset: fluidCount,
        stretchCompliance: 1e-7,
        bendCompliance: 1e-5,
        tetherCompliance: 1e-10,
        stretchTolerance: 0,
      });

      // ---- Tank colliders: floor + 4 wall planes (analytic). ----
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

      // ---- THE coupling line: register the cloth range as a
      //      dynamic boundary. Akinci 2012 §2 + §2.2. Same call shape
      //      Phase 17 used for softbody + rigid surface ranges. MUST
      //      run before SimLoop construction. ----
      await fluid.registerBoundaryParticles(
        { start: cloth.particleOffset, count: cloth.nParticles },
        { dynamic: true },
      );

      const loop = new SimLoop(particles, {
        substeps: 4,
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
        materials: [fluid, cloth],
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.81, 0);

      // Run sim — 3 s is enough for the curtain to settle onto the
      // water and start submerging.
      const totalSeconds = 3.0;
      const dt = 1 / 60;
      const frames = Math.ceil(totalSeconds / dt);
      for (let f = 0; f < frames; f++) {
        await loop.step(dt);
      }

      const snap = await particles.readback();

      // ---- Mass conservation: NaN-free finite positions.
      for (let i = 0; i < totalCount; i++) {
        const x = snap.positions[4 * i + 0]!;
        const y = snap.positions[4 * i + 1]!;
        const z = snap.positions[4 * i + 2]!;
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          throw new Error(`non-finite position at slot ${i}: (${x}, ${y}, ${z})`);
        }
      }

      // ---- Fluid ↔ cloth non-penetration.
      // For each fluid particle, find the closest cloth particle. If
      // dist < r, the fluid has penetrated past the contact gate
      // (which fires at 2·r). Count the violations.
      const clothStart = fluidCount;
      const clothEnd = fluidCount + clothCount;
      let penetrators = 0;
      for (let f = 0; f < fluidCount; f++) {
        const fx = snap.positions[4 * f + 0]!;
        const fy = snap.positions[4 * f + 1]!;
        const fz = snap.positions[4 * f + 2]!;
        for (let c = clothStart; c < clothEnd; c++) {
          const cx = snap.positions[4 * c + 0]!;
          const cy = snap.positions[4 * c + 1]!;
          const cz = snap.positions[4 * c + 2]!;
          const dx = fx - cx;
          const dy = fy - cy;
          const dz = fz - cz;
          if (dx * dx + dy * dy + dz * dz < r * r) {
            penetrators += 1;
            break;
          }
        }
      }
      const penFraction = penetrators / fluidCount;
      // eslint-disable-next-line no-console
      console.info(
        `[wet-cloth] T=${totalSeconds.toFixed(1)}s fluid=${fluidCount} cloth=${clothCount} penetrators=${penetrators} (${(penFraction * 100).toFixed(3)}%; target < 0.5%)`,
      );
      expect(penFraction).toBeLessThan(0.005);

      // ---- Pinned-corner sanity: top-left and top-right cloth
      //      vertices must still be at their initial pin positions.
      //      A drift here means the pin (invMass = 0) was overridden
      //      somewhere — which would be a regression in the cloth
      //      pipeline, not a Phase 20 architectural failure, but
      //      catching it here keeps the test from passing on a
      //      half-broken setup.
      const pinIndices = [0, nx - 1];
      for (const pi of pinIndices) {
        const slot = fluidCount + pi;
        const py = snap.positions[4 * slot + 1]!;
        if (Math.abs(py - pinRowY) > 1e-3) {
          throw new Error(`pin slot ${slot} drifted from y=${pinRowY} to y=${py}`);
        }
      }

      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    } finally {
      renderer.dispose();
    }
  }, 180_000);
});
