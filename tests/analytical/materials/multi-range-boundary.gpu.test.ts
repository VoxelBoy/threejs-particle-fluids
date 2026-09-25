import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
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
 * Phase 17 — multi-range dynamic boundary-volume coupling (G1).
 *
 * The Phase-17-novel architectural claim: one `FluidSystem` may hold
 * MULTIPLE boundary-particle ranges from MULTIPLE module systems
 * (softbody + rigid + fluid container), each contributing its own
 * per-substep dynamic boundary-volume kernel to the shared
 * `preIterKernels` pipeline, without any cross-talk between ranges.
 *
 *
 *   - Register one softbody surfaceRange (range A).
 *   - Register one rigid surface sub-range (range B).
 *   - Run one substep.
 *   - Assert:
 *       (a) every slot in range A has non-zero `boundaryVolume` (its
 *           kernel fired and computed `V_i = 1 / Σ_k W_ik` per Akinci
 *           2012 eq. 4).
 *       (b) every slot in range B has non-zero `boundaryVolume`.
 *       (c) every slot OUTSIDE both ranges (fluid slots, slot gaps
 *           between bodies, slots beyond either body) has
 *           `boundaryVolume = 0` — i.e. neither kernel wrote outside
 *           its own range.
 *
 * Failure modes this catches:
 *   - Kernel scoping bug: the second registration writes to slots
 *     [start..start+capacity) instead of [start..start+count), trampling
 *     fluid `boundaryVolume = 0` invariant.
 *   - Both kernels share state (e.g. a single shared register accidentally
 *     keyed by particleIndex without a range gate).
 *   - SimLoop snapshots only the FIRST registration's kernel; the second
 *     registration's per-substep kernel never fires.
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

describe('Phase 17 — multi-range dynamic boundary-volume (G1)', () => {
  it('two boundary registrations on one FluidSystem each update their own slot range without cross-contamination', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.025;
      const spacing = 2 * r;
      const h = 4 * r;
      const restDensity = 1000;

      // Voxelize one cube for softbody, one with SDF for rigid.
      // Tiny edge to keep the scene ~few hundred particles.
      const edge = 4 * spacing;
      const meshScaled: TriangleMesh = {
        vertices: unitCubeMesh().vertices.map((v) => v * edge),
        indices: unitCubeMesh().indices,
      };
      const softVox = voxelize(meshScaled, {
        particleRadius: r,
        spacingFactor: 1.0,
      });
      const rigidVox = voxelize(meshScaled, {
        particleRadius: r,
        bakeSdf: true,
        spacingFactor: 1.0,
      });

      // Tiny fluid blob — 3×3×3 = 27 particles. Just enough to
      // exercise FluidSystem construction; placement is deliberately
      // far from both bodies so no inter-particle pair fires.
      const fluidCount = 27;
      const fluidPositions: [number, number, number][] = [];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          for (let k = 0; k < 3; k++) {
            fluidPositions.push([-2.0 + i * spacing, 0.05 + j * spacing, -2.0 + k * spacing]);
          }
        }
      }

      // Body placements — both centred at the origin in y, separated
      // along x by 2×edge so neither contributes to the other's
      // density sum (they're outside each other's `h`).
      const softCount = softVox.count;
      const rigidCount = rigidVox.count;

      const totalCount = fluidCount + softCount + rigidCount;
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
      const softOffsetX = 0;
      for (let i = 0; i < softCount; i++) {
        const lx = softVox.positions[3 * i + 0]!;
        const ly = softVox.positions[3 * i + 1]!;
        const lz = softVox.positions[3 * i + 2]!;
        initial[softStart + i] = {
          position: [lx + softOffsetX, ly + 0.5, lz],
          velocity: [0, 0, 0],
          invMass: 100,
          phase: (2 << 16) >>> 0,
        };
      }
      const rigidStart = softStart + softCount;
      const rigidOffsetX = 1.0;
      for (let i = 0; i < rigidCount; i++) {
        const lx = rigidVox.positions[3 * i + 0]!;
        const ly = rigidVox.positions[3 * i + 1]!;
        const lz = rigidVox.positions[3 * i + 2]!;
        initial[rigidStart + i] = {
          position: [lx + rigidOffsetX, ly + 0.5, lz],
          velocity: [0, 0, 0],
          invMass: 50,
          phase: (3 << 16) >>> 0,
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
        xsph: { c: 0 },
      });

      const restSoftFlat = new Float32Array(softCount * 3);
      for (let i = 0; i < softCount * 3; i++) {
        restSoftFlat[i] = softVox.positions[i]!;
      }
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

      const restRigidFlat = new Float32Array(rigidCount * 3);
      for (let i = 0; i < rigidCount * 3; i++) {
        restRigidFlat[i] = rigidVox.positions[i]!;
      }
      const rigidBodies: RigidBodyDef[] = [
        {
          particleRange: { start: rigidStart, count: rigidCount },
          restPositions: restRigidFlat,
          restSDF: rigidVox.restSDF!.slice(),
          phaseId: 3,
          compliance: 0,
        },
      ];
      const rigid = new RigidBodySystem({
        particles,
        xpbd,
        bodies: rigidBodies,
      });

      // Two boundary registrations, both default `{dynamic: true}`.
      // These MUST run before SimLoop construction — SimLoop snapshots
      // FluidSystem.preIterKernels at construction, and the dynamic
      // boundary-volume kernels are appended on registration.
      const softRange = softbody.surfaceRange(0);
      const rigidSurfaceRange = {
        start: rigidStart,
        count: rigidVox.surfaceCount,
      };
      await fluid.registerBoundaryParticles(softRange);
      await fluid.registerBoundaryParticles(rigidSurfaceRange);

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        xpbd,
        hashGrid,
        contact: { hashGrid, maxContacts: 1024 },
        materials: [fluid, softbody, rigid],
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, 0, 0);

      // Run one substep so the per-substep dynamic boundary-volume
      // kernels fire (the registration seed runs on COMMITTED
      // positions; the per-substep kernel runs on PREDICTED
      // positions). With gravity off + zero velocity, predicted ==
      // committed, so V_i is the same value either way. What matters
      // here is whether each kernel ran and wrote into its own range.
      await loop.step(1 / 60);

      const bvBuf = await renderer.getArrayBufferAsync(particles.boundaryVolume.value);
      const bv = new Float32Array(bvBuf);

      // Range A — softbody surface particles. Every slot must be
      // strictly positive (the V_i = 1 / Σ W formula gives a finite
      // positive number for any non-empty same-range neighbour set;
      // a cube with 4³ samples has every surface particle within
      // h = 4r of at least its 4-neighbour ring).
      let maxA = -Infinity;
      let minA = Infinity;
      for (let i = softRange.start; i < softRange.start + softRange.count; i++) {
        const v = bv[i]!;
        if (v < minA) minA = v;
        if (v > maxA) maxA = v;
      }
      expect(minA).toBeGreaterThan(0);

      // Range B — rigid surface particles. Same expectation.
      let maxB = -Infinity;
      let minB = Infinity;
      for (
        let i = rigidSurfaceRange.start;
        i < rigidSurfaceRange.start + rigidSurfaceRange.count;
        i++
      ) {
        const v = bv[i]!;
        if (v < minB) minB = v;
        if (v > maxB) maxB = v;
      }
      expect(minB).toBeGreaterThan(0);

      // Outside both ranges — must remain at the buffer's
      // post-construction value (0). Slots checked:
      //   - fluid slots [0, fluidCount): 0
      //   - softbody INTERIOR slots (post-surfaceCount inside the
      //     softbody body's range): 0
      //   - rigid INTERIOR slots (post-surfaceCount inside the rigid
      //     body's range): 0
      //   - slots between softbody.end and rigid.start (gap): n/a
      //     (none in this layout — they're contiguous).
      // The interior-vs-surface slot ordering is enforced by
      // voxelize() which puts surface particles first.

      // Fluid range — must be 0.
      for (let i = 0; i < fluidCount; i++) {
        expect(bv[i]!).toBe(0);
      }

      // Softbody interior — slots [softStart + surfaceCount, softStart + count).
      for (let i = softStart + softVox.surfaceCount; i < softStart + softCount; i++) {
        expect(bv[i]!).toBe(0);
      }

      // Rigid interior — slots [rigidStart + surfaceCount, rigidStart + count).
      for (let i = rigidStart + rigidVox.surfaceCount; i < rigidStart + rigidCount; i++) {
        expect(bv[i]!).toBe(0);
      }

      // eslint-disable-next-line no-console
      console.info(
        `[multi-range-boundary] softRange=[${softRange.start}, +${softRange.count})  V_i in [${minA.toExponential(3)}, ${maxA.toExponential(3)}]`,
      );
      // eslint-disable-next-line no-console
      console.info(
        `[multi-range-boundary] rigidRange=[${rigidSurfaceRange.start}, +${rigidSurfaceRange.count})  V_i in [${minB.toExponential(3)}, ${maxB.toExponential(3)}]`,
      );

      particles.destroy();
      hashGrid.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
