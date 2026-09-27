import { describe, expect, it } from 'vitest';
import {
  FluidSystem,
  ParticleSystem,
  SimLoop,
  SoftbodySystem,
  createParticleRenderer,
  voxelize,
  type ParticleInit,
  type TriangleMesh,
} from '../../../src/index.js';

/*
 * Multiple dynamic boundary ranges on one fluid.
 *
 * One `FluidSystem` may hold several boundary-particle ranges owned by
 * different materials (here a soft body and a stiff, rigid-like body), each
 * with its own per-substep boundary-volume kernel, without any cross-talk
 * between ranges.
 *
 *   - Register the soft body's surface range as a boundary (range A).
 *   - Register the stiff body's surface range as a boundary (range B).
 *   - Run one substep.
 *   - Assert:
 *       (a) every slot in range A has a non-zero `boundaryVolume` (its
 *           kernel ran and computed `V_i = 1 / Σ_k W_ik`, Akinci 2012
 *           eq. 4).
 *       (b) every slot in range B has a non-zero `boundaryVolume`.
 *       (c) every slot OUTSIDE both ranges (fluid slots, and the interior
 *           slots of either body) has `boundaryVolume = 0` — neither
 *           kernel wrote outside its own range.
 *
 * Failure modes this catches:
 *   - Kernel scoping bug: a registration writes to slots
 *     [start, start + capacity) instead of [start, start + count),
 *     trampling the fluid's `boundaryVolume = 0` invariant.
 *   - Both kernels share state (e.g. a single shared buffer keyed by
 *     particle index without a range gate).
 *   - Only the FIRST registration's kernel is scheduled; the second
 *     registration's per-substep kernel never runs.
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

describe('materials: multiple dynamic boundary ranges on one fluid', () => {
  it('two boundary registrations on one FluidSystem each update their own slot range without cross-contamination', async () => {
    const renderer = await createParticleRenderer();
    try {
      const r = 0.025;
      const spacing = 2 * r;
      const h = 4 * r;
      const restDensity = 1000;

      // One small cube, voxelized once and used for both bodies. A tiny edge
      // keeps the scene at a few hundred particles.
      const edge = 4 * spacing;
      const cube: TriangleMesh = {
        vertices: unitCubeMesh().vertices.map((v) => v * edge),
        indices: unitCubeMesh().indices,
      };
      const vox = voxelize(cube, { particleRadius: r });

      // Tiny fluid blob — 3×3×3 = 27 particles. Just enough to exercise
      // FluidSystem construction; placed far from both bodies so no
      // fluid–body pair interacts.
      const fluidCount = 27;
      const fluidPositions: [number, number, number][] = [];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          for (let k = 0; k < 3; k++) {
            fluidPositions.push([-2.0 + i * spacing, 0.05 + j * spacing, -2.0 + k * spacing]);
          }
        }
      }

      // Body placements — both at the same height, separated along x by far
      // more than `h`, so neither contributes to the other's density sum.
      const bodyCount = vox.count;
      const totalCount = fluidCount + 2 * bodyCount;
      const initial: ParticleInit[] = new Array(totalCount);

      for (let i = 0; i < fluidCount; i++) {
        const p = fluidPositions[i]!;
        initial[i] = { position: [p[0], p[1], p[2]], velocity: [0, 0, 0], invMass: 1 };
      }
      const softStart = fluidCount;
      const softOffsetX = 0;
      for (let i = 0; i < bodyCount; i++) {
        const lx = vox.positions[3 * i + 0]!;
        const ly = vox.positions[3 * i + 1]!;
        const lz = vox.positions[3 * i + 2]!;
        initial[softStart + i] = {
          position: [lx + softOffsetX, ly + 0.5, lz],
          velocity: [0, 0, 0],
          invMass: 100,
        };
      }
      const stiffStart = softStart + bodyCount;
      const stiffOffsetX = 1.0;
      for (let i = 0; i < bodyCount; i++) {
        const lx = vox.positions[3 * i + 0]!;
        const ly = vox.positions[3 * i + 1]!;
        const lz = vox.positions[3 * i + 2]!;
        initial[stiffStart + i] = {
          position: [lx + stiffOffsetX, ly + 0.5, lz],
          velocity: [0, 0, 0],
          invMass: 50,
        };
      }

      const particles = new ParticleSystem(renderer, totalCount, r);
      particles.uploadParticles(initial);

      const fluid = new FluidSystem(particles, {
        range: { start: 0, count: fluidCount },
        restDensity,
        particleSpacing: spacing,
        smoothingRadius: h,
        compliance: 1e-4,
      });

      // Rest shapes default to the uploaded positions.
      const softbody = new SoftbodySystem(particles, {
        bodies: [
          {
            range: { start: softStart, count: bodyCount },
            surfaceCount: vox.surfaceCount,
            compliance: 1e-7,
          },
        ],
      });
      // A rigid-like body: global shape matching with zero compliance.
      const stiff = new SoftbodySystem(particles, {
        bodies: [
          {
            range: { start: stiffStart, count: bodyCount },
            surfaceCount: vox.surfaceCount,
            compliance: 0,
          },
        ],
      });

      // Two boundary registrations, both dynamic (the default). They must be
      // added before the SimLoop is created, which builds the fluid's
      // kernels, one boundary-volume kernel per registration.
      const softRange = softbody.surfaceRange(0);
      const stiffRange = stiff.surfaceRange(0);
      fluid.addBoundary(softRange);
      fluid.addBoundary(stiffRange);

      const loop = new SimLoop(particles, {
        substeps: 1,
        iterations: 1,
        contact: { maxContacts: 1024 },
        materials: [fluid, softbody, stiff],
      });
      loop.gravity.set(0, 0, 0);

      // Run one substep so the per-substep boundary-volume kernels run.
      // They read PREDICTED positions; with gravity off and zero velocity,
      // predicted == committed, so V_i is the same either way. What matters
      // here is whether each kernel ran and wrote into its own range.
      await loop.step(1 / 60);

      const bvBuf = await renderer.getArrayBufferAsync(particles.boundaryVolume.value);
      const bv = new Float32Array(bvBuf);

      // Range A — soft body surface particles. Every slot must be strictly
      // positive: V_i = 1 / Σ W is finite and positive for any non-empty
      // same-range neighbor set, and on a 4³ cube every surface particle
      // has neighbors within h = 4r.
      let maxA = -Infinity;
      let minA = Infinity;
      for (let i = softRange.start; i < softRange.start + softRange.count; i++) {
        const v = bv[i]!;
        if (v < minA) minA = v;
        if (v > maxA) maxA = v;
      }
      expect(minA).toBeGreaterThan(0);

      // Range B — stiff body surface particles. Same expectation.
      let maxB = -Infinity;
      let minB = Infinity;
      for (let i = stiffRange.start; i < stiffRange.start + stiffRange.count; i++) {
        const v = bv[i]!;
        if (v < minB) minB = v;
        if (v > maxB) maxB = v;
      }
      expect(minB).toBeGreaterThan(0);

      // Outside both ranges, slots keep their initial value (0):
      //   - fluid slots [0, fluidCount)
      //   - each body's INTERIOR slots (after its surfaceCount)
      // There is no gap between the bodies in this layout. Interior slots
      // follow the surface ones because voxelize() puts surface particles
      // first.

      // Fluid range — must be 0.
      for (let i = 0; i < fluidCount; i++) {
        expect(bv[i]!).toBe(0);
      }

      // Soft body interior — slots [softStart + surfaceCount, softStart + count).
      for (let i = softStart + vox.surfaceCount; i < softStart + bodyCount; i++) {
        expect(bv[i]!).toBe(0);
      }

      // Stiff body interior — slots [stiffStart + surfaceCount, stiffStart + count).
      for (let i = stiffStart + vox.surfaceCount; i < stiffStart + bodyCount; i++) {
        expect(bv[i]!).toBe(0);
      }

      console.info(
        `[multi-range-boundary] softRange=[${softRange.start}, +${softRange.count})  V_i in [${minA.toExponential(3)}, ${maxA.toExponential(3)}]`,
      );
      console.info(
        `[multi-range-boundary] stiffRange=[${stiffRange.start}, +${stiffRange.count})  V_i in [${minB.toExponential(3)}, ${maxB.toExponential(3)}]`,
      );

      loop.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
