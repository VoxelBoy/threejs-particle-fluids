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
import {
  RigidBodySystem,
  voxelize,
  type RigidBodyDef,
  type TriangleMesh,
} from '../../../src/softbody/index.js';

// Phase 15b G1 — top-of-stack lateral runaway regression detector.
//
// Locks the symptom Macklin 2014 §5.2 ¶ "perform mass modification only
// in the final solver iteration" warns about. Pre-fix, `RigidBodySystem`
// dispatched the §5.2 mass-scaling kernel from `preIterKernels` so
// every iter saw the scaled `contactInvMass`; with k=3 on an 8-cube
// × 0.4 m stack (mass ratio ~4500× across the height range) the per-
// iter asymmetric position corrections at height boundaries accumulated
// without shape-match's once-per-substep R having a chance to absorb
// them, and the top cube ran away laterally — measured |ω| ≈ 5.6 rad/s,
// |v_xz| ≈ 1.5 m/s, |Δcom_xz| ≈ 1.4 m at t=4 s.
//
// Phase 15b moves the §5.2 dispatch to `Material.lastIterPreContactKernels`,
// matching the paper's mitigation text exactly. Post-fix the runaway is
// gone: |ω| ≈ 0.12 rad/s, |v_xz| ≈ 0.19 m/s, |Δcom_xz| ≈ 0.12 m at the
// same parameters.
//
// The test measures three signals at t=4 s:
//   - Top-body angular speed |ω|, derived from a rotation-matrix finite
//     difference between two consecutive frames (exact for rigid bodies
//     up to f32 noise).
//   - Top-body lateral COM velocity |v_xz|, from the particle-velocity
//     readback averaged across the body's particles.
//   - Top-body lateral COM displacement |Δcom_xz| from origin.
//
// Thresholds are calibrated 2.5–3× above the post-fix measurement to
// give cross-run variance margin while keeping the regression band
// well below the pre-fix runaway magnitude.

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

/**
 * Extract |ω| from two row-major 3×3 rotation matrices captured one
 * frame apart at frame interval `dt`. Uses Rodrigues' formula in reverse:
 * `ΔR = R₂ · R₁ᵀ`, then `trace(ΔR) = 1 + 2 cos(θ)` so `θ = arccos((tr−1)/2)`,
 * and `|ω| = θ / dt`. Robust for the small rotations expected in a
 * settled stack (clamps the trace argument to [-1, 1] for f32 noise).
 */
function angularSpeedFromRotations(
  R1: readonly number[],
  R2: readonly number[],
  dt: number,
): number {
  // ΔR = R2 · R1ᵀ — row-major 3×3 multiplied by row-major-transpose.
  let traceDR = 0;
  for (let i = 0; i < 3; i++) {
    // (R2 · R1ᵀ)[i][i] = Σ_k R2[i][k] · R1ᵀ[k][i] = Σ_k R2[i][k] · R1[i][k]
    for (let k = 0; k < 3; k++) {
      traceDR += R2[3 * i + k]! * R1[3 * i + k]!;
    }
  }
  const cosTheta = Math.max(-1, Math.min(1, (traceDR - 1) / 2));
  const theta = Math.acos(cosTheta);
  return theta / dt;
}

/**
 * Read body `b`'s rotation as a row-major 3×3 from the
 * `bodyRotations` storage buffer. Layout matches
 * `RigidBodySystem.bodyRotations = instancedArray(bodies * 3, 'vec4')`:
 * three vec4 rows per body, with the fourth lane padding.
 */
function readBodyRotation(rotBuf: Float32Array, bodyIndex: number): number[] {
  const base = bodyIndex * 3 * 4;
  return [
    rotBuf[base + 0]!,
    rotBuf[base + 1]!,
    rotBuf[base + 2]!,
    rotBuf[base + 4]!,
    rotBuf[base + 5]!,
    rotBuf[base + 6]!,
    rotBuf[base + 8]!,
    rotBuf[base + 9]!,
    rotBuf[base + 10]!,
  ];
}

describe('Phase 15b — top-of-stack runaway (G1)', () => {
  it('top body of an 8-cube §5.2-on stack stays bounded — no lateral runaway', async () => {
    const renderer = await createParticleRenderer();
    try {
      const bodyCount = 8;
      const edgeLength = 0.4;
      const particleRadius = 0.04;
      const spacingFactor = 0.8;
      const stackK = 3;
      const dropHeight = 0;

      const mesh = unitCubeMesh();
      const scaled: TriangleMesh = {
        vertices: mesh.vertices.map((v) => v * edgeLength),
        indices: mesh.indices,
      };
      const cube = voxelize(scaled, {
        particleRadius,
        bakeSdf: true,
        spacingFactor,
      });
      const perBodyCount = cube.count;
      const totalCount = perBodyCount * bodyCount;

      let minLocalY = Infinity;
      let maxLocalY = -Infinity;
      for (let i = 0; i < perBodyCount; i++) {
        const ly = cube.positions[3 * i + 1]!;
        if (ly < minLocalY) minLocalY = ly;
        if (ly > maxLocalY) maxLocalY = ly;
      }
      const stackSpacing = maxLocalY - minLocalY + 2 * particleRadius;

      // Deterministic per-cube x/z jitter — matches the demo's pattern of
      // breaking perfect axis-edge alignment so the §5.1 SDF kernel sees
      // off-axis surface particles.
      let lcg = 1;
      const rng = (): number => {
        lcg = (lcg * 1664525 + 1013904223) | 0;
        return (lcg >>> 0) / 0x100000000;
      };

      const initial: ParticleInit[] = [];
      const restPerBody: Float32Array[] = [];
      const ranges: { start: number; count: number }[] = [];

      for (let b = 0; b < bodyCount; b++) {
        const cy = dropHeight + particleRadius - minLocalY + b * stackSpacing;
        const cx = (rng() - 0.5) * particleRadius * 0.5;
        const cz = (rng() - 0.5) * particleRadius * 0.5;

        const rest = new Float32Array(3 * perBodyCount);
        const start = b * perBodyCount;
        ranges.push({ start, count: perBodyCount });
        for (let i = 0; i < perBodyCount; i++) {
          const lx = cube.positions[3 * i + 0]!;
          const ly = cube.positions[3 * i + 1]!;
          const lz = cube.positions[3 * i + 2]!;
          rest[3 * i + 0] = lx;
          rest[3 * i + 1] = ly;
          rest[3 * i + 2] = lz;
          initial.push({
            position: [lx + cx, ly + cy, lz + cz],
            velocity: [0, 0, 0],
            invMass: 1,
            phase: ((b + 1) & 0xffff) << 16,
          });
        }
        restPerBody.push(rest);
      }

      const particles = new ParticleSystem(renderer, totalCount, particleRadius);
      particles.uploadParticles(initial);

      const xpbd = createXpbdUniforms(1 / 60);
      const hashGrid = new HashGrid(particles, {
        cellSize: particleRadius * 2.1,
      });

      const bodies: RigidBodyDef[] = ranges.map((range, b) => ({
        particleRange: range,
        restPositions: restPerBody[b]!,
        restSDF: cube.restSDF!,
        phaseId: b + 1,
        compliance: 0,
      }));

      const rigid = new RigidBodySystem({
        particles,
        xpbd,
        bodies,
        stackStabilization: { k: stackK, groundY: 0 },
      });

      const colliders = new PrimitiveSet(particles, { capacity: 1 });
      colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0), {
        muS: 0.6,
        muK: 0.4,
      });
      colliders.upload();

      const loop = new SimLoop(particles, {
        substeps: 4,
        iterations: 2,
        xpbd,
        hashGrid,
        colliders: { colliders },
        materials: [rigid],
        contact: {
          hashGrid,
          maxContacts: Math.max(8192, totalCount * 6),
          friction: { muS: 0.6, muK: 0.4 },
          stabIters: 1,
        },
      });
      loop.kernels.floorY.value = -1e9;
      loop.gravity.set(0, -9.8, 0);

      const frameDt = 1 / 60;
      const settleFrames = 240; // 4 s of integration.

      for (let f = 0; f < settleFrames; f++) {
        await loop.step(frameDt);
      }

      // Capture rotation R₁ at t=4 s.
      const rotBuf1 = new Float32Array(
        await renderer.getArrayBufferAsync(rigid.bodyRotations.value),
      );
      // Capture velocities + COM at t=4 s.
      const snap = await particles.readback();

      // Step one more frame to get R₂.
      await loop.step(frameDt);
      const rotBuf2 = new Float32Array(
        await renderer.getArrayBufferAsync(rigid.bodyRotations.value),
      );

      const topBody = bodyCount - 1;
      const topRange = ranges[topBody]!;

      const R1 = readBodyRotation(rotBuf1, topBody);
      const R2 = readBodyRotation(rotBuf2, topBody);
      const omegaMag = angularSpeedFromRotations(R1, R2, frameDt);

      // Top-body lateral COM speed from particle-velocity readback.
      let vx = 0,
        vz = 0;
      for (let i = 0; i < topRange.count; i++) {
        const slot = topRange.start + i;
        vx += snap.velocities[4 * slot + 0]!;
        vz += snap.velocities[4 * slot + 2]!;
      }
      const invN = 1 / topRange.count;
      vx *= invN;
      vz *= invN;
      const vXZ = Math.sqrt(vx * vx + vz * vz);

      // Lateral COM displacement from the spawn x/z (regression signal —
      // a settled stack has the top body within tens of micrometres of
      // its column position).
      let comX = 0,
        comZ = 0;
      for (let i = 0; i < topRange.count; i++) {
        const slot = topRange.start + i;
        comX += snap.positions[4 * slot + 0]!;
        comZ += snap.positions[4 * slot + 2]!;
      }
      comX *= invN;
      comZ *= invN;
      const comDispXZ = Math.sqrt(comX * comX + comZ * comZ);

      // eslint-disable-next-line no-console
      console.info(
        `[rigid-stack-jitter] T=${(settleFrames * frameDt).toFixed(2)}s |ω|=${omegaMag.toFixed(4)} rad/s |v_xz|=${vXZ.toExponential(3)} m/s |Δcom_xz|=${comDispXZ.toExponential(3)} m`,
      );

      // Bounded-motion thresholds. Post-fix calibration on this scene
      // measured |ω| ≈ 0.12 rad/s, |v_xz| ≈ 0.19 m/s,
      // |Δcom_xz| ≈ 0.12 m at t=4 s. The bounds below sit ≈3× above
      // those measurements — tight enough to fail the pre-fix runaway
      // (which produced |v_xz| ≈ 1.5 m/s, |Δcom_xz| ≈ 1.4 m) and loose
      // enough for cross-run f32-atomic order variance.
      //
      // Note: these are NOT zero-motion thresholds. An 8-cube PBD stack
      // at substeps=4, iters=2 retains some bounded oscillation (paper
      // §5.2 Figure 10 shows residual oscillation even with mass
      // scaling). The assertions encode "no lateral runaway", not
      // "perfectly stationary."
      expect(omegaMag).toBeLessThan(0.5);
      expect(vXZ).toBeLessThan(0.5);
      expect(comDispXZ).toBeLessThan(0.4);

      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    } finally {
      renderer.dispose();
    }
  }, 90_000);
});
