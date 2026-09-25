import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute } from 'three';

import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { ClothSystem, fromBufferGeometry } from '../../../src/cloth/index.js';

// Phase 18 G1 — bending rest test (plan §"Validation/Automatic G1").
//
// Build a four-vertex two-triangle mesh folded at 90° around the
// shared edge. Pin the two edge endpoints (so only the far vertices
// are free to move). Run with zero gravity, no perturbation. The
// dihedral angle should remain at the rest value (90°) within 2°
// after a long settle — this tests that the bending kernel correctly
// holds the rest angle and isn't injecting energy.
//
// We also verify a SECOND case: same geometry but folded at 0° (flat
// rest), perturb the far vertex slightly out of plane, run zero-grav,
// and assert the cloth returns to flat. The bending kernel must do
// the right thing in BOTH directions of perturbation, which is the
// gate against the signed-θ sign-flip bug caught at Phase 18 entry.

interface FoldedQuad {
  readonly geometry: BufferGeometry;
  /** Indices of the four particles in the constructed graph. */
  readonly p1: number;
  readonly p2: number;
  readonly p3: number;
  readonly p4: number;
}

/**
 * Build a four-vertex two-triangle mesh with the chosen dihedral.
 * Triangle 1 lies in the xy plane; triangle 2 rotates around the
 * shared edge by `dihedralRad`. Returns the BufferGeometry plus the
 * particle indices that the graph builder assigns to (p1, p2, p3, p4).
 *
 * Layout:
 *   - shared edge along x from (0,0,0) to (1,0,0)
 *   - triangle 1 (in xy): far vertex at (0.5, 1, 0)
 *   - triangle 2 (rotated): far vertex at (0.5, cos(α), sin(α))
 *     where α = π − dihedralRad ∈ (0, π) — sets the dihedral angle
 *     between the triangles' outward normals.
 */
function buildFoldedQuad(dihedralRad: number): FoldedQuad {
  const cos = Math.cos(Math.PI - dihedralRad);
  const sin = Math.sin(Math.PI - dihedralRad);
  const geom = new BufferGeometry();
  // prettier-ignore
  geom.setAttribute(
    'position',
    new Float32BufferAttribute(
      [
        0, 0, 0,           // 0 — shared edge endpoint
        1, 0, 0,           // 1 — shared edge endpoint
        0.5, 1, 0,         // 2 — far vertex of triangle 1 (xy plane)
        0.5, cos, sin,     // 3 — far vertex of triangle 2 (rotated)
      ],
      3,
    ),
  );
  // Triangle 1: (0, 1, 2) — vertices CCW in xy.
  // Triangle 2: (0, 3, 1) — far vertex 3 between edge endpoints.
  geom.setIndex(new Uint32BufferAttribute([0, 1, 2, 0, 3, 1], 1));
  // The graph builder records p1=min, p2=max for the shared edge,
  // and p3, p4 = far vertices in the order their triangles enumerate.
  // For edge (0, 1): p1=0, p2=1, p3 = far of triangle 1 = 2,
  // p4 = far of triangle 2 = 3.
  return { geometry: geom, p1: 0, p2: 1, p3: 2, p4: 3 };
}

function dihedralAngleSigned(
  pos: readonly [number, number, number][],
  i1: number,
  i2: number,
  i3: number,
  i4: number,
): number {
  const p1 = pos[i1]!;
  const p2 = pos[i2]!;
  const p3 = pos[i3]!;
  const p4 = pos[i4]!;
  const ex = p2[0] - p1[0];
  const ey = p2[1] - p1[1];
  const ez = p2[2] - p1[2];
  const eLen = Math.sqrt(ex * ex + ey * ey + ez * ez);
  const a3x = p3[0] - p1[0],
    a3y = p3[1] - p1[1],
    a3z = p3[2] - p1[2];
  const b3x = p3[0] - p2[0],
    b3y = p3[1] - p2[1],
    b3z = p3[2] - p2[2];
  const a4x = p4[0] - p2[0],
    a4y = p4[1] - p2[1],
    a4z = p4[2] - p2[2];
  const b4x = p4[0] - p1[0],
    b4y = p4[1] - p1[1],
    b4z = p4[2] - p1[2];
  const n1x = a3y * b3z - a3z * b3y;
  const n1y = a3z * b3x - a3x * b3z;
  const n1z = a3x * b3y - a3y * b3x;
  const n2x = a4y * b4z - a4z * b4y;
  const n2y = a4z * b4x - a4x * b4z;
  const n2z = a4x * b4y - a4y * b4x;
  const cx = n1y * n2z - n1z * n2y;
  const cy = n1z * n2x - n1x * n2z;
  const cz = n1x * n2y - n1y * n2x;
  const sinTimes = cx * ex + cy * ey + cz * ez;
  const cosTimes = (n1x * n2x + n1y * n2y + n1z * n2z) * eLen;
  return Math.atan2(sinTimes, cosTimes);
}

async function runRestTest(args: {
  readonly initialDihedralRad: number;
  readonly perturbedDihedralRad?: number;
  readonly frames: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly velocityDamp?: number;
}): Promise<{
  readonly initialAngle: number;
  readonly finalAngle: number;
  readonly maxKineticEnergy: number;
  readonly nanFree: boolean;
}> {
  const renderer = await createParticleRenderer();
  try {
    const { geometry, p1, p2, p3, p4 } = buildFoldedQuad(
      args.perturbedDihedralRad ?? args.initialDihedralRad,
    );
    const graph = fromBufferGeometry(geometry, {
      surfaceDensity: 0.2,
      // Pin the edge endpoints — only far vertices move.
      pinnedIndices: [p1, p2],
    });

    // Override the rest angle with the *target* rest, so that an initial
    // perturbation gives a non-zero C and the constraint pulls toward
    // the target rest. We mutate the readonly array via the graph object
    // (a defensive cast — read-only-ness is a TS hint, not a runtime
    // lock; in production code, build a separate ClothGraph from the
    // target geometry instead).
    if (args.perturbedDihedralRad !== undefined) {
      const restTarget = dihedralAngleSigned(
        [
          [0, 0, 0],
          [1, 0, 0],
          [0.5, 1, 0],
          [
            0.5,
            Math.cos(Math.PI - args.initialDihedralRad),
            Math.sin(Math.PI - args.initialDihedralRad),
          ],
        ],
        p1,
        p2,
        p3,
        p4,
      );
      (graph.bendingRestAngles as number[])[0] = restTarget;
    }

    const initial: ParticleInit[] = [];
    for (let i = 0; i < graph.positions.length; i++) {
      const pos = graph.positions[i]!;
      initial.push({
        position: [pos[0], pos[1], pos[2]],
        velocity: [0, 0, 0],
        invMass: graph.invMass[i]!,
        phase: 1,
      });
    }
    const particles = new ParticleSystem(renderer, graph.positions.length, 0.05);
    particles.uploadParticles(initial);

    const xpbd = createXpbdUniforms(1 / 60);
    const cloth = new ClothSystem({
      particles,
      xpbd,
      graph,
      particleOffset: 0,
      stretchCompliance: 1e-7,
      bendCompliance: 1e-5,
    });

    const loop = new SimLoop(particles, {
      substeps: args.substeps,
      iterations: args.iterations,
      xpbd,
      materials: [cloth],
    });
    loop.kernels.floorY.value = -1e9; // no floor
    loop.gravity.set(0, 0, 0); // zero-gravity rest test

    // GPU buffers aren't allocated until the first compute dispatch
    // (three.js StorageBufferAttribute lazy-allocates on first kernel
    // touch). Sample the initial angle from the CPU-side graph
    // positions instead — they are byte-identical to what we just
    // uploaded.
    const initialPos: [number, number, number][] = graph.positions.map((p) => [p[0], p[1], p[2]]);
    const initialAngle = dihedralAngleSigned(initialPos, p1, p2, p3, p4);

    let maxKE = 0;
    let nanFree = true;
    const frameDt = 1 / 60;
    const damp = args.velocityDamp ?? 1.0;
    for (let f = 0; f < args.frames; f++) {
      await loop.step(frameDt);
      // Sample KE every 10 frames to keep the readback overhead bounded.
      if (f % 10 === 0 || f === args.frames - 1) {
        const snap = await particles.readback();
        let frameKE = 0;
        for (let i = 0; i < graph.positions.length; i++) {
          const w = graph.invMass[i]!;
          if (w === 0) continue;
          const m = 1 / w;
          const vx = snap.velocities[4 * i + 0]!;
          const vy = snap.velocities[4 * i + 1]!;
          const vz = snap.velocities[4 * i + 2]!;
          const v2 = vx * vx + vy * vy + vz * vz;
          if (!Number.isFinite(v2)) {
            nanFree = false;
            break;
          }
          frameKE += 0.5 * m * v2;
        }
        if (frameKE > maxKE) maxKE = frameKE;
        if (!nanFree) break;
        // Optional CPU-side velocity damping: read, scale, re-upload.
        // Used by both rest tests so the cloth actually settles to the
        // rest angle within the test budget instead of oscillating
        // around it indefinitely.
        if (damp < 1.0 && nanFree) {
          for (let i = 0; i < graph.positions.length * 4; i++) {
            snap.velocities[i] = snap.velocities[i]! * damp;
          }
          (particles.velocities.value.array as Float32Array).set(snap.velocities);
          particles.velocities.value.needsUpdate = true;
        }
      }
    }
    const finalSnap = await particles.readback();
    const finalPos = readbackPositions(finalSnap.positions, 4);
    const finalAngle = dihedralAngleSigned(finalPos, p1, p2, p3, p4);

    particles.destroy();
    return { initialAngle, finalAngle, maxKineticEnergy: maxKE, nanFree };
  } finally {
    renderer.dispose();
  }
}

function readbackPositions(flat: Float32Array, count: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let i = 0; i < count; i++) {
    out.push([flat[4 * i + 0]!, flat[4 * i + 1]!, flat[4 * i + 2]!]);
  }
  return out;
}

describe('Phase 18 — bending rest test', () => {
  it('holds 90° rest angle with no perturbation (within 2°)', async () => {
    const result = await runRestTest({
      initialDihedralRad: Math.PI / 2,
      frames: 60, // 1 second at 60 Hz
      substeps: 8,
      iterations: 1,
    });
    expect(result.nanFree).toBe(true);
    // Edge endpoints are pinned and far vertices receive no force at
    // exact rest, so the angle should not drift. Allow 2° per the plan.
    const drift = Math.abs(result.finalAngle - result.initialAngle);
    // eslint-disable-next-line no-console
    console.info(
      `[bending-rest 90°] initial=${((result.initialAngle * 180) / Math.PI).toFixed(3)}° final=${((result.finalAngle * 180) / Math.PI).toFixed(3)}° drift=${((drift * 180) / Math.PI).toFixed(3)}° maxKE=${result.maxKineticEnergy.toExponential(2)}`,
    );
    expect((drift * 180) / Math.PI).toBeLessThan(2);
  }, 30_000);

  it('returns to 0° rest from a 30° perturbation (within 5°)', async () => {
    // Initial config is at 30° (the *perturbedDihedralRad*); rest
    // angle is overridden to 0° (flat). The bending constraint
    // should drive the cloth back toward flat. CPU-side velocity
    // damping at 0.92 / 10-frame so the cloth settles inside the
    // test budget instead of oscillating.
    const result = await runRestTest({
      initialDihedralRad: 0,
      perturbedDihedralRad: Math.PI / 6, // 30°
      frames: 240,
      substeps: 8,
      iterations: 1,
      velocityDamp: 0.92,
    });
    expect(result.nanFree).toBe(true);
    // eslint-disable-next-line no-console
    console.info(
      `[bending-rest 0°-from-30°] initial=${((result.initialAngle * 180) / Math.PI).toFixed(3)}° final=${((result.finalAngle * 180) / Math.PI).toFixed(3)}° maxKE=${result.maxKineticEnergy.toExponential(2)}`,
    );
    expect(Math.abs((result.finalAngle * 180) / Math.PI)).toBeLessThan(5);
  }, 60_000);
});
