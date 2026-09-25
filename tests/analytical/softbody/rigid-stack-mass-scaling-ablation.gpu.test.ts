import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import type WebGPURenderer from 'three/src/renderers/webgpu/WebGPURenderer.js';

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

// Phase 15b G1 — Macklin 2014 §5.2 stiff-stack ablation. Closes U-28's
// locking requirement.
//
// Reproduces the qualitative claim of paper Figure 10 caption ("with
// our method (red) the stack stabilizes quickly... without our mass
// modification the stack oscillates for a long time and shows
// significant compression"). 8-cube vertical stack at MVP iteration
// counts (substeps=4, iters=2). Two runs, scenes identical apart from
// `stackStabilization`.
//
// Metric: top-body lateral COM displacement at t=2 s. Pre-Phase-15b
// the §5.2-on code dispatched the mass-scaling kernel from
// `preIterKernels` (every iter saw scaled mass), which produced a top-
// body lateral runaway swamping the §5.2 settling benefit; that placement
// is the failure mode the paper itself warned about (§5.2 ¶ "if k is set
// too high"). Phase 15b moved the dispatch to
// `lastIterPreContactKernels` per the paper's mitigation text. Post-fix,
// §5.2-on settles the stack faster than §5.2-off — the assertion the
// plan's U-28 resolution path required.

const BODY_COUNT = 8;
const EDGE_LENGTH = 0.4;
const PARTICLE_RADIUS = 0.04;
const SPACING_FACTOR = 0.8;
const STACK_K = 3;
const FRAME_DT = 1 / 60;
const SETTLE_FRAMES = 120; // 2 s of integration.

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

interface AblationResult {
  readonly topComY: number;
  readonly settledTopY: number;
  readonly compression: number;
  readonly topSpeed: number;
}

async function runStackAblation(
  renderer: WebGPURenderer,
  enableStackStabilization: boolean,
): Promise<AblationResult> {
  const mesh = unitCubeMesh();
  const scaled: TriangleMesh = {
    vertices: mesh.vertices.map((v) => v * EDGE_LENGTH),
    indices: mesh.indices,
  };
  const cube = voxelize(scaled, {
    particleRadius: PARTICLE_RADIUS,
    bakeSdf: true,
    spacingFactor: SPACING_FACTOR,
  });
  const perBodyCount = cube.count;
  const totalCount = perBodyCount * BODY_COUNT;

  let minLocalY = Infinity;
  let maxLocalY = -Infinity;
  for (let i = 0; i < perBodyCount; i++) {
    const ly = cube.positions[3 * i + 1]!;
    if (ly < minLocalY) minLocalY = ly;
    if (ly > maxLocalY) maxLocalY = ly;
  }
  const stackSpacing = maxLocalY - minLocalY + 2 * PARTICLE_RADIUS;

  let lcg = 1;
  const rng = (): number => {
    lcg = (lcg * 1664525 + 1013904223) | 0;
    return (lcg >>> 0) / 0x100000000;
  };

  const initial: ParticleInit[] = [];
  const restPerBody: Float32Array[] = [];
  const ranges: { start: number; count: number }[] = [];

  for (let b = 0; b < BODY_COUNT; b++) {
    const cy = PARTICLE_RADIUS - minLocalY + b * stackSpacing;
    const cx = (rng() - 0.5) * PARTICLE_RADIUS * 0.5;
    const cz = (rng() - 0.5) * PARTICLE_RADIUS * 0.5;

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

  const particles = new ParticleSystem(renderer, totalCount, PARTICLE_RADIUS);
  particles.uploadParticles(initial);

  const xpbd = createXpbdUniforms(FRAME_DT);
  const hashGrid = new HashGrid(particles, {
    cellSize: PARTICLE_RADIUS * 2.1,
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
    ...(enableStackStabilization ? { stackStabilization: { k: STACK_K, groundY: 0 } } : {}),
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

  for (let f = 0; f < SETTLE_FRAMES; f++) {
    await loop.step(FRAME_DT);
  }

  const snap = await particles.readback();
  const top = ranges[BODY_COUNT - 1]!;
  let cy = 0;
  let speed = 0;
  for (let i = 0; i < top.count; i++) {
    const slot = top.start + i;
    cy += snap.positions[4 * slot + 1]!;
    const vx = snap.velocities[4 * slot + 0]!;
    const vy = snap.velocities[4 * slot + 1]!;
    const vz = snap.velocities[4 * slot + 2]!;
    const sp = Math.sqrt(vx * vx + vy * vy + vz * vz);
    if (sp > speed) speed = sp;
  }
  cy /= top.count;

  // Settled top-body COM y for an axis-aligned stack: each cube's COM
  // is at h = particleRadius - minLocalY + b·stackSpacing + 0 (cube
  // local COM is the origin since rest is centered).
  const settledTopY = PARTICLE_RADIUS - minLocalY + (BODY_COUNT - 1) * stackSpacing;
  const compression = settledTopY - cy; // positive = stack is compressed.

  particles.destroy();
  hashGrid.destroy();
  colliders.destroy();

  return {
    topComY: cy,
    settledTopY,
    compression,
    topSpeed: speed,
  };
}

describe('Phase 15b — §5.2 mass-scaling ablation (G1, closes U-28)', () => {
  it('§5.2 mass scaling settles the stack faster than no scaling', async () => {
    const renderer = await createParticleRenderer();
    try {
      const withScaling = await runStackAblation(renderer, true);
      const withoutScaling = await runStackAblation(renderer, false);

      // eslint-disable-next-line no-console
      console.info(
        `[stack-ablation] T=${(SETTLE_FRAMES * FRAME_DT).toFixed(2)}s ` +
          `with §5.2: topY=${withScaling.topComY.toFixed(4)} compression=${withScaling.compression.toFixed(4)} m maxSpeed=${withScaling.topSpeed.toExponential(3)} m/s | ` +
          `without §5.2: topY=${withoutScaling.topComY.toFixed(4)} compression=${withoutScaling.compression.toFixed(4)} m maxSpeed=${withoutScaling.topSpeed.toExponential(3)} m/s`,
      );

      // Paper Figure 10 claim: "with our method... stabilizes quickly...
      // without our mass modification the stack ... shows significant
      // compression." The ablation locks two parts of that claim:
      //
      //   1. §5.2-on shows LESS compression at t=2 s than §5.2-off
      //      (the stack is closer to its true rest configuration).
      //   2. §5.2-on top body has LOWER residual particle speed than
      //      §5.2-off (faster settling).
      //
      // Both assertions fail against pre-fix code (where §5.2-on
      // produced unbounded top-body runaway swamping the settling
      // benefit). Post-fix they pass because §5.2 mass scaling delivers
      // its paper-promised acceleration without the per-iter asymmetric
      // drift the paper warned about.
      expect(withScaling.compression).toBeLessThan(withoutScaling.compression);
      expect(withScaling.topSpeed).toBeLessThan(withoutScaling.topSpeed);
    } finally {
      renderer.dispose();
    }
  }, 120_000);
});
