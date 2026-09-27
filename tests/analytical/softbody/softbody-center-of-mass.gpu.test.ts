import { describe, expect, it } from 'vitest';
import { uniform } from 'three/tsl';

import {
  ParticleSystem,
  SoftbodySystem,
  createParticleRenderer,
  type ParticleInit,
  type SolverContext,
} from '../../../src/index.js';

// Global shape matching: per-body center of mass.
//
// Each body's center `c = Σ m_i x*_i / Σ m_i` is computed on the GPU by a
// workgroup-shared tree reduction, with a stride loop for bodies larger than
// the fixed workgroup size. This test feeds known positions and checks the
// GPU centers against a CPU reference for:
//   (a) a small body that fits in a single stride,
//   (b) a large body requiring multiple stride iterations,
//   (c) two bodies with non-contiguous ranges in the same dispatch — each
//       workgroup must pick up its own body's (start, count),
//   (d) a body with unequal particle masses.
//
// Tolerance: an f32 reduction over N summands has a bounded error. With
// `|x| < 2` and N ≤ 1024 the error envelope is N · |x| · 2^-23 ≈ 2.5e-4
// absolute; 1e-4 m is comfortable for the sizes used here.

const TOLERANCE = 1e-4;

/** The solver state SimLoop hands a material, so its kernels can run without a loop. */
function solverContext(particles: ParticleSystem, dt: number): SolverContext {
  let group = 0;
  return {
    particles,
    dt: uniform(dt, 'float'),
    get hashGrid(): never {
      throw new Error('soft bodies do not use the neighbor grid');
    },
    allocateCollisionGroup: () => ++group,
  };
}

function vec3Mean(positions: readonly (readonly number[])[]): [number, number, number] {
  let sx = 0,
    sy = 0,
    sz = 0;
  for (const p of positions) {
    sx += p[0]!;
    sy += p[1]!;
    sz += p[2]!;
  }
  const invN = 1 / positions.length;
  return [sx * invN, sy * invN, sz * invN];
}

async function runCenterOfMass(
  capacity: number,
  bodies: {
    readonly start: number;
    readonly positions: readonly (readonly number[])[];
    readonly invMass?: readonly number[];
  }[],
): Promise<{
  cpu: [number, number, number][];
  gpu: [number, number, number][];
  rest: (readonly [number, number, number])[];
}> {
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, capacity, 0.05);

    // Upload every body's particles into the shared ParticleSystem; the
    // trailing slots stay inert.
    const initData: ParticleInit[] = new Array(capacity)
      .fill(0)
      .map(() => ({ position: [0, 0, 0], invMass: 0 }));
    for (const body of bodies) {
      for (let i = 0; i < body.positions.length; i++) {
        const p = body.positions[i]!;
        initData[body.start + i] = {
          position: [p[0]!, p[1]!, p[2]!],
          invMass: body.invMass?.[i] ?? 1,
        };
      }
    }
    particles.uploadParticles(initData);

    // Rest shapes default to the uploaded positions.
    const softbody = new SoftbodySystem(particles, {
      bodies: bodies.map((body) => ({
        range: { start: body.start, count: body.positions.length },
        compliance: 1e-6,
      })),
    });

    // Run the soft body's kernels for one solver iteration, without a SimLoop.
    // The particles sit at their rest shape, so the shape-matching correction
    // is zero and only the fitted body frames change.
    const kernels = softbody.build(solverContext(particles, 1 / 60));
    await renderer.computeAsync([...(kernels.preSolve ?? []), ...(kernels.solve ?? [])]);

    const centers = new Float32Array(
      await renderer.getArrayBufferAsync(softbody.bodyCenters.value),
    );
    const gpu: [number, number, number][] = [];
    for (let b = 0; b < bodies.length; b++) {
      gpu.push([centers[4 * b + 0]!, centers[4 * b + 1]!, centers[4 * b + 2]!]);
    }
    const cpu: [number, number, number][] = bodies.map((body) => vec3Mean(body.positions));

    const rest = softbody.bodies.map((body) => body.restCenter);

    particles.dispose();
    return { cpu, gpu, rest };
  } finally {
    renderer.dispose();
  }
}

describe('SoftbodySystem per-body center of mass', () => {
  it('single small body fitting in one workgroup stride', async () => {
    // 8-particle cube at a non-origin offset so the mean is non-zero.
    const positions = [
      [2, 1, 0],
      [3, 1, 0],
      [2, 2, 0],
      [3, 2, 0],
      [2, 1, 1],
      [3, 1, 1],
      [2, 2, 1],
      [3, 2, 1],
    ];
    const { cpu, gpu } = await runCenterOfMass(16, [{ start: 0, positions }]);
    expect(gpu[0]![0]).toBeCloseTo(cpu[0]![0], 4);
    expect(gpu[0]![1]).toBeCloseTo(cpu[0]![1], 4);
    expect(gpu[0]![2]).toBeCloseTo(cpu[0]![2], 4);
    // Also verify we match the expected geometric center (2.5, 1.5, 0.5).
    expect(gpu[0]![0]).toBeCloseTo(2.5, 4);
    expect(gpu[0]![1]).toBeCloseTo(1.5, 4);
    expect(gpu[0]![2]).toBeCloseTo(0.5, 4);
  });

  it('large body requiring the stride loop (N > workgroup size)', async () => {
    // 400 particles on a 10×10×4 lattice — exceeds the 256 workgroup
    // size so the stride loop runs twice (ceil(400 / 256) = 2).
    const positions: number[][] = [];
    for (let iy = 0; iy < 10; iy++) {
      for (let ix = 0; ix < 10; ix++) {
        for (let iz = 0; iz < 4; iz++) {
          positions.push([ix * 0.1, iy * 0.1, iz * 0.1]);
        }
      }
    }
    expect(positions.length).toBe(400);
    const { cpu, gpu } = await runCenterOfMass(512, [{ start: 0, positions }]);
    expect(Math.abs(gpu[0]![0] - cpu[0]![0])).toBeLessThan(TOLERANCE);
    expect(Math.abs(gpu[0]![1] - cpu[0]![1])).toBeLessThan(TOLERANCE);
    expect(Math.abs(gpu[0]![2] - cpu[0]![2])).toBeLessThan(TOLERANCE);
  });

  it('weights particles by mass', async () => {
    // The bottom face (y = 1) is three times as heavy as the top (y = 2), so
    // the center sits a quarter of the way up: y = (3·1 + 1·2) / 4 = 1.25.
    const positions = [
      [2, 1, 0],
      [3, 1, 0],
      [2, 2, 0],
      [3, 2, 0],
      [2, 1, 1],
      [3, 1, 1],
      [2, 2, 1],
      [3, 2, 1],
    ];
    const invMass = positions.map((p) => (p[1] === 1 ? 1 / 3 : 1));
    const { gpu, rest } = await runCenterOfMass(16, [{ start: 0, positions, invMass }]);
    for (const center of [gpu[0]!, rest[0]!]) {
      expect(center[0]).toBeCloseTo(2.5, 4);
      expect(center[1]).toBeCloseTo(1.25, 4);
      expect(center[2]).toBeCloseTo(0.5, 4);
    }
  });

  it('two bodies with non-contiguous ranges', async () => {
    // Body A at slots [0, 8), body B at slots [64, 72), with different
    // geometric centers. Each workgroup must pick up its own (start, count).
    const aPositions = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
    ];
    const bPositions = [
      [10, 20, 30],
      [11, 20, 30],
      [10, 21, 30],
      [11, 21, 30],
      [10, 20, 31],
      [11, 20, 31],
      [10, 21, 31],
      [11, 21, 31],
    ];
    const { cpu, gpu } = await runCenterOfMass(128, [
      { start: 0, positions: aPositions },
      { start: 64, positions: bPositions },
    ]);
    expect(gpu[0]![0]).toBeCloseTo(cpu[0]![0], 4);
    expect(gpu[0]![1]).toBeCloseTo(cpu[0]![1], 4);
    expect(gpu[0]![2]).toBeCloseTo(cpu[0]![2], 4);
    expect(gpu[1]![0]).toBeCloseTo(cpu[1]![0], 4);
    expect(gpu[1]![1]).toBeCloseTo(cpu[1]![1], 4);
    expect(gpu[1]![2]).toBeCloseTo(cpu[1]![2], 4);
    // Sanity: body-B center lives at (10.5, 20.5, 30.5), not (0.5, 0.5, 0.5).
    expect(gpu[1]![0]).toBeCloseTo(10.5, 4);
    expect(gpu[1]![1]).toBeCloseTo(20.5, 4);
    expect(gpu[1]![2]).toBeCloseTo(30.5, 4);
  });

  it('odd count (not a power of two) stays correct', async () => {
    // 13-particle cloud — stride loop + tree reduction must handle
    // non-power-of-two counts without off-by-one errors.
    const positions = [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [2, 1, 0],
      [0, 2, 0],
      [1, 2, 0],
      [2, 2, 0],
      [0, 0, 1],
      [1, 0, 1],
      [2, 0, 1],
      [1, 1, 1],
    ];
    const { cpu, gpu } = await runCenterOfMass(32, [{ start: 0, positions }]);
    expect(gpu[0]![0]).toBeCloseTo(cpu[0]![0], 4);
    expect(gpu[0]![1]).toBeCloseTo(cpu[0]![1], 4);
    expect(gpu[0]![2]).toBeCloseTo(cpu[0]![2], 4);
  });
});
