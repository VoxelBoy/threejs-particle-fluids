import { describe, expect, it } from 'vitest';

import {
  ParticleSystem,
  createParticleRenderer,
  createXpbdUniforms,
} from '../../../src/core/index.js';
import { SoftbodySystem } from '../../../src/softbody/index.js';

const XPBD_FOR_TESTS = createXpbdUniforms(1 / 60);

// Phase 10 G1 — Pass 1 (center-of-mass) correctness.
//
// The kernel runs `c = (1/N) Σ x*_i` per body via a workgroup-shared
// tree reduction with a stride loop for bodies larger than the fixed
// workgroup size. This test feeds known positions and verifies the
// GPU-computed centers against a CPU reference for:
//   (a) a small body that fits in a single stride,
//   (b) a large body requiring multiple stride iterations,
//   (c) two bodies with different ranges in the same dispatch (non-
//       contiguous starts) — dispatches indexed by workgroupId map to
//       the right per-body (start, count).
//
// Tolerance: f32 reduction over N summands is Tier 2 bounded-max-error
// (ARCH §Guardrails G4). With `|x| < 2` and N ≤ 1024 the expected
// reduction error envelope is N · |x| · 2^-23 ≈ 2.5e-4 absolute; we
// allow 1e-4 m (comfortable).

const TOLERANCE = 1e-4;

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
  }[],
): Promise<{ cpu: [number, number, number][]; gpu: [number, number, number][] }> {
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, capacity, 0.05);

    // Upload every body's particles into the shared ParticleSystem.
    const initData: {
      position: [number, number, number];
      velocity: [number, number, number];
      invMass: number;
      phase: number;
    }[] = new Array(capacity).fill(0).map(() => ({
      position: [0, 0, 0],
      velocity: [0, 0, 0],
      invMass: 0, // trailing slots inert
      phase: 0,
    }));
    for (let b = 0; b < bodies.length; b++) {
      const body = bodies[b]!;
      for (let i = 0; i < body.positions.length; i++) {
        const p = body.positions[i]!;
        initData[body.start + i] = {
          position: [p[0]!, p[1]!, p[2]!],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: b + 1,
        };
      }
    }
    particles.uploadParticles(initData);

    // Build the SoftbodySystem and run just its preIterKernels (Pass 1
    // at the current commit). Bypass SimLoop — we want isolated-kernel
    // verification.
    const softbody = new SoftbodySystem({
      particles,
      xpbd: XPBD_FOR_TESTS,
      bodies: bodies.map((body, b) => {
        const rest = new Float32Array(body.positions.length * 3);
        for (let i = 0; i < body.positions.length; i++) {
          const p = body.positions[i]!;
          rest[3 * i + 0] = p[0]!;
          rest[3 * i + 1] = p[1]!;
          rest[3 * i + 2] = p[2]!;
        }
        return {
          particleRange: { start: body.start, count: body.positions.length },
          restPositions: rest,
          surfaceFlag: new Uint8Array(body.positions.length).fill(1),
          phaseId: b + 1,
          matchCompliance: 1e-6,
        };
      }),
    });

    await renderer.computeAsync([...softbody.preIterKernels]);

    const centers = new Float32Array(
      await renderer.getArrayBufferAsync(softbody.bodyCenters.value),
    );
    const gpu: [number, number, number][] = [];
    for (let b = 0; b < bodies.length; b++) {
      gpu.push([centers[4 * b + 0]!, centers[4 * b + 1]!, centers[4 * b + 2]!]);
    }
    const cpu: [number, number, number][] = bodies.map((body) => vec3Mean(body.positions));

    particles.destroy();
    return { cpu, gpu };
  } finally {
    renderer.dispose();
  }
}

describe('Phase 10 — SoftbodySystem Pass 1 center of mass', () => {
  it('single small body fitting in one workgroup stride', async () => {
    // 8-particle cube at a non-origin offset so mean is non-zero.
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

  it('large body requiring stride loop (N > workgroup size)', async () => {
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

  it('two bodies with non-contiguous ranges', async () => {
    // Body A at slots [0, 8), body B at slots [64, 72). Different
    // geometric centers. Workgroup dispatches must pick up the right
    // (start, count) per body.
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
