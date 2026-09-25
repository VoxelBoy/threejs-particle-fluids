import { describe, expect, it } from 'vitest';

import {
  ParticleSystem,
  createParticleRenderer,
  createXpbdUniforms,
} from '../../../src/core/index.js';
import { SoftbodySystem } from '../../../src/softbody/index.js';

const XPBD_FOR_TESTS = createXpbdUniforms(1 / 60);

// Phase 10 G1 — non-coplanar input precondition + range / upload guards.
//
// The SoftbodySystem constructor must reject rank-deficient particle
// clouds at construction — Mueller 2011 §3 (page 2) singular-A caveat,
// plan §"Particle representation". MVP has no oriented particles, so
// the guard is structural rather than a mid-simulation recovery path.
//
// Well-formed voxelized closed-mesh inputs (bounding box non-zero in
// all three axes) must construct cleanly.

function buildBody(
  positions: readonly (readonly [number, number, number])[],
  opts: {
    start?: number;
    phaseId?: number;
    matchCompliance?: number;
  } = {},
): {
  readonly particleRange: { readonly start: number; readonly count: number };
  readonly restPositions: Float32Array;
  readonly surfaceFlag: Uint8Array;
  readonly phaseId: number;
  readonly matchCompliance: number;
} {
  const start = opts.start ?? 0;
  const rest = new Float32Array(positions.length * 3);
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i]!;
    rest[3 * i + 0] = p[0];
    rest[3 * i + 1] = p[1];
    rest[3 * i + 2] = p[2];
  }
  const surfaceFlag = new Uint8Array(positions.length).fill(1); // all surface for the guard test
  return {
    particleRange: { start, count: positions.length },
    restPositions: rest,
    surfaceFlag,
    phaseId: opts.phaseId ?? 1,
    matchCompliance: opts.matchCompliance ?? 1e-6,
  };
}

// A 2×2×2 cube — minimal rank-3 voxelized cloud, 8 particles, bounding
// box non-zero in all three axes. Paper-faithful baseline (Mueller 2011
// §5.3 explicit shape matching) for any positive test.
const CUBE_8 = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 1],
] as const;

describe('Phase 10 — SoftbodySystem construction guard', () => {
  it('accepts a well-formed voxelized cube (rank 3)', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [buildBody(CUBE_8)],
          }),
      ).not.toThrow();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a coplanar particle cloud (rank 2)', async () => {
    // All particles in the z=0 plane — rank-deficient, smallest singular
    // value is 0.
    const coplanar = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0.5, 0.5, 0],
    ] as const;
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 16, 0.05);
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [buildBody(coplanar)],
          }),
      ).toThrow(/rank-deficient|singular/i);
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a collinear particle cloud (rank 1)', async () => {
    const collinear = [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ] as const;
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [buildBody(collinear)],
          }),
      ).toThrow(/rank-deficient|singular/i);
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a single-point cloud (rank 0)', async () => {
    const singlePoint = [[0, 0, 0]] as const;
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 4, 0.05);
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [buildBody(singlePoint)],
          }),
      ).toThrow(/rank-deficient|singular/i);
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a near-coplanar cloud below the particleRadius·1e-3 threshold', async () => {
    // Particles in the z=[-ε, +ε] band with ε = particleRadius · 5e-4,
    // well below the rank threshold. The out-of-plane singular value is
    // σ_z = sqrt(Σ (rz_i)^2) which for N=4 at ±ε is 2ε. At
    // particleRadius=0.05, ε=2.5e-5, σ_z=5e-5 = particleRadius·1e-3. We
    // need strictly below the threshold, so pick ε = 1e-5 which puts
    // σ_z = 2e-5 < particleRadius·1e-3 = 5e-5.
    const epsilon = 1e-5;
    const nearCoplanar = [
      [0, 0, +epsilon],
      [1, 0, -epsilon],
      [0, 1, +epsilon],
      [1, 1, -epsilon],
    ] as const;
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [buildBody(nearCoplanar)],
          }),
      ).toThrow(/rank-deficient|singular/i);
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('accepts a non-cubic but rank-3 cloud (wide aspect ratio)', async () => {
    // 10× longer in x than in y/z. Rank 3, just anisotropic — must pass.
    const elongated = [
      [0, 0, 0],
      [10, 0, 0],
      [0, 1, 0],
      [10, 1, 0],
      [0, 0, 1],
      [10, 0, 1],
      [0, 1, 1],
      [10, 1, 1],
    ] as const;
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [buildBody(elongated)],
          }),
      ).not.toThrow();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a particleRange that exceeds ParticleSystem capacity', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 4, 0.05);
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [buildBody(CUBE_8, { start: 0 })], // count=8 > capacity=4
          }),
      ).toThrow(/particleRange/i);
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects restPositions whose length does not match 3·count', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      const body = buildBody(CUBE_8);
      // Wrong restPositions length — truncate.
      const bad = {
        ...body,
        restPositions: body.restPositions.slice(0, 12),
      };
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [bad],
          }),
      ).toThrow(/restPositions/i);
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects overlapping particle ranges across bodies', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 16, 0.05);
      const a = buildBody(CUBE_8, { start: 0, phaseId: 1 });
      const b = buildBody(CUBE_8, { start: 4, phaseId: 2 }); // overlaps [4,8)
      expect(
        () =>
          new SoftbodySystem({
            particles,
            xpbd: XPBD_FOR_TESTS,
            bodies: [a, b],
          }),
      ).toThrow(/already used/i);
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('exposes particleRange and surfaceRange for accepted bodies', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      // 4 surface + 4 interior, contiguous [surface..., interior...].
      const rest = new Float32Array(8 * 3);
      for (let i = 0; i < 8; i++) {
        const p = CUBE_8[i]!;
        rest[3 * i + 0] = p[0];
        rest[3 * i + 1] = p[1];
        rest[3 * i + 2] = p[2];
      }
      const surfaceFlag = new Uint8Array([1, 1, 1, 1, 0, 0, 0, 0]);
      const body = {
        particleRange: { start: 0, count: 8 },
        restPositions: rest,
        surfaceFlag,
        phaseId: 1,
        matchCompliance: 1e-6,
      };
      const softbody = new SoftbodySystem({
        particles,
        xpbd: XPBD_FOR_TESTS,
        bodies: [body],
      });
      expect(softbody.particleRange(0)).toEqual({ start: 0, count: 8 });
      expect(softbody.surfaceRange(0)).toEqual({ start: 0, count: 4 });
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects surfaceFlag with interleaved surface/interior ordering', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      const rest = new Float32Array(8 * 3);
      for (let i = 0; i < 8; i++) {
        const p = CUBE_8[i]!;
        rest[3 * i + 0] = p[0];
        rest[3 * i + 1] = p[1];
        rest[3 * i + 2] = p[2];
      }
      // 1,1,0,1 — surface particle at index 3 after an interior at 2.
      const badFlag = new Uint8Array([1, 1, 0, 1, 0, 0, 0, 0]);
      const softbody = new SoftbodySystem({
        particles,
        xpbd: XPBD_FOR_TESTS,
        bodies: [
          {
            particleRange: { start: 0, count: 8 },
            restPositions: rest,
            surfaceFlag: badFlag,
            phaseId: 1,
            matchCompliance: 1e-6,
          },
        ],
      });
      expect(() => softbody.surfaceRange(0)).toThrow(/ordering/i);
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  });
});
