import { describe, expect, it } from 'vitest';

import {
  ParticleSystem,
  SoftbodySystem,
  createParticleRenderer,
  type SoftbodyDef,
} from '../../../src/index.js';

// SoftbodySystem construction guards.
//
// Shape matching extracts a rotation from the spread of a body's particles,
// which is only defined for a volume (Müller & Chentanez 2011 §3, the
// singular-A_pq caveat). The constructor must reject flat, collinear, and
// single-point rest shapes, as well as invalid ranges, rest data, and
// surface counts. Well-formed voxelized inputs (non-zero extent on all three
// axes) must construct cleanly.

function buildBody(
  positions: readonly (readonly [number, number, number])[],
  opts: { start?: number; compliance?: number } = {},
): SoftbodyDef {
  const start = opts.start ?? 0;
  const rest = new Float32Array(positions.length * 3);
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i]!;
    rest[3 * i + 0] = p[0];
    rest[3 * i + 1] = p[1];
    rest[3 * i + 2] = p[2];
  }
  // Every particle counts as surface (the default) for the guard tests.
  return {
    range: { start, count: positions.length },
    restPositions: rest,
    compliance: opts.compliance ?? 1e-6,
  };
}

// A 2×2×2 cube: the smallest rank-3 voxelized cloud, 8 particles, non-zero
// extent on all three axes.
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

function cubeRest(): Float32Array {
  const rest = new Float32Array(8 * 3);
  for (let i = 0; i < 8; i++) {
    const p = CUBE_8[i]!;
    rest[3 * i + 0] = p[0];
    rest[3 * i + 1] = p[1];
    rest[3 * i + 2] = p[2];
  }
  return rest;
}

describe('SoftbodySystem construction guards', () => {
  it('accepts a well-formed voxelized cube (rank 3)', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      expect(() => new SoftbodySystem(particles, { bodies: [buildBody(CUBE_8)] })).not.toThrow();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a coplanar particle cloud (rank 2)', async () => {
    // All particles in the z=0 plane: the smallest singular value is 0.
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
      expect(() => new SoftbodySystem(particles, { bodies: [buildBody(coplanar)] })).toThrow(
        /is flat; shape matching needs a 3D particle cloud/,
      );
      particles.dispose();
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
      expect(() => new SoftbodySystem(particles, { bodies: [buildBody(collinear)] })).toThrow(
        /is flat/,
      );
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a single-point cloud (rank 0)', async () => {
    const singlePoint = [[0, 0, 0]] as const;
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 4, 0.05);
      expect(() => new SoftbodySystem(particles, { bodies: [buildBody(singlePoint)] })).toThrow(
        /is flat/,
      );
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a near-coplanar cloud below the particleRadius·1e-3 threshold', async () => {
    // Particles in the z=[-ε, +ε] band. The out-of-plane singular value is
    // σ_z = sqrt(Σ rz_i²), which for N=4 at ±ε is 2ε. At particleRadius=0.05
    // the threshold is 5e-5; ε = 1e-5 puts σ_z = 2e-5 strictly below it.
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
      expect(() => new SoftbodySystem(particles, { bodies: [buildBody(nearCoplanar)] })).toThrow(
        /is flat/,
      );
      particles.dispose();
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
      expect(() => new SoftbodySystem(particles, { bodies: [buildBody(elongated)] })).not.toThrow();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a particle range that exceeds ParticleSystem capacity', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 4, 0.05);
      expect(
        // count=8 > capacity=4
        () => new SoftbodySystem(particles, { bodies: [buildBody(CUBE_8, { start: 0 })] }),
      ).toThrow(/invalid particle range/);
      particles.dispose();
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
      const bad = { ...body, restPositions: body.restPositions!.slice(0, 12) };
      expect(() => new SoftbodySystem(particles, { bodies: [bad] })).toThrow(
        /needs 24 rest coordinates/,
      );
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects overlapping particle ranges across bodies', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 16, 0.05);
      const a = buildBody(CUBE_8, { start: 0 });
      const b = buildBody(CUBE_8, { start: 4 }); // overlaps [4,8)
      expect(() => new SoftbodySystem(particles, { bodies: [a, b] })).toThrow(
        /particle 4 belongs to two bodies/,
      );
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('exposes particleRange and surfaceRange for accepted bodies', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      // 4 surface + 4 interior, surface particles first.
      const softbody = new SoftbodySystem(particles, {
        bodies: [{ range: { start: 0, count: 8 }, surfaceCount: 4, restPositions: cubeRest() }],
      });
      expect(softbody.particleRange(0)).toEqual({ start: 0, count: 8 });
      expect(softbody.surfaceRange(0)).toEqual({ start: 0, count: 4 });
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });

  it('rejects a surfaceCount outside [0, count] or not a whole number', async () => {
    const renderer = await createParticleRenderer();
    try {
      const particles = new ParticleSystem(renderer, 8, 0.05);
      const withSurfaceCount = (surfaceCount: number) => () =>
        new SoftbodySystem(particles, {
          bodies: [{ range: { start: 0, count: 8 }, surfaceCount, restPositions: cubeRest() }],
        });
      expect(withSurfaceCount(9)).toThrow(/surfaceCount 9 is out of range/);
      expect(withSurfaceCount(-1)).toThrow(/surfaceCount -1 is out of range/);
      expect(withSurfaceCount(2.5)).toThrow(/surfaceCount 2.5 is out of range/);
      // The whole range, or none of it, is valid.
      expect(withSurfaceCount(8)).not.toThrow();
      expect(withSurfaceCount(0)).not.toThrow();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  });
});
