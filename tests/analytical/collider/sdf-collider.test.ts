import { describe, expect, it } from 'vitest';
import { Matrix4, Quaternion, Vector3 } from 'three';
import { SDFCollider, type ParticleSystem, type SDFData } from '../../../src/index.js';

// CPU-side state of SDFCollider: placement bookkeeping and live options.

const particles = {} as ParticleSystem;
const field: SDFData = {
  data: new Float32Array(4 * 4 * 4).fill(1),
  resolution: [4, 4, 4],
  origin: [0, 0, 0],
  voxelSize: [0.25, 0.25, 0.25],
};

describe('SDFCollider', () => {
  it('counts placement changes in version', () => {
    const collider = new SDFCollider(particles, field, { position: new Vector3(1, 0, 0) });
    expect(collider.version).toBe(0);
    collider.setPosition(new Vector3(1, 0, 0));
    expect(collider.version).toBe(0);
    collider.setPosition(new Vector3(2, 0, 0));
    expect(collider.version).toBe(1);
    collider.setRotation(new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), 0.3));
    expect(collider.version).toBe(2);
    collider.setScale(2);
    expect(collider.version).toBe(3);
    // A static follower sets the same matrix every frame; only the first counts.
    const matrix = new Matrix4().makeRotationX(0.2).setPosition(0, 1, 0);
    collider.setTransform(matrix);
    const moved = collider.version;
    expect(moved).toBeGreaterThan(3);
    collider.setTransform(matrix);
    expect(collider.version).toBe(moved);
  });

  it('rejects transforms it can’t represent', () => {
    const collider = new SDFCollider(particles, field);
    expect(() => collider.setTransform(new Matrix4().makeScale(0, 0, 0))).toThrow(
      /scale must be non-zero/,
    );
    expect(() => collider.setTransform(new Matrix4().makeScale(-1, 1, 1))).toThrow(/mirrored/);
    expect(() => collider.setTransform(new Matrix4().makeScale(1, 2, 1))).toThrow(
      /scale must be uniform/,
    );
    const broken = new Matrix4();
    broken.elements[12] = NaN;
    expect(() => collider.setTransform(broken)).toThrow(/NaN or infinite/);
    collider.setTransform(new Matrix4().makeScale(2, 2, 2));
    expect(collider.scale).toBeCloseTo(2);
  });

  it('normalizes rotations and rejects zero ones', () => {
    const collider = new SDFCollider(particles, field);
    collider.setRotation(new Quaternion(0, 3, 0, 4));
    expect(collider.rotation.length()).toBeCloseTo(1, 6);
    expect(() => collider.setRotation(new Quaternion(0, 0, 0, 0))).toThrow(
      'SDFCollider.setRotation: rotation must be a finite, non-zero quaternion',
    );
  });

  it('exposes thickness as a live, validated setting', () => {
    const collider = new SDFCollider(particles, field, { thickness: 0.01 });
    expect(collider.thickness).toBe(0.01);
    collider.thickness = 0.02;
    expect(collider.thickness).toBe(0.02);
    collider.setScale(3);
    expect(collider.thickness).toBe(0.02);
    expect(() => {
      collider.thickness = -1;
    }).toThrow('SDFCollider: thickness must be non-negative');
  });

  it('prefixes friction errors', () => {
    expect(() => new SDFCollider(particles, field, { muK: -1 })).toThrow(
      'SDFCollider: friction coefficients must be non-negative',
    );
  });
});
