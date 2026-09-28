import { describe, expect, it } from 'vitest';
import { Object3D, Quaternion, Vector3 } from 'three';
import { PrimitiveSet, type ParticleSystem } from '../../../src/index.js';
import { FLAG_SWEEP } from '../../../src/core/collision/primitives.js';

// CPU-side bookkeeping of PrimitiveSet: what attach, update, and the add*
// methods write into the (not yet uploaded) GPU arrays.

const particles = {} as ParticleSystem;
const dt = 1 / 60;

function slotData(set: PrimitiveSet, slot: number): { data0: number[]; data1: number[] } {
  const data0 = set.gpu.data0.value.array as Float32Array;
  const data1 = set.gpu.data1.value.array as Float32Array;
  return {
    data0: Array.from(data0.subarray(slot * 4, slot * 4 + 4)),
    data1: Array.from(data1.subarray(slot * 4, slot * 4 + 4)),
  };
}

function flags(set: PrimitiveSet, slot: number): number {
  return (set.gpu.packed.value.array as Uint32Array)[slot]! & 0xffff;
}

describe('PrimitiveSet', () => {
  it('turns an attached plane normal with its object', () => {
    const set = new PrimitiveSet(particles);
    const object = new Object3D();
    const slot = set.addPlane(new Vector3(0, 1, 0), new Vector3());
    set.attach(slot, object);
    object.rotation.z = Math.PI / 2;
    object.position.set(1, 2, 3);
    object.updateMatrixWorld(true);
    set.update(dt);
    const { data0, data1 } = slotData(set, slot);
    expect(data0[0]).toBeCloseTo(-1, 5);
    expect(data0[1]).toBeCloseTo(0, 5);
    expect(data1.slice(0, 3)).toEqual([1, 2, 3]);
  });

  it('keeps a capsule axis at attach time, then turns it with its object', () => {
    const set = new PrimitiveSet(particles);
    const object = new Object3D();
    object.rotation.y = Math.PI / 3; // already turned; the axis must not jump
    object.updateMatrixWorld(true);
    const slot = set.addCapsule(new Vector3(-1, 0, 0), new Vector3(1, 0, 0), 0.1);
    set.attach(slot, object);
    let { data0, data1 } = slotData(set, slot);
    expect(data0[0]).toBeCloseTo(-1, 5);
    expect(data1[0]).toBeCloseTo(1, 5);
    object.rotation.y += Math.PI / 2;
    object.updateMatrixWorld(true);
    set.update(dt);
    ({ data0, data1 } = slotData(set, slot));
    // +x turned 90° about +y is −z.
    expect(data1[0]).toBeCloseTo(0, 5);
    expect(data1[2]).toBeCloseTo(-1, 5);
    expect(data0[2]).toBeCloseTo(1, 5);
    expect(data0[3]).toBeCloseTo(0.1, 6);
  });

  it('gives a box its object orientation', () => {
    const set = new PrimitiveSet(particles);
    const object = new Object3D();
    object.rotation.x = 0.4;
    object.updateMatrixWorld(true);
    const slot = set.addBox(new Vector3(), new Vector3(1, 1, 1));
    set.attach(slot, object);
    expect(set.gpu.rotations[slot]!.angleTo(object.quaternion)).toBeLessThan(1e-6);
  });

  it('derives angular velocity from the object turning', () => {
    const set = new PrimitiveSet(particles);
    const object = new Object3D();
    const slot = set.addSphere(new Vector3(), 0.5);
    set.attach(slot, object);
    object.rotation.y = 0.05;
    object.updateMatrixWorld(true);
    set.update(dt);
    const spin = set.gpu.spins[slot]!;
    expect(spin.x).toBeCloseTo(0, 5);
    expect(spin.y).toBeCloseTo(0.05 / dt, 3);
    expect(spin.z).toBeCloseTo(0, 5);
  });

  it('sweeps only primitives that move', () => {
    const set = new PrimitiveSet(particles);
    const belt = set.addPlane(new Vector3(0, 1, 0), new Vector3(), {
      velocity: new Vector3(1, 0, 0),
    });
    const attached = set.addSphere(new Vector3(), 0.5);
    const placed = set.addSphere(new Vector3(), 0.5);
    set.attach(attached, new Object3D());
    set.setSphere(placed, new Vector3(1, 0, 0), 0.5, new Vector3(1, 0, 0));
    set.update(dt);
    expect(flags(set, belt) & FLAG_SWEEP).toBe(0);
    expect(flags(set, attached) & FLAG_SWEEP).toBe(FLAG_SWEEP);
    expect(flags(set, placed) & FLAG_SWEEP).toBe(FLAG_SWEEP);
  });

  it('throws at add time once a reserved capacity is full', () => {
    const set = new PrimitiveSet(particles, { capacity: 1 });
    set.addSphere(new Vector3(), 1);
    expect(() => set.addSphere(new Vector3(), 1)).toThrow('PrimitiveSet: capacity 1 is full');
  });

  it('exposes version read-only', () => {
    const set = new PrimitiveSet(particles);
    set.addSphere(new Vector3(), 1);
    const before = set.version;
    expect(set.gpu.capacity).toBe(1);
    expect(set.version).toBe(before + 1);
    expect(() => {
      (set as { version: number }).version = 0;
    }).toThrow();
  });

  it('normalizes box rotations and rejects zero ones', () => {
    const set = new PrimitiveSet(particles);
    const slot = set.addBox(new Vector3(), new Vector3(1, 1, 1), {
      rotation: new Quaternion(0, 0, 2, 2),
    });
    expect(set.gpu.rotations[slot]!.length()).toBeCloseTo(1, 6);
    expect(() =>
      set.addBox(new Vector3(), new Vector3(1, 1, 1), { rotation: new Quaternion(0, 0, 0, 0) }),
    ).toThrow('PrimitiveSet.addBox: rotation must be a finite, non-zero quaternion');
  });

  it('prefixes friction errors', () => {
    const set = new PrimitiveSet(particles);
    expect(() => set.addSphere(new Vector3(), 1, { muS: -1 })).toThrow(
      'PrimitiveSet: friction coefficients must be non-negative',
    );
  });

  it('cannot be used after dispose', () => {
    const set = new PrimitiveSet(particles);
    const slot = set.addSphere(new Vector3(), 1);
    set.attach(slot, new Object3D());
    set.dispose();
    expect(() => set.update(dt)).not.toThrow();
    expect(() => set.addSphere(new Vector3(), 1)).toThrow('PrimitiveSet has been disposed');
    expect(() => set.gpu).toThrow('PrimitiveSet has been disposed');
  });
});
