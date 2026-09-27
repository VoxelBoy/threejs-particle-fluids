import { Vector3, type BufferGeometry } from 'three';
import { PrimitiveSet, type ParticleInit, type ParticleSystem } from '../../src/core/index.js';
import type { TriangleMesh } from '../../src/softbody/index.js';

export function lattice(
  min: readonly number[],
  max: readonly number[],
  spacing: number,
  accept?: (x: number, y: number, z: number) => boolean,
): ParticleInit[] {
  const values: ParticleInit[] = [];
  for (let y = min[1]! + spacing / 2; y < max[1]!; y += spacing) {
    for (let z = min[2]! + spacing / 2; z < max[2]!; z += spacing) {
      for (let x = min[0]! + spacing / 2; x < max[0]!; x += spacing) {
        if (accept && !accept(x, y, z)) continue;
        values.push({ position: [x, y, z], velocity: [0, 0, 0], invMass: 1, phase: 0 });
      }
    }
  }
  return values;
}

export function tank(
  particles: ParticleSystem,
  halfX = 0.8,
  halfZ = 0.55,
  capacity = 12,
  friction = { muS: 0.08, muK: 0.04 },
): PrimitiveSet {
  const colliders = new PrimitiveSet(particles, { capacity });
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(), friction);
  colliders.addPlane(new Vector3(1, 0, 0), new Vector3(-halfX, 0, 0), friction);
  colliders.addPlane(new Vector3(-1, 0, 0), new Vector3(halfX, 0, 0), friction);
  colliders.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -halfZ), friction);
  colliders.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, halfZ), friction);
  return colliders;
}

export function triangleMesh(geometry: BufferGeometry): TriangleMesh {
  const position = geometry.getAttribute('position');
  const vertices = new Float32Array(position.count * 3);
  for (let i = 0; i < position.count; i++)
    vertices.set([position.getX(i), position.getY(i), position.getZ(i)], i * 3);
  const index = geometry.getIndex();
  const indices = index
    ? Uint32Array.from(index.array)
    : Uint32Array.from({ length: position.count }, (_, i) => i);
  return { vertices, indices };
}

/**
 * Particle radius for which `fill(radius)` produces about `target` particles.
 * Lattice counts scale with 1/r³, so a few cube-root corrections converge.
 */
export function fitRadius(
  fill: (radius: number) => readonly unknown[],
  target: number,
  guess: number,
): number {
  let radius = guess;
  for (let i = 0; i < 5; i++) radius *= Math.cbrt(Math.max(1, fill(radius).length) / target);
  return radius;
}

/**
 * Substeps for a particle budget. Presets fill a fixed volume (or area, for
 * cloth), so spacing shrinks as the count grows while speeds stay the same;
 * keeping motion per substep a fixed fraction of the spacing means scaling
 * substeps by 1/spacing. `base` is the preset's setting at Medium (10k).
 */
export function scaledSubsteps(base: number, particles: number, dimensions: 2 | 3 = 3): number {
  const scale = (particles / 10000) ** (1 / dimensions);
  return Math.max(base, Math.ceil(base * scale - 0.1));
}
