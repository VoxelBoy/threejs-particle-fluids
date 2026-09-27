import { Vector3, type InstancedMesh } from 'three';
import {
  PrimitiveSet,
  createParticleMesh,
  type ParticleInit,
  type ParticleMeshOptions,
  type ParticleSystem,
} from '../../src/index.js';

/** Particles on a cubic lattice filling a box, optionally clipped by `accept`. */
export function lattice(
  min: readonly number[],
  max: readonly number[],
  spacing: number,
  accept?: (x: number, y: number, z: number) => boolean,
): ParticleInit[] {
  const points: ParticleInit[] = [];
  for (let y = min[1]! + spacing / 2; y < max[1]!; y += spacing)
    for (let z = min[2]! + spacing / 2; z < max[2]!; z += spacing)
      for (let x = min[0]! + spacing / 2; x < max[0]!; x += spacing)
        if (!accept || accept(x, y, z)) points.push({ position: [x, y, z] });
  return points;
}

/** A floor and four walls around the origin. */
export function tank(
  particles: ParticleSystem,
  halfX: number,
  halfZ: number,
  friction = { muS: 0.08, muK: 0.04 },
): PrimitiveSet {
  const walls = new PrimitiveSet(particles);
  walls.addPlane(new Vector3(0, 1, 0), new Vector3(), friction);
  walls.addPlane(new Vector3(1, 0, 0), new Vector3(-halfX, 0, 0), friction);
  walls.addPlane(new Vector3(-1, 0, 0), new Vector3(halfX, 0, 0), friction);
  walls.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -halfZ), friction);
  walls.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, halfZ), friction);
  return walls;
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

/** Deterministic random numbers in [0, 1), so every run starts the same. */
export function seededRandom(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Spheres showing the raw particles, hidden until the particle view is switched on. */
export function particleView(
  particles: ParticleSystem,
  options: ParticleMeshOptions = {},
): InstancedMesh {
  const mesh = createParticleMesh(particles, options);
  mesh.visible = false;
  return mesh;
}
