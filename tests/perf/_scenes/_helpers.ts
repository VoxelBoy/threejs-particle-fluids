// Shared pieces of the benchmark scenes: the built-scene shape and
// initial-state generators.

import type { ParticleInit } from '../../../src/index.js';
import type { PerfSceneSpec } from '../_helpers/PerfRunner.js';

/** A scene ready to benchmark, and the cleanup to run after it. */
export interface BuiltScene {
  readonly spec: PerfSceneSpec;
  readonly dispose: () => void;
}

/** Every scene steps at 60 frames per second. */
export const FRAME_DT = 1 / 60;

/** Particles per side of a cubic lattice holding `count` particles. */
export function latticeSide(count: number): number {
  return Math.ceil(Math.cbrt(count));
}

/**
 * `count` particles at rest in a cubic lattice with `spacing` between
 * neighbors; `origin` is the lattice's minimum corner.
 */
export function fluidColumn(args: {
  readonly count: number;
  readonly spacing: number;
  readonly origin: readonly [number, number, number];
}): ParticleInit[] {
  const { count, spacing } = args;
  const [ox, oy, oz] = args.origin;
  const side = latticeSide(count);
  const out: ParticleInit[] = [];
  for (let z = 0; z < side && out.length < count; z++) {
    for (let y = 0; y < side && out.length < count; y++) {
      for (let x = 0; x < side && out.length < count; x++) {
        out.push({
          position: [ox + (x + 0.5) * spacing, oy + (y + 0.5) * spacing, oz + (z + 0.5) * spacing],
        });
      }
    }
  }
  return out;
}

/**
 * Open-top box of pinned particles (`invMass` 0) for use as a fluid
 * boundary: an `nxz × nxz` floor at `floorY` and four walls `height`
 * particles tall, `nxz² + 4·nxz·height` particles in all.
 */
export function boxBoundary(args: {
  readonly nxz: number;
  readonly height: number;
  readonly spacing: number;
  readonly center: readonly [number, number];
  readonly floorY: number;
}): ParticleInit[] {
  const { nxz, height, spacing, center, floorY } = args;
  const [cx, cz] = center;
  const half = (nxz * spacing) / 2;
  const out: ParticleInit[] = [];
  const pinned = (x: number, y: number, z: number): void => {
    out.push({ position: [x, y, z], invMass: 0 });
  };
  for (let z = 0; z < nxz; z++) {
    for (let x = 0; x < nxz; x++) {
      pinned(cx - half + (x + 0.5) * spacing, floorY, cz - half + (z + 0.5) * spacing);
    }
  }
  for (let h = 0; h < height; h++) {
    const y = floorY + (h + 0.5) * spacing;
    for (let i = 0; i < nxz; i++) {
      const t = -half + (i + 0.5) * spacing;
      pinned(cx + t, y, cz - half);
      pinned(cx + t, y, cz + half);
      pinned(cx - half, y, cz + t);
      pinned(cx + half, y, cz + t);
    }
  }
  return out;
}

/**
 * A `side³` lattice cube centered on `center`, surface particles first (the
 * order {@link SoftbodySystem} expects). Returns the particles and how many
 * lie on the surface.
 */
export function particleCube(args: {
  readonly center: readonly [number, number, number];
  readonly side: number;
  readonly spacing: number;
  readonly invMass?: number;
}): { readonly particles: ParticleInit[]; readonly surfaceCount: number } {
  const { center, side, spacing } = args;
  const invMass = args.invMass ?? 1;
  const offset = (i: number): number => (i - (side - 1) / 2) * spacing;
  const surface: ParticleInit[] = [];
  const interior: ParticleInit[] = [];
  for (let z = 0; z < side; z++) {
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        const onSurface = [x, y, z].some((i) => i === 0 || i === side - 1);
        (onSurface ? surface : interior).push({
          position: [center[0] + offset(x), center[1] + offset(y), center[2] + offset(z)],
          invMass,
        });
      }
    }
  }
  return { particles: [...surface, ...interior], surfaceCount: surface.length };
}
