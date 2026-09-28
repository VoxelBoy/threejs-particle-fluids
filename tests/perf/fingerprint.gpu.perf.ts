// Fingerprint suite: runs each scene for a fixed number of frames, reads the
// particles back, and prints summary statistics of the moving ones between
// sentinel lines for `_helpers/fingerprint.ts` to compare against a saved
// baseline. An optimization that changes the physics shifts these numbers
// by more than the run-to-run noise (the grid's scatter order varies between
// runs, so float sums differ slightly and the motion drifts apart).
//
// VITE_PROFILE_SCENES and VITE_PROFILE_LEVEL select scenes as in the profile
// suite; VITE_FINGERPRINT_FRAMES sets the frame count (default 90).

import { describe, it } from 'vitest';

import { PARTICLE_LEVELS, type ParticleLevel } from '../../demo/types.js';
import { PerfRenderer } from './_helpers/PerfRenderer.js';
import { PRESET_IDS, SCENES } from './_scenes/registry.js';

const JSON_BEGIN = '__PARTICLE_FLUIDS_FINGERPRINT_JSON_BEGIN__';
const JSON_END = '__PARTICLE_FLUIDS_FINGERPRINT_JSON_END__';

export interface Fingerprint {
  readonly id: string;
  readonly moving: number;
  readonly nonFinite: number;
  readonly centroid: readonly [number, number, number];
  /** RMS distance from the centroid. */
  readonly spread: number;
  readonly minY: number;
  readonly maxY: number;
  readonly meanSpeed: number;
  readonly rmsSpeed: number;
  readonly p99Speed: number;
}

function fingerprint(
  id: string,
  positions: Float32Array,
  velocities: Float32Array,
  invMass: Float32Array,
): Fingerprint {
  let moving = 0;
  let nonFinite = 0;
  const c = [0, 0, 0];
  let minY = Infinity;
  let maxY = -Infinity;
  const speeds: number[] = [];
  for (let i = 0; i < invMass.length; i++) {
    if (!(invMass[i]! > 0)) continue;
    const x = positions[i * 4]!;
    const y = positions[i * 4 + 1]!;
    const z = positions[i * 4 + 2]!;
    const vx = velocities[i * 4]!;
    const vy = velocities[i * 4 + 1]!;
    const vz = velocities[i * 4 + 2]!;
    if (![x, y, z, vx, vy, vz].every(Number.isFinite)) {
      nonFinite++;
      continue;
    }
    moving++;
    c[0]! += x;
    c[1]! += y;
    c[2]! += z;
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    speeds.push(Math.hypot(vx, vy, vz));
  }
  const n = Math.max(moving, 1);
  const centroid: [number, number, number] = [c[0]! / n, c[1]! / n, c[2]! / n];
  let spreadSq = 0;
  for (let i = 0; i < invMass.length; i++) {
    if (!(invMass[i]! > 0)) continue;
    const dx = positions[i * 4]! - centroid[0];
    const dy = positions[i * 4 + 1]! - centroid[1];
    const dz = positions[i * 4 + 2]! - centroid[2];
    if (Number.isFinite(dx + dy + dz)) spreadSq += dx * dx + dy * dy + dz * dz;
  }
  speeds.sort((a, b) => a - b);
  const sum = speeds.reduce((s, v) => s + v, 0);
  const sumSq = speeds.reduce((s, v) => s + v * v, 0);
  return {
    id,
    moving,
    nonFinite,
    centroid,
    spread: Math.sqrt(spreadSq / n),
    minY,
    maxY,
    meanSpeed: sum / n,
    rmsSpeed: Math.sqrt(sumSq / n),
    p99Speed: speeds[Math.floor(0.99 * (speeds.length - 1))] ?? 0,
  };
}

describe('physics fingerprint', () => {
  it(
    'runs every selected scene and prints particle statistics',
    async () => {
      const ids = String(import.meta.env['VITE_PROFILE_SCENES'] ?? '')
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const selected = ids.length > 0 ? ids : PRESET_IDS;
      const levelName = String(import.meta.env['VITE_PROFILE_LEVEL'] ?? 'ultra');
      const level = (PARTICLE_LEVELS.find((l) => l.id === levelName)?.id ??
        'ultra') as ParticleLevel;
      const frames = Number(import.meta.env['VITE_FINGERPRINT_FRAMES'] ?? 90);

      const perf = await PerfRenderer.create();
      const results: Fingerprint[] = [];
      try {
        for (const id of selected) {
          const scene = await SCENES[id]!(perf, level);
          try {
            if (!scene.particles) throw new Error(`${id} has no particles to read back`);
            for (let i = 0; i < frames; i++) {
              await scene.simulate();
              await perf.discardGpuTimings();
            }
            await perf.device.queue.onSubmittedWorkDone();
            perf.assertNoErrors(id);
            const snapshot = await scene.particles.readback();
            const result = fingerprint(
              id,
              snapshot.positions,
              snapshot.velocities,
              snapshot.invMass,
            );
            results.push(result);
            console.log(`[fingerprint] ${id}: ${result.moving} moving, spread ${result.spread}`);
          } finally {
            scene.dispose();
          }
        }
      } finally {
        perf.dispose();
      }
      console.log(JSON_BEGIN);
      console.log(JSON.stringify({ level, frames, scenes: results }));
      console.log(JSON_END);
    },
    60 * 60_000,
  );
});
