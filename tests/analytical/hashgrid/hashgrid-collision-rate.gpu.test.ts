import { Fn, If, instanceIndex, instancedArray, uint } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  emitForEachNeighbor,
  type ParticleInit,
} from '../../../src/core/index.js';

// HashGrid collision-rate sweep (G1).
//
// The hash-table-size knob trades memory for narrow-phase false-positive
// rate. This test pins the curve numerically for the chosen default
// (`hashTableSize = 2·capacity`) under the current Morton bucket function.
//
// Scene: 16³ = 4096 particles, one per cell, on a uniform lattice of cell-
// size spacing. Each particle sits squarely inside its own cell so the
// cell-index computation is unambiguous.
//
// For each `hashTableSize` in the sweep:
//   - Build the grid.
//   - Count the total number of `onCandidate` invocations across all
//     queries (excluding self, since every caller filters self anyway).
//   - Compare to the CPU-computed ground-truth count (27-cell lattice
//     neighbors minus self, clipped at boundaries).
//   - Collision rate = (visited - true) / true. A positive rate means
//     the hash surfaced candidates from cells outside the 27-neighbor
//     window — the bucket-aliasing tax.
//
// Measured curve under Morton bucketing on the reference platform
// (capacity = 4096):
//   n=1024   rate=370.0%  (table much smaller than capacity, dense aliasing)
//   n=2048   rate=132.8%
//   n=4096   rate= 14.2%
//   n=8192   rate=  9.3%  ← n = 2·capacity, the production default
//   n=16384  rate=  4.5%
//   n=32768  rate=  0.0%  ← n = 8·capacity, flat regime
//
// The previous Teschner XOR-mix bucket function pinned ~54% rate at
// `n = 2·capacity` (Phase 07a Plan deviation 1: birthday-paradox floor).
// Morton bucketing improves this by ~6× because spatially-clustered cells
// share lower-bit Morton-code structure, reducing within-walk false-
// positives. Same lattice scene, same hashTableSize.
//
// Assertion thresholds below stay loose so the test is not brittle to
// minor Morton-distribution variations across platforms:
//   - At `n = 2·capacity`  (n = 8192):  rate ≤ 0.60. Generous ceiling;
//     Morton actually lands well below this on the reference platform.
//   - At `n = 8·capacity`  (n = 32768): rate ≤ 0.10. Flat regime reachable
//     at this load factor under both Teschner and Morton.
//   - At `n = 0.5·capacity` (n = 2048): rate > 0.50. Test-sanity lower
//     bound — confirms the hash is real, not a unique-bucket stub.
//
// Real-scene collision rates are expected to be substantially below these
// figures because particle clustering leaves most cells empty; this test
// deliberately measures the densest possible worst case to ground the
// upper bound.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const N_CUBE = 16;
const N = N_CUBE ** 3;
const CELL = 0.05;

function buildLattice(): ParticleInit[] {
  const data: ParticleInit[] = [];
  for (let iz = 0; iz < N_CUBE; iz++) {
    for (let iy = 0; iy < N_CUBE; iy++) {
      for (let ix = 0; ix < N_CUBE; ix++) {
        data.push({
          position: [ix * CELL + CELL / 2, iy * CELL + CELL / 2, iz * CELL + CELL / 2],
          velocity: [0, 0, 0],
          invMass: 1,
          phase: 0,
        });
      }
    }
  }
  return data;
}

function cpuTrueCandidateTotal(): number {
  let total = 0;
  for (let iz = 0; iz < N_CUBE; iz++) {
    for (let iy = 0; iy < N_CUBE; iy++) {
      for (let ix = 0; ix < N_CUBE; ix++) {
        for (let dz = -1; dz <= 1; dz++) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0 && dz === 0) continue; // exclude self
              const nx = ix + dx;
              const ny = iy + dy;
              const nz = iz + dz;
              if (nx >= 0 && nx < N_CUBE && ny >= 0 && ny < N_CUBE && nz >= 0 && nz < N_CUBE) {
                total += 1;
              }
            }
          }
        }
      }
    }
  }
  return total;
}

async function measureCollisionRate(
  renderer: Awaited<ReturnType<typeof createParticleRenderer>>,
  particles: ParticleSystem,
  hashTableSize: number,
): Promise<number> {
  const grid = new HashGrid(particles, { cellSize: CELL, hashTableSize });
  await grid.rebuild();

  const countsOut = instancedArray(N, 'uint');
  const countKernel = Fn(() => {
    const p: Any = instanceIndex;
    const pos: Any = particles.positions.element(p).xyz;
    const count: Any = uint(0).toVar();
    emitForEachNeighbor({
      queryPosXyz: pos,
      hashOrigin: grid.hashOriginUniform,
      cellSize: grid.cellSizeUniform,
      hashTableSize: grid.hashTableSize,
      cellStart: grid.cellStart,
      cellEnd: grid.cellEnd,
      sortedIndices: grid.sortedIndices,
      onCandidate: (n) => {
        If((n as Any).notEqual(p), () => {
          count.addAssign(uint(1));
        });
      },
    });
    countsOut.element(p).assign(count);
  })().compute(N);

  await renderer.computeAsync(countKernel);
  const gpu = new Uint32Array(await renderer.getArrayBufferAsync(countsOut.value));

  let visited = 0;
  for (let p = 0; p < N; p++) visited += gpu[p]!;

  const trueTotal = cpuTrueCandidateTotal();
  grid.destroy();
  return (visited - trueTotal) / trueTotal;
}

describe('HashGrid: collision-rate sweep (G1)', () => {
  it('false-positive rate decreases monotonically with hashTableSize under Morton bucketing', async () => {
    const renderer = await createParticleRenderer();
    try {
      const data = buildLattice();
      const particles = new ParticleSystem(renderer, N, 0.005);
      particles.uploadParticles(data);

      // Sweep covers load factors 4×, 2×, 1×, 0.5×, 0.25×, 0.125× the
      // particle count. `hashTableSize = 2·capacity = 8192` is the plan's
      // default; the sweep around it grounds the default choice.
      const sizes = [1024, 2048, 4096, 8192, 16384, 32768];
      const results: { n: number; rate: number }[] = [];
      for (const n of sizes) {
        const rate = await measureCollisionRate(renderer, particles, n);
        results.push({ n, rate });
      }

      // eslint-disable-next-line no-console
      console.info(
        `[hashgrid-collision-rate] capacity=${N} ` +
          results.map((r) => `n=${r.n}:rate=${(r.rate * 100).toFixed(1)}%`).join(' '),
      );

      // Monotonicity: larger tables produce ≤ smaller collision rates.
      // Cannot be strict — hash collisions are position-dependent and can
      // tie or invert slightly across adjacent sizes. We assert the
      // coarse trend: the quartile boundaries must be ordered.
      const byN = new Map(results.map((r) => [r.n, r.rate]));
      expect(byN.get(1024)!).toBeGreaterThan(byN.get(32768)!);
      expect(byN.get(2048)!).toBeGreaterThan(byN.get(16384)!);

      // See the file-level comment for the plan-vs-measurement rationale.
      expect(byN.get(8192)!).toBeLessThanOrEqual(0.6);
      expect(byN.get(32768)!).toBeLessThanOrEqual(0.1);
      expect(byN.get(2048)!).toBeGreaterThan(0.5);

      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 180_000);
});
