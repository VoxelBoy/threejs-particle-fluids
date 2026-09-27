import { Fn, If, instanceIndex, instancedArray, uint } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  emitForEachNeighbor,
  type ParticleInit,
} from '../../../src/index.js';

// Lattice correctness.
//
// 4×4×4 particles on a 0.05 lattice with h=0.08. h² = 0.0064 is cleanly between
// the 2-axis pair distance² (0.005) and the 3-axis pair distance² (0.0075), so
// f32/f64 precision differences cannot flip a pair's neighbor status. CPU
// brute force with |Δx|² < h² gives the ground truth.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

describe('HashGrid: 4×4×4 lattice correctness', () => {
  it('per-particle neighbor count matches CPU brute force', async () => {
    const renderer = await createParticleRenderer();
    try {
      const LATTICE = 4;
      const SPACING = 0.05;
      const H = 0.08;
      const H2 = H * H;
      const N = LATTICE ** 3;

      const data: ParticleInit[] = [];
      for (let k = 0; k < LATTICE; k++) {
        for (let j = 0; j < LATTICE; j++) {
          for (let i = 0; i < LATTICE; i++) {
            data.push({
              position: [i * SPACING, j * SPACING, k * SPACING],
              velocity: [0, 0, 0],
              invMass: 1,
            });
          }
        }
      }

      const particles = new ParticleSystem(renderer, N, 0.02);
      particles.uploadParticles(data);

      const grid = new HashGrid(particles, {
        cellSize: H,
      });

      await grid.rebuild();

      // Invariant pass: readback and validate the sort structure before
      // running the neighbor kernel. A failure here isolates bugs to the
      // build pipeline (cellIndex / histogram / scan / scatter) rather than
      // to `emitForEachNeighbor`.
      const snap = await grid.readback();
      let totalCounts = 0;
      for (let c = 0; c < grid.hashTableSize; c++) totalCounts += snap.counts[c]!;
      expect(totalCounts).toBe(N);

      let prefix = 0;
      for (let c = 0; c < grid.hashTableSize; c++) {
        expect(snap.cellStart[c]).toBe(prefix);
        expect(snap.cellEnd[c]).toBe(prefix + snap.counts[c]!);
        prefix += snap.counts[c]!;
      }

      const seen = new Uint8Array(N);
      for (let k = 0; k < N; k++) {
        const p = snap.sortedIndices[k]!;
        expect(p).toBeLessThan(N);
        expect(seen[p]).toBe(0);
        seen[p] = 1;
      }
      for (let p = 0; p < N; p++) expect(seen[p]).toBe(1);

      for (let p = 0; p < N; p++) {
        const c = snap.cellIndex[p]!;
        expect(c).toBeLessThan(grid.hashTableSize);
      }

      // Neighbor-walk pass: per-particle neighbor count, matched against
      // CPU brute force using squared-distance comparison. Within-cell sort
      // order doesn't matter — the set of visited candidates does.
      const countsOut = instancedArray(N, 'uint');
      const countKernel = Fn(() => {
        const p: Any = instanceIndex;
        const pos: Any = particles.positions.element(p).xyz;
        const count: Any = uint(0).toVar();
        emitForEachNeighbor(grid, pos, (n) => {
          If(n.notEqual(p), () => {
            const npos: Any = particles.positions.element(n).xyz;
            const diff: Any = pos.sub(npos);
            const d2: Any = diff.dot(diff);
            If(d2.lessThan(H2), () => {
              count.addAssign(uint(1));
            });
          });
        });
        countsOut.element(p).assign(count);
      })().compute(N);

      await renderer.computeAsync(countKernel);
      const gpuCounts = new Uint32Array(await renderer.getArrayBufferAsync(countsOut.value));

      const cpuCounts = new Uint32Array(N);
      for (let p = 0; p < N; p++) {
        const dp = data[p]!.position;
        let c = 0;
        for (let q = 0; q < N; q++) {
          if (q === p) continue;
          const dq = data[q]!.position;
          const dx = dp[0] - dq[0];
          const dy = dp[1] - dq[1];
          const dz = dp[2] - dq[2];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < H2) c++;
        }
        cpuCounts[p] = c;
      }

      for (let p = 0; p < N; p++) {
        expect(gpuCounts[p]).toBe(cpuCounts[p]);
      }

      grid.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
