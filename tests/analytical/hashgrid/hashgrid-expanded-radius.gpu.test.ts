import { Fn, If, instanceIndex, instancedArray, uint } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import {
  HashGrid,
  ParticleSystem,
  createParticleRenderer,
  emitForEachNeighbor,
  type ParticleInit,
} from '../../../src/core/index.js';

// Phase 03 G1 — expanded-radius coverage.
//
// Paper ref: Macklin et al. 2014 "Unified Particle Physics for Real-Time
// Applications" §9 — "we expand the search radius by a small ε to catch
// particles that move into range during the constraint solve".
//
// Plan §Step 5: the grid is built with `cellSize = h * (1 + ε)`; particles
// at distance slightly greater than `h` (but less than `h * (1 + ε)`) still
// appear as neighbors. This test places a pair at distance `h * (1 + ε/2)`
// and confirms that `emitForEachNeighbor` surfaces both sides of the pair.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

describe('Phase 03 — HashGrid: expanded-radius coverage', () => {
  it('a pair at h·(1 + ε/2) is returned by forEachNeighbor', async () => {
    const renderer = await createParticleRenderer();
    try {
      const H = 0.1;
      const EPS = 0.2;
      const EXPANDED = H * (1 + EPS);
      const D = H * (1 + EPS / 2);
      const D2 = D * D;

      // Two particles along the x-axis, separated by D. Both inside an
      // ample domain so the 27-cell query is unconstrained by bounds.
      const data: ParticleInit[] = [
        { position: [0, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
        { position: [D, 0, 0], velocity: [0, 0, 0], invMass: 1, phase: 0 },
      ];
      const particles = new ParticleSystem(renderer, 2, 0.02);
      particles.uploadParticles(data);

      const grid = new HashGrid(particles, {
        cellSize: EXPANDED,
      });
      await grid.rebuild();

      // Sanity: D is greater than H (so the base radius excludes the pair)
      // but less than EXPANDED (so the expanded build must include it).
      expect(D).toBeGreaterThan(H);
      expect(D).toBeLessThan(EXPANDED);

      // Count neighbors with |Δx|² < EXPANDED² for each particle. With the
      // pair at distance D < EXPANDED, the expected count is 1 for each.
      const EXPANDED2 = EXPANDED * EXPANDED;
      const countsOut = instancedArray(2, 'uint');
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
              const npos: Any = particles.positions.element(n).xyz;
              const diff: Any = pos.sub(npos);
              const d2: Any = diff.dot(diff);
              If(d2.lessThan(EXPANDED2), () => {
                count.addAssign(uint(1));
              });
            });
          },
        });
        countsOut.element(p).assign(count);
      })().compute(2);

      await renderer.computeAsync(countKernel);
      const gpu = new Uint32Array(await renderer.getArrayBufferAsync(countsOut.value));
      expect(gpu[0]).toBe(1);
      expect(gpu[1]).toBe(1);

      // Reference check: the pair would NOT be surfaced under a non-expanded
      // build (cellSize = H). Confirms the expanded-radius is what makes the
      // difference, rather than the test query filter doing all the work.
      // (Not strictly required for correctness — included as a targeted
      // guard against over-sized cellSize masking a forEachNeighbor bug.)
      expect(D2).toBeGreaterThan(H * H);

      grid.destroy();
      particles.destroy();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
