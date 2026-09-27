import { describe, expect, it } from 'vitest';
import { Fn, instanceIndex, uniform } from 'three/tsl';
import {
  ParticleSystem,
  constraintKernels,
  createDistanceConstraints,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

// Residual decreases monotonically across solver iterations.
//
// For a random network of distance constraints, XPBD's Gauss-Seidel solve
// must produce a non-increasing sequence of `L2 residual = Σ C(x)²`
// measurements across iterations within a single substep. Any iteration
// that INCREASES residual is a correctness failure — the most common
// causes are sign errors in ∇C, missing the `α̃·λ` regularization term, or
// a non-coloring-compliant group partition letting two constraints write
// the same particle.
//
// We allow a small ULP tolerance (1e-6 relative) — a strictly-decreasing
// assertion would be defeated by f32 round-off once the residual has
// converged to its final value.

interface RandomGraph {
  readonly nParticles: number;
  readonly positions: readonly (readonly [number, number, number])[];
  readonly pairs: readonly [number, number][];
  readonly restLengths: readonly number[];
}

function buildRandomGraph(seed: number): RandomGraph {
  let rng = seed >>> 0 || 1;
  const rand = (): number => {
    rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
    return rng / 0x100000000;
  };
  const N = 24;
  const positions: [number, number, number][] = [];
  for (let i = 0; i < N; i++) {
    positions.push([rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1]);
  }
  // Build a connected-ish graph: spanning tree + random extra edges. Reject
  // duplicate and self edges.
  const edgeSet = new Set<string>();
  const pairs: [number, number][] = [];
  const key = (a: number, b: number): string => (a < b ? `${a}-${b}` : `${b}-${a}`);
  const pushEdge = (a: number, b: number): void => {
    if (a === b) return;
    const k = key(a, b);
    if (edgeSet.has(k)) return;
    edgeSet.add(k);
    pairs.push([a, b]);
  };
  // Spanning tree (chain) — guarantees connectivity so the residual
  // landscape is non-trivial.
  for (let i = 0; i < N - 1; i++) pushEdge(i, i + 1);
  // Add random extras
  const extra = 20;
  let attempts = 0;
  while (pairs.length < N - 1 + extra && attempts < extra * 20) {
    attempts++;
    pushEdge(Math.floor(rand() * N), Math.floor(rand() * N));
  }

  // Rest lengths are set to a perturbed version of the current separation,
  // so the graph is already slightly stretched — otherwise the initial
  // residual is zero and the test measures nothing.
  const restLengths: number[] = [];
  for (const [i, j] of pairs) {
    const dx = positions[i]![0] - positions[j]![0];
    const dy = positions[i]![1] - positions[j]![1];
    const dz = positions[i]![2] - positions[j]![2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    restLengths.push(len * (0.85 + 0.3 * rand())); // ±15% stretch
  }

  return { nParticles: N, positions, pairs, restLengths };
}

function computeL2Residual(
  pairs: readonly [number, number][],
  restLengths: readonly number[],
  positionsFlat: Float32Array,
): number {
  let sumSq = 0;
  for (let e = 0; e < pairs.length; e++) {
    const [i, j] = pairs[e]!;
    const dx = positionsFlat[i * 4]! - positionsFlat[j * 4]!;
    const dy = positionsFlat[i * 4 + 1]! - positionsFlat[j * 4 + 1]!;
    const dz = positionsFlat[i * 4 + 2]! - positionsFlat[j * 4 + 2]!;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const C = len - restLengths[e]!;
    sumSq += C * C;
  }
  return sumSq;
}

async function measureIterationResiduals(graph: RandomGraph): Promise<{
  residuals: number[];
  numGroups: number;
}> {
  const renderer = await createParticleRenderer();
  try {
    const particles = new ParticleSystem(renderer, graph.nParticles, 0.02);
    const data: ParticleInit[] = graph.positions.map((p) => ({
      position: p,
      velocity: [0, 0, 0],
      invMass: 1,
    }));
    particles.uploadParticles(data);

    // Substep length; only enters the solve through α̃ = α / dt².
    const dt = uniform(1 / 60, 'float');
    const dist = createDistanceConstraints({
      particles,
      pairs: graph.pairs,
      compliance: 1e-6, // very stiff — exercises the solver, not the compliance regularization
      restLength: [...graph.restLengths],
      dt,
    });
    // One solver iteration = one pass over every color group. The λ reset
    // runs once below, NOT between iterations: re-zeroing the multipliers
    // would violate XPBD's accumulate-across-iterations invariant (eq. 13).
    const { solve: oneIteration } = constraintKernels([dist]);

    // `uploadParticles` already wrote predicted = positions on the CPU, but
    // the GPU buffer only exists once a kernel touches it; a trivial copy
    // creates it (and re-seeds it) before the first readback.
    const initPredicted = Fn(() => {
      const i: Any = instanceIndex;
      particles.predictedPositions.element(i).assign(particles.positions.element(i));
    })().compute(graph.nParticles);
    await renderer.computeAsync([initPredicted, dist.resetLambdaKernel]);

    const readPredicted = async (): Promise<Float32Array> => {
      const buf = await renderer.getArrayBufferAsync(particles.predictedPositions.value);
      return new Float32Array(buf);
    };

    const residuals: number[] = [];
    const predicted0 = await readPredicted();
    residuals.push(computeL2Residual(graph.pairs, graph.restLengths, predicted0));

    const maxIter = 10;
    for (let it = 0; it < maxIter; it++) {
      await renderer.computeAsync(oneIteration);
      const pred = await readPredicted();
      residuals.push(computeL2Residual(graph.pairs, graph.restLengths, pred));
    }

    particles.dispose();
    return { residuals, numGroups: dist.groups.length };
  } finally {
    renderer.dispose();
  }
}

describe('XPBD: residual decreases monotonically', () => {
  it('L2 residual non-increasing across iterations on 10 random graphs', async () => {
    const tolerance = 1e-6;
    for (let t = 0; t < 10; t++) {
      const graph = buildRandomGraph(0x51de0000 + t);
      const { residuals, numGroups } = await measureIterationResiduals(graph);

      console.info(
        `[xpbd-residual] trial ${t}: groups=${numGroups} edges=${graph.pairs.length} ` +
          `residuals=[${residuals.map((r) => r.toExponential(2)).join(', ')}]`,
      );

      for (let i = 1; i < residuals.length; i++) {
        const prev = residuals[i - 1]!;
        const curr = residuals[i]!;
        // Allow `curr ≤ prev + |prev|·tolerance` so f32 round-off near
        // convergence doesn't false-fail. The first few iterations should
        // decrease by orders of magnitude, so this tolerance only matters
        // in the tail.
        const allowed = prev + Math.max(Math.abs(prev), 1e-20) * tolerance;
        expect(
          curr,
          `trial ${t} iter ${i}: residual ${curr.toExponential(3)} exceeds prev ${prev.toExponential(3)}`,
        ).toBeLessThanOrEqual(allowed);
      }

      // Sanity: solver must actually be doing something — the final
      // residual should be measurably smaller than the initial.
      expect(residuals.at(-1)!).toBeLessThan(residuals[0]! * 0.5);
    }
  }, 300_000);
});
