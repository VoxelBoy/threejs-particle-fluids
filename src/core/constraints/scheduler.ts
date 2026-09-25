import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type { ConstraintType } from './types.js';

/**
 * Orchestrates the XPBD solve loop around a set of registered constraint
 * types. Phase 04 scope: owns the group-serial / within-group-parallel
 * kernel dispatch order. The outer substep loop and predict/advect live in
 * `SimLoop`; the scheduler is called once per substep with the list of
 * compute nodes to run.
 *
 * The dispatch sequence for a single substep (Macklin 2016 Algorithm 1
 * lines 3–13):
 *   1. `resetLambdaKernel` for every registered constraint type — sets
 *      `λ_0 ← 0` per Algorithm 1 line 4.
 *   2. Outer iteration loop `i = 1 … I`:
 *        For each type in registration order:
 *          For each of the type's groups in group index order:
 *            Dispatch `group.solveKernel`.
 *
 * Group order within a type is deterministic (construction order from
 * {@link colorConstraints}). Type order is the order the caller registered
 * them in. Type registration order matters: types registered earlier have
 * their Δx applied first within an iteration — for MVP (one constraint
 * family per material), this only matters at the fluid-cloth coupling
 * boundary (Phase 20, post-MVP), where the documented convention is fluid-density
 * before cloth-distance (Akinci 2012 + Macklin 2014 §9).
 */
export class ConstraintScheduler {
  private readonly types: ConstraintType[] = [];

  register(type: ConstraintType): void {
    this.types.push(type);
  }

  get registeredTypes(): readonly ConstraintType[] {
    return this.types;
  }

  /**
   * Flatten the per-substep kernel chain into a single ordered array that
   * the caller dispatches in one `renderer.computeAsync([...])` call.
   *
   * Ordering within the returned array:
   *   [resetLambda(t0), resetLambda(t1), …,
   *    iter 1: [solve(t0.g0), solve(t0.g1), …, solve(t1.g0), …],
   *    iter 2: [same],
   *    … up to `iterations` copies.]
   */
  buildSubstepPipeline(iterations: number): ComputeNode[] {
    if (!Number.isInteger(iterations) || iterations <= 0) {
      throw new Error(
        `ConstraintScheduler: iterations must be a positive integer, got ${iterations}`,
      );
    }
    const kernels: ComputeNode[] = [];
    for (const t of this.types) kernels.push(t.resetLambdaKernel);
    for (let i = 0; i < iterations; i++) {
      for (const t of this.types) {
        for (const g of t.groups) kernels.push(g.solveKernel);
      }
    }
    return kernels;
  }
}
