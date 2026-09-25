import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

/**
 * One color-class of a constraint graph partitioned by {@link colorConstraints}.
 *
 * Within a group no two constraints share a participating particle, so the
 * group's {@link solveKernel} can be dispatched in parallel over all
 * `capacity` particles with no write conflict on `predictedPositions` and no
 * atomics — a particle thread is ever only a participant in at most one
 * constraint of the group.
 *
 * `particleToConstraint` is the "inverted index" required by plan §Solve
 * mode: for particle `p`, `particleToConstraint[p]` is either (a) the
 * constraint index (into the owning {@link ConstraintType}'s per-constraint
 * storage) that `p` participates in for this group, or (b) `NO_CONSTRAINT`
 * (= 0xFFFFFFFF) if `p` participates in no constraint of this group.
 */
export interface ConstraintGroup {
  readonly particleToConstraint: StorageBufferNode<'uint'>;
  readonly solveKernel: ComputeNode;
}

/**
 * Sentinel for "this particle participates in no constraint of this group."
 * Stored in {@link ConstraintGroup.particleToConstraint}; solve kernels
 * compare against this before indexing into the constraint-data buffers.
 */
export const NO_CONSTRAINT = 0xffffffff;

/**
 * A registered constraint family (distance, bending, contact, density, …).
 *
 * Storage layout (SoA, per §Constraint interface of plan):
 *   `particleIndices[arity·nConstraints]` — flat `uint` array; participant
 *     `k` of constraint `c` lives at index `c·arity + k`.
 *   `compliance[nConstraints]`  — `α`, inverse stiffness, in s²/kg for
 *     distance (units match the particular constraint family).
 *   `restValue[nConstraints]`   — rest length / rest angle / offset — the
 *     constant term of `C(x)`. Family-specific interpretation.
 *   `lambda[nConstraints]`      — Macklin 2016 "total Lagrange multiplier";
 *     reset to 0 each substep (Algorithm 1 line 4), accumulated across
 *     solver iterations via `λ_{i+1} = λ_i + Δλ` (eq. 13).
 *
 * `resetLambdaKernel` is dispatched once per substep over `nConstraints`
 * slots before the first iteration. The scheduler owns when to call it.
 */
export interface ConstraintType {
  readonly arity: number;
  readonly nConstraints: number;
  readonly particleIndices: StorageBufferNode<'uint'>;
  readonly compliance: StorageBufferNode<'float'>;
  readonly restValue: StorageBufferNode<'float'>;
  readonly lambda: StorageBufferNode<'float'>;
  readonly groups: readonly ConstraintGroup[];
  readonly resetLambdaKernel: ComputeNode;
}

/**
 * Greedy graph coloring — partitions a set of constraints into groups such
 * that no two constraints in the same group share a particle.
 *
 * Input: flat `participantsPerConstraint[c·arity + k]` = particle index for
 * participant `k` of constraint `c`, plus `arity` and `nConstraints`.
 *
 * Output: `groupOf[c]` = color index assigned to constraint `c` (0-indexed),
 * plus `numGroups` = one past the largest color used.
 *
 * The greedy algorithm is O(nConstraints · arity · maxDegree) worst-case; for
 * structured topologies (cloth lattice, chains) it produces an optimal or
 * near-optimal coloring. Contact (Phase 05) rebuilds the coloring every
 * substep — ok because contact count is O(n) and the greedy pass is single-
 * digit microseconds for our target scene sizes.
 *
 * Guarantee: `numGroups ≤ 1 + maxDegree` where `maxDegree` is the largest
 * number of constraints any single particle participates in. Tested by
 * `tests/analytical/xpbd-group-partition.test.ts`.
 */
export function colorConstraints(args: {
  readonly arity: number;
  readonly nConstraints: number;
  readonly participantsPerConstraint: readonly number[] | Uint32Array;
}): { groupOf: Uint32Array; numGroups: number } {
  const { arity, nConstraints, participantsPerConstraint } = args;
  if (nConstraints === 0) return { groupOf: new Uint32Array(0), numGroups: 0 };

  const groupOf = new Uint32Array(nConstraints);
  // `particleLastGroupTouched[p]` is a per-color sparse map kept as
  // a flat `Map<particle, Set<group>>`. For the MVP scene sizes this is fine;
  // a bit-packed array can replace it if Phase 18 (cloth, post-MVP) measurement demands it.
  const particleGroups: Map<number, Set<number>> = new Map();

  let numGroups = 0;
  for (let c = 0; c < nConstraints; c++) {
    let g = 0;
    // Find the lowest-indexed group that conflicts with none of this
    // constraint's participants.
    while (true) {
      let conflict = false;
      for (let k = 0; k < arity; k++) {
        const p = participantsPerConstraint[c * arity + k]!;
        const owned = particleGroups.get(p);
        if (owned !== undefined && owned.has(g)) {
          conflict = true;
          break;
        }
      }
      if (!conflict) break;
      g++;
    }
    groupOf[c] = g;
    if (g + 1 > numGroups) numGroups = g + 1;
    for (let k = 0; k < arity; k++) {
      const p = participantsPerConstraint[c * arity + k]!;
      let owned = particleGroups.get(p);
      if (owned === undefined) {
        owned = new Set();
        particleGroups.set(p, owned);
      }
      owned.add(g);
    }
  }

  return { groupOf, numGroups };
}
