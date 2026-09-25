import { Continue, Fn, If, atomicAdd, float, instanceIndex, min, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

import type { ParticleSystem } from '../particles.js';
import { emitForEachNeighbor } from '../hashGrid/query.js';
import type { HashGrid } from '../hashGrid/HashGrid.js';
import type { ContactBuffer } from './ContactBuffer.js';
import type { FrictionTable } from './FrictionTable.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface BuildContactGenerateArgs {
  readonly particles: ParticleSystem;
  readonly hashGrid: HashGrid;
  readonly contacts: ContactBuffer;
  /**
   * Phase 21 — per-phase-group Coulomb-friction LUT consumed at emit time.
   * The kernel reads `LUT[groupI]` and `LUT[groupJ]`, applies the
   * `min(μ_i, μ_j)` combine rule (paper-allowed per Macklin 2020 §3.5;
   * U-51 resolution), and writes the per-pair `(μ_s, μ_k)` into the
   * claimed `ContactRecord.muS`/`muK` slots. Required.
   */
  readonly frictionTable: FrictionTable;
  /**
   * Multiplicative expansion applied to the contact radius `r = r_i + r_j =
   * 2·particleRadius` during the candidate-distance filter. Paper Macklin
   * 2014 §9 (Implementation Details, p. 10): "we allow expanding the
   * collision radius by a fixed percentage during the overlap checks. This
   * increases the number of potential colliders that are processed so should
   * be set as a small fraction of the particle radius." Defaults to 1.1.
   *
   * The expansion widens the set of *candidate* pairs emitted here. The
   * solve/stabilize kernels still gate the positional projection on the
   * strict `C = |x_ij| - 2r < 0` test (paper eq. 22), so expanding is safe —
   * extra candidates become no-ops in the solve, not spurious corrections.
   */
  readonly radiusExpansion?: number;
}

/**
 * Build the per-particle contact emission TSL kernel.
 *
 *
 *   Algorithm 1 line 7–8: "find neighboring particles N_i(x*_i) … find solid
 *   contacts". We split this into two stages — the hash grid was rebuilt in
 *   the prior step of the substep pipeline on start-of-substep `positions`,
 *   and the candidate filter here queries with `predictedPositions` so the
 *   emitted set matches the post-predict state (paper eq. 22 is evaluated
 *   against `x*`, not `x`).
 *
 *
 *   §9 expanded radius: "we allow expanding the collision radius by a fixed
 *   percentage during the overlap checks." See {@link BuildContactGenerateArgs.radiusExpansion}.
 *
 *
 * Dispatch size: `particles.capacity` — one thread per particle slot, same
 * shape as every other core compute kernel.
 *
 * Finding forward-reference: the "TSL atomic buffer access rule" from Phase
 * 03 applies here — `counter` is declared `atomic<u32>` so `atomicAdd` is
 * the only legal write path. `pairs` is a plain `u32` buffer (no atomic
 * declaration) because coloring guarantees distinct `i` / `j` per slot and
 * slot indices are partitioned by the atomicAdd's return value.
 */
export function buildContactGenerateKernel(args: BuildContactGenerateArgs): ComputeNode {
  const { particles, hashGrid, contacts, frictionTable } = args;
  const radiusExpansion = args.radiusExpansion ?? 1.1;

  if (!Number.isFinite(radiusExpansion) || radiusExpansion < 1) {
    throw new Error(
      `buildContactGenerateKernel: radiusExpansion must be ≥ 1 (got ${radiusExpansion}) — ` +
        `values below 1 would shrink the candidate set below the paper's eq. 22 contact radius.`,
    );
  }

  // Scene-global contact diameter `r = r_i + r_j` for uniform-radius particles
  // (Macklin 2014 §3, p. 2): "We restrict ourselves to a fixed particle
  // radius per scene in order to leverage efficient collision detection
  // based on uniform grids." Squared form for the candidate filter.
  const contactDiameter = 2 * particles.particleRadius;
  const candidateRadius = contactDiameter * radiusExpansion;
  const candidateRadiusSq = candidateRadius * candidateRadius;
  const maxContacts = contacts.maxContacts;

  return Fn(() => {
    const i: Any = instanceIndex;

    // Early-out: kinematic particles with invMass=0 still emit contacts so
    // the other participant (with positive invMass) can resolve penetration
    // against them. Skip only if BOTH participants are kinematic — which we
    // re-check in the neighbor loop to avoid dominating pinned-heavy scenes.
    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const phaseI: Any = particles.phase.element(i).toVar();
    const wi: Any = particles.invMass.element(i).toVar();
    // Self-collision group: high 16 bits of `phase`. Zero = "no self-
    // collision suppression" (MVP default; fluid+cloth scenes).
    const groupI: Any = phaseI.shiftRight(uint(16)).toVar();

    emitForEachNeighbor({
      queryPosXyz: xi,
      hashOrigin: hashGrid.hashOriginUniform,
      cellSize: hashGrid.cellSizeUniform,
      hashTableSize: hashGrid.hashTableSize,
      cellStart: hashGrid.cellStart,
      cellEnd: hashGrid.cellEnd,
      sortedIndices: hashGrid.sortedIndices,
      onCandidate: (j: Any) => {
        // Every `Continue()` below skips THIS candidate and moves to the
        // next particle in the current cell's TSL `Loop`. Using `Return()`
        // here would exit the enclosing `Fn()` entirely, aborting the
        // remaining 26 cells' walk (Phase 03 Finding #3 / Phase 04 Finding
        // #5 — `Return()` inside `If(cond, () => ...)` is an Fn-level
        // exit, not a loop-iteration skip). The first-run Phase 05 bug was
        // exactly this confusion.

        // Undirected-pair dedup: only particle `i` with `i < j` claims the
        // pair. The other side (`j > i`) skips — its partner has already
        // claimed. Also skips self (`j == i`, included by `<=`).
        If(j.lessThanEqual(i), () => {
          Continue();
        });

        const phaseJ: Any = particles.phase.element(j).toVar();
        const groupJ: Any = phaseJ.shiftRight(uint(16)).toVar();
        If(groupI.notEqual(uint(0)).and(groupI.equal(groupJ)), () => {
          Continue();
        });

        // Two kinematic particles have zero degrees of freedom between
        // them; the contact is a no-op and a wasted solve slot.
        const wj: Any = particles.invMass.element(j).toVar();
        If(wi.lessThanEqual(float(0.0)).and(wj.lessThanEqual(float(0.0))), () => {
          Continue();
        });

        // Paper §9 expanded-radius candidate filter. Squared form.
        const xj: Any = particles.predictedPositions.element(j).xyz.toVar();
        const diff: Any = xi.sub(xj).toVar();
        const dsq: Any = diff.dot(diff);
        If(dsq.greaterThanEqual(float(candidateRadiusSq)), () => {
          Continue();
        });

        // Claim a slot. `counter` is atomic<u32>, returning the pre-increment
        // value — the exclusive slot this thread owns.
        const slot: Any = atomicAdd(contacts.counter.element(uint(0)), uint(1));

        // Overflow drop: slots past the cap silently no-op. The atomic still
        // counts past `maxContacts`, so a CPU readback of `counter` after
        // the kernel surfaces how many pairs were lost. Plan §Contact
        // generation — "log a diagnostic … loud failure to silent dropped
        // contacts" is implemented at the CPU-side driver (SimLoop).
        If(slot.lessThan(uint(maxContacts)), () => {
          const rec: Any = contacts.records.element(slot);
          rec.get('i').assign(i);
          rec.get('j').assign(j);
          // Phase 21 — per-friction-group Coulomb friction. The LUT key is
          // the LOW 16 bits of `phase` (decoupled from the high-16-bit
          // self-collision group used by the phase-mask above). This
          // decoupling lets a pile demo set self-collision = 0 (intra-pile
          // contacts emit) while still differentiating friction across
          // groups via the low bits. Scenes that don't differentiate
          // friction leave the low bits at 0 → every pair reads `LUT[0]`,
          // which the FrictionTable seeds with the scalar default.
          //
          // Combine rule: `min` per Macklin 2020 §3.5 (paper-allowed
          // alternative to the arithmetic-mean default; U-51
          // resolved-verified — chosen because the demo motivation requires
          // zero pair-friction at a μ_i=0 surface, which arithmetic mean
          // cannot deliver). Same rule applies to μ_s and μ_k.
          const frictionMask = uint(0xffff);
          const frictionI: Any = phaseI.bitAnd(frictionMask);
          const frictionJ: Any = phaseJ.bitAnd(frictionMask);
          const fI: Any = frictionTable.lut.element(frictionI);
          const fJ: Any = frictionTable.lut.element(frictionJ);
          rec.get('muS').assign(min(fI.x, fJ.x));
          rec.get('muK').assign(min(fI.y, fJ.y));
        });
      },
    });
  })().compute(particles.capacity);
}

// ---------------------------------------------------------------------------
// Phase Perf-14 — ranged dispatch variant.
// ---------------------------------------------------------------------------

export interface BuildContactGenerateRangedArgs {
  readonly particles: ParticleSystem;
  readonly hashGrid: HashGrid;
  readonly contacts: ContactBuffer;
  /**
   * Phase 21 — per-phase-group Coulomb-friction LUT. Same shape and
   * semantics as in {@link BuildContactGenerateArgs.frictionTable}.
   */
  readonly frictionTable: FrictionTable;
  /**
   * Indirection table mapping `instanceIndex` → global particle index for
   * threads that should dispatch. Length equals {@link emittingCount}.
   * Built once at SimLoop construction from the union of
   * `ContactOptions.emittingRanges` (no per-frame writes).
   */
  readonly emittingIndices: StorageBufferNode<'uint'>;
  /**
   * Number of particles that dispatch — sum of `range.count` across
   * `ContactOptions.emittingRanges`. Becomes the kernel's dispatch size.
   */
  readonly emittingCount: number;
  /**
   * Per-particle flag (`u32[capacity]`): 1 if the particle is in the
   * emitting set, 0 otherwise. Used inside the candidate loop to dedup
   * pairs where `j` is also a dispatched thread.
   */
  readonly isEmittingFlag: StorageBufferNode<'uint'>;
  /** See {@link BuildContactGenerateArgs.radiusExpansion}. */
  readonly radiusExpansion?: number;
}

/**
 * Build the per-emitting-particle contact emission TSL kernel — Phase
 * Perf-14 dispatch-shape reduction.
 *
 * Differs from {@link buildContactGenerateKernel} in two ways:
 *   - Dispatch size is `emittingCount` (typically ~600) instead of
 *     `particles.capacity` (typically 50,000+).
 *   - Per-thread `i = emittingIndices[instanceIndex]` (an indirection
 *     through a static u32 table built at SimLoop construction).
 *   - The dedup `j ≤ i ⇒ skip` becomes `(isEmittingFlag[j] == 1 && j ≤ i) ⇒ skip`,
 *     plus an explicit self-skip `j == i ⇒ skip`. When `j ∉ E` (the
 *     emitting set), the partner thread doesn't dispatch, so `i` must
 *     emit the pair unconditionally.
 *
 * Pair-set output is set-identical to the legacy kernel by construction
 * — every `(i, j)` pair survives the same phase-mask, double-kinematic,
 * and distance filters; the dedup change only redistributes which
 * thread emits which pair. The order within `pairs[]` differs because
 * atomic-counter increments interleave differently across the smaller
 * dispatch; downstream solve scatter is order-independent (G4 tier-1).
 *
 *
 * Algorithmic outline (one thread per emitting-set element):
 *   1. `i = emittingIndices[instanceIndex]` — global particle index.
 *   2. Read `xi`, `phaseI`, `wi`, `groupI`. Same as legacy.
 *   3. Walk the 27-cell neighborhood via {@link emitForEachNeighbor}.
 *   4. For each candidate `j`:
 *        a. Skip self (`j == i`).
 *        b. Augmented dedup: skip if `isEmittingFlag[j] == 1 && j < i`
 *           (the lower-index dispatched thread emits the pair).
 *        c. Phase mask, double-kinematic, distance filter — verbatim
 *           from legacy.
 *        d. Atomic emit. Verbatim.
 */
export function buildContactGenerateRangedKernel(
  args: BuildContactGenerateRangedArgs,
): ComputeNode {
  const {
    particles,
    hashGrid,
    contacts,
    frictionTable,
    emittingIndices,
    emittingCount,
    isEmittingFlag,
  } = args;
  const radiusExpansion = args.radiusExpansion ?? 1.1;

  if (!Number.isFinite(radiusExpansion) || radiusExpansion < 1) {
    throw new Error(
      `buildContactGenerateRangedKernel: radiusExpansion must be ≥ 1 (got ${radiusExpansion}) — ` +
        `values below 1 would shrink the candidate set below the paper's eq. 22 contact radius.`,
    );
  }
  if (!Number.isInteger(emittingCount) || emittingCount <= 0) {
    throw new Error(
      `buildContactGenerateRangedKernel: emittingCount must be a positive integer, got ${emittingCount}`,
    );
  }
  if (emittingCount > particles.capacity) {
    throw new Error(
      `buildContactGenerateRangedKernel: emittingCount=${emittingCount} exceeds capacity=${particles.capacity}`,
    );
  }

  const contactDiameter = 2 * particles.particleRadius;
  const candidateRadius = contactDiameter * radiusExpansion;
  const candidateRadiusSq = candidateRadius * candidateRadius;
  const maxContacts = contacts.maxContacts;

  return Fn(() => {
    // Indirection: read the global index from the emit-set table.
    const i: Any = emittingIndices.element(instanceIndex as Any).toVar();

    const xi: Any = particles.predictedPositions.element(i).xyz.toVar();
    const phaseI: Any = particles.phase.element(i).toVar();
    const wi: Any = particles.invMass.element(i).toVar();
    const groupI: Any = phaseI.shiftRight(uint(16)).toVar();

    emitForEachNeighbor({
      queryPosXyz: xi,
      hashOrigin: hashGrid.hashOriginUniform,
      cellSize: hashGrid.cellSizeUniform,
      hashTableSize: hashGrid.hashTableSize,
      cellStart: hashGrid.cellStart,
      cellEnd: hashGrid.cellEnd,
      sortedIndices: hashGrid.sortedIndices,
      onCandidate: (j: Any) => {
        // Self-skip — `j == i` always skipped, regardless of emit-set
        // membership. Without this, a thread would emit `(i, i)` because
        // the legacy `j ≤ i ⇒ skip` previously caught self via the `≤`
        // path; we've split that into a separate self-check + an emit-
        // set-aware dedup below.
        If(j.equal(i), () => {
          Continue();
        });

        // Emit-set-aware dedup. If `j ∈ E`, both `i` and `j` dispatch;
        // only the lower-index thread emits the pair. If `j ∉ E`, `j`
        // doesn't dispatch — the pair is seen only here, so emit
        // unconditionally.
        const jEmits: Any = isEmittingFlag.element(j).equal(uint(1));
        If(jEmits.and(j.lessThan(i)), () => {
          Continue();
        });

        // Phase mask suppression — verbatim from legacy. Symmetric in
        // i, j: the suppression decision is the same regardless of
        // which thread dispatches.
        const phaseJ: Any = particles.phase.element(j).toVar();
        const groupJ: Any = phaseJ.shiftRight(uint(16)).toVar();
        If(groupI.notEqual(uint(0)).and(groupI.equal(groupJ)), () => {
          Continue();
        });

        // Double-kinematic skip — verbatim.
        const wj: Any = particles.invMass.element(j).toVar();
        If(wi.lessThanEqual(float(0.0)).and(wj.lessThanEqual(float(0.0))), () => {
          Continue();
        });

        // Distance filter (paper §9 expanded radius) — verbatim.
        const xj: Any = particles.predictedPositions.element(j).xyz.toVar();
        const diff: Any = xi.sub(xj).toVar();
        const dsq: Any = diff.dot(diff);
        If(dsq.greaterThanEqual(float(candidateRadiusSq)), () => {
          Continue();
        });

        // Atomic emit — verbatim.
        const slot: Any = atomicAdd(contacts.counter.element(uint(0)), uint(1));
        If(slot.lessThan(uint(maxContacts)), () => {
          const rec: Any = contacts.records.element(slot);
          rec.get('i').assign(i);
          rec.get('j').assign(j);
          // Phase 21 — friction LUT keyed on LOW 16 bits of phase (see
          // legacy kernel for the full rationale; same behavior here).
          const frictionMask = uint(0xffff);
          const frictionI: Any = phaseI.bitAnd(frictionMask);
          const frictionJ: Any = phaseJ.bitAnd(frictionMask);
          const fI: Any = frictionTable.lut.element(frictionI);
          const fJ: Any = frictionTable.lut.element(frictionJ);
          rec.get('muS').assign(min(fI.x, fJ.x));
          rec.get('muK').assign(min(fI.y, fJ.y));
        });
      },
    });
  })().compute(emittingCount);
}
