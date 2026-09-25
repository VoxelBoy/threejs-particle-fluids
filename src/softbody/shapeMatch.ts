import {
  Fn,
  If,
  Loop,
  float,
  instanceIndex,
  localId,
  uint,
  vec3,
  vec4,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { ParticleSystem, XpbdUniforms } from '../core/index.js';

import { emitPolarDecomposition, type Mat3Nodes } from './polarDecomp.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Workgroup size for per-body reductions. Picked 2026-04-24 at Phase 10
 * landing on the basis of MVP body sizes (Macklin 2014 Figure 9 bunnies:
 * 44 particles; plan §"Particle representation": 50–500 particles per
 * body). 256 covers the common case in a single stride and is a power of
 * two so the tree reduction halves cleanly. Larger bodies are handled by
 * the stride loop inside the kernel.
 *
 * Not exported — changing this value needs a paired tree-reduction audit.
 */
const SOFTBODY_WORKGROUP_SIZE = 256;

export interface BuildCenterOfMassKernelArgs {
  readonly particles: ParticleSystem;
  /** Per-body starting index into `particles.predictedPositions`. */
  readonly bodyStart: StorageBufferNode<'uint'>;
  /** Per-body particle count. */
  readonly bodyCount: StorageBufferNode<'uint'>;
  /**
   * Per-body output: current center of mass, `xyz` valid, `w` unused.
   * Written once per dispatch (once per substep).
   */
  readonly bodyCenters: StorageBufferNode<'vec4'>;
  /** Number of softbodies — sets the dispatch's workgroup count. */
  readonly numBodies: number;
}

/**
 * Build the Pass 1 center-of-mass kernel (Mueller 2011 eq. 2, uniform mass).
 *
 *
 * Algorithmic outline (one workgroup per body; `SOFTBODY_WORKGROUP_SIZE`
 * threads per workgroup):
 *   1. Thread `tid` sums a stride of the body's particles from its
 *      `predictedPositions`. Stride is `W = SOFTBODY_WORKGROUP_SIZE` so
 *      bodies larger than W require multiple stride iterations; bodies
 *      smaller than W leave the trailing threads with zero partial sums.
 *   2. Each thread writes its partial sum into the workgroup-shared
 *      accumulator `shared[tid]` and barriers.
 *   3. Tree reduction halves the active range each round, halting when
 *      thread 0 holds the full body sum.
 *   4. Thread 0 divides by the body's particle count and writes the final
 *      centre `c = (1/N) Σ x*_i` into `bodyCenters[bodyIndex]`.
 *
 *
 * Determinism (ARCH §Guardrails G4): the workgroup-shared tree reduction
 * produces an order-independent sum per body (binary tree is associative
 * and commutative under f32 only approximately — tier 2 bounded-max-error
 * applies, with tolerance `N_body · |x*_max| · ~1e-7`). The divide-by-N
 * is a per-body scalar op, tier 1 on top of the tier 2 sum.
 */
export function buildCenterOfMassKernel(args: BuildCenterOfMassKernelArgs): ComputeNode {
  const { particles, bodyStart, bodyCount, bodyCenters, numBodies } = args;
  if (!Number.isInteger(numBodies) || numBodies <= 0) {
    throw new Error(
      `buildCenterOfMassKernel: numBodies must be a positive integer, got ${numBodies}`,
    );
  }

  const W = SOFTBODY_WORKGROUP_SIZE;

  return Fn(() => {
    const tid: Any = localId.x;
    const bi: Any = workgroupId.x;
    const start: Any = bodyStart.element(bi).toVar();
    const count: Any = bodyCount.element(bi).toVar();

    // Per-thread partial sum over a stride of the body's particles.
    const localSum: Any = vec3(0.0, 0.0, 0.0).toVar();

    // Number of strides = ceil(count / W). Integer div with manual round-up.
    const numStrides: Any = count.add(uint(W - 1)).div(uint(W));
    Loop(
      { start: uint(0), end: numStrides, type: 'uint', condition: '<' },
      ({ i: stride }: { i: Any }) => {
        const localIdx: Any = stride.mul(uint(W)).add(tid).toVar();
        If(localIdx.lessThan(count), () => {
          const globalIdx: Any = start.add(localIdx);
          const xi: Any = particles.predictedPositions.element(globalIdx).xyz;
          localSum.addAssign(xi);
        });
      },
    );

    // Workgroup-shared tree reduction. `vec3` has 16-byte stride in WGSL
    // workgroup memory so it's effectively `vec4` storage — we only
    // read/write the xyz lanes.
    const shared: Any = workgroupArray('vec3', W);
    shared.element(tid).assign(localSum);
    workgroupBarrier();

    // Halve the active range each round. `for` loop runs at shader-build
    // time (TSL unrolls); the `If` inside is the per-thread gate.
    for (let s = W >> 1; s > 0; s >>= 1) {
      const ss = s;
      If(tid.lessThan(uint(ss)), () => {
        const other: Any = shared.element(tid.add(uint(ss)));
        shared.element(tid).assign(shared.element(tid).add(other));
      });
      workgroupBarrier();
    }

    // Thread 0 writes the body centre.
    If(tid.equal(uint(0)), () => {
      const totalSum: Any = shared.element(uint(0));
      const countF: Any = float(count);
      const c: Any = totalSum.div(countF);
      bodyCenters.element(bi).assign(vec4(c.x, c.y, c.z, float(0.0)));
    });
  })().compute(numBodies * W, [W]);
}

export interface BuildMomentAndPolarDecompKernelArgs {
  readonly particles: ParticleSystem;
  /** Pre-centered rest offsets `r_i` (per particle, vec3 + pad). */
  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly bodyStart: StorageBufferNode<'uint'>;
  readonly bodyCount: StorageBufferNode<'uint'>;
  /**
   * Per-body output: rotation `R` from the polar decomposition. Stored
   * row-major as three contiguous `vec4` slots per body —
   * `bodyRotations[3·b + row].xyz` holds row `row` of `R_b`, `w` unused.
   * Total length = `3 · numBodies`.
   */
  readonly bodyRotations: StorageBufferNode<'vec4'>;
  readonly numBodies: number;
}

/**
 * Build the Pass 2 moment-matrix + polar-decomposition kernel.
 *
 *
 * Algorithmic outline (one workgroup per body; `SOFTBODY_WORKGROUP_SIZE`
 * threads per workgroup):
 *   1. Each thread stride-loops over its share of the body's particles,
 *      accumulating three `vec3` row sums of the outer product
 *      `x*_i · r_i^T` into local registers.
 *   2. Three workgroup-shared `vec3` arrays (`sharedRow0/1/2`) collect
 *      the per-thread partial rows. Barrier.
 *   3. Tree reduction halves the active range each round across all
 *      three rows in lock-step.
 *   4. Thread 0 packs the reduced rows into a `Mat3Nodes`, calls
 *      {@link emitPolarDecomposition}, and writes the resulting
 *      rotation `R` row-major into `bodyRotations[3·b + 0..2].xyz`.
 *
 * Cadence: once per substep via {@link SoftbodySystem.preIterKernels}
 * (U-35 resolved-assumed 2026-04-24 — per-substep R + c are frozen for
 * the whole iter loop; the Pass 3 apply kernel reads them).
 *
 * Determinism (ARCH §Guardrails G4): the workgroup-shared tree
 * reduction of 9-component f32 sums per body is order-dependent; Tier 2
 * bounded-max-error applies with the same ULP-envelope as Pass 1. The
 * 3×3 polar decomposition is deterministic given A_pq.
 */
export function buildMomentAndPolarDecompKernel(
  args: BuildMomentAndPolarDecompKernelArgs,
): ComputeNode {
  const { particles, restOffsets, bodyStart, bodyCount, bodyRotations, numBodies } = args;
  if (!Number.isInteger(numBodies) || numBodies <= 0) {
    throw new Error(
      `buildMomentAndPolarDecompKernel: numBodies must be a positive integer, got ${numBodies}`,
    );
  }

  const W = SOFTBODY_WORKGROUP_SIZE;

  return Fn(() => {
    const tid: Any = localId.x;
    const bi: Any = workgroupId.x;
    const start: Any = bodyStart.element(bi).toVar();
    const count: Any = bodyCount.element(bi).toVar();

    // Per-thread partial row sums of A_pq = Σ x*_i · r_i^T.
    //   row0_i = x*_i.x · r_i,    row1_i = x*_i.y · r_i,    row2_i = x*_i.z · r_i
    const localRow0: Any = vec3(0.0, 0.0, 0.0).toVar();
    const localRow1: Any = vec3(0.0, 0.0, 0.0).toVar();
    const localRow2: Any = vec3(0.0, 0.0, 0.0).toVar();

    const numStrides: Any = count.add(uint(W - 1)).div(uint(W));
    Loop(
      { start: uint(0), end: numStrides, type: 'uint', condition: '<' },
      ({ i: stride }: { i: Any }) => {
        const localIdx: Any = stride.mul(uint(W)).add(tid).toVar();
        If(localIdx.lessThan(count), () => {
          const globalIdx: Any = start.add(localIdx);
          const xi: Any = particles.predictedPositions.element(globalIdx).xyz;
          const ri: Any = restOffsets.element(globalIdx).xyz;
          localRow0.addAssign(ri.mul(xi.x));
          localRow1.addAssign(ri.mul(xi.y));
          localRow2.addAssign(ri.mul(xi.z));
        });
      },
    );

    // Workgroup-shared accumulators (vec3 has 16-byte stride in WGSL
    // workgroup memory: 3 · 16 · W = 12 KB at W=256, under the 16 KB
    // maxComputeWorkgroupStorageSize WebGPU default).
    const sharedRow0: Any = workgroupArray('vec3', W);
    const sharedRow1: Any = workgroupArray('vec3', W);
    const sharedRow2: Any = workgroupArray('vec3', W);
    sharedRow0.element(tid).assign(localRow0);
    sharedRow1.element(tid).assign(localRow1);
    sharedRow2.element(tid).assign(localRow2);
    workgroupBarrier();

    for (let s = W >> 1; s > 0; s >>= 1) {
      const ss = s;
      If(tid.lessThan(uint(ss)), () => {
        sharedRow0
          .element(tid)
          .assign(sharedRow0.element(tid).add(sharedRow0.element(tid.add(uint(ss)))));
        sharedRow1
          .element(tid)
          .assign(sharedRow1.element(tid).add(sharedRow1.element(tid.add(uint(ss)))));
        sharedRow2
          .element(tid)
          .assign(sharedRow2.element(tid).add(sharedRow2.element(tid.add(uint(ss)))));
      });
      workgroupBarrier();
    }

    // Thread 0 extracts A_pq, runs the polar decomposition, and writes R.
    If(tid.equal(uint(0)), () => {
      const row0: Any = sharedRow0.element(uint(0));
      const row1: Any = sharedRow1.element(uint(0));
      const row2: Any = sharedRow2.element(uint(0));
      const aPq: Mat3Nodes = {
        m00: row0.x,
        m01: row0.y,
        m02: row0.z,
        m10: row1.x,
        m11: row1.y,
        m12: row1.z,
        m20: row2.x,
        m21: row2.y,
        m22: row2.z,
      };
      const R: Mat3Nodes = emitPolarDecomposition(aPq);

      const baseSlot: Any = bi.mul(uint(3));
      bodyRotations.element(baseSlot.add(uint(0))).assign(vec4(R.m00, R.m01, R.m02, float(0.0)));
      bodyRotations.element(baseSlot.add(uint(1))).assign(vec4(R.m10, R.m11, R.m12, float(0.0)));
      bodyRotations.element(baseSlot.add(uint(2))).assign(vec4(R.m20, R.m21, R.m22, float(0.0)));
    });
  })().compute(numBodies * W, [W]);
}

export interface BuildResetLambdaKernelArgs {
  readonly particles: ParticleSystem;
  readonly lambda: StorageBufferNode<'vec4'>;
}

/**
 * Zero the per-particle shape-matching Lagrange multipliers at the start
 * of each substep. Paper reference: Macklin 2016 "XPBD" Algorithm 1
 * line 4 — `λ_0 ← 0` once per substep, then accumulate Δλ across solver
 * iterations within the substep.
 *
 * Sized to `particles.capacity` (matches the λ buffer's indexing with
 * every other per-particle core buffer); non-softbody slots are written
 * to zero harmlessly.
 */
export function buildResetLambdaKernel(args: BuildResetLambdaKernelArgs): ComputeNode {
  const { particles, lambda } = args;
  return Fn(() => {
    const i: Any = instanceIndex;
    lambda.element(i).assign(vec4(0.0, 0.0, 0.0, 0.0));
  })().compute(particles.capacity);
}

export interface BuildShapeMatchDeltaApplyKernelArgs {
  readonly particles: ParticleSystem;
  /** Pre-centered rest offsets `r_i` (per particle, vec3 + pad). */
  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly bodyStart: StorageBufferNode<'uint'>;
  readonly bodyCount: StorageBufferNode<'uint'>;
  /** Per-body centre `c` — written by Pass 1 earlier in the substep. */
  readonly bodyCenters: StorageBufferNode<'vec4'>;
  /** Per-body rotation `R` — written by Pass 2 earlier in the substep. */
  readonly bodyRotations: StorageBufferNode<'vec4'>;
  /**
   * Per-body XPBD compliance `α` in s²/kg. Kernel derives
   * `α̃ = α / dt²` per dispatch.
   */
  readonly bodyCompliance: StorageBufferNode<'float'>;
  /**
   * Per-particle Lagrange multiplier `λ_i` as a 3-vector (xyz used, w
   * padding). Reset to zero at the start of each substep, accumulates
   * Δλ across the iter loop.
   */
  readonly lambda: StorageBufferNode<'vec4'>;
  /** XPBD uniforms (substep `dt`), shared with {@link SimLoop.xpbd}. */
  readonly xpbd: XpbdUniforms;
  readonly numBodies: number;
}

/**
 * Build the Pass 3 per-particle Δx apply kernel.
 *
 *
 * Dispatch shape: one workgroup per body (`workgroupId.x = body index`),
 * `SOFTBODY_WORKGROUP_SIZE` threads per workgroup, stride loop over
 * bodies whose particle counts exceed the workgroup size. Threads read
 * the substep-frozen `(R, c)` for their body (uniform across the
 * workgroup — each thread reads independently; hardware-level broadcast
 * or scalar-cache elides the redundancy).
 *
 * Apply path: the kernel writes the Δx correction **directly to
 * `predictedPositions`** in-place. Shape matching has no cross-particle
 * dependency — each particle's goal and Δx are functions of its own
 * rest offset, the body's `(R, c)`, and the particle's own `(x*, λ)` —
 * so scatter/atomic is unnecessary and the per-thread-per-slot write
 * is safe. This mirrors `FluidSystem`'s direct `predictedPositions`
 * update in its perIter apply kernel (`buildApplyDeltaKernel`);
 * downstream contact/collider scatter kernels in the same iter pick up
 * the shape-matched state via the accumulator-apply step that runs
 * after all materials but before the next iter begins.
 *
 * Cadence: once per solver iteration via {@link
 * SoftbodySystem.perIterKernels}. Standard XPBD small-steps pattern —
 * accumulate Δλ into `λ` across iters within a substep, reset `λ = 0`
 * once per substep at the start (see {@link buildResetLambdaKernel}).
 *
 * Determinism (ARCH §Guardrails G4): per-particle in-place write is
 * Tier 1 bit-exact — no cross-thread reduction, no atomic scheduling
 * order. The substep-frozen `(R, c)` inputs are Tier 2 bounded (they
 * come from Pass 1 + Pass 2's f32 reductions) but that propagates
 * through rather than introducing new order-dependence.
 */
export function buildShapeMatchDeltaApplyKernel(
  args: BuildShapeMatchDeltaApplyKernelArgs,
): ComputeNode {
  const {
    particles,
    restOffsets,
    bodyStart,
    bodyCount,
    bodyCenters,
    bodyRotations,
    bodyCompliance,
    lambda,
    xpbd,
    numBodies,
  } = args;
  if (!Number.isInteger(numBodies) || numBodies <= 0) {
    throw new Error(
      `buildShapeMatchDeltaApplyKernel: numBodies must be a positive integer, got ${numBodies}`,
    );
  }

  const W = SOFTBODY_WORKGROUP_SIZE;
  const dt: UniformNode<'float', number> = xpbd.dt;

  return Fn(() => {
    const tid: Any = localId.x;
    const bi: Any = workgroupId.x;
    const start: Any = bodyStart.element(bi).toVar();
    const count: Any = bodyCount.element(bi).toVar();

    // Read (R, c, α) for this body. Uniform across the workgroup.
    const baseRotSlot: Any = bi.mul(uint(3));
    const R0: Any = bodyRotations.element(baseRotSlot).xyz.toVar();
    const R1: Any = bodyRotations.element(baseRotSlot.add(uint(1))).xyz.toVar();
    const R2: Any = bodyRotations.element(baseRotSlot.add(uint(2))).xyz.toVar();
    const c: Any = bodyCenters.element(bi).xyz.toVar();
    const alpha: Any = bodyCompliance.element(bi).toVar();
    // α̃ = α / dt². The plan's §"XPBD adaptation" derivation treats
    // `matchCompliance` as the artist-facing α (s²/kg); this per-
    // dispatch scalar divide is the standard Macklin 2016 mapping.
    const alphaTilde: Any = alpha.div((dt as Any).mul(dt as Any));

    const numStrides: Any = count.add(uint(W - 1)).div(uint(W));
    Loop(
      { start: uint(0), end: numStrides, type: 'uint', condition: '<' },
      ({ i: stride }: { i: Any }) => {
        const localIdx: Any = stride.mul(uint(W)).add(tid).toVar();
        If(localIdx.lessThan(count), () => {
          const globalIdx: Any = start.add(localIdx);
          const ri: Any = restOffsets.element(globalIdx).xyz.toVar();

          // goal = R · r + c  (row-major R → three dots against r)
          const goal: Any = vec3(R0.dot(ri), R1.dot(ri), R2.dot(ri)).add(c).toVar();

          const xStar: Any = particles.predictedPositions.element(globalIdx).xyz.toVar();

          // Vector constraint with identity Jacobian:
          //   C_ij = x*_ij − goal_ij
          //   Δλ_ij = (−C_ij − α̃ · λ_ij) / (w_i + α̃)
          //   Δx_ij = w_i · Δλ_ij
          const C: Any = xStar.sub(goal);
          const lambdaOld: Any = lambda.element(globalIdx).xyz.toVar();
          const w: Any = particles.invMass.element(globalIdx).toVar();
          const denom: Any = w.add(alphaTilde);

          const deltaLambda: Any = C.negate().sub(lambdaOld.mul(alphaTilde)).div(denom);
          const newLambda: Any = lambdaOld.add(deltaLambda);
          const deltaX: Any = deltaLambda.mul(w);

          lambda.element(globalIdx).assign(vec4(newLambda.x, newLambda.y, newLambda.z, float(0.0)));
          particles.predictedPositions
            .element(globalIdx)
            .assign(vec4(xStar.add(deltaX), float(0.0)));
        });
      },
    );
  })().compute(numBodies * W, [W]);
}

export { SOFTBODY_WORKGROUP_SIZE };
