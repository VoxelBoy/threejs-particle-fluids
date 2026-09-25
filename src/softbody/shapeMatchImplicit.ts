import { Fn, If, Loop, atomicAdd, float, instanceIndex, int, uint, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { ContactAccumulator, ParticleSystem, XpbdUniforms } from '../core/index.js';

import { emitPolarDecomposition, type Mat3Nodes } from './polarDecomp.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 *
 *
 * The four passes below replace the §5.3 explicit kernel set when
 * `SoftbodySystem` is constructed with `shapeMatchMode: 'implicit'`.
 * Each pass dispatches one thread per particle in the system; the
 * neighborhood walk reads a per-particle CSR built at construction from
 * the voxel-grid 6-face edge graph (F-12.2). Per-particle output buffers
 * mean Phase 13's mesh skinner can consume `R_i` with no shape-match-mode
 * branch (plan §"Coexistence with §5.3").
 *
 * Paper-fidelity findings encoded inline:
 *   F-12.1 — Eq. 7's `Aᵢ = (1/5)·m·r²·Rᵢ` is mandatory; without it the
 *     kernel goes singular for the single-particle group case (Pass 2).
 *   F-12.3 — `qp_i` is replaced by the optimal rotation in Pass 4. The
 *     §4.1 epilogue dispatched by SimLoop reads this `qp_i` and writes
 *     `ω_i` via the Eq. 14 finite-difference; this kernel does not write
 *     ω directly.
 *   F-12.6 — Δx applies to all `j ∈ N(i)`, but the orientation update
 *     applies only to particle `i` (the group center).
 *
 * Mass treatment (MVP simplification): with uniform per-particle mass the
 * `m·` factor scales `A_pq_i` by a positive scalar, which the polar
 * decomposition is invariant under. The `Aᵢ` term and the Σ term are both
 * linear in `m`, so `m` factors out of the eigen-decomposition; this
 * kernel computes the `m`-cancelled form and accepts `(r²/5)` as a uniform.
 * Heterogeneous mass within one body is out of MVP scope.
 *
 * Per-pair Lagrange multiplier: each (i, j) constraint where particle `j`
 * is in particle `i`'s group carries its own `λ_{i,j}` (Macklin 2016 §4
 * eq. 18 with identity Jacobian, applied per CSR entry). Per-particle
 * λ would race when multiple groups touch the same particle's slot;
 * per-pair λ is the smallest correct unit. Indexed by CSR entry `k`, so
 * the buffer is sized `totalDegree` (sum of `|N(i)|` over particles).
 */

export interface BuildImplicitNeighborhoodCenterKernelArgs {
  readonly particles: ParticleSystem;
  /**
   * CSR offsets into {@link neighborIndices}; length `capacity + 1`.
   * `neighborOffsets[i+1] − neighborOffsets[i]` is `|N(i)|` (= 0 for
   * particles that are not part of any implicit-mode body, in which case
   * `c_i` is left zero — Pass 2's `Aᵢ` term still produces a valid R for
   * those slots, but the per-particle output is unused).
   */
  readonly neighborOffsets: StorageBufferNode<'uint'>;
  /**
   * Flat global particle indices for every CSR entry, including each
   * particle's own self-entry. Length = `totalDegree`.
   */
  readonly neighborIndices: StorageBufferNode<'uint'>;
  /**
   * Output: per-particle current-frame neighborhood centroid `c_i`. Sized
   * `capacity`; `xyz` valid, `w` unused.
   */
  readonly particleCenters: StorageBufferNode<'vec4'>;
}

/**
 * Pass 1 — per-particle neighborhood centre `c_i = (1/|N(i)|) Σ x*_j`.
 *
 * Uniform-mass simplification of plan §"Pass 1": with `m_j = m` for all
 * particles in the body, `c_i = (Σ m_j x*_j) / (Σ m_j)` collapses to the
 * arithmetic mean. Particles with `|N(i)| = 0` (slots outside any
 * implicit-mode body) skip the divide and write zero.
 *
 * Cadence: once per substep via {@link SoftbodySystem.preIterKernels} —
 * the same per-substep cadence as §5.3 Pass 1 (U-35 transferred to §5.1
 * per plan §"XPBD cadence"). Per-iter recomputation would dispatch I
 * times per substep without changing the qualitative behavior; the
 * I-independence test in `tests/analytical/softbody/` validates this.
 *
 * Determinism: per-particle thread-local sum, no atomics or workgroup-
 * shared memory. Tier 1 bit-exact (ARCH §Guardrails G4) — the per-thread
 * sum is determined by the CSR walk order which is fixed at construction.
 */
export function buildImplicitNeighborhoodCenterKernel(
  args: BuildImplicitNeighborhoodCenterKernelArgs,
): ComputeNode {
  const { particles, neighborOffsets, neighborIndices, particleCenters } = args;

  return Fn(() => {
    const i: Any = instanceIndex;
    const start: Any = neighborOffsets.element(i).toVar();
    const end: Any = neighborOffsets.element(i.add(uint(1))).toVar();
    const count: Any = end.sub(start).toVar();

    const sum: Any = vec3(0.0, 0.0, 0.0).toVar();
    Loop({ start: start, end: end, type: 'uint', condition: '<' }, ({ i: k }: { i: Any }) => {
      const j: Any = neighborIndices.element(k);
      sum.addAssign(particles.predictedPositions.element(j).xyz);
    });
    const safeCount: Any = count.max(uint(1));
    const inv: Any = float(1.0).div(safeCount.toFloat());
    const c: Any = sum.mul(inv);
    const isEmpty: Any = count.equal(uint(0));
    const cx: Any = isEmpty.select(float(0.0), c.x);
    const cy: Any = isEmpty.select(float(0.0), c.y);
    const cz: Any = isEmpty.select(float(0.0), c.z);
    particleCenters.element(i).assign(vec4(cx, cy, cz, float(0.0)));
  })().compute(particles.capacity);
}

export interface BuildImplicitMomentPolarKernelArgs {
  readonly particles: ParticleSystem;
  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly neighborOffsets: StorageBufferNode<'uint'>;
  readonly neighborIndices: StorageBufferNode<'uint'>;
  readonly particleCenters: StorageBufferNode<'vec4'>;
  /**
   * Per-particle rest-frame neighborhood centroid `c̄_i = (1/|N(i)|) Σ
   * x̃_j`, computed CPU-side at SoftbodySystem construction (the rest
   * graph is static). Sized `capacity`; `xyz` valid, `w` unused.
   */
  readonly restNeighborhoodCenters: StorageBufferNode<'vec4'>;
  /**
   * Output: per-particle rotation `R_i` row-major (3 vec4 per particle,
   * `xyz` of each is one row, `w` unused). Sized `3 · capacity`.
   */
  readonly particleRotations: StorageBufferNode<'vec4'>;
  /**
   * Constant `r²/5` uniform — the scalar coefficient of the `Aᵢ` term
   * (Eq. 8 with sphere inertia `I = (2/5)·m·r²`, mass cancelled). Equals
   * `particleRadius² / 5` at scene-global radius.
   */
  readonly aiScalar: UniformNode<'float', number>;
}

/**
 * Pass 2 — per-particle moment matrix `A_pq_i` and polar decomposition.
 *
 * Eqs. 7 + 8 (mass-cancelled, uniform-mass MVP):
 * ```
 *   A_pq_i = (r²/5) · Σ_{j ∈ N(i)} R_j_prev
 *          + Σ_{j ∈ N(i)} (x*_j − c_i) · (x̃_j − c̄_i)^T          (Eq. 7)
 *   R_i    = polarDecomp(A_pq_i)                                  (Eq. 8)
 * ```
 *
 * The `(r²/5) · Σ R_j_prev` term sums each neighbor's own previous
 * rotation matrix, NOT just the centre particle's. Mueller 2011 Eq. 7
 * is explicit: `A_pq = Σ_i (A_i + m_i · x_i · x̃_i^T) − M c c̄^T` —
 * the index runs over every particle in the group, and each `A_i` uses
 * that particle's own rotation. Including only the centre (an earlier
 * implementation bug) under-regularizes the polar decomp by a factor
 * of |N(i)| ~7×; for uniformly-oriented bodies this is invisible because
 * the polar decomp is scale-invariant on the diagonal, but for bodies
 * where particles have accumulated different rotation history the
 * missing per-neighbor `R_j_prev` lets the polar decomp pick up noise
 * from the outer-product sum and the body drifts / explodes.
 *
 * Reading `R_prev` from the substep-start `rotation[j]` (NOT
 * `predictedRotation`) is the F-12.1 fidelity point — `A_j` represents
 * the particle's own rotational state at substep entry, the contribution
 * that keeps the kernel non-singular for chains and single-particle
 * groups. Reading from `predictedRotation` would feed back the post-
 * Pass-4 result into the next iter's reduction and break the once-per-
 * substep semantics of `A_j`.
 *
 * The polar decomposition is shared with §5.3's Pass 2 via
 * {@link emitPolarDecomposition}; the only difference is the per-particle
 * dispatch shape and the additional `Aᵢ` initialisation.
 *
 * Cadence: once per substep via {@link SoftbodySystem.preIterKernels}
 * (U-35 transferred to §5.1; I-independence test re-verifies).
 *
 * Determinism: per-particle thread-local 9-component reduction with no
 * cross-thread reads. Tier 2 bounded-max-error (f32 sums over `N(i)`),
 * with the same ULP envelope as §5.3 Pass 2 — `|N(i)| ≤ 7` for voxel-grid
 * 6-face neighborhoods, so the bound is two orders of magnitude tighter
 * than §5.3's per-body sums.
 */
export function buildImplicitMomentPolarKernel(
  args: BuildImplicitMomentPolarKernelArgs,
): ComputeNode {
  const {
    particles,
    restOffsets,
    neighborOffsets,
    neighborIndices,
    particleCenters,
    restNeighborhoodCenters,
    particleRotations,
    aiScalar,
  } = args;

  return Fn(() => {
    const i: Any = instanceIndex;
    const start: Any = neighborOffsets.element(i).toVar();
    const end: Any = neighborOffsets.element(i.add(uint(1))).toVar();
    const count: Any = end.sub(start).toVar();

    // A starts at zero — both the per-neighbor `(r²/5)·R_j_prev` sum AND
    // the outer-product sum accumulate inside the Loop below. Eq. 7
    // explicitly sums over EVERY particle in the group, with each
    // particle's own R_j_prev contributing its own A_j term.
    const a00: Any = float(0.0).toVar();
    const a01: Any = float(0.0).toVar();
    const a02: Any = float(0.0).toVar();
    const a10: Any = float(0.0).toVar();
    const a11: Any = float(0.0).toVar();
    const a12: Any = float(0.0).toVar();
    const a20: Any = float(0.0).toVar();
    const a21: Any = float(0.0).toVar();
    const a22: Any = float(0.0).toVar();

    const c: Any = particleCenters.element(i).xyz.toVar();
    const cBar: Any = restNeighborhoodCenters.element(i).xyz.toVar();

    Loop({ start: start, end: end, type: 'uint', condition: '<' }, ({ i: k }: { i: Any }) => {
      const j: Any = neighborIndices.element(k);

      // (r²/5) · R_j_prev — the A_j term from this neighbor (Eq. 8).
      // R_j_prev = mat3FromQuat(particles.rotation[j]). Standard formula
      // for unit quaternion q = (x, y, z, w):
      //   m00 = 1 − 2(y² + z²)     m01 = 2(xy − wz)     m02 = 2(xz + wy)
      //   m10 = 2(xy + wz)         m11 = 1 − 2(x² + z²) m12 = 2(yz − wx)
      //   m20 = 2(xz − wy)         m21 = 2(yz + wx)     m22 = 1 − 2(x² + y²)
      const qj: Any = particles.rotation.element(j).toVar();
      const xx: Any = qj.x.mul(qj.x);
      const yy: Any = qj.y.mul(qj.y);
      const zz: Any = qj.z.mul(qj.z);
      const xy: Any = qj.x.mul(qj.y);
      const xz: Any = qj.x.mul(qj.z);
      const yz: Any = qj.y.mul(qj.z);
      const wx: Any = qj.w.mul(qj.x);
      const wy: Any = qj.w.mul(qj.y);
      const wz: Any = qj.w.mul(qj.z);
      const rj00: Any = float(1.0).sub(yy.add(zz).mul(float(2.0)));
      const rj01: Any = xy.sub(wz).mul(float(2.0));
      const rj02: Any = xz.add(wy).mul(float(2.0));
      const rj10: Any = xy.add(wz).mul(float(2.0));
      const rj11: Any = float(1.0).sub(xx.add(zz).mul(float(2.0)));
      const rj12: Any = yz.sub(wx).mul(float(2.0));
      const rj20: Any = xz.sub(wy).mul(float(2.0));
      const rj21: Any = yz.add(wx).mul(float(2.0));
      const rj22: Any = float(1.0).sub(xx.add(yy).mul(float(2.0)));
      a00.addAssign(aiScalar.mul(rj00));
      a01.addAssign(aiScalar.mul(rj01));
      a02.addAssign(aiScalar.mul(rj02));
      a10.addAssign(aiScalar.mul(rj10));
      a11.addAssign(aiScalar.mul(rj11));
      a12.addAssign(aiScalar.mul(rj12));
      a20.addAssign(aiScalar.mul(rj20));
      a21.addAssign(aiScalar.mul(rj21));
      a22.addAssign(aiScalar.mul(rj22));

      // Outer product accumulate: A += (x*_j - c) (x̃_j - c̄)^T.
      const xj: Any = particles.predictedPositions.element(j).xyz.sub(c);
      const xtj: Any = restOffsets.element(j).xyz.sub(cBar);
      a00.addAssign(xj.x.mul(xtj.x));
      a01.addAssign(xj.x.mul(xtj.y));
      a02.addAssign(xj.x.mul(xtj.z));
      a10.addAssign(xj.y.mul(xtj.x));
      a11.addAssign(xj.y.mul(xtj.y));
      a12.addAssign(xj.y.mul(xtj.z));
      a20.addAssign(xj.z.mul(xtj.x));
      a21.addAssign(xj.z.mul(xtj.y));
      a22.addAssign(xj.z.mul(xtj.z));
    });

    const aPq: Mat3Nodes = {
      m00: a00,
      m01: a01,
      m02: a02,
      m10: a10,
      m11: a11,
      m12: a12,
      m20: a20,
      m21: a21,
      m22: a22,
    };
    const R: Mat3Nodes = emitPolarDecomposition(aPq);

    // Particles outside any implicit-mode body (count == 0): write
    // identity. Otherwise write the polar-decomp rotation. `select` per
    // component avoids a Loop-in-Else nesting.
    const isEmpty: Any = count.equal(uint(0));
    const r00: Any = isEmpty.select(float(1.0), R.m00);
    const r01: Any = isEmpty.select(float(0.0), R.m01);
    const r02: Any = isEmpty.select(float(0.0), R.m02);
    const r10: Any = isEmpty.select(float(0.0), R.m10);
    const r11: Any = isEmpty.select(float(1.0), R.m11);
    const r12: Any = isEmpty.select(float(0.0), R.m12);
    const r20: Any = isEmpty.select(float(0.0), R.m20);
    const r21: Any = isEmpty.select(float(0.0), R.m21);
    const r22: Any = isEmpty.select(float(1.0), R.m22);

    const baseSlot: Any = i.mul(uint(3));
    particleRotations.element(baseSlot).assign(vec4(r00, r01, r02, float(0.0)));
    particleRotations.element(baseSlot.add(uint(1))).assign(vec4(r10, r11, r12, float(0.0)));
    particleRotations.element(baseSlot.add(uint(2))).assign(vec4(r20, r21, r22, float(0.0)));
  })().compute(particles.capacity);
}

export interface BuildImplicitShapeMatchScatterKernelArgs {
  readonly particles: ParticleSystem;
  readonly restOffsets: StorageBufferNode<'vec4'>;
  readonly neighborOffsets: StorageBufferNode<'uint'>;
  readonly neighborIndices: StorageBufferNode<'uint'>;
  readonly particleCenters: StorageBufferNode<'vec4'>;
  readonly restNeighborhoodCenters: StorageBufferNode<'vec4'>;
  readonly particleRotations: StorageBufferNode<'vec4'>;

  readonly compliance: UniformNode<'float', number>;
  /**
   * Per-CSR-entry Lagrange multiplier `λ_{i,j}` (vec3 in xyz, w padding).
   * Sized `totalDegree`. Reset to zero at substep start by
   * {@link buildImplicitResetPairLambdaKernel}.
   */
  readonly pairLambda: StorageBufferNode<'vec4'>;
  readonly accumulator: ContactAccumulator;
  readonly xpbd: XpbdUniforms;
}

/**
 * Pass 3 — per group `i`, scatter Δx into all `j ∈ N(i)`.
 *
 * For every CSR entry `k` belonging to particle `i`'s row, with
 * `j = neighborIndices[k]`:
 * ```
 *   goal_{j,i} = R_i · (x̃_j − c̄_i) + c_i                       (Eq. 4)
 *   C          = x*_j − goal_{j,i}                              (vec3)
 *   α̃         = particleCompliance[i] / dt²
 *   Δλ         = (−C − α̃ · λ_{i,j}_old) / (w_j + α̃)            per-component
 *   λ_{i,j}    ← λ_{i,j}_old + Δλ
 *   Δx         = w_j · Δλ
 *   atomicAdd(accumulator[j], Δx)
 * ```
 *
 * Per-pair λ buffer (one slot per CSR entry) lets multiple groups touch
 * the same particle's predicted position without racing on a single per-
 * particle λ slot. Each thread is the only writer to its `pairLambda[k]`
 * entries (`k` ranges over particle `i`'s row of the CSR), and every
 * `Δx` contribution lands in the per-particle accumulator via
 * `atomicAdd`. Apply (`buildApplyAccumulatorToPredictedKernel`) commits
 * the sum to `predictedPositions` once per iter and zeros the
 * accumulator for the next iter.
 *
 * Cadence: once per solver iteration via
 * {@link SoftbodySystem.perIterKernels}.
 *
 * Determinism: i32 atomicAdd into the accumulator is order-independent
 * (Tier 1 bit-exact, ARCH §G4 — same property contact scatter relies on).
 * Per-pair λ updates are per-thread-local (no race).
 */
export function buildImplicitShapeMatchScatterKernel(
  args: BuildImplicitShapeMatchScatterKernelArgs,
): ComputeNode {
  const {
    particles,
    restOffsets,
    neighborOffsets,
    neighborIndices,
    particleCenters,
    restNeighborhoodCenters,
    particleRotations,
    compliance,
    pairLambda,
    accumulator,
    xpbd,
  } = args;

  const dt = xpbd.dt;
  const accScale = accumulator.scale;
  const accDelta = accumulator.delta;

  return Fn(() => {
    const i: Any = instanceIndex;
    const start: Any = neighborOffsets.element(i).toVar();
    const end: Any = neighborOffsets.element(i.add(uint(1))).toVar();

    // Read per-particle inputs unconditionally; the Loop is naturally a
    // no-op when start == end so the inputs are unused for slots outside
    // any implicit-mode body. Avoiding `Loop` inside `If(...)` matches
    // the pattern in the other §5.1 passes.
    const baseSlot: Any = i.mul(uint(3));
    const R0: Any = particleRotations.element(baseSlot).xyz.toVar();
    const R1: Any = particleRotations.element(baseSlot.add(uint(1))).xyz.toVar();
    const R2: Any = particleRotations.element(baseSlot.add(uint(2))).xyz.toVar();
    const c: Any = particleCenters.element(i).xyz.toVar();
    const cBar: Any = restNeighborhoodCenters.element(i).xyz.toVar();
    const alphaTilde: Any = compliance.div((dt as Any).mul(dt as Any));

    Loop({ start: start, end: end, type: 'uint', condition: '<' }, ({ i: k }: { i: Any }) => {
      const j: Any = neighborIndices.element(k);
      const restRel: Any = restOffsets.element(j).xyz.sub(cBar).toVar();
      // goal = R · restRel + c.
      const goal: Any = vec3(R0.dot(restRel), R1.dot(restRel), R2.dot(restRel)).add(c).toVar();
      const xStar: Any = particles.predictedPositions.element(j).xyz;
      const C: Any = xStar.sub(goal).toVar();
      const wj: Any = particles.invMass.element(j).toVar();
      const denom: Any = wj.add(alphaTilde);

      const lambdaOld: Any = pairLambda.element(k).xyz.toVar();
      const deltaLambda: Any = C.negate().sub(lambdaOld.mul(alphaTilde)).div(denom);
      const newLambda: Any = lambdaOld.add(deltaLambda);
      pairLambda.element(k).assign(vec4(newLambda.x, newLambda.y, newLambda.z, float(0.0)));
      // Constraint averaging: particle j receives a Δx contribution
      // from every group it belongs to (|N(j)| = own self-entry +
      // neighbor groups). Without averaging, contributions sum and
      // overshoot the goal by ~|N(j)|× per iter, which feeds back via
      // c_i and explodes the body within a handful of frames. Dividing
      // by m_j = |N(j)| recovers the per-particle constraint balance
      // — a standard XPBD treatment for multiply-constrained particles
      // (Macklin 2014 §4.2, applied here to the §5.1 group fan-in).
      const offJlo: Any = neighborOffsets.element(j).toVar();
      const offJhi: Any = neighborOffsets.element(j.add(uint(1))).toVar();
      const mj: Any = offJhi.sub(offJlo).toFloat().max(float(1.0));
      const deltaXraw: Any = deltaLambda.mul(wj).div(mj).toVar();

      // Per-pair Δx clamp. Standard PBD safety net: caps per-iter
      // correction magnitude so a single ill-conditioned constraint
      // can't teleport its target particle. Triggers when |C| is large
      // (e.g. high-velocity floor impacts where bottom particles get
      // pushed up while top particles still falling produces a goal
      // far from x*) or when polar decomp returns a noisy R that makes
      // goal_{j,i} = R · (x̃_j - c̄_i) + c_i wildly unrealistic.
      //
      // The 0.05 m cap is calibrated to MVP body sizes (rest extents
      // ~0.2-1 m, particle radius ~0.05 m): well above typical
      // per-iter Δx (~mm at α=1e-6) but tight enough to prevent
      // single-iter teleports across the body. Without this, a
      // high-acceleration drop (gravity ≥20 m/s² or drop height
      // ≥1 m) consistently kablooies; with it, the body relaxes
      // toward its goal over multiple iters and substeps.
      const PER_PAIR_DELTA_X_CAP = 0.05;
      const dxMagSq: Any = deltaXraw.x
        .mul(deltaXraw.x)
        .add(deltaXraw.y.mul(deltaXraw.y))
        .add(deltaXraw.z.mul(deltaXraw.z));
      const cap: Any = float(PER_PAIR_DELTA_X_CAP);
      const dxScale: Any = cap
        .mul(cap)
        .div(dxMagSq.max(cap.mul(cap)))
        .sqrt();
      const deltaX: Any = deltaXraw.mul(dxScale).toVar();

      const base: Any = j.mul(uint(3));
      const dxTicks: Any = deltaX.x.mul(accScale).toInt();
      const dyTicks: Any = deltaX.y.mul(accScale).toInt();
      const dzTicks: Any = deltaX.z.mul(accScale).toInt();
      atomicAdd(accDelta.element(base), dxTicks);
      atomicAdd(accDelta.element(base.add(uint(1))), dyTicks);
      atomicAdd(accDelta.element(base.add(uint(2))), dzTicks);
    });
  })().compute(particles.capacity);
}

export interface BuildImplicitQpWriteKernelArgs {
  readonly particles: ParticleSystem;
  readonly particleRotations: StorageBufferNode<'vec4'>;
  readonly neighborOffsets: StorageBufferNode<'uint'>;
}

/**
 * Pass 4 — per particle `i` (as the centre of its own group): write the
 * solver-modified `qp_i` into {@link ParticleSystem.predictedRotation}.
 *
 * Reads the particle's own `R_i` row (Mueller 2011 §5.1: "we only update
 * the orientation of the centre particle by replacing it with the
 * optimal rotation"), converts to a quaternion via the standard
 * `quatFromMat3` formula, and applies the shorter-rotation rule against
 * the substep-start `q_i` (`rotation[i]`). The §4.1 epilogue
 * (`SimLoop.advectRotation`) then reads `predictedRotation[i]` and
 * back-propagates `ω_i` via the Eq. 14 finite-difference. F-12.3.
 *
 * Cadence: ran every iter via {@link SoftbodySystem.perIterKernels}.
 * Plan §"Cadence note" preferred last-iter-only as a write-elision trade,
 * but the SimLoop has no slot for "post-iter, pre-advect" material
 * kernels — putting the write in `perIterKernels` lets the last iter's
 * value win naturally. The `R_i` written by Pass 2 once per substep does
 * not change across iters, so the `qp_i` written here is also stable
 * across iters; the cost is one extra dispatch per iter, ~30 µs at 50k
 * particles. If a future profile shows this matters, add a `lastIterOnly`
 * SimLoop hook.
 *
 * Particles outside any implicit-mode body (`|N(i)| = 0`) skip the write
 * entirely so the `predictedRotation` set by the §4.1 prologue
 * (`SimLoop.predictRotation`) survives unchanged for them.
 *
 * Determinism: per-particle thread-local computation, no atomics. Tier 1
 * bit-exact (ARCH §G4) given the Tier 2 `R_i` input.
 */
export function buildImplicitQpWriteKernel(args: BuildImplicitQpWriteKernelArgs): ComputeNode {
  const { particles, particleRotations, neighborOffsets } = args;

  return Fn(() => {
    const i: Any = instanceIndex;
    const start: Any = neighborOffsets.element(i).toVar();
    const end: Any = neighborOffsets.element(i.add(uint(1))).toVar();

    const baseSlot: Any = i.mul(uint(3));
    const r0: Any = particleRotations.element(baseSlot).xyz.toVar();
    const r1: Any = particleRotations.element(baseSlot.add(uint(1))).xyz.toVar();
    const r2: Any = particleRotations.element(baseSlot.add(uint(2))).xyz.toVar();

    // quatFromMat3 — branched form to avoid division by small numbers.
    // trace = m00 + m11 + m22; if trace > 0 use the trace-based form;
    // otherwise use the largest-diagonal-element form. See Shoemake 1985.
    const m00: Any = r0.x;
    const m01: Any = r0.y;
    const m02: Any = r0.z;
    const m10: Any = r1.x;
    const m11: Any = r1.y;
    const m12: Any = r1.z;
    const m20: Any = r2.x;
    const m21: Any = r2.y;
    const m22: Any = r2.z;
    const trace: Any = m00.add(m11).add(m22);

    const qx: Any = float(0.0).toVar();
    const qy: Any = float(0.0).toVar();
    const qz: Any = float(0.0).toVar();
    const qw: Any = float(1.0).toVar();

    If(trace.greaterThan(float(0.0)), () => {
      const s: Any = float(0.5).div(trace.add(float(1.0)).sqrt());
      qw.assign(float(0.25).div(s));
      qx.assign(m21.sub(m12).mul(s));
      qy.assign(m02.sub(m20).mul(s));
      qz.assign(m10.sub(m01).mul(s));
    })
      .ElseIf(m00.greaterThan(m11).and(m00.greaterThan(m22)), () => {
        const s: Any = float(2.0).mul(float(1.0).add(m00).sub(m11).sub(m22).sqrt());
        qw.assign(m21.sub(m12).div(s));
        qx.assign(float(0.25).mul(s));
        qy.assign(m01.add(m10).div(s));
        qz.assign(m02.add(m20).div(s));
      })
      .ElseIf(m11.greaterThan(m22), () => {
        const s: Any = float(2.0).mul(float(1.0).add(m11).sub(m00).sub(m22).sqrt());
        qw.assign(m02.sub(m20).div(s));
        qx.assign(m01.add(m10).div(s));
        qy.assign(float(0.25).mul(s));
        qz.assign(m12.add(m21).div(s));
      })
      .Else(() => {
        const s: Any = float(2.0).mul(float(1.0).add(m22).sub(m00).sub(m11).sqrt());
        qw.assign(m10.sub(m01).div(s));
        qx.assign(m02.add(m20).div(s));
        qy.assign(m12.add(m21).div(s));
        qz.assign(float(0.25).mul(s));
      });

    // Shorter-rotation rule against the substep-start `q_i`.
    const qStart: Any = particles.rotation.element(i).toVar();
    const dot: Any = qStart.x
      .mul(qx)
      .add(qStart.y.mul(qy))
      .add(qStart.z.mul(qz))
      .add(qStart.w.mul(qw));
    const flip: Any = dot.lessThan(float(0.0)).select(float(-1.0), float(1.0));

    // Slots outside any implicit-mode body keep the §4.1 prologue's
    // predictedRotation — leave the buffer untouched (use a separate
    // `If` here, NOT `Loop`-inside-`Else`; the body is just a single
    // assign so this is safe).
    If(end.greaterThan(start), () => {
      particles.predictedRotation
        .element(i)
        .assign(vec4(qx.mul(flip), qy.mul(flip), qz.mul(flip), qw.mul(flip)));
    });
  })().compute(particles.capacity);
}

export interface BuildImplicitResetPairLambdaKernelArgs {
  readonly pairLambda: StorageBufferNode<'vec4'>;
  /** Total CSR entries (sum of per-particle degree, including self). */
  readonly totalDegree: number;
}

/**
 * Reset the per-pair λ buffer to zero at the start of each substep.
 * Macklin 2016 Algorithm 1 line 4 — `λ_0 ← 0` per substep, then
 * accumulate Δλ across solver iterations.
 *
 * Sized to `totalDegree` (one slot per CSR entry). Dispatched once per
 * substep alongside the existing per-body λ reset in
 * {@link SoftbodySystem.preIterKernels}.
 */
export function buildImplicitResetPairLambdaKernel(
  args: BuildImplicitResetPairLambdaKernelArgs,
): ComputeNode {
  const { pairLambda, totalDegree } = args;
  if (!Number.isInteger(totalDegree) || totalDegree < 0) {
    throw new Error(
      `buildImplicitResetPairLambdaKernel: totalDegree must be a non-negative integer, got ${totalDegree}`,
    );
  }
  if (totalDegree === 0) {
    // Empty body or no-edge config — emit a one-thread no-op kernel so the
    // caller's pipeline shape is uniform (avoids a conditional dispatch
    // append at the SoftbodySystem call site).
    return Fn(() => {
      // Intentional no-op; one-thread dispatch is cheap and keeps the
      // pipeline shape regular.
      void int(0);
    })().compute(1);
  }
  return Fn(() => {
    const k: Any = instanceIndex;
    pairLambda.element(k).assign(vec4(0.0, 0.0, 0.0, 0.0));
  })().compute(totalDegree);
}
