import { Fn, If, acos, cos, float, instanceIndex, uint, uniform, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import { emitForEachNeighbor } from '../../../core/index.js';
import type { FluidSystem } from '../../FluidSystem.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Phase 14c — Yu & Turk 2010 *Reconstructing Surfaces of Particle-Based
 * Fluids Using Anisotropic Kernels* §4 anisotropy compute pass.
 *
 * Per render frame, for each fluid particle i, this kernel:
 *
 * 1. Walks the 27-cell hash neighbourhood (radius `r_i = 2·h_i`,
 *    paper §4.2 last paragraph).
 * 2. Eq. 11 polynomial weight `w_ij = 1 − (||x_i − x_j|| / r_i)³`
 *    on each candidate within `r_i`. (Paper writes the function with
 *    the cube once; do not square or re-cube.)
 * 3. One-pass relative-coordinate PCA — accumulates `Σw`, `Σ w·y`,
 *    `Σ w·y⊗y` with `y_j = x_j − x_i` so float32 precision stays
 *    at ~1e-7 even for world-scale scenes; recovers
 *    `x_i^w = x_i + Σ(w·y)/Σw` (eq. 10) and
 *    `C_i = Σ(w·y⊗y)/Σw − y_w·y_w^T` (eq. 9 algebraic identity
 *    `Var = E[xx^T] − E[x]·E[x]^T`).
 * 4. Eq. 12 SVD `C = R Σ R^T`, computed via Kopp 2008 / Smith 1961
 *    closed-form symmetric 3×3 eigendecomposition (trigonometric
 *    formula on the depressed cubic). Eigenvectors via the
 *    cross-product-of-rows method on `A − λI` with the largest
 *    cross-product magnitude picked for stability; v_2 reconstructed
 *    as `cross(v_3, v_1)` after Gram-Schmidt re-orthogonalisation.
 *    Closed-form, not Jacobi-iteration — see U-FR-5 for the
 *    repeated-eigenvalue stability caveat.
 * 5. Eq. 15 eigenvalue clamp + scale:
 *      `σ̃_k = max(σ_k, σ_1/k_r)` for k = 2,3;  `σ̃_1 = σ_1` itself
 *      if `N > N_ε`:  `Σ̃ = k_s · diag(σ_1, σ̃_2, σ̃_3)`
 *      else:           `Σ̃ = k_n · I`
 * 6. Eq. 16 `G_i = (1/h_i) R Σ̃^{-1} R^T`, stored as its inverse
 *    `G_i^{-1} = h_i · R · Σ̃ · R^T` (the depth pass needs `G^{-1}`
 *    to size the imposter quad and as the affine map for the
 *    ray-vs-ellipsoid intersection). Six unique floats split
 *    diagonal (`anisotropyDiag`) + off-diagonal (`anisotropyOff`).
 * 7. Eq. 6 Laplacian-smoothed kernel centre
 *      `x̄_i = (1 − λ) · x_i + λ · x_i^w`
 *    with `λ ∈ [0.9, 1.0]` (paper §4.1 last paragraph; default 0.95
 *    is the midpoint of the paper's recommended range). Note:
 *    `x_i^w` and `x̄_i` share the same Σ-weights as the covariance
 *    walk, so we reuse the accumulators rather than running a second
 *    pass.
 *
 * The single dispatch reads `particles.positions` (post-advect
 * committed state — the renderer runs after the substep loop so
 * `positions` is current) and writes `anisotropyDiag`,
 * `anisotropyOff`, `smoothedPositions`. The depth pass reads all
 * three when `anisotropy.enabled = true`.
 */
export interface AnisotropyKernelUniforms {
  /** Yu & Turk 2010 §4.2 `k_r` — eigenvalue ratio cap; paper default 4. */
  readonly kr: UniformNode<'float', number>;
  /** §4.2 `k_s` — covariance scale factor; auto-derived from median |C_i|. */
  readonly ks: UniformNode<'float', number>;
  /** §4.2 `k_n` — isolated-particle radius; paper default 0.5. */
  readonly kn: UniformNode<'float', number>;
  /** §4.2 `N_ε` — minimum-neighbor threshold for interior; paper default 25. */
  readonly nEpsilon: UniformNode<'float', number>;
  /** §4.1 `λ` — Laplacian centre-smoothing strength; paper [0.9, 1.0]. */
  readonly lambda: UniformNode<'float', number>;
}

export function createAnisotropyKernelUniforms(initial: {
  readonly kr: number;
  readonly ks: number;
  readonly kn: number;
  readonly nEpsilon: number;
  readonly lambda: number;
}): AnisotropyKernelUniforms {
  return {
    kr: uniform(initial.kr, 'float'),
    ks: uniform(initial.ks, 'float'),
    kn: uniform(initial.kn, 'float'),
    nEpsilon: uniform(initial.nEpsilon, 'float'),
    lambda: uniform(initial.lambda, 'float'),
  };
}

export interface BuildAnisotropyKernelArgs {
  readonly fluidSystem: FluidSystem;
  readonly aniso: AnisotropyKernelUniforms;
}

/**
 * Build the anisotropy compute kernel. `fluidSystem.enableAnisotropyBuffers()`
 * MUST have been called before this — the kernel writes to
 * `anisotropyDiag`, `anisotropyOff`, `smoothedPositions`. Throws if any
 * is undefined.
 */
export function buildAnisotropyKernel(args: BuildAnisotropyKernelArgs): ComputeNode {
  const { fluidSystem, aniso } = args;
  const { particles, hashGrid, fluidParticles, h } = fluidSystem;
  const anisotropyDiag = fluidSystem.anisotropyDiag;
  const anisotropyOff = fluidSystem.anisotropyOff;
  const smoothedPositions = fluidSystem.smoothedPositions;
  const anisotropyDiagnostic = fluidSystem.anisotropyDiagnostic;
  if (!anisotropyDiag || !anisotropyOff || !smoothedPositions || !anisotropyDiagnostic) {
    throw new Error(
      'buildAnisotropyKernel: fluidSystem.enableAnisotropyBuffers() must be called before constructing this kernel',
    );
  }

  const fluidStart = fluidParticles.start;
  // r_i = 2·h_i — paper §4.2: "we choose r_i to be 2·h_i in order to
  // include enough neighborhood particles". Constant across all fluid
  // particles (single-h scene).
  const rI = float(2 * h);
  const rISq = float(2 * h * 2 * h);

  return Fn(() => {
    const i: Any = (instanceIndex as Any).add(uint(fluidStart)).toVar();
    const xi: Any = particles.positions.element(i).xyz.toVar();

    // Accumulators — six unique covariance entries plus three weighted-
    // mean components plus weight sum and neighbour count. All in
    // relative-y coordinates so the catastrophic-cancellation magnitude
    // for `Var = E[yy^T] − y_w·y_w^T` stays at ~1e-7.
    const sumW: Any = float(0.0).toVar();
    const sumWYx: Any = float(0.0).toVar();
    const sumWYy: Any = float(0.0).toVar();
    const sumWYz: Any = float(0.0).toVar();
    const sumWYY00: Any = float(0.0).toVar();
    const sumWYY11: Any = float(0.0).toVar();
    const sumWYY22: Any = float(0.0).toVar();
    const sumWYY01: Any = float(0.0).toVar();
    const sumWYY02: Any = float(0.0).toVar();
    const sumWYY12: Any = float(0.0).toVar();
    const N: Any = float(0.0).toVar();

    emitForEachNeighbor({
      queryPosXyz: xi,
      hashOrigin: hashGrid.hashOriginUniform,
      cellSize: hashGrid.cellSizeUniform,
      hashTableSize: hashGrid.hashTableSize,
      cellStart: hashGrid.cellStart,
      cellEnd: hashGrid.cellEnd,
      sortedIndices: hashGrid.sortedIndices,
      onCandidate: (j: Any) => {
        const xj: Any = particles.positions.element(j).xyz;
        const y: Any = xj.sub(xi).toVar();
        const rSq: Any = y.dot(y).toVar();
        // Paper eq. 11: w = 1 − (r/r_i)³ when r < r_i, else 0. Convert
        // to (r/r_i)² for the predicate (avoids sqrt on the rejection
        // path) — the surviving branch still takes one sqrt.
        If(rSq.lessThan(rISq), () => {
          const r: Any = rSq.sqrt();
          const t: Any = r.div(rI);
          const w: Any = float(1.0).sub(t.mul(t).mul(t));
          sumW.addAssign(w);
          sumWYx.addAssign(w.mul(y.x));
          sumWYy.addAssign(w.mul(y.y));
          sumWYz.addAssign(w.mul(y.z));
          sumWYY00.addAssign(w.mul(y.x).mul(y.x));
          sumWYY11.addAssign(w.mul(y.y).mul(y.y));
          sumWYY22.addAssign(w.mul(y.z).mul(y.z));
          sumWYY01.addAssign(w.mul(y.x).mul(y.y));
          sumWYY02.addAssign(w.mul(y.x).mul(y.z));
          sumWYY12.addAssign(w.mul(y.y).mul(y.z));
          N.addAssign(float(1.0));
        });
      },
    });

    // Weighted mean offset (Yu & Turk eq. 10 written in relative-y form):
    //   x_i^w − x_i = Σ(w·y) / Σw
    // Guard `Σw > 0` (a fluid particle with zero neighbours within r_i
    // would produce 1.0 from its self-term, but defensive max is cheap).
    const invSumW: Any = float(1.0).div(sumW.max(float(1e-20)));
    const yWx: Any = sumWYx.mul(invSumW).toVar();
    const yWy: Any = sumWYy.mul(invSumW).toVar();
    const yWz: Any = sumWYz.mul(invSumW).toVar();

    // Eq. 6 Laplacian-smoothed centre: x̄_i = x_i + λ · y_w. The
    // (1 − λ)·x_i + λ·x_i^w form simplifies to this when x_i^w = x_i + y_w.
    const lambdaU: Any = aniso.lambda as Any;
    smoothedPositions
      .element(i)
      .assign(
        vec4(
          xi.x.add(lambdaU.mul(yWx)),
          xi.y.add(lambdaU.mul(yWy)),
          xi.z.add(lambdaU.mul(yWz)),
          float(0.0),
        ),
      );

    // Eq. 9 covariance via the algebraic identity
    //   C_ab = E[y_a · y_b] − y_w_a · y_w_b
    // C is symmetric; six unique entries.
    const c00: Any = sumWYY00.mul(invSumW).sub(yWx.mul(yWx)).toVar();
    const c11: Any = sumWYY11.mul(invSumW).sub(yWy.mul(yWy)).toVar();
    const c22: Any = sumWYY22.mul(invSumW).sub(yWz.mul(yWz)).toVar();
    const c01: Any = sumWYY01.mul(invSumW).sub(yWx.mul(yWy)).toVar();
    const c02: Any = sumWYY02.mul(invSumW).sub(yWx.mul(yWz)).toVar();
    const c12: Any = sumWYY12.mul(invSumW).sub(yWy.mul(yWz)).toVar();

    // Closed-form 3×3 symmetric eigendecomposition (Kopp 2008 /
    // Smith 1961 trigonometric formula on the depressed cubic). All
    // three eigenvalues come out sorted descending.
    //
    //   p1 = c01² + c02² + c12²
    //   q  = trace(C) / 3
    //   p2 = (c00−q)² + (c11−q)² + (c22−q)² + 2·p1
    //   p  = sqrt(p2 / 6)
    //   B  = (C − q·I) / p   (deviatoric, traceless, |B| = √2)
    //   r  = det(B) / 2  ∈ [-1, 1]
    //   φ  = acos(clamp(r, -1, 1)) / 3
    //   λ_1 = q + 2p·cos(φ)            (largest)
    //   λ_3 = q + 2p·cos(φ + 2π/3)     (smallest)
    //   λ_2 = 3q − λ_1 − λ_3            (middle, by trace-sum)
    //
    // Degenerate cases (rare, but Yu & Turk eq. 15's `N < N_ε` branch
    // catches truly isolated particles before the eigenvalues are
    // touched). When `p ≈ 0` (already-diagonal C) the trig branch
    // collapses; we guard with `p.max(1e-20)`.
    const trC: Any = c00.add(c11).add(c22);
    const q: Any = trC.div(float(3.0)).toVar();
    const p1: Any = c01.mul(c01).add(c02.mul(c02)).add(c12.mul(c12));
    const aq: Any = c00.sub(q).toVar();
    const bq: Any = c11.sub(q).toVar();
    const cq: Any = c22.sub(q).toVar();
    const p2: Any = aq
      .mul(aq)
      .add(bq.mul(bq))
      .add(cq.mul(cq))
      .add(p1.mul(float(2.0)));
    const p: Any = p2.div(float(6.0)).max(float(1e-30)).sqrt().toVar();
    // det(C − q·I): cofactor expansion across the first row, with the
    // diagonal entries replaced by their q-shifted forms.
    const detAmQI: Any = aq
      .mul(bq.mul(cq).sub(c12.mul(c12)))
      .sub(c01.mul(c01.mul(cq).sub(c12.mul(c02))))
      .add(c02.mul(c01.mul(c12).sub(bq.mul(c02))));
    const rDet: Any = detAmQI.div(p.mul(p).mul(p).mul(float(2.0))).toVar();
    const rClamped: Any = rDet.clamp(float(-1.0), float(1.0));
    const phi: Any = acos(rClamped).div(float(3.0)).toVar();
    const TWO_PI_OVER_3 = float(2.0943951023931953); // 2π/3
    const sigma1: Any = q.add(p.mul(float(2.0)).mul(cos(phi))).toVar();
    const sigma3: Any = q.add(p.mul(float(2.0)).mul(cos(phi.add(TWO_PI_OVER_3)))).toVar();
    const sigma2: Any = trC.sub(sigma1).sub(sigma3).toVar();

    // Eigenvectors. For each eigenvalue λ, M = C − λ·I has rank ≤ 2;
    // the kernel is recovered as `cross(row_a, row_b)` of M, picking
    // the row pair giving the largest cross-product magnitude. When
    // all three candidate cross products have FP-noise magnitude (the
    // case for repeated eigenvalues — uniform fluid interior, or two
    // equal eigenvalues for axis-aligned flat sheets), the kernel
    // returns a sentinel `{ valid: false }` so the caller can splice
    // in a basis-completion fallback. We use a meaningful-magnitude
    // check on `max(m1, m2, m3)` rather than an additive epsilon on
    // norm² — the additive form swamps genuine FP-noise candidates
    // (|c|² ≈ 1e-44 under repeated eigenvalues) and produces non-unit
    // outputs when divided through.
    const MEANINGFUL_M_THRESHOLD = float(1e-20);
    const computeEigvec = (
      lambda: Any,
    ): {
      readonly x: Any;
      readonly y: Any;
      readonly z: Any;
      readonly valid: Any;
    } => {
      const m00: Any = c00.sub(lambda).toVar();
      const m11: Any = c11.sub(lambda).toVar();
      const m22: Any = c22.sub(lambda).toVar();
      // Three candidate cross products (rows of M = C − λI; off-
      // diagonals same as C since I has none on those slots):
      //   c1 = row0 × row1, c2 = row0 × row2, c3 = row1 × row2
      const c1x: Any = c01.mul(c12).sub(c02.mul(m11));
      const c1y: Any = c02.mul(c01).sub(m00.mul(c12));
      const c1z: Any = m00.mul(m11).sub(c01.mul(c01));

      const c2x: Any = c01.mul(m22).sub(c02.mul(c12));
      const c2y: Any = c02.mul(c02).sub(m00.mul(m22));
      const c2z: Any = m00.mul(c12).sub(c01.mul(c02));

      const c3x: Any = m11.mul(m22).sub(c12.mul(c12));
      const c3y: Any = c12.mul(c02).sub(c01.mul(m22));
      const c3z: Any = c01.mul(c12).sub(m11.mul(c02));

      const m1: Any = c1x.mul(c1x).add(c1y.mul(c1y)).add(c1z.mul(c1z)).toVar();
      const m2: Any = c2x.mul(c2x).add(c2y.mul(c2y)).add(c2z.mul(c2z)).toVar();
      const m3: Any = c3x.mul(c3x).add(c3y.mul(c3y)).add(c3z.mul(c3z)).toVar();

      const useC1: Any = m1.greaterThanEqual(m2).and(m1.greaterThanEqual(m3));
      const useC2: Any = m2.greaterThanEqual(m3);
      const vx: Any = useC1.select(c1x, useC2.select(c2x, c3x)).toVar();
      const vy: Any = useC1.select(c1y, useC2.select(c2y, c3y)).toVar();
      const vz: Any = useC1.select(c1z, useC2.select(c2z, c3z)).toVar();
      const maxM: Any = m1.max(m2).max(m3).toVar();
      const valid: Any = maxM.greaterThan(MEANINGFUL_M_THRESHOLD);
      // Safe norm: when the cross product is below threshold, `norm`
      // floors to a finite small value so the divide doesn't NaN; the
      // caller will discard the result via `valid` anyway.
      const norm: Any = maxM.sqrt().max(float(1e-15)).toVar();
      return {
        x: vx.div(norm).toVar(),
        y: vy.div(norm).toVar(),
        z: vz.div(norm).toVar(),
        valid,
      };
    };

    const v1raw = computeEigvec(sigma1);
    const v3raw = computeEigvec(sigma3);

    // v1 fallback: ê_x is a valid (arbitrary) unit vector when the
    // σ_1 cross-product is FP-noise. For repeated-eigenvalue cases the
    // matrix is invariant under ANY rotation, so a fixed-axis pick is
    // mathematically correct.
    const v1x: Any = v1raw.valid.select(v1raw.x, float(1.0)).toVar();
    const v1y: Any = v1raw.valid.select(v1raw.y, float(0.0)).toVar();
    const v1z: Any = v1raw.valid.select(v1raw.z, float(0.0)).toVar();

    // v3 path: try Gram-Schmidt against v1 if v3raw is meaningful and
    // the residual is non-trivial; otherwise compute v1-perpendicular
    // via cross-with-least-aligned-world-axis (always produces a unit
    // vector with |result| ≥ √(2/3)).
    const v3dotv1: Any = v3raw.x.mul(v1x).add(v3raw.y.mul(v1y)).add(v3raw.z.mul(v1z));
    const v3ox: Any = v3raw.x.sub(v3dotv1.mul(v1x));
    const v3oy: Any = v3raw.y.sub(v3dotv1.mul(v1y));
    const v3oz: Any = v3raw.z.sub(v3dotv1.mul(v1z));
    const v3oNormSq: Any = v3ox.mul(v3ox).add(v3oy.mul(v3oy)).add(v3oz.mul(v3oz)).toVar();
    const v3oNorm: Any = v3oNormSq.sqrt().max(float(1e-15)).toVar();

    // v1-perpendicular fallback: cross v1 with the world axis whose
    // |dot| with v1 is smallest. The result is bounded |·| ≥ √(2/3).
    const absV1x: Any = v1x.abs();
    const absV1y: Any = v1y.abs();
    const absV1z: Any = v1z.abs();
    const useEx: Any = absV1x.lessThanEqual(absV1y).and(absV1x.lessThanEqual(absV1z));
    const useEy: Any = absV1y.lessThanEqual(absV1z);
    // cross(v1, ê_k) for k = x, y, z respectively:
    //   cross(v1, (1,0,0)) = (0, v1.z, -v1.y)
    //   cross(v1, (0,1,0)) = (-v1.z, 0, v1.x)
    //   cross(v1, (0,0,1)) = (v1.y, -v1.x, 0)
    const fbx: Any = useEx.select(float(0.0), useEy.select(v1z.negate(), v1y));
    const fby: Any = useEx.select(v1z, useEy.select(float(0.0), v1x.negate()));
    const fbz: Any = useEx.select(v1y.negate(), useEy.select(v1x, float(0.0)));
    const fbNormSq: Any = fbx.mul(fbx).add(fby.mul(fby)).add(fbz.mul(fbz)).toVar();
    const fbNorm: Any = fbNormSq.sqrt().max(float(1e-15)).toVar();
    const fbX: Any = fbx.div(fbNorm);
    const fbY: Any = fby.div(fbNorm);
    const fbZ: Any = fbz.div(fbNorm);

    // Use Gram-Schmidt result only when both v3raw is valid AND its
    // residual is non-trivial (so v3raw isn't ±v1). Otherwise fall back.
    const useGS: Any = v3raw.valid.and(v3oNormSq.greaterThan(float(1e-6)));
    const v3x: Any = useGS.select(v3ox.div(v3oNorm), fbX).toVar();
    const v3y: Any = useGS.select(v3oy.div(v3oNorm), fbY).toVar();
    const v3z: Any = useGS.select(v3oz.div(v3oNorm), fbZ).toVar();

    // v2 = v3 × v1 — orthogonal to both, unit length when v3 ⊥ v1 and
    // both are unit (guaranteed by the construction above).
    const v2x: Any = v3y.mul(v1z).sub(v3z.mul(v1y)).toVar();
    const v2y: Any = v3z.mul(v1x).sub(v3x.mul(v1z)).toVar();
    const v2z: Any = v3x.mul(v1y).sub(v3y.mul(v1x)).toVar();

    // Yu & Turk eq. 15 — clamp + scale eigenvalues. σ_1 stays as itself
    // (it's already the largest); σ_2, σ_3 are clamped against σ_1/k_r
    // so the worst-case eigenvalue ratio doesn't exceed `k_r` (default 4).
    // Then either the interior scaling `Σ̃ = k_s · diag(...)` or the
    // isolated fallback `Σ̃ = k_n · I` based on `N` vs `N_ε`.
    const krU: Any = aniso.kr as Any;
    const ksU: Any = aniso.ks as Any;
    const knU: Any = aniso.kn as Any;
    const nEps: Any = aniso.nEpsilon as Any;

    const sigmaFloor: Any = sigma1.div(krU);
    const s1Clamped: Any = sigma1; // largest, no clamp
    const s2Clamped: Any = sigma2.max(sigmaFloor);
    const s3Clamped: Any = sigma3.max(sigmaFloor);

    const interior: Any = N.greaterThan(nEps);
    // Interior: use clamped eigenvalues × k_s. Isolated: k_n in all
    // three slots (so the rotation R drops out — R k_n·I R^T = k_n·I).
    const sTilde1: Any = interior.select(s1Clamped.mul(ksU), knU);
    const sTilde2: Any = interior.select(s2Clamped.mul(ksU), knU);
    const sTilde3: Any = interior.select(s3Clamped.mul(ksU), knU);

    // Eq. 16 inverse: G_i^{-1} = h_i · R · Σ̃ · R^T. With R = [v1 v2 v3]
    // as columns, the symmetric output is
    //   M_ab = Σ_k σ̃_k · v_k[a] · v_k[b]
    // Six unique entries. The h_i scalar multiplies all six.
    const hScalar: Any = float(h);
    const invG00: Any = hScalar.mul(
      sTilde1
        .mul(v1x.mul(v1x))
        .add(sTilde2.mul(v2x.mul(v2x)))
        .add(sTilde3.mul(v3x.mul(v3x))),
    );
    const invG11: Any = hScalar.mul(
      sTilde1
        .mul(v1y.mul(v1y))
        .add(sTilde2.mul(v2y.mul(v2y)))
        .add(sTilde3.mul(v3y.mul(v3y))),
    );
    const invG22: Any = hScalar.mul(
      sTilde1
        .mul(v1z.mul(v1z))
        .add(sTilde2.mul(v2z.mul(v2z)))
        .add(sTilde3.mul(v3z.mul(v3z))),
    );
    const invG01: Any = hScalar.mul(
      sTilde1
        .mul(v1x.mul(v1y))
        .add(sTilde2.mul(v2x.mul(v2y)))
        .add(sTilde3.mul(v3x.mul(v3y))),
    );
    const invG02: Any = hScalar.mul(
      sTilde1
        .mul(v1x.mul(v1z))
        .add(sTilde2.mul(v2x.mul(v2z)))
        .add(sTilde3.mul(v3x.mul(v3z))),
    );
    const invG12: Any = hScalar.mul(
      sTilde1
        .mul(v1y.mul(v1z))
        .add(sTilde2.mul(v2y.mul(v2z)))
        .add(sTilde3.mul(v3y.mul(v3z))),
    );

    anisotropyDiag.element(i).assign(vec4(invG00, invG11, invG22, float(0.0)));
    anisotropyOff.element(i).assign(vec4(invG01, invG02, invG12, float(0.0)));

    // Diagnostic — feeds the anisotropy debug views (#aniso.*). Stored
    // RAW (pre eq. 15 clamp + scale) so the artist sees what the PCA
    // produced before the clamp policy mangles things.
    const offsetMag: Any = yWx.mul(yWx).add(yWy.mul(yWy)).add(yWz.mul(yWz)).sqrt().mul(lambdaU);
    anisotropyDiagnostic.element(i).assign(vec4(N, sigma1, sigma3, offsetMag));
  })().compute(fluidParticles.count);
}
