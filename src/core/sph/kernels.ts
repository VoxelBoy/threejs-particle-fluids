import { float, uniform } from 'three/tsl';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

// TSL's @types surface many GPGPU nodes as bare `Node`, stripping the
// proxy-provided `.element()/.mul()/.dot()/...` methods. The loose alias
// matches the pattern already used in `src/core/src/contact/*`.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * CPU-side precomputed constants for the Müller 2003 Poly6 / Spiky SPH
 * kernels used by Macklin 2013 Position Based Fluids §3.
 *
 * The coefficients scale as high powers of `h` (`h^-9` for Poly6, `h^-6`
 * for Spiky), so precomputing them CPU-side avoids a `pow()` on every
 * neighbor pair. `h` is exposed as a mutable uniform; `setH(newH)`
 * rewrites all four entries in lock-step so they cannot drift apart.
 *
 *

 */
export interface SphKernelUniforms {
  /** Smoothing length `h` in metres. */
  readonly h: UniformNode<'float', number>;
  /** `h²`, precomputed for Poly6's `(h² − r²)³` form. */
  readonly hSq: UniformNode<'float', number>;
  /** Poly6 normalization `315 / (64·π·h⁹)`. */
  readonly poly6Coef: UniformNode<'float', number>;
  /** Spiky-gradient magnitude coefficient `45 / (π·h⁶)`; sign applied in-kernel. */
  readonly spikyCoef: UniformNode<'float', number>;
  /**
   * Rewrite `h` and every derived coefficient in lock-step. The four
   * uniforms MUST move together — independent mutation of any one produces
   * SPH integrals that no longer correspond to a single kernel.
   */
  readonly setH: (newH: number) => void;
}

function poly6CoefFor(h: number): number {
  return 315 / (64 * Math.PI * Math.pow(h, 9));
}

function spikyCoefFor(h: number): number {
  return 45 / (Math.PI * Math.pow(h, 6));
}

export function createSphKernelUniforms(h: number): SphKernelUniforms {
  if (!Number.isFinite(h) || h <= 0) {
    throw new Error(`createSphKernelUniforms: h must be a positive finite number, got ${h}`);
  }
  const hU = uniform(h, 'float');
  const hSqU = uniform(h * h, 'float');
  const poly6U = uniform(poly6CoefFor(h), 'float');
  const spikyU = uniform(spikyCoefFor(h), 'float');
  return {
    h: hU,
    hSq: hSqU,
    poly6Coef: poly6U,
    spikyCoef: spikyU,
    setH(newH: number): void {
      if (!Number.isFinite(newH) || newH <= 0) {
        throw new Error(`SphKernelUniforms.setH: h must be a positive finite number, got ${newH}`);
      }
      (hU as Any).value = newH;
      (hSqU as Any).value = newH * newH;
      (poly6U as Any).value = poly6CoefFor(newH);
      (spikyU as Any).value = spikyCoefFor(newH);
    },
  };
}

/**
 * Emit Poly6 `W(|r|, h) = poly6Coef · (h² − |r|²)³` as a scalar TSL node.
 *
 * Returns 0 for `|r| ≥ h` (the `max(·, 0)` clamp on `(h² − r²)`). Direction-
 * independent — only `|r|²` matters. Takes `r_vec` for caller convenience;
 * if the caller already has `rSq` from a proximity filter, prefer
 * {@link emitPoly6FromRSq} to skip the redundant dot product.
 *
 * Paper: Müller 2003 Poly6; cited by Macklin 2013 §3 as the density
 * estimator. See the {@link SphKernelUniforms} module docstring.
 */
export function emitPoly6(r_vec: Any, u: SphKernelUniforms): Any {
  const rSq: Any = r_vec.dot(r_vec);
  return emitPoly6FromRSq(rSq, u);
}

/**
 * {@link emitPoly6} variant that reuses a pre-computed `|r|²` scalar.
 */
export function emitPoly6FromRSq(rSq: Any, u: SphKernelUniforms): Any {
  const d: Any = (u.hSq as Any).sub(rSq).max(float(0.0));
  return d
    .mul(d)
    .mul(d)
    .mul(u.poly6Coef as Any);
}

/**
 * Emit Spiky `∇W(r_vec, h) = −spikyCoef · (h − |r|)² · r̂` as a vec3 TSL
 * node.
 *
 * **Sign convention**: the returned vector is the gradient taken wrt the
 * argument `r_vec`. If a caller uses `r_vec = x_i − x_j`, this returns
 * `∇_{p_i} W(p_i − p_j, h)`. For the `∇_{p_j}` form (as paper eq. 8 writes
 * for the `k = j` case, and paper eq. 15 writes for vorticity), negate the
 * returned vector — or equivalently pass `r_vec = x_j − x_i`.
 *
 * Returns zero-vector for `|r| ≥ h` (via the `(h − r)²` clamp) and for
 * `|r| → 0` (direction is undefined at the origin; the SPH convention is
 * that a particle contributes zero gradient to itself, which is consistent
 * with the gradient at `r = 0` being a point of ambiguity in the kernel
 * itself). The `rSafe = max(r, 1e-20)` guard prevents NaN at true-zero
 * separation.
 *
 */
export function emitSpikyGrad(r_vec: Any, u: SphKernelUniforms): Any {
  const rSq: Any = r_vec.dot(r_vec).toVar();
  const r: Any = rSq.sqrt().toVar();
  // `(h − r)²`, clamped to 0 so outside-radius pairs produce zero.
  const t: Any = (u.h as Any).sub(r).max(float(0.0)).toVar();
  // Guard against `r = 0`: the spiky-gradient magnitude stays finite as
  // `r → 0` (it tends to `spikyCoef · h²`), but the direction `r̂` is
  // undefined. Returning zero there matches the conventional SPH choice
  // (Müller 2003, Monaghan 1992) — a particle contributes no gradient to
  // itself.
  const rSafe: Any = r.max(float(1e-20));
  const rHat: Any = r_vec.div(rSafe);
  // Combine magnitude · direction. `.negate()` applies the paper's
  // explicit minus sign in `−(45/π·h⁶)·(h − r)² · r̂`.
  const mag: Any = t
    .mul(t)
    .mul(u.spikyCoef as Any)
    .negate();
  return rHat.mul(mag);
}
