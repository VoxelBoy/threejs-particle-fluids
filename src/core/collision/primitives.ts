import { If, float, max, uint, vec3, vec4 } from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Collider kind discriminator. Stored in {@link PrimitiveSet.kinds} as `u32`.
 * Values are numbered 0..3 so the kernel can dispatch on them with a plain
 * If-chain (TSL has no Switch). Warp coherence is preserved because adjacent
 * threads (= adjacent particles) iterate the same collider slot in the same
 * order — every thread in a warp hits the same branch at the same step.
 */
export const KIND_PLANE = 0;
export const KIND_SPHERE = 1;
export const KIND_BOX = 2;
export const KIND_CAPSULE = 3;

/** Bit flags packed into {@link PrimitiveSet.flags}. */
export const FLAG_INVERT = 1 << 0;

/**
 * Rotate vector `v` by unit quaternion `q = (x, y, z, w)` via the optimized
 * two-cross-product form (Hamilton convention, q·v·q*).
 *
 *   t = 2 · (q.xyz × v)
 *   v' = v + q.w · t + (q.xyz × t)
 *
 * 15 fused multiply-adds vs. the 18 of the build-a-mat3-first formulation.
 * Identity quaternion `(0, 0, 0, 1)` collapses to `v' = v + 0 + 0 = v`, so
 * a default-identity collider incurs the ops but returns the input
 * unchanged — matches the "pay for what you use" shape of the inline
 * If-chain dispatch.
 *
 * Used twice per box SDF evaluation: once to rotate the particle position
 * into the box's local frame (via the conjugate `(-x, -y, -z, w)`), and
 * once to rotate the resulting gradient back to world space (via `q`).
 */
function emitQuatRotate(q: Any, v: Any): Any {
  // 2 · (q.xyz × v)
  const qxyz: Any = q.xyz;
  const t: Any = float(2.0).mul(qxyz.cross(v)).toVar();
  return v.add(t.mul(q.w)).add(qxyz.cross(t));
}

/** Quaternion conjugate — inverse for unit quaternions. */
function emitQuatConjugate(q: Any): Any {
  return vec4(q.x.negate(), q.y.negate(), q.z.negate(), q.w);
}

/**
 * Numerical floor below which a gradient length is treated as degenerate.
 * Used by sphere / capsule / inside-AABB SDFs where the gradient is undefined
 * at the exact shape center or on a non-principal interior face. Particles
 * should never touch these measure-zero loci in practice, but the guard
 * prevents NaN propagation if they do (e.g. from rounded f32 landing exactly
 * at the center).
 */
export const EPSILON_GRADIENT = 1e-8;

/**
 * Emit TSL code that evaluates the signed-distance function `phi(x)` and its
 * outward unit gradient `n(x)` for the collider at `colliderSlot`.
 *
 * Returns both values as `toVar`-bound locals; the caller threads them into
 * the contact-projection arithmetic in `collision/solve.ts`.
 *
 * SDFs implemented:
 *   - Plane       `phi = n · (x - p)`, `∇phi = n`                 — plan §Primitives
 *   - Sphere      `phi = |x - c| - R`, `∇phi = (x - c) / |x - c|` — plan §Primitives
 *   - Box         `q = |x_local| - e`
 *                 `phi = |max(q, 0)| + min(max(q.x, q.y, q.z), 0)` — plan §Primitives
 *                 (closed-form, axis-aligned; rotation is applied by the
 *                  caller at CPU-side upload — see `PrimitiveSet.addBox`'s
 *                  Unknown note regarding rotated boxes).
 *   - Capsule     closest-point-on-segment + sphere displacement — plan §Primitives
 *
 *
 * Invert flag (bit `FLAG_INVERT`): negates both `phi` and `gradient`. Used
 * by inverted-sphere bowls and inverted-box tanks where the interior is the
 * valid region (plan §"Inside-flip for containers").
 *
 * Written as a TSL helper that mutates `phiVar` and `gradientVar` in place
 * rather than returning a struct — TSL's expression builders interact
 * poorly with ad-hoc object returns inside If-chains, and mutating vars is
 * the established pattern in the Phase 05 contact kernels.
 *
 * @param fields  Parallel storage buffers from the owning `PrimitiveSet`.
 *                Only the fields the kernel needs are required here.
 * @param colliderSlot  `u32` TSL node indexing the collider.
 * @param x       `vec3` TSL node — the particle position to test.
 * @param phiVar  `float` TSL var the kernel writes the SDF value into.
 * @param gradientVar  `vec3` TSL var the kernel writes the outward unit normal into.
 */
export interface ColliderFields {
  /**
   * `packed[c] = (kind << 16) | flags` — see
   * `PrimitiveSet.ts::packKindFlags`. The GPU-side unpack below shifts out
   * the 16-bit fields.
   */
  readonly packed: Any;
  readonly data0: Any;
  readonly data1: Any;
  /**
   * Per-slot unit quaternion `(x, y, z, w)`. Identity `(0, 0, 0, 1)` for
   * plane / sphere / capsule (they have no meaningful rotation). For box,
   * the quaternion defines the box's world rotation: a particle position
   * is rotated into the box's local frame via the **conjugate** before
   * the axis-aligned SDF test, and the gradient is rotated back by the
   * forward quaternion after. Resolves U-21.
   */
  readonly rotation: Any;
}

export function emitColliderSdf(
  fields: ColliderFields,
  colliderSlot: Any,
  x: Any,
  phiVar: Any,
  gradientVar: Any,
): void {
  const packedVal: Any = fields.packed.element(colliderSlot);
  const kind: Any = packedVal.shiftRight(uint(16)).bitAnd(uint(0xffff));
  const flags: Any = packedVal.bitAnd(uint(0xffff));
  const data0: Any = fields.data0.element(colliderSlot);
  const data1: Any = fields.data1.element(colliderSlot);

  // -------- Plane --------
  // data0.xyz = unit normal n, data1.xyz = reference point p.
  If(kind.equal(uint(KIND_PLANE)), () => {
    const n: Any = data0.xyz;
    const p: Any = data1.xyz;
    phiVar.assign(n.dot(x.sub(p)));
    gradientVar.assign(n);
  });

  // -------- Sphere --------
  // data0.xyz = center c, data0.w = radius R.
  If(kind.equal(uint(KIND_SPHERE)), () => {
    const c: Any = data0.xyz;
    const R: Any = data0.w;
    const diff: Any = x.sub(c).toVar();
    const len: Any = diff.length().toVar();
    phiVar.assign(len.sub(R));
    // Guarded divide — see EPSILON_GRADIENT doc above.
    const safeLen: Any = len.max(float(EPSILON_GRADIENT));
    gradientVar.assign(diff.div(safeLen));
  });

  // -------- Oriented Box --------
  // data0.xyz = center c, data1.xyz = half-extents e, rotation = unit
  // quaternion defining the box's world orientation (identity = AABB).
  // Rotation is applied in two places: (1) the particle position is
  // rotated into the box's local frame via the conjugate quaternion
  // before the axis-aligned SDF, (2) the resulting gradient is rotated
  // back to world space by the forward quaternion. Resolves U-21.
  If(kind.equal(uint(KIND_BOX)), () => {
    const c: Any = data0.xyz;
    const e: Any = data1.xyz;
    const rot: Any = fields.rotation.element(colliderSlot);
    // World → box-local. Identity quaternion passes `x - c` through
    // unchanged, so the axis-aligned case costs two cross products
    // but returns the same answer it would have without rotation.
    const xWorldOffset: Any = x.sub(c).toVar();
    const xLocal: Any = emitQuatRotate(emitQuatConjugate(rot), xWorldOffset).toVar();
    const q: Any = xLocal.abs().sub(e).toVar();
    // Exterior term: length of positive components of q.
    const qPos: Any = max(q, vec3(float(0.0))).toVar();
    const outside: Any = qPos.length();
    // Interior term: max of q's components, clamped at zero.
    const inside: Any = q.x.max(q.y).max(q.z).min(float(0.0));
    phiVar.assign(outside.add(inside));
    // Gradient — closed-form in the box's LOCAL frame, then rotated back
    // to world space below.
    //   Outside (any q > 0): unit(max(q, 0)) with per-axis sign from
    //     xLocal (local-space position).
    //   Inside (all q ≤ 0): outward direction is the axis with the
    //     greatest q (closest to escaping); sign taken from xLocal.
    const signs: Any = vec3(
      xLocal.x.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0)),
      xLocal.y.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0)),
      xLocal.z.greaterThanEqual(float(0.0)).select(float(1.0), float(-1.0)),
    ).toVar();
    const outsideLen: Any = qPos.length().toVar();
    const isOutside: Any = outsideLen.greaterThan(float(EPSILON_GRADIENT));
    const domX: Any = q.x.greaterThanEqual(q.y).and(q.x.greaterThanEqual(q.z));
    const domY: Any = q.y.greaterThanEqual(q.x).and(q.y.greaterThanEqual(q.z));
    const insideDir: Any = vec3(
      domX.select(signs.x, float(0.0)),
      domX.not().and(domY).select(signs.y, float(0.0)),
      domX.not().and(domY.not()).select(signs.z, float(0.0)),
    ).toVar();
    const localGrad: Any = isOutside
      .select(qPos.mul(signs).div(outsideLen.max(float(EPSILON_GRADIENT))), insideDir)
      .toVar();
    // Rotate the local-frame gradient back to world. For identity
    // quaternion this is a no-op (passes `localGrad` through).
    gradientVar.assign(emitQuatRotate(rot, localGrad));
  });

  // -------- Capsule --------
  // data0.xyz = endpoint a, data0.w = radius R, data1.xyz = endpoint b.
  If(kind.equal(uint(KIND_CAPSULE)), () => {
    const a: Any = data0.xyz;
    const R: Any = data0.w;
    const b: Any = data1.xyz;
    const ab: Any = b.sub(a).toVar();
    const ax: Any = x.sub(a).toVar();
    // Project x onto the segment, clamp to [0, 1].
    const abLenSq: Any = ab.dot(ab).max(float(EPSILON_GRADIENT));
    const t: Any = ax.dot(ab).div(abLenSq).clamp(float(0.0), float(1.0));
    const closest: Any = a.add(ab.mul(t)).toVar();
    const diff: Any = x.sub(closest).toVar();
    const len: Any = diff.length().toVar();
    phiVar.assign(len.sub(R));
    const safeLen: Any = len.max(float(EPSILON_GRADIENT));
    gradientVar.assign(diff.div(safeLen));
  });

  // -------- Invert --------
  // Flip sign of both phi and gradient. The collider's "outside" becomes
  // the interior half-space. Plan §"Inside-flip for containers".
  const invertBit: Any = flags.bitAnd(uint(FLAG_INVERT));
  const isInverted: Any = invertBit.notEqual(uint(0));
  phiVar.assign(isInverted.select(phiVar.negate(), phiVar));
  gradientVar.assign(isInverted.select(gradientVar.negate(), gradientVar));
}
