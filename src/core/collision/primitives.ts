import { If, float, max, uint, vec3, vec4 } from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Primitive kinds, stored in the high 16 bits of each slot's packed word. */
export const KIND_PLANE = 0;
export const KIND_SPHERE = 1;
export const KIND_BOX = 2;
export const KIND_CAPSULE = 3;

/** Flag bit (low 16 bits of the packed word): the inside is the valid region. */
export const FLAG_INVERT = 1 << 0;

/** Rotate `v` by unit quaternion `q`: `v + 2w(q × v) + q × (2(q × v))`. */
function emitQuatRotate(q: Any, v: Any): Any {
  const qxyz: Any = q.xyz;
  const t: Any = float(2.0).mul(qxyz.cross(v)).toVar();
  return v.add(t.mul(q.w)).add(qxyz.cross(t));
}

function emitQuatConjugate(q: Any): Any {
  return vec4(q.x.negate(), q.y.negate(), q.z.negate(), q.w);
}

/** Lengths below this are treated as zero when normalizing a gradient. */
const EPSILON_GRADIENT = 1e-8;

/** GPU buffers {@link emitColliderSdf} reads; see {@link PrimitiveSet}. */
export interface ColliderFields {
  readonly packed: Any;
  readonly data0: Any;
  readonly data1: Any;
  readonly rotation: Any;
  readonly velocity?: Any;
}

/**
 * Emit TSL that evaluates the signed distance `phi` to the primitive in
 * `colliderSlot` and its outward unit gradient, writing both into the given
 * vars. Distances are positive outside the shape; inverted primitives flip
 * both signs.
 *
 * - Plane: `phi = n · (x − p)`
 * - Sphere: `phi = |x − c| − r`
 * - Box: `q = |x_local| − e`, `phi = |max(q, 0)| + min(max(q.x, q.y, q.z), 0)`
 * - Capsule: sphere around the closest point on the segment
 *
 * `rewind` (seconds) moves the query point forward along the primitive's
 * velocity, which is the same as moving the primitive back in time.
 */
export function emitColliderSdf(
  fields: ColliderFields,
  colliderSlot: Any,
  xWorld: Any,
  phiVar: Any,
  gradientVar: Any,
  rewind?: Any,
): void {
  // Evaluating at x + v·t is the same as moving a translating collider back by v·t.
  const x: Any =
    rewind !== undefined && fields.velocity
      ? xWorld.add(fields.velocity.element(colliderSlot).xyz.mul(rewind)).toVar()
      : xWorld;
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
  // back to world space by the forward quaternion.
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
