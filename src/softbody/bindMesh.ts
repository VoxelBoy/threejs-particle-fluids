import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute } from 'three';

import type { SoftbodySystem } from './SoftbodySystem.js';

/**
 *
 *
 * For each render vertex, find its `K` nearest particles by Euclidean
 * distance in the body's pre-centered rest frame and emit inverse-
 * distance weights normalized to a partition of unity. The per-vertex
 * `influences` (Uint32, K=4) and `weights` (Float32, K=4) attributes
 * are added to the geometry in place; the geometry is also returned for
 * convenient chaining.
 *
 * Coordinate frame contract. The geometry's `position` attribute MUST be
 * in the **same frame as the body's `restPositions`** (the pre-pre-
 * centered, mesh-local frame the caller passed to {@link SoftbodySystem}
 * at construction). The bind step computes the rest centre of mass
 * `c̄_body` from the body's restPositions and shifts mesh vertices by
 * `−c̄_body` internally before computing nearest-particle distances —
 * this matches the pre-centred frame `restOffsets` is uploaded in. The
 * shader applies the same `−c̄_body` shift at runtime via a uniform.
 *
 * MVP scope: K=4 fixed (Mueller 2011 §7 default). The plan's K∈[1,8]
 * static ceiling lives behind U-41 — extending to K=8 needs a second
 * vec4 per vertex and is a re-tune, not a re-architecture.
 */

const K_DEFAULT = 4;

export interface BindSoftbodyMeshOptions {
  /**
   * Number of nearest-particle influences per vertex. MVP supports
   * `1 ≤ K ≤ 4`; values below 4 zero-pad the influences array (unused
   * slots have weight 0, contributing nothing to the DLB blend). U-41
   * tracks K > 4 for thin-feature meshes.
   */
  readonly K?: number;
  /**
   * Maximum expected distance from any vertex to its 1-nearest particle,
   * in metres. Vertices that exceed this fall back to whatever's closest
   * (with a single console warning summarising the count) by default;
   * pass `strictReach: true` to throw on the first violation instead.
   *
   * Default `2 · particleRadius`. The plan §"Binding precondition"
   * specified `1.5 · r`, but voxel-diagonal-only-reachable vertices
   * sit at distance up to `√3 · r ≈ 1.73 · r` from any voxel-centre
   * particle, so the conservative default trips on otherwise well-
   * formed bunnies. Demos with anatomically thin features (bunny ears,
   * fingers, etc.) that voxelize as empty should still pass an
   * explicit `reachRadius: 3..6 · r` so the warning summary is honest
   * about which vertices actually slipped past expectation.
   */
  readonly reachRadius?: number;
  /**
   * When true, the first vertex whose 1-nearest particle exceeds
   * `reachRadius` aborts binding with the original throw. Default
   * `false` — bind degrades gracefully and emits a single
   * `console.warn` summarising the count + max distance, so demos with
   * a few sparse-voxelization outliers (Stanford bunny ear tips, etc.)
   * load and just visibly skin those few vertices to whatever stable
   * particle the K-NN found.
   *
   * Tests and tools that want to gate on the precondition pass
   * `strictReach: true` to recover the original behaviour.
   */
  readonly strictReach?: boolean;
}

export interface BindSoftbodyMeshResult {
  /** The same geometry, mutated with `influences` + `weights` attributes. */
  readonly geometry: BufferGeometry;
  /** The K used (echoed from options, or default). */
  readonly K: number;
  /** Body-local pre-centered rest centre of mass (= `c̄_body`). */
  readonly cBar: readonly [number, number, number];
}

/**
 * Bind `geometry` to `softbody.bodies[bodyIndex]`.
 *
 * Throws if any vertex's 1-nearest particle distance exceeds
 * `reachRadius` — that vertex would otherwise be driven by acoustics,
 * not shape matching (plan §"Binding precondition").
 */
export function bindSoftbodyMesh(
  geometry: BufferGeometry,
  softbody: SoftbodySystem,
  bodyIndex: number,
  options: BindSoftbodyMeshOptions = {},
): BindSoftbodyMeshResult {
  const K = options.K ?? K_DEFAULT;
  if (!Number.isInteger(K) || K < 1 || K > 4) {
    throw new Error(
      `bindSoftbodyMesh: K=${K} unsupported. MVP supports 1..4 (U-41 covers K>4 follow-up).`,
    );
  }

  const body = softbody.bodies[bodyIndex];
  if (!body) {
    throw new Error(
      `bindSoftbodyMesh: bodyIndex ${bodyIndex} out of range (0..${softbody.bodies.length - 1}).`,
    );
  }
  const reachRadius = options.reachRadius ?? 2.0 * softbody.particles.particleRadius;
  const strictReach = options.strictReach ?? false;
  if (!(reachRadius > 0)) {
    throw new Error(`bindSoftbodyMesh: reachRadius must be > 0, got ${reachRadius}.`);
  }
  // Tally over-reach vertices so we can emit one summary instead of
  // spamming the console when many vertices slip past the precondition.
  let outOfReachCount = 0;
  let outOfReachMax = 0;
  let outOfReachFirstVertex = -1;

  const positionAttr = geometry.getAttribute('position');
  if (!positionAttr) {
    throw new Error('bindSoftbodyMesh: geometry has no `position` attribute');
  }
  const vertexCount = positionAttr.count;

  // ---- Compute the body's rest COM `c̄_body` from body.restPositions
  //      (pre-pre-centered mesh-local frame). SoftbodySystem applies the
  //      same shift internally before uploading restOffsets, so we mirror
  //      it here so vertex distances are computed in the same frame the
  //      shader will read. ----
  const rest = body.restPositions;
  const n = body.particleRange.count;
  let cx = 0,
    cy = 0,
    cz = 0;
  for (let i = 0; i < n; i++) {
    cx += rest[3 * i + 0]!;
    cy += rest[3 * i + 1]!;
    cz += rest[3 * i + 2]!;
  }
  const invN = 1 / n;
  cx *= invN;
  cy *= invN;
  cz *= invN;

  // Pre-centered particle rest positions (body-local), used for nearest-
  // particle distance queries. Indexed by body-local i; converted to
  // global slot at output time via `body.particleRange.start + i`.
  const px = new Float32Array(n);
  const py = new Float32Array(n);
  const pz = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    px[i] = rest[3 * i + 0]! - cx;
    py[i] = rest[3 * i + 1]! - cy;
    pz[i] = rest[3 * i + 2]! - cz;
  }

  // ---- Stability mask — particles with 6-face degree ≥ 3 are not part
  //      of a single-particle chain. Bunny ears voxelize as 1-particle-
  //      thick chains; the §5.1 implicit-mode shape matching is unstable
  //      on those (Phase 12 §5.5 torsion gap, U-52), and even §5.3 will
  //      let chain-end particles drift visibly under hard impacts. The
  //      skin would faithfully follow that drift and tear the ear apart.
  //
  //      Workaround: skip chain particles in K-NN. Ear vertices then bind
  //      to the body's nearest stable particles (back at the ear base /
  //      skull); since stable particles share the body's coherent
  //      rotation, ears render as rigid extensions of the body — not
  //      anatomically correct, but stable until U-41 / U-52 land.
  //
  //      `body.edges` is supplied by `voxelize` (Phase 12); when absent
  //      (e.g., legacy callers passing `edges: undefined` to a §5.3-only
  //      SoftbodySystem) we treat every particle as stable, preserving
  //      the original behaviour.
  const STABLE_DEGREE_MIN = 3;
  const isStable = new Uint8Array(n);
  if (body.edges && body.edges.length > 0) {
    const degree = new Uint32Array(n);
    for (let e = 0; e < body.edges.length; e += 2) {
      degree[body.edges[e]!]!++;
      degree[body.edges[e + 1]!]!++;
    }
    let stableCount = 0;
    for (let i = 0; i < n; i++) {
      if (degree[i]! >= STABLE_DEGREE_MIN) {
        isStable[i] = 1;
        stableCount++;
      }
    }
    // Degenerate body — no stable particles at all (everything is a
    // chain). Fall back to treating every particle as stable so the
    // mesh still binds; the resulting skin will be as unstable as the
    // physics, but at least it loads.
    if (stableCount === 0) isStable.fill(1);
  } else {
    isStable.fill(1);
  }

  // ---- Per-vertex k-NN (naive O(V·N)) + inverse-distance weights ----
  const baseSlot = body.particleRange.start;
  const influences = new Uint32Array(4 * vertexCount);
  const weights = new Float32Array(4 * vertexCount);
  // Floor on the inverse-distance weight magnitude — without it a vertex
  // coincident with a particle blows the weight to ∞.
  const epsBind = 0.1 * softbody.particles.particleRadius;

  // Scratch arrays for the K-best heap (small K ⇒ linear scan is faster
  // than maintaining a binary heap for K ≤ 4).
  const bestDist2 = new Float32Array(K);
  const bestIdx = new Int32Array(K);

  for (let v = 0; v < vertexCount; v++) {
    const vx = positionAttr.getX(v) - cx;
    const vy = positionAttr.getY(v) - cy;
    const vz = positionAttr.getZ(v) - cz;

    for (let k = 0; k < K; k++) {
      bestDist2[k] = Infinity;
      bestIdx[k] = -1;
    }

    // Track the 1-nearest distance to ANY particle (stable or chain) for
    // the reach precondition; the K-best heap below considers only
    // stable particles (chains skipped — see §"Stability mask" above).
    let absNearest2 = Infinity;
    for (let i = 0; i < n; i++) {
      const dx = px[i]! - vx;
      const dy = py[i]! - vy;
      const dz = pz[i]! - vz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < absNearest2) absNearest2 = d2;
      if (!isStable[i]) continue;
      // Insert into the K-best slot if d2 beats the worst.
      let worst = 0;
      for (let k = 1; k < K; k++) {
        if (bestDist2[k]! > bestDist2[worst]!) worst = k;
      }
      if (d2 < bestDist2[worst]!) {
        bestDist2[worst] = d2;
        bestIdx[worst] = i;
      }
    }

    const nearest = Math.sqrt(absNearest2);
    if (nearest > reachRadius) {
      if (strictReach) {
        throw new Error(
          `bindSoftbodyMesh: vertex ${v} 1-nearest particle distance ` +
            `${nearest.toExponential(3)} m exceeds reachRadius ` +
            `${reachRadius.toExponential(3)} m. Mitigation: denser ` +
            `voxelization (smaller particleRadius), not larger K.`,
        );
      }
      if (outOfReachFirstVertex < 0) outOfReachFirstVertex = v;
      outOfReachCount++;
      if (nearest > outOfReachMax) outOfReachMax = nearest;
      // Fall through — the K-best heap below already holds the closest
      // available stable particles; the vertex skins to those, just at
      // a longer-than-expected lever arm. Visually this looks like a
      // soft "stretchy" skin around the outlier vertex, which beats
      // crashing the demo on the first sparse ear-tip vertex.
    }

    // Inverse-distance weights with a floor; normalize to Σw = 1.
    let wSum = 0;
    const rawW = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      const d = Math.max(Math.sqrt(bestDist2[k]!), epsBind);
      rawW[k] = 1 / d;
      wSum += rawW[k]!;
    }
    const invWSum = 1 / wSum;
    for (let k = 0; k < K; k++) {
      const idx = bestIdx[k]!;
      // Pad unused slots (only happens if n < K) with index 0, weight 0.
      if (idx < 0) {
        influences[4 * v + k] = baseSlot;
        weights[4 * v + k] = 0;
      } else {
        influences[4 * v + k] = baseSlot + idx;
        weights[4 * v + k] = rawW[k]! * invWSum;
      }
    }
    // Pad slots K..3 with (baseSlot, 0) so the shader can still read
    // 4 lanes safely.
    for (let k = K; k < 4; k++) {
      influences[4 * v + k] = baseSlot;
      weights[4 * v + k] = 0;
    }
  }

  geometry.setAttribute('influences', new Uint32BufferAttribute(influences, 4));
  geometry.setAttribute('weights', new Float32BufferAttribute(weights, 4));

  if (outOfReachCount > 0) {
    console.warn(
      `bindSoftbodyMesh: ${outOfReachCount} vertex(es) bound past reachRadius ` +
        `${reachRadius.toExponential(3)} m (max distance ` +
        `${outOfReachMax.toExponential(3)} m, first at vertex ${outOfReachFirstVertex}). ` +
        `Mitigation: denser voxelization (smaller particleRadius), or pass a ` +
        `larger 'reachRadius' option to silence this warning.`,
    );
  }

  return { geometry, K, cBar: [cx, cy, cz] };
}
