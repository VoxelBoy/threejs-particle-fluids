import { If, abs, bool, float, uint, vec3 } from 'three/tsl';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { ContactGeometryExtension } from '../core/index.js';

import { FLAG_RIGID } from './flags.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Phase 15a — Macklin 2014 §5.1 sparse-SDF rigid-rigid contact emitted as
 * a {@link ContactGeometryExtension}.
 *
 * The extension claims pairs whose participants both set
 * {@link FLAG_RIGID}. For those pairs it computes the contact normal and
 * penetration depth per paper eqs. 17–20 and writes them into the
 * caller's `outN` / `outD` cells; the shared `emitContactSolveCorrection`
 * then performs Δλ_n accumulation, normal scatter, and the §3.5 static-
 * friction cone. Friction therefore fires on rigid pairs through the
 * same kernel that handles fluid / softbody / cloth — no rigid-only
 * friction code exists anywhere.
 *
 * Algorithm per rigid-rigid pair `(i, j)`:
 *
 *   1. Bail if `|x_ij| ≥ 2·radius`, `|x_ij| < 1e-8`, or
 *      `contactInvMass[i] + contactInvMass[j] ≤ 0` (degenerate / non-
 *      penetrating / kinematic).
 *   2. Eq. 17 — pick the minimum-|φ| gradient as the contact normal.
 *      Strict `<` per paper; tie falls into the `−∇φ_j` branch.
 *   3. Boundary test — `|φ_i| < r OR |φ_j| < r` selects the eq. 20
 *      reflected-normal form and the `d = |x_ij| − r` penetration depth
 *      (paper §5.1 ¶3 "treating boundary particles as one-sided hard
 *      spheres"). Otherwise use the raw eq. 17 normal and `d_int =
 *      min(|φ|)`.
 *   4. Convert to the PBD-standard convention `emitContactSolveCorrection`
 *      expects: `outN` is the direction in which `i` separates and
 *      `outD > 0` is the penetration depth.
 *      - Interior branch: paper eq. 18's Δx_i ∝ -n_eq17 with d_int > 0
 *        translates to `outN = -n_eq17`, `outD = +d_int`.
 *      - Boundary branch: paper's `d = |x_ij| - r < 0` and the eq. 20
 *        reflected normal n_eq20 already point in the separating
 *        direction, so `outN = +n_eq20`, `outD = -d = r - |x_ij| > 0`.
 *   5. Set `outHandled = true`.
 *
 * Determinism: read-only inside the kernel (the per-pair scatter happens
 * downstream in `correction.ts`). Tier 1 bit-exact.
 */
export interface RigidGeometryExtensionArgs {
  /**
   * Per-particle `(φ, ∇φ_x, ∇φ_y, ∇φ_z)` baked at construction by the
   * voxelizer's `--sdf` mode (paper §5.1 Figure 7). Sized to
   * `particles.capacity`; non-rigid slots are unread (the extension
   * gates on `flags & FLAG_RIGID`).
   *
   * Sign convention (per voxelize.ts §"Gradient direction"): `φ ≥ 0`
   * inside, `∇φ` points INWARD into the body — the direction eq. 18's
   * `Δx_i ∝ -n` interprets correctly.
   */
  readonly restSDF: StorageBufferNode<'vec4'>;
}

export function buildRigidGeometryExtension(
  args: RigidGeometryExtensionArgs,
): ContactGeometryExtension {
  const { restSDF } = args;

  return {
    emit({ i, j, particles, outN, outD, outHandled }) {
      const r = 2 * particles.particleRadius; // contact diameter, paper §3
      const flagsI: Any = particles.flags.element(i).toVar();
      const flagsJ: Any = particles.flags.element(j).toVar();
      const bothRigid: Any = flagsI.bitAnd(flagsJ).bitAnd(uint(FLAG_RIGID)).notEqual(uint(0));

      // Only act on rigid-rigid pairs; everything else falls through to
      // the next extension or the spherical default.
      If(bothRigid, () => {
        const xiStar: Any = particles.predictedPositions.element(i).xyz.toVar();
        const xjStar: Any = particles.predictedPositions.element(j).xyz.toVar();
        const xij: Any = xiStar.sub(xjStar).toVar();
        const distSq: Any = xij.dot(xij).toVar();

        // Mirror solve.ts's spherical guards: pair generator emits out to
        // ~`r·radiusExpansion`, so non-penetrating candidates inside the
        // emit band must be skipped — eqs. 17–20 are sign-correct only
        // for penetrating pairs.
        const wSum: Any = particles.contactInvMass
          .element(i)
          .add(particles.contactInvMass.element(j))
          .toVar();
        const dist: Any = distSq.sqrt().toVar();
        const candidate: Any = distSq
          .lessThan(float(r * r))
          .and(dist.greaterThan(float(1e-8)))
          .and(wSum.greaterThan(float(0.0)));

        If(candidate, () => {
          const sdfI: Any = restSDF.element(i).toVar();
          const phiI: Any = sdfI.x.toVar();
          const gradI: Any = vec3(sdfI.y, sdfI.z, sdfI.w).toVar();
          const absPhiI: Any = abs(phiI).toVar();

          const sdfJ: Any = restSDF.element(j).toVar();
          const phiJ: Any = sdfJ.x.toVar();
          const gradJ: Any = vec3(sdfJ.y, sdfJ.z, sdfJ.w).toVar();
          const absPhiJ: Any = abs(phiJ).toVar();

          // Eq. 17 — minimum-|φ| gradient. Strict `<` per paper; tie
          // falls into the `−∇φ_j` branch.
          const useI: Any = absPhiI.lessThan(absPhiJ);
          const nRawX: Any = useI.select(gradI.x, gradJ.x.negate());
          const nRawY: Any = useI.select(gradI.y, gradJ.y.negate());
          const nRawZ: Any = useI.select(gradI.z, gradJ.z.negate());
          const nRawVec: Any = vec3(nRawX, nRawY, nRawZ).toVar();

          // Boundary case (paper §5.1 ¶3) — use eq. 20 reflected normal +
          // `d = |x_ij| − r` (signed; negative penetrating).
          const isBoundary: Any = absPhiI.lessThan(float(r)).or(absPhiJ.lessThan(float(r)));

          // Eq. 20 reflection: when xij·nRaw < 0, reflect xij about the
          // plane perpendicular to nRaw; otherwise pass xij through. Then
          // normalise so eq. 18 sees a unit vector regardless of branch.
          const xDotN: Any = xij.dot(nRawVec).toVar();
          const reflected: Any = xij.sub(nRawVec.mul(xDotN.mul(float(2.0)))).toVar();
          const boundaryRaw: Any = vec3(
            xDotN.lessThan(float(0.0)).select(reflected.x, xij.x),
            xDotN.lessThan(float(0.0)).select(reflected.y, xij.y),
            xDotN.lessThan(float(0.0)).select(reflected.z, xij.z),
          ).toVar();
          const invDist: Any = float(1.0).div(dist.add(float(1e-20)));
          const boundaryUnit: Any = boundaryRaw.mul(invDist).toVar();

          // Convert paper convention → PBD-standard expected by
          // `emitContactSolveCorrection`. See header doc step 4.
          //
          //   Interior: paper Δx_i ∝ -n_eq17 with d_int > 0, so to match
          //             the helper's Δx_i = +n · wi · d/wSum we pass
          //             outN = -n_eq17 and outD = +d_int.
          //   Boundary: paper's reflected normal already points in the
          //             separating direction; outN = +n_eq20 and
          //             outD = r - |x_ij| > 0.
          const dInteriorPositive: Any = absPhiI.min(absPhiJ);
          const dBoundaryPositive: Any = float(r).sub(dist);
          const finalNX: Any = isBoundary.select(boundaryUnit.x, nRawVec.x.negate());
          const finalNY: Any = isBoundary.select(boundaryUnit.y, nRawVec.y.negate());
          const finalNZ: Any = isBoundary.select(boundaryUnit.z, nRawVec.z.negate());
          const finalD: Any = isBoundary.select(dBoundaryPositive, dInteriorPositive);

          outN.assign(vec3(finalNX, finalNY, finalNZ));
          outD.assign(finalD);
          outHandled.assign(bool(true));
        });
      });
    },
  };
}
