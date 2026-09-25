import { InstancedMesh, Matrix4, PlaneGeometry } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  Discard,
  Fn,
  cameraPosition,
  cameraProjectionMatrix,
  cameraViewMatrix,
  cameraWorldMatrix,
  float,
  instanceIndex,
  positionLocal,
  positionWorld,
  uniform,
  uv,
  vec3,
  vec4,
} from 'three/tsl';
import type { FluidSystem } from '../../FluidSystem.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Pass 1 — particle-imposter depth.
 *
 * Two paths share this module:
 *
 *   - **Sphere imposter** (van der Laan, Green, Sainz 2009 §3.1) — each
 *     particle renders as a camera-aligned 1×1 quad. The fragment
 *     recovers `r²` from the unit-disk uv, discards outside the disk,
 *     and emits the sphere-surface view-space depth.
 *   - **Anisotropic-ellipsoid imposter** (Yu & Turk 2010 §4) — Phase 14c.
 *     Each particle reads its smoothed centre `x̄_i` and per-particle
 *     `G_i^{-1}` (6-float symmetric 3×3) from FluidSystem buffers; the
 *     vertex sizes the quad to the ellipsoid's screen-space AABB along
 *     the camera-right and camera-up basis, and the fragment runs a
 *     per-pixel ray-vs-ellipsoid quadratic. Activated via
 *     `buildDepthMaterial({ useAnisotropy: true })`. Falls back to the
 *     sphere path when the FluidSystem's anisotropy buffers haven't
 *     been allocated.
 *
 * Background pixels in the bound RT carry the NRF far-sentinel
 * (`NRF_FAR_SENTINEL = 1e6`); see `passes/smoothing.ts`.
 *
 * Switching paths requires rebuilding the material (different TSL node
 * graphs); the renderer flips on `params.anisotropy.enabled` change
 * (carried under U-FR-7's surface-mode flip cost analogue).
 */
export interface CreateDepthPassMaterialOptions {
  readonly fluidSystem: FluidSystem;
  /** Imposter radius uniform (m). Owned by `FluidSurfaceRenderer`. */
  readonly radiusUniform: ReturnType<typeof uniform<'float', number>>;
  /**
   * When `true`, build the Yu & Turk 2010 ellipsoid imposter path.
   * Requires `fluidSystem.enableAnisotropyBuffers()` to have been
   * called and the renderer's anisotropy compute kernel to have been
   * dispatched at least once before this material renders. Defaults
   * to `false` (sphere path).
   */
  readonly useAnisotropy?: boolean;
}

/**
 * Build the production sphere-imposter depth material. Renders to
 * `pass1RT`; reads `particles.positions` (post-advect committed state).
 */
export function buildDepthMaterial(options: CreateDepthPassMaterialOptions): MeshBasicNodeMaterial {
  const { fluidSystem, useAnisotropy = false } = options;
  if (useAnisotropy) {
    if (
      !fluidSystem.anisotropyDiag ||
      !fluidSystem.anisotropyOff ||
      !fluidSystem.smoothedPositions
    ) {
      throw new Error(
        'buildDepthMaterial: useAnisotropy=true requires fluidSystem.enableAnisotropyBuffers() to have been called',
      );
    }
    return buildEllipsoidImposterMaterial(options);
  }
  return buildSphereImposterMaterial(options);
}

function buildSphereImposterMaterial(
  options: CreateDepthPassMaterialOptions,
): MeshBasicNodeMaterial {
  const { fluidSystem, radiusUniform } = options;
  const { particles, fluidParticles } = fluidSystem;

  const material = new MeshBasicNodeMaterial();
  material.transparent = false;
  material.depthWrite = true;
  material.depthTest = true;

  const fluidStart = float(fluidParticles.start).toUint();
  const slotIdx = instanceIndex.add(fluidStart);
  const particleWorld: Any = (particles.positions as Any).element(slotIdx).xyz;

  const cameraRight: Any = (cameraWorldMatrix as Any).element(0).xyz;
  const cameraUp: Any = (cameraWorldMatrix as Any).element(1).xyz;

  const quadOffsetXY: Any = positionLocal.xy.mul(2.0).mul(radiusUniform);
  const offsetWorld: Any = cameraRight.mul(quadOffsetXY.x).add(cameraUp.mul(quadOffsetXY.y));

  const billboardedWorld: Any = particleWorld.add(offsetWorld);
  material.positionNode = billboardedWorld;

  const viewCenterView: Any = cameraViewMatrix.mul(vec4(particleWorld, 1.0)).xyz;
  const viewCenterZ: Any = viewCenterView.z.toVarying('depthPassViewCenterZ');

  const surfaceDepth: Any = Fn(() => {
    const diskUv: Any = uv().mul(2.0).sub(1.0);
    const r2: Any = diskUv.dot(diskUv);
    Discard(r2.greaterThan(1.0));

    const zOffset: Any = r2.oneMinus().sqrt().mul(radiusUniform);
    const viewSurfaceZ: Any = viewCenterZ.add(zOffset);

    return viewSurfaceZ.negate();
  })().toVar();
  material.outputNode = vec4(surfaceDepth, 0, 0, 1);
  // Depth-test the actual front of the sphere/ellipsoid, not its billboard center.
  material.depthNode = (cameraProjectionMatrix as Any)
    .element(3)
    .z.div(surfaceDepth)
    .sub((cameraProjectionMatrix as Any).element(2).z);

  return material;
}

/**
 * Yu & Turk 2010 §4 ellipsoid imposter material. Reads:
 *   - `smoothedPositions[i]` — eq. 6 Laplacian-smoothed centre `x̄_i`
 *   - `anisotropyDiag[i]` — diagonal of `G_i^{-1}` (g_xx, g_yy, g_zz)
 *   - `anisotropyOff[i]` — off-diagonal of `G_i^{-1}` (g_xy, g_xz, g_yz)
 *
 * Vertex stage: sizes the imposter quad to the ellipsoid's screen-space
 * AABB along the camera-right / camera-up basis. The half-extent along
 * a screen direction `ê` (world-space) is `||G^{-1} · ê||` — the
 * standard projection-of-an-ellipsoid-onto-a-direction formula derived
 * from `ellipsoid = { x̄ + G^{-1} · u : |u| ≤ 1 }`.
 *
 * Fragment stage: for each pixel, build the world-space ray from the
 * camera through the fragment's interpolated billboard position, then
 * solve `|u(t)|² = 1` where `u(t) = G · (cam + t·d − x̄)`. `G` is
 * obtained as the inverse of the symmetric 3×3 `G^{-1}` (cofactor /
 * det formula, ~10 ops). Discard pixels with negative discriminant
 * (the ray misses the ellipsoid). Emit view-space z of the front-side
 * intersection point as `vec4(-viewZ, 0, 0, 1)` matching the sphere-
 * imposter encoding.
 *
 * Phase 14b finding (TSL `positionView`/`positionWorld` don't track
 * `positionNode` overrides) applies here: we use `positionWorld` from
 * the auto-pipeline's geometry vertex (the local-vertex × instance ×
 * model matrix product) which IS the quad's world-space position
 * because the quad lives at the origin and the `positionNode` override
 * supplies the world position directly. Empirically `positionWorld`
 * gives the interpolated billboard point in this code path.
 */
function buildEllipsoidImposterMaterial(
  options: CreateDepthPassMaterialOptions,
): MeshBasicNodeMaterial {
  const { fluidSystem, radiusUniform } = options;
  const { fluidParticles } = fluidSystem;
  const anisotropyDiag = fluidSystem.anisotropyDiag!;
  const anisotropyOff = fluidSystem.anisotropyOff!;
  const smoothedPositions = fluidSystem.smoothedPositions!;

  const material = new MeshBasicNodeMaterial();
  material.transparent = false;
  material.depthWrite = true;
  material.depthTest = true;

  const fluidStart = float(fluidParticles.start).toUint();
  const slotIdx = instanceIndex.add(fluidStart);

  // Per-particle anisotropy state (read once per vertex; identical
  // across all four quad vertices).
  const xBar: Any = (smoothedPositions as Any).element(slotIdx).xyz;
  const aniDiag: Any = (anisotropyDiag as Any).element(slotIdx).xyz;
  const aniOff: Any = (anisotropyOff as Any).element(slotIdx).xyz;
  // G^{-1} as a 3×3 symmetric matrix:
  //   [ aniDiag.x  aniOff.x  aniOff.y ]
  //   [ aniOff.x   aniDiag.y aniOff.z ]
  //   [ aniOff.y   aniOff.z  aniDiag.z ]

  const cameraRight: Any = (cameraWorldMatrix as Any).element(0).xyz;
  const cameraUp: Any = (cameraWorldMatrix as Any).element(1).xyz;

  // Screen-space AABB extents — half-axis lengths of the projected
  // ellipsoid along (cameraRight, cameraUp). With Yu & Turk's
  // dimensionless `Σ̃ = k_s · diag(σ_k)` and `G^-1 = h · R · Σ̃ · R^T`,
  // the result already carries metres; multiplying by `radiusUniform`
  // (which has the per-particle radius `r` baked in) would compound
  // dimensions and squash the imposter to sub-millimetre size. The
  // user's "Imposter Radius" slider only meaningfully applies to
  // sphere imposters; the ellipsoid path is sized by `k_s` instead.
  const gInvCRight: Any = symMatVec(aniDiag, aniOff, cameraRight);
  const gInvCUp: Any = symMatVec(aniDiag, aniOff, cameraUp);
  const extX: Any = gInvCRight.length();
  const extY: Any = gInvCUp.length();
  void radiusUniform;

  // Quad position: x̄ + cameraRight·(2·s·extX) + cameraUp·(2·t·extY)
  // where (s, t) ∈ [-0.5, 0.5] from positionLocal (PlaneGeometry(1,1)).
  const quadOffsetXY: Any = positionLocal.xy.mul(2.0);
  const offsetWorld: Any = cameraRight
    .mul(quadOffsetXY.x.mul(extX))
    .add(cameraUp.mul(quadOffsetXY.y.mul(extY)));
  const billboardedWorld: Any = xBar.add(offsetWorld);
  material.positionNode = billboardedWorld;

  // Pass per-instance scalars to the fragment as varyings — they're
  // identical across the quad's four vertices (per-instance constant)
  // so interpolation collapses to the same value.
  const xBarVx: Any = xBar.x.toVarying('aniXBarX');
  const xBarVy: Any = xBar.y.toVarying('aniXBarY');
  const xBarVz: Any = xBar.z.toVarying('aniXBarZ');
  const aniDiagVx: Any = aniDiag.x.toVarying('aniDiagX');
  const aniDiagVy: Any = aniDiag.y.toVarying('aniDiagY');
  const aniDiagVz: Any = aniDiag.z.toVarying('aniDiagZ');
  const aniOffVx: Any = aniOff.x.toVarying('aniOffX');
  const aniOffVy: Any = aniOff.y.toVarying('aniOffY');
  const aniOffVz: Any = aniOff.z.toVarying('aniOffZ');

  const surfaceDepth: Any = Fn(() => {
    const xBarF: Any = vec3(xBarVx, xBarVy, xBarVz);
    const aniDiagF: Any = vec3(aniDiagVx, aniDiagVy, aniDiagVz);
    const aniOffF: Any = vec3(aniOffVx, aniOffVy, aniOffVz);

    // Invert the symmetric 3×3 G^{-1} → G via cofactor / det.
    // For symmetric M = [[a, d, e], [d, b, f], [e, f, c]]:
    //   det = a(bc − f²) − d(dc − ef) + e(df − be)
    //   inv = (1/det) · cofactor^T  (cofactor of symmetric matrix is symmetric)
    const a: Any = aniDiagF.x;
    const b: Any = aniDiagF.y;
    const c: Any = aniDiagF.z;
    const d: Any = aniOffF.x;
    const e: Any = aniOffF.y;
    const f: Any = aniOffF.z;
    const detM: Any = a
      .mul(b.mul(c).sub(f.mul(f)))
      .sub(d.mul(d.mul(c).sub(f.mul(e))))
      .add(e.mul(d.mul(f).sub(b.mul(e))));
    const invDet: Any = float(1.0).div(detM.abs().max(float(1e-20)).mul(detM.sign()));
    // Cofactor entries (symmetric):
    const ga: Any = b.mul(c).sub(f.mul(f)).mul(invDet);
    const gb: Any = a.mul(c).sub(e.mul(e)).mul(invDet);
    const gc: Any = a.mul(b).sub(d.mul(d)).mul(invDet);
    const gd: Any = e.mul(f).sub(d.mul(c)).mul(invDet);
    const ge: Any = d.mul(f).sub(b.mul(e)).mul(invDet);
    const gf: Any = d.mul(e).sub(a.mul(f)).mul(invDet);
    const gDiag: Any = vec3(ga, gb, gc);
    const gOff: Any = vec3(gd, ge, gf);

    // Ray from camera through the interpolated quad world position.
    // `positionWorld` IS the quad's world-space position because
    // `positionNode = billboardedWorld` supplies it directly.
    const rayO: Any = cameraPosition;
    const rayD: Any = positionWorld.sub(rayO).normalize();

    // Transform into u-space: u(t) = G · (rayO + t·rayD − x̄)
    //                              = G · (rayO − x̄) + t · G · rayD
    const oRel: Any = rayO.sub(xBarF);
    const uO: Any = symMatVec(gDiag, gOff, oRel);
    const uD: Any = symMatVec(gDiag, gOff, rayD);

    // Quadratic |u_o + t·u_d|² = 1:
    //   A·t² + B·t + C = 0
    const A: Any = uD.dot(uD);
    const B: Any = uO.dot(uD).mul(2.0);
    const C: Any = uO.dot(uO).sub(1.0);
    const disc: Any = B.mul(B).sub(A.mul(C).mul(4.0));
    Discard(disc.lessThan(0.0));

    // Near-side intersection: t = (-B − sqrt(disc)) / (2A).
    // For A > 0 (always — uD is non-zero unless degenerate), the
    // smaller t corresponds to the camera-near surface.
    const sqrtD: Any = disc.sqrt();
    const t: Any = B.negate().sub(sqrtD).div(A.mul(2.0));
    const hit: Any = rayO.add(rayD.mul(t));

    // View-space z of the hit point (matches sphere-imposter encoding).
    const hitView: Any = cameraViewMatrix.mul(vec4(hit, 1.0)).xyz;
    const viewSurfaceZ: Any = hitView.z;

    return viewSurfaceZ.negate();
  })().toVar();
  material.outputNode = vec4(surfaceDepth, 0, 0, 1);
  // Depth-test the actual front of the sphere/ellipsoid, not its billboard center.
  material.depthNode = (cameraProjectionMatrix as Any)
    .element(3)
    .z.div(surfaceDepth)
    .sub((cameraProjectionMatrix as Any).element(2).z);

  return material;
}

/**
 * Symmetric 3×3 matrix–vector product.
 *   M = [[d.x, o.x, o.y], [o.x, d.y, o.z], [o.y, o.z, d.z]]
 *   M · v = (d.x·v.x + o.x·v.y + o.y·v.z,
 *            o.x·v.x + d.y·v.y + o.z·v.z,
 *            o.y·v.x + o.z·v.y + d.z·v.z)
 */
function symMatVec(diag: Any, off: Any, v: Any): Any {
  const x: Any = diag.x.mul(v.x).add(off.x.mul(v.y)).add(off.y.mul(v.z));
  const y: Any = off.x.mul(v.x).add(diag.y.mul(v.y)).add(off.z.mul(v.z));
  const z: Any = off.y.mul(v.x).add(off.z.mul(v.y)).add(diag.z.mul(v.z));
  return vec3(x, y, z);
}

/** A depth-pass mesh wrapper holding the `InstancedMesh` + production material. */
export interface DepthPassMesh {
  readonly mesh: InstancedMesh;
  readonly material: MeshBasicNodeMaterial;
}

/**
 * Construct the InstancedMesh that drives the depth pass. Identity-seed
 * the per-instance matrices so the standard pipeline's
 * `positionNode × instanceMatrix` doesn't collapse every quad to a point.
 */
export function createDepthPassMesh(options: CreateDepthPassMaterialOptions): DepthPassMesh {
  const material = buildDepthMaterial(options);
  const geometry = new PlaneGeometry(1, 1);
  const mesh = new InstancedMesh(geometry, material, options.fluidSystem.fluidParticles.count);
  mesh.name = 'FluidDepthPassMesh';
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;

  const identity = new Matrix4();
  for (let i = 0; i < options.fluidSystem.fluidParticles.count; i++) {
    mesh.setMatrixAt(i, identity);
  }
  mesh.instanceMatrix.needsUpdate = true;

  return { mesh, material };
}
