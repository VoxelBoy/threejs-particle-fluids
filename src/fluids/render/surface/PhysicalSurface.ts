import { Color, Mesh, PlaneGeometry, type CubeTexture, type Texture } from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import {
  Fn,
  cameraProjectionMatrix,
  cameraWorldMatrix,
  cross,
  float,
  int,
  normalize,
  positionLocal,
  screenUV,
  select,
  texture,
  textureSize,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { NRF_FAR_SENTINEL_THRESHOLD } from '../passes/smoothing.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * PBR fluid surface backed by `MeshPhysicalNodeMaterial`. Lives in the
 * user's scene; standard render handles lights / IBL / transmission.
 *
 * The mesh is a tessellated `PlaneGeometry` placed at world origin with
 * identity transform. Each vertex's `positionNode` reads NRF-smoothed
 * depth at the vertex UV and reconstructs the world-space surface
 * point per van der Laan eq. 3 + the camera's inverse view matrix.
 *
 * **Phase 14b fix — `normalNode` override.** Phase 14a's PBR composite
 * shipped only `positionNode`, leaving `normalNode` to fall back to
 * the geometry's flat (0, 0, 1) plane normal. Every fragment then saw
 * the same normal → uniform Fresnel + uniform reflection + no surface
 * relief. Green 2010 GDC slide *"Calculating Normals (code)"*
 * prescribes per-fragment `dFdx` / `dFdy` of view-space position +
 * cross product as the depth-derived normal source — same recipe the
 * Phase 14a custom composite (`composite.ts`) used. The view-space
 * position is reconstructed *locally* in the fragment shader (from
 * the depth sample + projection-matrix focal lengths + screen
 * coordinate) rather than read from the `positionView` TSL accessor;
 * three.js's `positionView` varying tracks the original `positionLocal`
 * vertex coord, not the `positionNode` override, so its derivatives
 * give the flat plane's normal — which would then "rotate with the
 * camera" after the world-space basis change.
 *
 * What the PBR material gets us "for free" once normalNode is right:
 *   - Per-fragment lighting from scene lights and IBL irradiance
 *   - IBL specular radiance from `scene.environment`
 *   - Beer-Lambert transmission via `attenuationColor` +
 *     `attenuationDistance` + `thicknessNode`
 *   - Fresnel from `ior`
 *   - Roughness / clearcoat / etc. through the standard PBR knobs
 *
 * What still runs as pre-passes (caller responsibility):
 *   - Pass 1 imposters → `pass1RT`
 *   - Pass 2 NRF → `finalRT` (consumed via `smoothedDepthTexture`)
 *   - Pass 3 thickness → `thicknessRT` (consumed via `thicknessTexture`)
 *
 * Sentinel-depth vertices (background = 1e6) are alpha-discarded via
 * `opacityNode` — vertex still emits a position (degenerate quads
 * stretch to far-Z) but every fragment in those triangles fails the
 * alpha test and is discarded by three.js. Keeps the silhouette
 * pixel-accurate even when tessellation is much coarser than canvas
 * resolution.
 */

export interface PhysicalSurfaceOptions {
  readonly smoothedDepthTexture: Texture;
  readonly thicknessTexture: Texture;
  readonly fluidColor?: Color;
  /** Tessellation segments per axis. 128 ≈ 8 px/triangle at 1080p. */
  readonly tessellation?: number;
  /** Index of refraction. Drives Fresnel + transmission direction. */
  readonly ior?: number;
  /** Beer-Lambert decay distance toward the absorption tint. */
  readonly attenuationDistance?: number;
  /** PBR microfacet roughness. 0 = mirror surface. */
  readonly roughness?: number;
  /** Initial multiplier mapping accumulated thickness alpha → metres. */
  readonly thicknessScale?: number;
  /** Initial environment-reflection intensity. */
  readonly envIntensity?: number;
  /**
   * Scene environment (typically `scene.environment` — the PMREM-
   * prefiltered IBL source). When provided, assigned to the material's
   * `envMap` slot so `envMapIntensity` becomes a per-material override
   * per three.js's `materialEnvIntensity` accessor (which falls back to
   * `scene.environmentIntensity` when `material.envMap` is unset).
   */
  readonly environment?: CubeTexture | Texture | null;
}

export interface PhysicalSurfaceMesh {
  readonly mesh: Mesh;
  readonly material: MeshPhysicalNodeMaterial;
  /** Live multiplier mapping accumulated thickness alpha to metres. */
  readonly thicknessScaleUniform: ReturnType<typeof uniform<'float', number>>;
}

export function createPhysicalSurfaceMesh(options: PhysicalSurfaceOptions): PhysicalSurfaceMesh {
  const tessellation = options.tessellation ?? 256;
  const fluidColor = options.fluidColor ?? new Color(0x0a2840);

  const material = new MeshPhysicalNodeMaterial();
  material.transparent = true;
  // depthWrite stays true (set explicitly with the depthNode override
  // below) so the fluid surface participates in the depth buffer with
  // the SAMPLED depth, not the per-vertex interpolated depth. Required
  // for correct depth integration against scene geometry inside or
  // behind the fluid.
  material.depthWrite = true;
  material.transmission = 1.0;
  material.roughness = options.roughness ?? 0.0;
  material.metalness = 0.0;
  material.ior = options.ior ?? 1.33;
  material.attenuationColor = fluidColor;
  material.attenuationDistance = options.attenuationDistance ?? 3.0;
  material.envMapIntensity = options.envIntensity ?? 1.0;
  // three.js's `materialEnvIntensity` accessor (used by EnvironmentNode
  // for IBL radiance / irradiance) reads `material.envMapIntensity`
  // ONLY when `material.envMap` is set; otherwise it falls back to
  // `scene.environmentIntensity`. We want a per-material intensity
  // override, so assign the harness's `scene.environment` (PMREM-
  // prefiltered IBL texture) to `material.envMap`. This keeps the
  // sampled environment identical to the scene-wide IBL while making
  // the intensity slider effective.
  if (options.environment) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    material.envMap = options.environment as any;
  }
  material.alphaTest = 0.5;

  const finalRT: Any = options.smoothedDepthTexture;
  const SENTINEL_THRESHOLD = float(NRF_FAR_SENTINEL_THRESHOLD);

  // ---- Vertex stage: positionNode reads NRF-smoothed depth and
  // reconstructs the world-space surface point.
  //
  // positionLocal.xy ∈ [-1, 1] from the tessellated 2×2 plane. Map to
  // UV [0, 1], sample finalRT for the positive view-space distance,
  // negate to get viewZ, then apply paper eq. 3:
  //   P_view = (W_x · z, W_y · z, z), W_x = NDC_x / F_x, W_y = NDC_y / F_y
  // World-space = cameraWorldMatrix × view-space (camera → world).
  //
  // Sentinel-depth vertices clamp to a moderate-far depth so the
  // triangles don't stretch to infinity; the fragment opacity discard
  // handles them anyway.
  material.positionNode = Fn(() => {
    const uvCoord: Any = positionLocal.xy.mul(0.5).add(0.5);
    const depthSample: Any = texture(finalRT, uvCoord).r;
    const isBg: Any = depthSample.greaterThanEqual(SENTINEL_THRESHOLD);
    const safeDepth: Any = select(isBg, float(10.0), depthSample);
    const viewZ: Any = safeDepth.negate();

    // Inverse-projection: NDC.x = Fx · viewX / clip.w, with clip.w =
    // -viewZ (camera looks down -Z view). Therefore
    //   viewX = NDC.x · (-viewZ) / Fx = NDC.x · safeDepth / Fx
    // (same for y). The earlier formula `viewX = NDC.x · viewZ / Fx`
    // was off by a sign — for a horizontal water surface viewed at an
    // oblique angle, that mirrored the reconstructed worldPos across
    // the camera's right/up axes, which fed into three.js's IBL
    // refraction and made `getIBLVolumeRefraction` sample the
    // framebuffer at flipped screen UVs (= the wrong/empty region of
    // the opaque-pass texture, which reads as black).
    const Fx: Any = (cameraProjectionMatrix as Any).element(0).x;
    const Fy: Any = (cameraProjectionMatrix as Any).element(1).y;
    const Wx: Any = positionLocal.x.div(Fx);
    const Wy: Any = positionLocal.y.div(Fy);
    const viewPos: Any = vec3(Wx.mul(safeDepth), Wy.mul(safeDepth), viewZ);

    const worldPos4: Any = (cameraWorldMatrix as Any).mul(vec4(viewPos, 1.0));
    return worldPos4.xyz;
  })();

  // ---- Fragment opacity: discard sentinel pixels so the scene-behind
  // shows through. Sample the smoothed depth at canvas-pixel UV
  // (interpolated from vertex UV); coarse tessellation can't punch
  // accurate silhouette holes by itself, but the per-fragment sample
  // does.
  material.opacityNode = Fn(() => {
    const depthSample: Any = texture(finalRT, screenUV).r;
    return select(depthSample.greaterThanEqual(SENTINEL_THRESHOLD), float(0.0), float(1.0));
  })();

  // ---- Fragment normal: Green 2010 *Calculating Normals (code)*.
  // Reconstruct view-space position locally from the depth sample
  // (NOT from `positionView` — that varying tracks the original
  // `positionLocal` in three.js TSL, not the positionNode override,
  // so its derivatives are the flat plane's, not the depth-derived
  // surface's). Then take screen-space derivatives of the local view
  // position and cross them: `cross(ddx, ddy)` gives the view-space
  // outward normal directly (same recipe three.js uses internally
  // for `getNormalFromDepth` at three.webgpu.js:37892).
  //
  // Return the **view-space** normal. three.js's NodeMaterial pipeline
  // treats `normalNode` as view-space and computes
  // `normalWorld = normalView.transformDirection(cameraViewMatrix)`
  // (three.webgpu.js:15031), where `cameraViewMatrix` is
  // `camera.matrixWorldInverse`. `transformDirection(M)` evaluates as
  // `vec4(v, 0) * M` (= `M^T · v`), which for the orthonormal rotation
  // is the view→world rotation. If we returned a world-space normal
  // here, three.js would re-rotate it as if it were view-space,
  // producing a normal locked to the camera's local axes (= the
  // visible "shading rotates with the camera" symptom). Phase 14b
  // shipped that bug; the IBL reflection direction read the wrong
  // env hemisphere, which is invisible against a uniform sky but
  // shows as wrong-side-of-env reflections under a real PMREM
  // environment map. Fix: return Nview, let three.js do view→world.
  material.normalNode = Fn(() => {
    // ±1-texel central differences on the depth texture (instead of
    // dFdx / dFdy of viewPos). Hardware screen-space derivatives gave
    // a globally-tilted normal that didn't match per-face geometry —
    // top-down view of a flat surface produced (0.7, 0.9, −0.1) view
    // instead of (0, 0, 1). Sampling the depth texture at the four
    // neighbour texels and reconstructing viewPos at each gives a
    // direct geometric finite difference that matches what the depth
    // actually encodes.
    const Fx: Any = (cameraProjectionMatrix as Any).element(0).x;
    const Fy: Any = (cameraProjectionMatrix as Any).element(1).y;
    const texSizeXY: Any = textureSize(texture(finalRT), int(0));
    const texelDx: Any = float(1.0).div(texSizeXY.x.toFloat());
    const texelDy: Any = float(1.0).div(texSizeXY.y.toFloat());

    const reconstruct = (uv: Any): Any => {
      const d: Any = texture(finalRT, uv).r;
      const bg: Any = d.greaterThanEqual(SENTINEL_THRESHOLD);
      const sd: Any = select(bg, float(10.0), d);
      const ndcX: Any = uv.x.mul(2.0).sub(1.0);
      const ndcY: Any = uv.y.mul(2.0).sub(1.0);
      const Wx: Any = ndcX.div(Fx);
      const Wy: Any = ndcY.div(Fy);
      return vec3(Wx.mul(sd), Wy.mul(sd), sd.negate());
    };

    const uvL: Any = vec2(screenUV.x.sub(texelDx), screenUV.y);
    const uvR: Any = vec2(screenUV.x.add(texelDx), screenUV.y);
    const uvD: Any = vec2(screenUV.x, screenUV.y.sub(texelDy));
    const uvU: Any = vec2(screenUV.x, screenUV.y.add(texelDy));

    const ddx: Any = reconstruct(uvR).sub(reconstruct(uvL));
    const ddy: Any = reconstruct(uvU).sub(reconstruct(uvD));
    return normalize(cross(ddx, ddy));
  })();

  // ---- Fragment depth override.
  //
  // Without this, three.js auto-derives `gl_FragDepth` from the
  // rasterised triangle's interpolated `gl_Position.z`. The surface
  // mesh's triangles span from sentinel vertices (clamped to viewZ
  // = -10 m above) to fluid vertices (viewZ ≈ -1 m), so a fragment
  // alpha-passed as fluid (with screenUV-sampled depth ≈ 1 m) might
  // get an interpolated triangle depth corresponding to ~5 m, then
  // fail the depth test against opaque scene geometry that's
  // actually behind the fluid in 3D — the visible "fluid renders
  // behind submerged objects" artifact, plus stair-stepping at the
  // silhouette.
  //
  // The fix: write the per-fragment SAMPLED depth directly via
  // `material.depthNode` so the depth buffer integration matches the
  // visible surface. Setup confirmed against three.js 0.184
  // `NodeMaterial.setupDepth()` — when `depthNode` is non-null the
  // builder assigns it to `depth.assign(depthNode).toStack()`,
  // overriding the rasteriser-derived value.
  //
  // Convention: WebGPU NDC depth ∈ [0, 1]. For a perspective
  // projection, `clipZ / clipW = (m22·viewZ + m23) / -viewZ` lands in
  // [0, 1] (verified for `viewZ = -near → 0` and `viewZ = -far → 1`).
  //
  // For sentinel pixels we hand back `1.0` (far plane) — they get
  // alpha-discarded anyway, so the value is just a safe fallback.
  material.depthNode = Fn(() => {
    const depthSample: Any = texture(finalRT, screenUV).r;
    const isBg: Any = depthSample.greaterThanEqual(SENTINEL_THRESHOLD);
    const safePos: Any = select(isBg, float(1.0), depthSample);
    const viewZ: Any = safePos.negate();
    const m22: Any = (cameraProjectionMatrix as Any).element(2).z;
    const m23: Any = (cameraProjectionMatrix as Any).element(3).z;
    const fragDepth: Any = m22.mul(viewZ).add(m23).div(viewZ.negate()).clamp(0.0, 1.0);
    return select(isBg, float(1.0), fragDepth);
  })();

  // ---- Thickness drives Beer-Lambert via three's PBR volume model.
  // `thicknessNode` is in metres along the view ray; our `thicknessRT`
  // accumulates a unitless Gaussian-summed alpha, mapped to metres
  // through `thicknessScale`.
  const thicknessScaleUniform = uniform(options.thicknessScale ?? 0.01, 'float');
  material.thicknessNode = Fn(() => {
    const t: Any = texture(options.thicknessTexture, screenUV).r;
    return t.mul(thicknessScaleUniform);
  })();

  const geometry = new PlaneGeometry(2, 2, tessellation, tessellation);
  const mesh = new Mesh(geometry, material);
  mesh.name = 'FluidPhysicalSurfaceMesh';
  mesh.frustumCulled = false;

  return { mesh, material, thicknessScaleUniform };
}
