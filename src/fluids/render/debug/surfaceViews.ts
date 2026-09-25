import { Mesh, PlaneGeometry, Scene, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  Discard,
  Fn,
  cameraProjectionMatrix,
  cross,
  dFdx,
  dFdy,
  float,
  int,
  normalize,
  positionLocal,
  screenCoordinate,
  screenSize,
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
 * Geometry-stage debug views (#13 viewPos, #14 normal). Each view runs
 * a full-screen-quad pass that reconstructs the per-pixel view-space
 * surface position from the smoothed depth, then visualizes either:
 *
 *   - `viewPos` (#13) — view-space position as RGB. R/G = clamp((x or
 *     y)*0.5 + 0.5), B = -z / depthScale (so far → 1, near → 0).
 *   - `normal`  (#14) — depth-derived view-space normal (cross of
 *     dFdx/dFdy of viewPos) as (N+1)/2 RGB. This is exactly the
 *     normal three.js's PBR pipeline consumes via
 *     `material.normalNode` for IBL — useful when the silhouette /
 *     reflection looks wrong on the production surface.
 *
 * Sentinel pixels alpha-discard so the canvas's existing scene shows
 * through.
 */
export type SurfaceDebugMode = 'viewPos' | 'normal' | 'ddxViewPos' | 'ddyViewPos';

export interface BuildSurfaceDebugMaterialArgs {
  readonly mode: SurfaceDebugMode;
  readonly smoothedDepthTexture: Texture;
  /** Depth-normalisation distance for the `viewPos` blue channel. */
  readonly depthScaleUniform: ReturnType<typeof uniform<'float', number>>;
  /**
   * Multiplier for the `ddxViewPos` / `ddyViewPos` modes. Derivatives
   * are O(metres / pixel) (~1e-4 at typical scenes) so the raw values
   * collapse to mid-grey without scaling. Crank `derivativeScale`
   * (default 20) up to ~5000 for these views.
   */
  readonly derivativeScaleUniform: ReturnType<typeof uniform<'float', number>>;
}

export function buildSurfaceDebugMaterial(
  args: BuildSurfaceDebugMaterialArgs,
): MeshBasicNodeMaterial {
  const depthScaleUniform = args.depthScaleUniform;
  const mode = args.mode;

  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthTest = false;
  material.depthWrite = false;
  material.vertexNode = vec4(positionLocal.xy, 0.0, 1.0);

  material.outputNode = Fn(() => {
    const tex: Any = texture(args.smoothedDepthTexture, screenUV);
    const W: Any = screenSize.x.toFloat();
    const H: Any = screenSize.y.toFloat();

    const depthSample: Any = tex.r.toVar();
    const isBg: Any = depthSample.greaterThanEqual(float(NRF_FAR_SENTINEL_THRESHOLD));
    Discard(isBg);

    const safeDepth: Any = select(isBg, float(1.0), depthSample);
    const viewZ: Any = safeDepth.negate();

    const Fx: Any = (cameraProjectionMatrix as Any).element(0).x;
    const Fy: Any = (cameraProjectionMatrix as Any).element(1).y;
    const xPix: Any = screenCoordinate.x.toFloat();
    const yPix: Any = screenCoordinate.y.toFloat();
    const Wx: Any = xPix.div(W).mul(2.0).sub(1.0).div(Fx);
    const Wy: Any = yPix.div(H).mul(2.0).sub(1.0).div(Fy);
    // Inverse-projection: viewX = NDC.x · safeDepth / Fx (positive
    // depth, NOT viewZ which is negative). Mirrors the corrected
    // formula in surface/PhysicalSurface.ts so #14 normal matches
    // what the production PBR shader sees.
    const viewPos: Any = vec3(Wx.mul(safeDepth), Wy.mul(safeDepth), viewZ).toVar();

    if (mode === 'viewPos') {
      const r: Any = viewPos.x.mul(0.5).add(0.5).clamp(0.0, 1.0);
      const g: Any = viewPos.y.mul(0.5).add(0.5).clamp(0.0, 1.0);
      const b: Any = viewPos.z.negate().div(depthScaleUniform).clamp(0.0, 1.0);
      return vec4(r, g, b, 1.0);
    }

    // Hardware-derivative path (kept for the ddxViewPos / ddyViewPos
    // diagnostic views — we explicitly want to see what dFdx / dFdy
    // produce). The 'normal' mode below uses neighbour-texel finite
    // differences instead, mirroring PhysicalSurface.ts.
    const ddx_hw: Any = dFdx(viewPos);
    const ddy_hw: Any = dFdy(viewPos);

    if (mode === 'ddxViewPos' || mode === 'ddyViewPos') {
      const d: Any = mode === 'ddxViewPos' ? ddx_hw : ddy_hw;
      const s: Any = args.derivativeScaleUniform;
      const r: Any = d.x.mul(s).mul(0.5).add(0.5).clamp(0.0, 1.0);
      const g: Any = d.y.mul(s).mul(0.5).add(0.5).clamp(0.0, 1.0);
      const b: Any = d.z.mul(s).mul(0.5).add(0.5).clamp(0.0, 1.0);
      return vec4(r, g, b, 1.0);
    }

    // Neighbour-texel finite differences for the production normal
    // mode — see PhysicalSurface.ts:material.normalNode for rationale.
    const texSizeXY: Any = textureSize(texture(args.smoothedDepthTexture), int(0));
    const texelDx: Any = float(1.0).div(texSizeXY.x.toFloat());
    const texelDy: Any = float(1.0).div(texSizeXY.y.toFloat());

    const reconstruct = (uv: Any): Any => {
      const d: Any = texture(args.smoothedDepthTexture, uv).r;
      const bg: Any = d.greaterThanEqual(float(NRF_FAR_SENTINEL_THRESHOLD));
      const sd: Any = select(bg, float(1.0), d);
      const ndcX: Any = uv.x.mul(2.0).sub(1.0);
      const ndcY: Any = uv.y.mul(2.0).sub(1.0);
      const wx2: Any = ndcX.div(Fx);
      const wy2: Any = ndcY.div(Fy);
      return vec3(wx2.mul(sd), wy2.mul(sd), sd.negate());
    };

    const uvL: Any = vec2(screenUV.x.sub(texelDx), screenUV.y);
    const uvR: Any = vec2(screenUV.x.add(texelDx), screenUV.y);
    const uvD: Any = vec2(screenUV.x, screenUV.y.sub(texelDy));
    const uvU: Any = vec2(screenUV.x, screenUV.y.add(texelDy));

    const ddxN: Any = reconstruct(uvR).sub(reconstruct(uvL));
    const ddyN: Any = reconstruct(uvU).sub(reconstruct(uvD));
    const N: Any = normalize(cross(ddxN, ddyN));
    const r: Any = N.x.mul(0.5).add(0.5).clamp(0.0, 1.0);
    const g: Any = N.y.mul(0.5).add(0.5).clamp(0.0, 1.0);
    const b: Any = N.z.mul(0.5).add(0.5).clamp(0.0, 1.0);
    return vec4(r, g, b, 1.0);
  })();

  return material;
}

export function buildSurfaceDebugScene(material: MeshBasicNodeMaterial): {
  readonly mesh: Mesh;
  readonly scene: Scene;
} {
  const mesh = new Mesh(new PlaneGeometry(2, 2), material);
  mesh.frustumCulled = false;
  const scene = new Scene();
  scene.add(mesh);
  return { mesh, scene };
}
