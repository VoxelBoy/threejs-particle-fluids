import { Mesh, PlaneGeometry, Scene, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { Discard, Fn, positionLocal, screenUV, texture, uniform, vec4 } from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Debug-blit material for view #5 (`pass1.rtResOverlay`).
 *
 * Renders `pass1RT.texture` directly to the canvas via a full-screen
 * quad with `screenUV`. The source RT lives at smoothing resolution
 * with `NearestFilter`; sampling at canvas resolution exposes the
 * resolution gap as a visible pixel grid. If the artist toggles
 * `smoothingResolution` from `full` → `quarter` and the grid does NOT
 * coarsen, the sub-resolution path is silently disabled — the direct
 * test for U-FR-1.
 *
 * Output: grayscale of `viewSurfaceZ / depthScale` clamped to [0, 1]
 * (matches view #3 so the two views are directly comparable). Pixels
 * with `depth ≤ 0` are discarded so the scene shows through.
 */
export interface BuildDepthBlitMaterialArgs {
  readonly inputTexture: Texture;
  readonly depthScaleUniform: ReturnType<typeof uniform<'float', number>>;
}

export function buildDepthBlitMaterial(args: BuildDepthBlitMaterialArgs): MeshBasicNodeMaterial {
  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthTest = false;
  material.depthWrite = false;
  material.vertexNode = vec4(positionLocal.xy, 0.0, 1.0);

  material.outputNode = Fn(() => {
    const sample: Any = texture(args.inputTexture, screenUV);
    const z: Any = sample.r;
    Discard(z.lessThanEqual(0.0));
    const t: Any = z.div(args.depthScaleUniform).clamp(0.0, 1.0);
    return vec4(t, t, t, 1.0);
  })();

  return material;
}

export function buildDepthBlitScene(material: MeshBasicNodeMaterial): {
  readonly mesh: Mesh;
  readonly scene: Scene;
} {
  const mesh = new Mesh(new PlaneGeometry(2, 2), material);
  mesh.frustumCulled = false;
  const scene = new Scene();
  scene.add(mesh);
  return { mesh, scene };
}
