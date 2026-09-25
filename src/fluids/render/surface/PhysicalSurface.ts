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
  positionGeometry,
  screenUV,
  select,
  texture,
  textureLoad,
  textureSize,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { NRF_FAR_SENTINEL_THRESHOLD } from '../passes/smoothing.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface PhysicalSurfaceOptions {
  readonly smoothedDepthTexture: Texture;
  readonly thicknessTexture: Texture;
  readonly fluidColor?: Color;
  readonly tessellation?: number;
  readonly ior?: number;
  readonly attenuationDistance?: number;
  readonly roughness?: number;
  readonly thicknessScale?: number;
  readonly envIntensity?: number;
  readonly environment?: CubeTexture | Texture | null;
}

export interface PhysicalSurfaceMesh {
  readonly mesh: Mesh;
  readonly material: MeshPhysicalNodeMaterial;
  readonly thicknessScaleUniform: ReturnType<typeof uniform<'float', number>>;
}

/** Screen-space depth reconstructed into a lit, refractive Three.js surface. */
export function createPhysicalSurfaceMesh(options: PhysicalSurfaceOptions): PhysicalSurfaceMesh {
  const material = new MeshPhysicalNodeMaterial();
  material.transparent = true;
  material.depthWrite = true;
  material.fog = false;
  material.transmission = 1;
  material.roughness = options.roughness ?? 0.08;
  material.ior = options.ior ?? 1.33;
  material.attenuationColor = options.fluidColor ?? new Color(0x0a2840);
  material.attenuationDistance = options.attenuationDistance ?? 3;
  material.envMapIntensity = options.envIntensity ?? 1;
  if (options.environment) material.envMap = options.environment;
  material.alphaTest = 0.5;

  const depthMap = options.smoothedDepthTexture;
  const threshold = float(NRF_FAR_SENTINEL_THRESHOLD);
  // Interpolate only actual surface samples: mixing the far sentinel into a
  // valid depth creates false cliffs and stretched refraction at the silhouette.
  const readDepth: Any = Fn(([uv]: Any[]) => {
    const size: Any = textureSize(texture(depthMap), int(0));
    const pixel: Any = uv.mul(size).sub(0.5).toVar();
    const base: Any = pixel.floor();
    const f: Any = pixel.fract();
    const sum: Any = float(0).toVar(),
      weights: Any = float(0).toVar();
    for (let y = 0; y < 2; y++)
      for (let x = 0; x < 2; x++) {
        const sample: Any = textureLoad(
          depthMap,
          base.add(vec2(x, y)).clamp(vec2(0), size.sub(1)).toIVec2(),
        ).r;
        const weight: Any = (x ? f.x : f.x.oneMinus())
          .mul(y ? f.y : f.y.oneMinus())
          .mul(select(sample.lessThan(threshold), 1, 0));
        sum.addAssign(sample.mul(weight));
        weights.addAssign(weight);
      }
    return select(weights.greaterThan(0.001), sum.div(weights.max(0.001)), float(1e6));
  });
  const reconstruct = (uv: Any, depth: Any): Any => {
    const fx = (cameraProjectionMatrix as Any).element(0).x;
    const fy = (cameraProjectionMatrix as Any).element(1).y;
    // Render-target coordinates run downwards in WebGPU, unlike NDC Y.
    return vec3(
      uv.x.mul(2).sub(1).mul(depth).div(fx),
      uv.y.mul(-2).add(1).mul(depth).div(fy),
      depth.negate(),
    );
  };
  material.positionNode = Fn(() => {
    const uv = vec2(
      positionGeometry.x.mul(0.5).add(0.5),
      float(0.5).sub(positionGeometry.y.mul(0.5)),
    );
    const depth: Any = readDepth(uv).toVar();
    // Extend depth under the discarded border triangles so lighting never
    // interpolates between the visible surface and an arbitrary far position.
    const step = 2 / (options.tessellation ?? 256);
    const nearest: Any = depth
      .min(readDepth(uv.add(vec2(step, 0))))
      .min(readDepth(uv.sub(vec2(step, 0))))
      .min(readDepth(uv.add(vec2(0, step))))
      .min(readDepth(uv.sub(vec2(0, step))));
    depth.assign(select(depth.lessThan(threshold), depth, nearest));
    const safe = select(depth.lessThan(threshold), depth, float(10));
    return cameraWorldMatrix.mul(vec4(reconstruct(uv, safe), 1)).xyz;
  })();
  material.opacityNode = select(readDepth(screenUV).lessThan(threshold), 1, 0);
  material.normalNode = Fn(() => {
    const size: Any = textureSize(texture(depthMap), int(0));
    const dx = vec2(float(1).div(size.x), 0);
    const dy = vec2(0, float(1).div(size.y));
    const d = readDepth(screenUV).toVar();
    const center = reconstruct(screenUV, d);
    const leftD = readDepth(screenUV.sub(dx));
    const rightD = readDepth(screenUV.add(dx));
    const upD = readDepth(screenUV.sub(dy));
    const downD = readDepth(screenUV.add(dy));
    // One-sided differences at silhouettes avoid normals reaching into empty space.
    const tx: Any = select(
      leftD.sub(d).abs().lessThan(rightD.sub(d).abs()),
      center.sub(reconstruct(screenUV.sub(dx), leftD)),
      reconstruct(screenUV.add(dx), rightD).sub(center),
    );
    const ty: Any = select(
      upD.sub(d).abs().lessThan(downD.sub(d).abs()),
      center.sub(reconstruct(screenUV.sub(dy), upD)),
      reconstruct(screenUV.add(dy), downD).sub(center),
    );
    const centralX: Any = reconstruct(screenUV.add(dx), rightD).sub(
      reconstruct(screenUV.sub(dx), leftD),
    );
    const centralY: Any = reconstruct(screenUV.add(dy), downD).sub(
      reconstruct(screenUV.sub(dy), upD),
    );
    return normalize(
      cross(
        select(upD.lessThan(threshold).and(downD.lessThan(threshold)), centralY, ty) as Any,
        select(leftD.lessThan(threshold).and(rightD.lessThan(threshold)), centralX, tx) as Any,
      ),
    );
  })();
  material.depthNode = Fn(() => {
    const d = readDepth(screenUV);
    const m22 = (cameraProjectionMatrix as Any).element(2).z;
    const m23 = (cameraProjectionMatrix as Any).element(3).z;
    return m23.div(d).sub(m22).clamp(0, 1);
  })();
  const thicknessScaleUniform = uniform(options.thicknessScale ?? 0.01, 'float');
  material.thicknessNode = texture(options.thicknessTexture, screenUV).r.mul(thicknessScaleUniform);
  const mesh = new Mesh(
    new PlaneGeometry(2, 2, options.tessellation ?? 256, options.tessellation ?? 256),
    material,
  );
  mesh.name = 'FluidPhysicalSurfaceMesh';
  mesh.frustumCulled = false;
  return { mesh, material, thicknessScaleUniform };
}
