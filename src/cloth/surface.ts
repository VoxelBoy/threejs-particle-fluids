import { DoubleSide, Mesh, PlaneGeometry } from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import {
  Fn,
  cameraViewMatrix,
  cross,
  faceDirection,
  float,
  mat3,
  normalize,
  uv,
  vec3,
  vec4,
} from 'three/tsl';
import type { ParticleSystem } from '../core/particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface ClothSurfaceOptions {
  readonly particles: ParticleSystem;
  /** Number of simulation vertices in each row and column, in row-major order. */
  readonly columns: number;
  readonly rows: number;
  readonly offset?: number;
  readonly subdivisions?: number;
  readonly material?: MeshPhysicalNodeMaterial;
}

/** Bicubic geometry and analytic smooth normals from the live simulation grid. */
export function createClothSurface(options: ClothSurfaceOptions): Mesh {
  const { particles, columns, rows, offset = 0, subdivisions = 3 } = options;
  if (columns < 2 || rows < 2 || offset < 0 || offset + columns * rows > particles.capacity)
    throw new Error('Cloth surface grid must fit in the particle buffer.');
  const weights = (t: Any) => {
    const t2 = t.mul(t),
      t3 = t2.mul(t);
    return [
      t.mul(-0.5).add(t2).sub(t3.mul(0.5)),
      t2.mul(-2.5).add(t3.mul(1.5)).add(1),
      t.mul(0.5).add(t2.mul(2)).sub(t3.mul(1.5)),
      t3.sub(t2).mul(0.5),
    ];
  };
  const derivatives = (t: Any) => [
    t.mul(2).sub(t.mul(t).mul(1.5)).sub(0.5),
    t.mul(-5).add(t.mul(t).mul(4.5)),
    t.mul(4).sub(t.mul(t).mul(4.5)).add(0.5),
    t.mul(t).mul(1.5).sub(t),
  ];
  const frame: Any = Fn(() => {
    const x: Any = uv()
      .x.mul(columns - 1)
      .min(columns - 1.0001)
      .toVar();
    const y: Any = float(1)
      .sub(uv().y)
      .mul(rows - 1)
      .min(rows - 1.0001)
      .toVar();
    const ix: Any = x.floor(),
      iy: Any = y.floor();
    const wx = weights(x.fract()),
      wy = weights(y.fract());
    const dx = derivatives(x.fract()),
      dy = derivatives(y.fract());
    const p: Any = vec3(0).toVar(),
      tu: Any = vec3(0).toVar(),
      tv: Any = vec3(0).toVar();
    for (let v = 0; v < 4; v++)
      for (let u = 0; u < 4; u++) {
        const index = iy
          .add(v - 1)
          .clamp(0, rows - 1)
          .mul(columns)
          .add(ix.add(u - 1).clamp(0, columns - 1))
          .add(offset)
          .toUint();
        const sample: Any = particles.positions.element(index).xyz.toVar();
        p.addAssign(sample.mul(wx[u]).mul(wy[v]));
        tu.addAssign(sample.mul(dx[u]).mul(wy[v]));
        tv.addAssign(sample.mul(wx[u]).mul(dy[v]));
      }
    return mat3(p, tu, tv);
  })();
  const material =
    options.material ??
    new MeshPhysicalNodeMaterial({ side: DoubleSide, roughness: 0.4, sheen: 1 });
  material.flatShading = false;
  material.positionNode = frame.element(0);
  const normal = normalize(cross(frame.element(2), frame.element(1))).toVarying();
  // Custom normalNode bypasses the built-in two-sided normal adjustment.
  material.normalNode = normalize(cameraViewMatrix.mul(vec4(normal, 0)).xyz).mul(
    faceDirection as Any,
  );
  const geometry = new PlaneGeometry(1, 1, (columns - 1) * subdivisions, (rows - 1) * subdivisions);
  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}
