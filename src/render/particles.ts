import {
  BufferAttribute,
  BufferGeometry,
  Color,
  InstancedMesh,
  Matrix4,
  Points,
  SphereGeometry,
} from 'three';
import { MeshPhongNodeMaterial, PointsNodeMaterial } from 'three/webgpu';
import { instanceIndex, positionLocal, vertexIndex } from 'three/tsl';
import type { ParticleSystem } from '../core/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * GPU-direct instanced particle mesh. Each sphere reads its centre
 * straight from `particles.positions` on the GPU via a TSL
 * `positionNode` — no per-frame CPU readback, no `setMatrixAt` loop,
 * no CPU→GPU re-upload of `instanceMatrix`. Phase 08 dam-break
 * Findings 9–11 pinned this as the only animate-loop pattern that
 * keeps steady frame rate at 7 000+ particles.
 *
 * `instanceMatrix` is seeded to identity once at construction;
 * without that seed the zero-initialised matrices collapse every
 * instance to the origin (instanceMatrix · positionNode = 0).
 */
export interface ParticleMeshOptions {
  readonly particles: ParticleSystem;
  readonly radius: number;
  /** Static color — ignored if `colorNode` is provided. */
  readonly color?: number | string;
  /**
   * Optional per-particle color as a TSL node. Receives the vec3
   * world-space centre of the instance (the `particlePos` node). Use
   * for height-gradient coloring, speed-based tinting, etc. Return a
   * TSL vec3 (RGB, 0–1).
   */

  readonly colorNode?: (particlePos: Any) => Any;
  /** Sphere tessellation. Default 14 × 10 — cheap, looks smooth enough. */
  readonly widthSegments?: number;
  readonly heightSegments?: number;
  /** Visual radius multiplier. 0.9 keeps adjacent particles distinct. */
  readonly visualScale?: number;
  /**
   * Cast shadows from particles onto scene geometry (floor, walls,
   * colliders). Default true. Three.js derives the shadow depth
   * material from the node graph, so the `positionNode`-based
   * per-instance displacement is honored in the shadow pass as long
   * as this flag is set on the InstancedMesh.
   */
  readonly castShadow?: boolean;
  /** Receive shadows onto particles. Default true. */
  readonly receiveShadow?: boolean;
}

/**
 * Options for the lightweight Points renderer. Use this for large
 * particle counts (10 000+) where the geometric cost of instanced
 * spheres outweighs their visual clarity. 1-pixel WebGPU points wash
 * toward white without saturated colors — picking an explicit `color`
 * (or a colorNode) keeps them visible on dark backgrounds.
 */
export interface ParticlePointsOptions {
  readonly particles: ParticleSystem;
  /**
   * Optional per-particle color as a function of vertex position.
   * Receives the TSL vec3 position node, returns a TSL vec3 color.
   * For the simple case, pass a static color via the `color` option.
   */

  readonly colorNode?: (posXyz: Any) => Any;
  /** Static color if `colorNode` is not provided. */
  readonly color?: number;
}

/**
 * GPU-direct Points renderer for large particle counts. Every point
 * reads its position from `particles.positions` via a TSL
 * `positionNode` (same pattern as instanced spheres, but without the
 * sphere geometry or instance matrix). Much cheaper per particle.
 */
export function createParticlePoints(options: ParticlePointsOptions): Points {
  const { particles, colorNode, color = 0xffffff } = options;

  const geom = new BufferGeometry();
  geom.setAttribute('position', new BufferAttribute(new Float32Array(particles.capacity * 3), 3));
  geom.setDrawRange(0, particles.capacity);

  const mat = new PointsNodeMaterial();
  const pos: Any = (particles.positions as Any).element(vertexIndex).xyz;
  (mat as Any).positionNode = pos;
  if (colorNode) {
    (mat as Any).colorNode = colorNode(pos);
  } else {
    (mat as Any).color = new Color(color);
  }

  const points = new Points(geom, mat);
  points.frustumCulled = false;
  return points;
}

export function createParticleMesh(options: ParticleMeshOptions): InstancedMesh {
  const {
    particles,
    radius,
    color = 0x5fb9ff,
    colorNode,
    widthSegments = 14,
    heightSegments = 10,
    visualScale = 0.9,
    castShadow = true,
    receiveShadow = true,
  } = options;

  const geom = new SphereGeometry(radius * visualScale, widthSegments, heightSegments);
  // Node-native material — required for Three.js node-based postprocessing to
  // write view-space normals to its MRT target. Legacy
  // MeshPhongMaterial + positionNode silently skips the normal output,
  // which makes GTAO's depth + normal sampling return "no occlusion"
  // everywhere.
  const mat = new MeshPhongNodeMaterial({
    color: new Color(color),
    shininess: 60,
    specular: new Color(0x334466),
  });

  const particlePos: Any = (particles.positions as Any).element(instanceIndex).xyz;
  (mat as Any).positionNode = particlePos.add(positionLocal);
  if (colorNode) {
    (mat as Any).colorNode = colorNode(particlePos);
  }

  const mesh = new InstancedMesh(geom, mat, particles.capacity);
  mesh.frustumCulled = false;
  mesh.castShadow = castShadow;
  mesh.receiveShadow = receiveShadow;

  // Identity seed — zero-initialised matrices would collapse every
  // instance to the origin (`instanceMatrix * positionNode = 0`).
  const identity = new Matrix4();
  for (let i = 0; i < particles.capacity; i++) mesh.setMatrixAt(i, identity);
  mesh.instanceMatrix.needsUpdate = true;

  return mesh;
}
