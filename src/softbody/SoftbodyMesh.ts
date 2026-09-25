import { BufferGeometry, Mesh } from 'three';
import type { MeshStandardMaterial } from 'three';
import type { MeshStandardNodeMaterial } from 'three/webgpu';

import { bindSoftbodyMesh, type BindSoftbodyMeshOptions } from './bindMesh.js';
import { createSoftbodySkinMaterial } from './skinMaterial.js';
import type { SoftbodySystem } from './SoftbodySystem.js';

/**
 *
 *
 * Construction:
 *   1. Bind the geometry to body `bodyIndex`'s particle cloud (CPU
 *      k-NN, partition-of-unity weights).
 *   2. Build a TSL node material whose vertex stage runs DQB skinning
 *      (Kavan 2008 Eq. 11, mode-aware quaternion source per
 *      {@link SoftbodySystem.shapeMatchMode}).
 *
 * Works identically for `'implicit'` (§5.1) and `'explicit'` (§5.3)
 * bodies — DQB collapses to LBS algebraically on `'explicit'` so no
 * branch is needed at the call site.
 */
export interface SoftbodyMeshOptions extends BindSoftbodyMeshOptions {
  readonly geometry: BufferGeometry;
  readonly softbody: SoftbodySystem;
  readonly bodyIndex: number;
  /** PBR base colour. Default 0xd07030. */
  readonly color?: number | string;
  /** PBR roughness. Default 0.6. */
  readonly roughness?: number;
  /** PBR metalness. Default 0.0. */
  readonly metalness?: number;
  /**
   * Optional source `MeshStandardMaterial` whose PBR maps + colour are
   * copied onto the skin material. See
   * {@link CreateSoftbodySkinMaterialOptions.sourceMaterial}.
   */
  readonly sourceMaterial?: MeshStandardMaterial;
}

export class SoftbodyMesh extends Mesh {
  readonly softbody: SoftbodySystem;
  readonly bodyIndex: number;

  constructor(options: SoftbodyMeshOptions) {
    const {
      geometry,
      softbody,
      bodyIndex,
      K,
      reachRadius,
      color,
      roughness,
      metalness,
      sourceMaterial,
    } = options;
    const bindOptions: BindSoftbodyMeshOptions = {};
    if (K !== undefined) (bindOptions as { K: number }).K = K;
    if (reachRadius !== undefined) {
      (bindOptions as { reachRadius: number }).reachRadius = reachRadius;
    }
    const { cBar } = bindSoftbodyMesh(geometry, softbody, bodyIndex, bindOptions);
    const skinOptions: Parameters<typeof createSoftbodySkinMaterial>[0] = {
      softbody,
      bodyIndex,
      cBar,
    };
    if (color !== undefined) {
      (skinOptions as { color: number | string }).color = color;
    }
    if (roughness !== undefined) {
      (skinOptions as { roughness: number }).roughness = roughness;
    }
    if (metalness !== undefined) {
      (skinOptions as { metalness: number }).metalness = metalness;
    }
    if (sourceMaterial !== undefined) {
      (skinOptions as { sourceMaterial: MeshStandardMaterial }).sourceMaterial = sourceMaterial;
    }
    const material: MeshStandardNodeMaterial = createSoftbodySkinMaterial(skinOptions);

    super(geometry, material);

    this.softbody = softbody;
    this.bodyIndex = bodyIndex;
    // Per-vertex skinning displaces every vertex; pre-baked bounds
    // would clip the body the moment it moves.
    this.frustumCulled = false;
    this.castShadow = true;
    this.receiveShadow = true;
  }
}
