import { Color, Vector3 } from 'three';
import type { MeshStandardMaterial } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { If, float, uint, uniform, vec4 } from 'three/tsl';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { SoftbodySystem } from './SoftbodySystem.js';
import { buildSkinNormalFn, buildSkinPositionFn } from './dlb.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 *
 *
 * Mode handling. The skin shader needs a unit quaternion per influence.
 *  - `'implicit'` (Mueller 2011 §5.1): per-particle quaternion is
 *     written by Phase 12's `qpWriteKernel` into `particles.rotation`.
 *     The shader reads `particles.rotation.element(idx)` per influence.
 *  - `'explicit'` (Mueller 2011 §5.3): only one rotation per body
 *     exists, stored as a 3x3 matrix in `bodyRotations[3·bodyIndex..]`.
 *     The shader converts that matrix to a quaternion once (Shoemake-
 *     style branched extraction, identical to `qpWriteKernel`'s code)
 *     and re-uses the result for all 4 influences. Because all q_k are
 *     identical the antipodality flips and post-blend normalization in
 *     {@link buildSkinPositionFn} collapse to no-ops, and the DLB blend
 *     reduces algebraically to LBS — Kavan 2008 Eq. 11 with
 *     `q_r,j = q_r ∀j`. Plan §"LBS equivalence on §5.3 bodies".
 */

export interface CreateSoftbodySkinMaterialOptions {
  readonly softbody: SoftbodySystem;
  readonly bodyIndex: number;
  /**
   * Pre-centring offset for the bound mesh: `c̄_body = (1/n) Σ x_i^0`,
   * computed by {@link bindSoftbodyMesh}. Subtracted from
   * `positionLocal` before any rest-frame math runs.
   */
  readonly cBar: readonly [number, number, number];
  /** PBR base colour. Defaults to a soft warm tone. */
  readonly color?: number | string;
  /** PBR roughness. Default 0.6. */
  readonly roughness?: number;
  /** PBR metalness. Default 0.0. */
  readonly metalness?: number;
  /**
   * Optional source `MeshStandardMaterial` (typically lifted from a glTF
   * via `mesh.material`) — its `.color`, `.map`, `.normalMap`,
   * `.roughnessMap`, `.metalnessMap`, `.aoMap`, `.emissive`,
   * `.emissiveMap`, `.roughness`, and `.metalness` are copied onto the
   * skin material so the bound mesh inherits the asset's PBR textures.
   * Caller is responsible for ensuring the geometry has a `uv` attribute
   * so the maps actually sample meaningfully.
   *
   * When provided the explicit `color` / `roughness` / `metalness`
   * options below are ignored — the source material wins.
   */
  readonly sourceMaterial?: MeshStandardMaterial;
}

export function createSoftbodySkinMaterial(
  options: CreateSoftbodySkinMaterialOptions,
): MeshStandardNodeMaterial {
  const {
    softbody,
    bodyIndex,
    cBar,
    color = 0xd07030,
    roughness = 0.6,
    metalness = 0.0,
    sourceMaterial,
  } = options;
  if (bodyIndex < 0 || bodyIndex >= softbody.bodies.length) {
    throw new Error(`createSoftbodySkinMaterial: bodyIndex ${bodyIndex} out of range`);
  }

  // `uniform()` takes a value (Vector3, number, etc.) — NOT a TSL node.
  // Passing `vec3(...)` would produce a uniform whose initial value is an
  // unparseable node, which TSL silently treats as zero — the symptom is
  // a "squished" rest pose where the cBar offset is missing.
  const cBarUniform: UniformNode<'vec3', Vector3> = uniform(new Vector3(cBar[0], cBar[1], cBar[2]));

  let createGetRotationQuat: () => (idx: Any) => Any;
  if (softbody.shapeMatchMode === 'implicit') {
    // §5.1: per-particle quaternion buffer (Phase 12). The factory
    // returns a stateless closure — every lane reads its own slot.
    const rotation = softbody.particles.rotation;
    createGetRotationQuat = () => (idx: Any) => rotation.element(idx);
  } else {
    // §5.3: shared body rotation. Convert R → q ONCE in the vertex
    // shader and return the same value for every influence. Identical
    // Shoemake branching to `shapeMatchImplicit.ts` Pass 4 — kept inline
    // because the only sharable abstraction would be a helper that
    // takes nine TSL scalars and returns a vec4, and that's no clearer
    // than the inline form.
    const bodyRotations = softbody.bodyRotations;
    const baseRow: Any = uint(3 * bodyIndex);
    const sharedQuat = (): Any => {
      const r0: Any = bodyRotations.element(baseRow).xyz.toVar();
      const r1: Any = bodyRotations.element(baseRow.add(uint(1))).xyz.toVar();
      const r2: Any = bodyRotations.element(baseRow.add(uint(2))).xyz.toVar();
      const m00: Any = r0.x;
      const m01: Any = r0.y;
      const m02: Any = r0.z;
      const m10: Any = r1.x;
      const m11: Any = r1.y;
      const m12: Any = r1.z;
      const m20: Any = r2.x;
      const m21: Any = r2.y;
      const m22: Any = r2.z;
      const trace: Any = m00.add(m11).add(m22);
      const qx: Any = float(0.0).toVar();
      const qy: Any = float(0.0).toVar();
      const qz: Any = float(0.0).toVar();
      const qw: Any = float(1.0).toVar();
      If(trace.greaterThan(float(0.0)), () => {
        // s = 0.5 / sqrt(trace+1) — same formulation as
        // shapeMatchImplicit.ts qpWriteKernel.
        const s: Any = float(0.5).div(trace.add(float(1.0)).sqrt());
        qw.assign(float(0.25).div(s));
        qx.assign(m21.sub(m12).mul(s));
        qy.assign(m02.sub(m20).mul(s));
        qz.assign(m10.sub(m01).mul(s));
      })
        .ElseIf(m00.greaterThan(m11).and(m00.greaterThan(m22)), () => {
          const s: Any = float(2.0).mul(float(1.0).add(m00).sub(m11).sub(m22).sqrt());
          qw.assign(m21.sub(m12).div(s));
          qx.assign(float(0.25).mul(s));
          qy.assign(m01.add(m10).div(s));
          qz.assign(m02.add(m20).div(s));
        })
        .ElseIf(m11.greaterThan(m22), () => {
          const s: Any = float(2.0).mul(float(1.0).add(m11).sub(m00).sub(m22).sqrt());
          qw.assign(m02.sub(m20).div(s));
          qx.assign(m01.add(m10).div(s));
          qy.assign(float(0.25).mul(s));
          qz.assign(m12.add(m21).div(s));
        })
        .Else(() => {
          const s: Any = float(2.0).mul(float(1.0).add(m22).sub(m00).sub(m11).sqrt());
          qw.assign(m10.sub(m01).div(s));
          qx.assign(m02.add(m20).div(s));
          qy.assign(m12.add(m21).div(s));
          qz.assign(float(0.25).mul(s));
        });
      return vec4(qx, qy, qz, qw);
    };
    // Per-`Fn`-invocation closure. The first lane materialises the
    // shared quaternion as a `.toVar()` inside the current Fn body;
    // subsequent lanes reuse the same node reference. The closure is
    // re-created for the position and normal Fns separately, so the
    // memoised node never crosses Fn boundaries.
    createGetRotationQuat = () => {
      let cached: Any | null = null;
      return (_idx: Any): Any => {
        if (cached === null) cached = sharedQuat().toVar();
        return cached;
      };
    };
  }

  const material = new MeshStandardNodeMaterial({
    color: new Color(color),
    roughness,
    metalness,
  });

  // Honour a glTF source material — copy PBR scalars + every texture
  // map slot the upstream `MeshStandardMaterial` exposes. Three.js's
  // node material extends the legacy material so direct property
  // assignment works; the standard PBR shader graph reads these without
  // any explicit `colorNode` / `mapNode` wiring on our side.
  if (sourceMaterial) {
    if (sourceMaterial.color) material.color.copy(sourceMaterial.color);
    if (sourceMaterial.emissive) material.emissive.copy(sourceMaterial.emissive);
    material.roughness = sourceMaterial.roughness;
    material.metalness = sourceMaterial.metalness;
    material.map = sourceMaterial.map;
    material.normalMap = sourceMaterial.normalMap;
    material.roughnessMap = sourceMaterial.roughnessMap;
    material.metalnessMap = sourceMaterial.metalnessMap;
    material.aoMap = sourceMaterial.aoMap;
    material.aoMapIntensity = sourceMaterial.aoMapIntensity;
    material.emissiveMap = sourceMaterial.emissiveMap;
    material.emissiveIntensity = sourceMaterial.emissiveIntensity;
    if (sourceMaterial.normalScale) {
      material.normalScale.copy(sourceMaterial.normalScale);
    }
  }

  const skinArgs = {
    particles: softbody.particles,
    restOffsets: softbody.restOffsets,
    cBar: cBarUniform,
    createGetRotationQuat,
  };
  // `Fn` returns a "callable" node — invoke once with no args to get
  // the actual TSL node graph. Three.js sees the call site as a single
  // node; the body is inlined into the vertex shader.
  (material as Any).positionNode = buildSkinPositionFn(skinArgs)();
  (material as Any).normalNode = buildSkinNormalFn(skinArgs)();

  return material;
}
