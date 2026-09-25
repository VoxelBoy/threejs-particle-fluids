import { Color, InstancedMesh, Matrix4, PlaneGeometry } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  cameraWorldMatrix,
  exp,
  float,
  instanceIndex,
  max,
  positionLocal,
  uniform,
  uv,
} from 'three/tsl';

import type { GasSystem } from '../sim/GasSystem.js';
import type { GasRenderer } from './types.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface PointSpritesGasRendererOptions {
  readonly gas: GasSystem;
  /**
   * Smoke colour (RGB hex). MVP renders solid-coloured billboards; for
   * a lit / shaded look use the future `VolumetricGasRenderer`. Default
   * `0xeeeeee` (off-white).
   */
  readonly color?: number | string;
  /**
   * Billboard full edge length in world units — the rendered quad
   * spans `[−size/2, +size/2]` along its camera-aligned x and y axes.
   * Default `0.08` m. Tune for the scene scale: 0.08 reads as a small
   * puff in a 1 m container; bump to 0.2+ for room-scale plumes.
   */
  readonly size?: number;
  /**
   * Peak opacity at age 0 (`α_0` in `α = α_0 · exp(−age / τ)`). Default
   * `0.6`. Lowering it is the cheapest way to make a dense smoke plume
   * read as semi-transparent without ordering issues — billboards
   * don't depth-sort in this MVP renderer (paper §7.2.4 minimal recipe).
   */
  readonly initialOpacity?: number;
  /**
   * Opacity decay time-constant `τ` (seconds). Smoke alpha decays as
   * `α_0 · exp(−age / τ)`. Default `lifetime / 3` so a particle is at
   * `~5% α_0` by the time the lifetime gate kills it. Pass an explicit
   * value to override. Paper §7.2.4: "decrease opacity over the
   * lifetime of the particle."
   */
  readonly opacityTau?: number;
  /**
   * Optional per-instance color callback. When provided, the constant
   * {@link color} is ignored and `mat.colorNode` is set to the result
   * of `colorNode(centerWorld, velocity)`, where both args are TSL
   * vec3 nodes read from the gas-owned storage buffers at the current
   * `instanceIndex`. Use this to color sprites by speed
   * (`velocity.length()`), height (`centerWorld.y`), radial position,
   * or any other per-instance signal — return any TSL vec3 (RGB, 0–1).
   *
   * Velocity is the post-substep SPH-interpolated drift the
   * `smokeAdvect` kernel writes (Macklin 2014 §7.2.1 eq. 28); centre
   * is the post-substep position. Both are in world space.
   *
   * Stays optional so the simple "constant white smoke" path doesn't
   * pay any extra node-graph cost.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly colorNode?: (centerWorld: any, velocity: any) => any;
}

/**
 * Camera-facing billboard renderer for smoke (MVP). Each smoke slot
 * renders as one `PlaneGeometry` quad oriented in screen-space. The
 * vertex shader pulls the smoke's world-space centre from the
 * `GasSystem` storage buffer, then displaces the quad's local (x, y)
 * along the camera's right / up axes — `cameraWorldMatrix`'s first two
 * columns — to produce a billboard that always faces the camera.
 *
 * Why an `InstancedMesh` of quads rather than a `THREE.Points`: WebGPU
 * does not support `gl_PointSize`-style fixed-function point sprites.
 * `PointsMaterial.size` works in WebGL but renders 1-pixel points in
 * WebGPU regardless of the value — visually invisible at MVP smoke
 * counts. An instanced quad reaches every WebGPU device the same way.
 *
 * Visual model — Macklin 2014 §7.2.4 minimal recipe:
 *   - dead slots (`smokeAlive == 0`) collapse to zero size — the quad
 *     still rasterises but at zero footprint
 *   - alive slots fade as `α(age) = α_0 · exp(−age / τ)`; freshly-
 *     emitted smoke is opaque and gradually thins as it ages out
 *   - the quad is masked to a soft circular puff via a radial
 *     `(1 − r²)²` falloff on alpha, with `r = 0` at the centre and
 *     `r = 1` at the inscribed disk. The falloff goes to exactly zero
 *     at the disk edge so the square quad is invisible — the smoke
 *     reads as a fuzzy ball rather than a sprite billboard
 *   - colour is constant (paper Figure 14 / 15 use white smoke)
 *
 * Out of scope (post-MVP volumetric renderer):
 *   - depth sort for correct alpha compositing under occlusion
 *   - shadow-map self-shadowing (paper §7.2.4 second paragraph)
 *   - light-space scattering / Mie / silver-lining lighting
 *
 * `update()` is a no-op — TSL re-evaluates the position / opacity
 * nodes against the current GPU buffer contents every frame, so the
 * visual stays in sync with `GasSystem` automatically.
 */
export class PointSpritesGasRenderer implements GasRenderer {
  readonly object: InstancedMesh;
  readonly material: MeshBasicNodeMaterial;
  readonly geometry: PlaneGeometry;
  /**
   * Live billboard edge-length uniform (m). Mutate
   * `sizeUniform.value = ...` between frames to resize every smoke
   * sprite without rebuilding the renderer — the vertex shader reads
   * this uniform every dispatch. Useful for UI-driven tuning.
   */
  readonly sizeUniform: ReturnType<typeof uniform<'float', number>>;

  constructor(options: PointSpritesGasRendererOptions) {
    const {
      gas,
      color = 0xeeeeee,
      size = 0.08,
      initialOpacity = 0.6,
      opacityTau = gas.lifetime / 3,
      colorNode,
    } = options;

    if (!Number.isFinite(size) || size <= 0) {
      throw new Error(
        `PointSpritesGasRenderer: size must be a positive finite number, got ${size}`,
      );
    }
    if (!Number.isFinite(initialOpacity) || initialOpacity <= 0 || initialOpacity > 1) {
      throw new Error(
        `PointSpritesGasRenderer: initialOpacity must be in (0, 1], got ${initialOpacity}`,
      );
    }
    if (!Number.isFinite(opacityTau) || opacityTau <= 0) {
      throw new Error(
        `PointSpritesGasRenderer: opacityTau must be a positive finite number, got ${opacityTau}`,
      );
    }

    // Unit quad in the local xy plane — vertices at (±0.5, ±0.5, 0).
    // The vertex shader displaces along the camera's right / up axes
    // so the geometry's z is irrelevant; we still set 1×1 so any
    // post-process that reads `geometry.boundingSphere` gets a sane
    // default for frustum culling fallback.
    const geom = new PlaneGeometry(1, 1);

    const mat = new MeshBasicNodeMaterial();
    mat.transparent = true;
    mat.depthWrite = false; // additive-style; depth-sort is the post-MVP fix
    // Constant fallback colour. When `colorNode` is supplied below it
    // overrides this via `mat.colorNode`; otherwise this is the tint
    // every sprite uses. Per-vertex alpha lives on `opacityNode`.
    const c = new Color(color);
    mat.color = c;

    const sizeU = uniform(size, 'float');

    // --- Billboard math (camera-facing quad in world space) ---
    // `cameraWorldMatrix` is the camera's local-to-world transform; its
    // first two columns are the camera's right and up axes expressed in
    // world coordinates. Multiplying the quad-local (x, y) by those
    // axes and adding the smoke's world-space centre puts every
    // instance flat-on to the camera, regardless of camera orbit.
    const centerWS: Any = (gas.smokePositions as Any).element(instanceIndex).xyz;
    const cameraRight: Any = (cameraWorldMatrix as Any).element(0).xyz;
    const cameraUp: Any = (cameraWorldMatrix as Any).element(1).xyz;

    // Alive gate folds into the size. A dead slot (alive=0) gets a
    // size of 0, collapsing the quad to a point at the smoke's last
    // position — invisible but cheap. (Setting `discard` would also
    // work but adds a per-fragment branch we don't need here.)
    const alive: Any = (gas.smokeAlive as Any).element(instanceIndex);
    const aliveFloat: Any = max(float(0), alive.toFloat());
    const effectiveSize: Any = (sizeU as Any).mul(aliveFloat);

    const offset: Any = cameraRight
      .mul((positionLocal as Any).x)
      .mul(effectiveSize)
      .add(cameraUp.mul((positionLocal as Any).y).mul(effectiveSize));
    (mat as Any).positionNode = centerWS.add(offset);

    // Optional per-instance colorNode. The callback receives the
    // smoke's world-space centre + its SPH-interpolated drift velocity
    // (Macklin 2014 §7.2.1 eq. 28 output, written by `smokeAdvect`)
    // as TSL vec3 nodes. Demos use it to colour by speed, height,
    // radius, etc. When omitted, `mat.color` carries the constant
    // tint set above.
    if (colorNode !== undefined) {
      const velocity: Any = (gas.smokeVelocities as Any).element(instanceIndex).xyz;
      (mat as Any).colorNode = colorNode(centerWS, velocity);
    }

    // Per-instance opacity. Exponential decay against age, modulated
    // by alive flag (so dead slots drop to zero opacity even though
    // their size is also zero — belt + braces).
    const age: Any = (gas.smokeAge as Any).element(instanceIndex);
    const fade: Any = exp(age.div(float(opacityTau)).negate())
      .mul(float(initialOpacity))
      .mul(aliveFloat);

    // Soft circular puff mask. `uv()` is the per-fragment quad UV in
    // [0, 1]², so `2·uv − 1` is in [−1, +1]² and `r² = dot(d, d)` is
    // the squared distance to the quad centre. The radial falloff
    // `(1 − r²)²` clamped to ≥ 0 starts at 1 at the centre, hits 0 at
    // the inscribed-disk edge (r = 1), and stays 0 beyond — so the
    // square corners of the quad blend out completely. The squaring
    // gives the falloff a bit of "shoulder" so the puff reads soft
    // rather than as a hard alpha disk. opacityNode-side scalar
    // multiply: TSL handles the vertex-stage `fade` and fragment-stage
    // `radial` halves automatically.
    const diskUv: Any = (uv() as Any).mul(2.0).sub(1.0);
    const r2: Any = diskUv.dot(diskUv);
    const radial: Any = float(1.0).sub(r2).max(float(0));
    const radialSoft: Any = radial.mul(radial);
    (mat as Any).opacityNode = fade.mul(radialSoft);

    const mesh = new InstancedMesh(geom, mat, gas.capacity);
    mesh.frustumCulled = false;

    // Identity-seed `instanceMatrix` so it doesn't collapse instances
    // to the origin (zero matrices · positionNode = 0). The actual per-
    // instance placement comes from the positionNode above; the matrix
    // is unused for placement but three.js multiplies through it.
    const identity = new Matrix4();
    for (let i = 0; i < gas.capacity; i++) mesh.setMatrixAt(i, identity);
    mesh.instanceMatrix.needsUpdate = true;

    this.geometry = geom;
    this.material = mat;
    this.object = mesh;
    this.sizeUniform = sizeU;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
