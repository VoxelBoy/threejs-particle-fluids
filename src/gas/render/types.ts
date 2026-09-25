import type { Object3D } from 'three';

/**
 * Polymorphic interface for any smoke-particle renderer.
 *
 * The MVP shipper is {@link PointSpritesGasRenderer} — a single
 * `THREE.Points` mesh whose vertex shader reads the gas-owned
 * `smokePositions` / `smokeAge` / `smokeAlive` storage buffers
 * directly. A future `VolumetricGasRenderer` (raymarched smoke with
 * shadow-map scattering, post-MVP per Macklin 2014 §7.2.4) plugs into
 * the same interface, so demos can swap renderers without touching the
 * physics-side code.
 *
 * Lifecycle:
 *   1. Construct with a `GasSystem`. Read its public `smokePositions` /
 *      `smokeAge` / `smokeAlive` / `smokeVelocities` buffers as inputs.
 *   2. Add `renderer.object` to the host scene.
 *   3. Call `renderer.update?.()` after each `loop.step()` if defined.
 *      Renderers that read the GPU buffers via TSL nodes don't need an
 *      `update` — three.js re-renders against the live buffers every
 *      frame. The hook exists for renderers that need CPU-side work
 *      (depth sort, billboard re-orient, etc.).
 *   4. Call `renderer.dispose()` when the demo tears down.
 */
export interface GasRenderer {
  /**
   * The Three.js node to add to the scene. `Points`, `Mesh`, `Group`,
   * etc. — the renderer chooses the underlying primitive.
   */
  readonly object: Object3D;
  /**
   * Optional per-frame hook. Called after each `loop.step()` so CPU-
   * side state (depth sort, etc.) can refresh. Renderers that read GPU
   * buffers directly via TSL nodes don't need this; three.js re-runs
   * the node graph against current buffer contents every frame.
   */
  update?: () => void;
  /** Release any GPU resources / event listeners. */
  dispose(): void;
}
