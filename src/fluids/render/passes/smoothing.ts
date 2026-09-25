import { Mesh, PlaneGeometry, Scene, type Texture } from 'three';
import { MeshBasicNodeMaterial, type WebGPURenderer } from 'three/webgpu';
import type { RenderTarget } from 'three';
import {
  Break,
  Fn,
  If,
  Loop,
  cameraProjectionMatrix,
  exp,
  float,
  int,
  ivec2,
  positionLocal,
  screenCoordinate,
  screenSize,
  select,
  textureLoad,
  uniform,
  vec4,
} from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 *
 *
 * Pipeline: 4 separable 1D passes (X-Y-X-Y) + 1 fixed-kernel 2D cleanup
 * pass (paper §3.4). Total 5 dispatches.
 *
 * Equations transcribed verbatim from the paper:
 *
 *   z'_i = Σ_j ω_ij f(z_i, z_j) / Σ_j ω_ij                         (eq. 1)
 *   f(z_i, z_j) = z_j           if z_j ≥ z_i − δ                   (eq. 2)
 *               = z_i − μ        otherwise
 *   ω_ij = 0                     if z_j > z_i + δ                  (eq. 3)
 *        = G(p_i, p_j, σ_i)      otherwise
 *   G(p_i, p_j, σ_i) = exp(−|p_j − p_i|² / (2 σ_i²))                (eq. 4)
 *   σ_i = ⌈H · σ / (2 |z_i| · tan(α/2))⌉                            (eq. 5)
 *   ω_ij = 0       if z_j > z_i + δ ∨ z_k > z_i + δ                 (eq. 6, bias)
 *        = G       otherwise   where p_k = p_i + (p_i − p_j)
 *   z_i + δ_high ≥ z_j ≥ z_i − δ_low                                (eq. 7, dyn.)
 *   δ_low  ← max(δ_low,  z_i − z_j + δ)                              (eq. 8)
 *   δ_high ← max(δ_high, z_j − z_i + δ)                              (eq. 9)
 *
 * **Convention adapter — our positive-distance encoding.** Our
 * `pass1RT` stores `-viewSurfaceZ` (positive metres-from-camera) per
 * Phase 14 Pass 1 (`passes/depth.ts`). The paper's `z` is NEGATIVE
 * eye-space (smaller = farther). The math is sign-symmetric under the
 * flip; the operator translation is:
 *
 *   Truong (negative z)         Ours (positive z' = -z)
 *   ───────────────────         ────────────────────────
 *   z_j > z_i + δ  → ignore     z'_j < z'_i − δ_near
 *   z_j < z_i − δ  → clamp      z'_j > z'_i + δ_far
 *   f = z_i − μ                 f = z'_i + μ
 *   eq. 8 update                δ_far  ← max(δ_far,  z'_j − z'_i + δ)
 *   eq. 9 update                δ_near ← max(δ_near, z'_i − z'_j + δ)
 *
 * Variable naming: `deltaNear` corresponds to Truong's δ_high (tolerance
 * to closer-than-i); `deltaFar` corresponds to δ_low (tolerance to
 * farther-than-i).
 *
 * **Background sentinel.** `pass1RT` is cleared to `NRF_FAR_SENTINEL`
 * (1e6). Background pixels then naturally trigger the eq. 2 clamp
 * branch; the clamp value `z'_i + μ` keeps them as "slightly farther
 * than centre" contributions that don't bias the filter. Centre-pixel
 * background detection (skip the whole pixel) uses
 * `NRF_FAR_SENTINEL_THRESHOLD` (1e5).
 */

export const NRF_FAR_SENTINEL = 1e6;
export const NRF_FAR_SENTINEL_THRESHOLD = 1e5;

export type FilterAxis = 'x' | 'y';

/**
 * Maximum 1D kernel half-width (pixels). Truong §3.1 sets the kernel
 * half-width to 3σ_i per eq. 5; this caps the dynamic σ_i at ~21
 * (3·21 = 63 ≈ 64). At quarter-res with default `σ = 0.7r`, computed
 * σ_i is typically 3–11, well below the cap.
 */
const MAX_KERNEL_HALF_WIDTH = 64;

export interface SmoothingUniforms {
  /** World-space filter size σ. Default 0.7r per Truong §4. */
  readonly sigma: ReturnType<typeof uniform<'float', number>>;
  /** Range threshold δ. Default 10r per Truong §4. */
  readonly delta: ReturnType<typeof uniform<'float', number>>;
  /** Clamp magnitude μ. Default r per Truong §4. */
  readonly mu: ReturnType<typeof uniform<'float', number>>;
}

export interface BuildNRF1DMaterialArgs {
  readonly inputTexture: Texture;
  readonly axis: FilterAxis;
  readonly uniforms: SmoothingUniforms;
}

/**
 * Build the 1D NRF material for one separable pass (X or Y). Reads
 * `inputTexture`, writes filtered depth to the bound render target.
 */
export function buildNRF1DMaterial(args: BuildNRF1DMaterialArgs): MeshBasicNodeMaterial {
  const material = new MeshBasicNodeMaterial();
  material.transparent = false;
  material.depthTest = false;
  material.depthWrite = false;
  material.vertexNode = vec4(positionLocal.xy, 0.0, 1.0);

  const tex: Any = args.inputTexture;
  const sigmaUni: Any = args.uniforms.sigma;
  const deltaUni: Any = args.uniforms.delta;
  const muUni: Any = args.uniforms.mu;

  const dx: number = args.axis === 'x' ? 1 : 0;
  const dy: number = args.axis === 'y' ? 1 : 0;

  material.outputNode = Fn(() => {
    const px: Any = screenCoordinate.x.toUint();
    const py: Any = screenCoordinate.y.toUint();
    const W: Any = screenSize.x.toUint();
    const Hres: Any = screenSize.y.toUint();

    const z_i: Any = textureLoad(tex, ivec2(px, py)).r.toVar();

    const Fy: Any = (cameraProjectionMatrix as Any).element(1).y;
    const sigma_i_f: Any = Hres.toFloat()
      .mul(sigmaUni)
      .mul(Fy)
      .div(z_i.mul(2.0).max(float(1e-4)))
      .max(float(1.0));
    const halfWidth: Any = sigma_i_f.mul(3.0).ceil().toUint();

    const sum: Any = z_i.toVar();
    const weightSum: Any = float(1.0).toVar();
    const deltaFar: Any = deltaUni.toVar();
    const deltaNear: Any = deltaUni.toVar();
    const twoSigmaSq: Any = sigma_i_f.mul(sigma_i_f).mul(2.0);

    Loop(
      { start: int(1), end: int(MAX_KERNEL_HALF_WIDTH), type: 'int', condition: '<=' },
      ({ i: k }: { i: Any }) => {
        If(k.toUint().greaterThan(halfWidth), () => {
          Break();
        });

        const offX: Any = int(dx).mul(k);
        const offY: Any = int(dy).mul(k);
        const xPos: Any = px
          .toInt()
          .add(offX)
          .max(int(0))
          .min(W.toInt().sub(int(1)));
        const yPos: Any = py
          .toInt()
          .add(offY)
          .max(int(0))
          .min(Hres.toInt().sub(int(1)));
        const xNeg: Any = px
          .toInt()
          .sub(offX)
          .max(int(0))
          .min(W.toInt().sub(int(1)));
        const yNeg: Any = py
          .toInt()
          .sub(offY)
          .max(int(0))
          .min(Hres.toInt().sub(int(1)));

        const z_j_pos: Any = textureLoad(tex, ivec2(xPos, yPos)).r;
        const z_j_neg: Any = textureLoad(tex, ivec2(xNeg, yNeg)).r;

        const tooClose_pos: Any = z_j_pos.lessThan(z_i.sub(deltaNear));
        const tooClose_neg: Any = z_j_neg.lessThan(z_i.sub(deltaNear));
        const dropPair: Any = tooClose_pos.or(tooClose_neg);

        If(dropPair.not(), () => {
          const f_pos: Any = select(
            z_j_pos.greaterThan(z_i.add(deltaFar)),
            z_i.add(muUni),
            z_j_pos,
          );
          const f_neg: Any = select(
            z_j_neg.greaterThan(z_i.add(deltaFar)),
            z_i.add(muUni),
            z_j_neg,
          );

          const k_f: Any = k.toFloat();
          const weight: Any = exp(k_f.mul(k_f).div(twoSigmaSq).negate());

          sum.assign(sum.add(weight.mul(f_pos.add(f_neg))));
          weightSum.assign(weightSum.add(weight.mul(2.0)));

          const inRange_pos: Any = z_j_pos.lessThanEqual(z_i.add(deltaFar));
          const inRange_neg: Any = z_j_neg.lessThanEqual(z_i.add(deltaFar));

          If(inRange_pos, () => {
            const newFar: Any = z_j_pos.sub(z_i).add(deltaUni);
            const newNear: Any = z_i.sub(z_j_pos).add(deltaUni);
            deltaFar.assign(deltaFar.max(newFar));
            deltaNear.assign(deltaNear.max(newNear));
          });
          If(inRange_neg, () => {
            const newFar: Any = z_j_neg.sub(z_i).add(deltaUni);
            const newNear: Any = z_i.sub(z_j_neg).add(deltaUni);
            deltaFar.assign(deltaFar.max(newFar));
            deltaNear.assign(deltaNear.max(newNear));
          });
        });
      },
    );

    const filtered: Any = select(
      z_i.greaterThanEqual(float(NRF_FAR_SENTINEL_THRESHOLD)),
      float(NRF_FAR_SENTINEL),
      sum.div(weightSum),
    );

    return vec4(filtered, 0.0, 0.0, 1.0);
  })();

  return material;
}

export interface BuildNRF2DCleanupMaterialArgs {
  readonly inputTexture: Texture;
  readonly uniforms: SmoothingUniforms;
}

/**
 * 2D cleanup (Truong §3.4). Fixed 5×5 kernel applied as a single 2D NRF
 * after the four 1D passes. Hides axis-aligned streaks that the
 * separable approximation introduces near silhouettes.
 */
export function buildNRF2DCleanupMaterial(
  args: BuildNRF2DCleanupMaterialArgs,
): MeshBasicNodeMaterial {
  const material = new MeshBasicNodeMaterial();
  material.transparent = false;
  material.depthTest = false;
  material.depthWrite = false;
  material.vertexNode = vec4(positionLocal.xy, 0.0, 1.0);

  const tex: Any = args.inputTexture;
  const sigmaUni: Any = args.uniforms.sigma;
  const deltaUni: Any = args.uniforms.delta;
  const muUni: Any = args.uniforms.mu;

  material.outputNode = Fn(() => {
    const px: Any = screenCoordinate.x.toUint();
    const py: Any = screenCoordinate.y.toUint();
    const W: Any = screenSize.x.toUint();
    const Hres: Any = screenSize.y.toUint();

    const z_i: Any = textureLoad(tex, ivec2(px, py)).r.toVar();

    const Fy: Any = (cameraProjectionMatrix as Any).element(1).y;
    const sigma_i_f: Any = Hres.toFloat()
      .mul(sigmaUni)
      .mul(Fy)
      .div(z_i.mul(2.0).max(float(1e-4)))
      .max(float(1.0));
    const twoSigmaSq: Any = sigma_i_f.mul(sigma_i_f).mul(2.0);

    const sum: Any = z_i.toVar();
    const weightSum: Any = float(1.0).toVar();
    const deltaFar: Any = deltaUni.toVar();
    const deltaNear: Any = deltaUni.toVar();

    // 12 mirror-pair offsets (5×5 kernel = 24 non-centre pixels in 12 pairs).
    const pairOffsets: Array<readonly [number, number]> = [
      [1, 0],
      [0, 1],
      [1, 1],
      [1, -1],
      [2, 0],
      [0, 2],
      [2, 1],
      [2, -1],
      [1, 2],
      [-1, 2],
      [2, 2],
      [2, -2],
    ];

    for (const [ox, oy] of pairOffsets) {
      const xPos: Any = px
        .toInt()
        .add(int(ox))
        .max(int(0))
        .min(W.toInt().sub(int(1)));
      const yPos: Any = py
        .toInt()
        .add(int(oy))
        .max(int(0))
        .min(Hres.toInt().sub(int(1)));
      const xNeg: Any = px
        .toInt()
        .sub(int(ox))
        .max(int(0))
        .min(W.toInt().sub(int(1)));
      const yNeg: Any = py
        .toInt()
        .sub(int(oy))
        .max(int(0))
        .min(Hres.toInt().sub(int(1)));

      const z_j_pos: Any = textureLoad(tex, ivec2(xPos, yPos)).r;
      const z_j_neg: Any = textureLoad(tex, ivec2(xNeg, yNeg)).r;

      const tooClose_pos: Any = z_j_pos.lessThan(z_i.sub(deltaNear));
      const tooClose_neg: Any = z_j_neg.lessThan(z_i.sub(deltaNear));
      const dropPair: Any = tooClose_pos.or(tooClose_neg);

      If(dropPair.not(), () => {
        const f_pos: Any = select(z_j_pos.greaterThan(z_i.add(deltaFar)), z_i.add(muUni), z_j_pos);
        const f_neg: Any = select(z_j_neg.greaterThan(z_i.add(deltaFar)), z_i.add(muUni), z_j_neg);

        const dist_sq: Any = float(ox * ox + oy * oy);
        const weight: Any = exp(dist_sq.div(twoSigmaSq).negate());

        sum.assign(sum.add(weight.mul(f_pos.add(f_neg))));
        weightSum.assign(weightSum.add(weight.mul(2.0)));

        const inRange_pos: Any = z_j_pos.lessThanEqual(z_i.add(deltaFar));
        const inRange_neg: Any = z_j_neg.lessThanEqual(z_i.add(deltaFar));

        If(inRange_pos, () => {
          const newFar: Any = z_j_pos.sub(z_i).add(deltaUni);
          const newNear: Any = z_i.sub(z_j_pos).add(deltaUni);
          deltaFar.assign(deltaFar.max(newFar));
          deltaNear.assign(deltaNear.max(newNear));
        });
        If(inRange_neg, () => {
          const newFar: Any = z_j_neg.sub(z_i).add(deltaUni);
          const newNear: Any = z_i.sub(z_j_neg).add(deltaUni);
          deltaFar.assign(deltaFar.max(newFar));
          deltaNear.assign(deltaNear.max(newNear));
        });
      });
    }

    const filtered: Any = select(
      z_i.greaterThanEqual(float(NRF_FAR_SENTINEL_THRESHOLD)),
      float(NRF_FAR_SENTINEL),
      sum.div(weightSum),
    );

    return vec4(filtered, 0.0, 0.0, 1.0);
  })();

  return material;
}

/** Number of NRF dispatches (4 separable 1D + 1 fixed-kernel 2D cleanup). */
export const NRF_TOTAL_DISPATCHES = 5;

/**
 * Orchestrates the five NRF dispatches against three RTs (smoothA,
 * smoothB, finalRT). The owner allocates the RTs and supplies them
 * here; the pass holds all five materials and swaps them onto a shared
 * full-screen mesh as it walks the dispatch sequence.
 *
 * RT layout per dispatch K:
 *   K=1 (X1):  pass1RT  → smoothA
 *   K=2 (Y1):  smoothA  → smoothB
 *   K=3 (X2):  smoothB  → smoothA   (overwrites K=1)
 *   K=4 (Y2):  smoothA  → smoothB   (overwrites K=2)
 *   K=5 (2D):  smoothB  → finalRT
 *
 * The X1/Y/X2 materials fan in different input textures so each is
 * built once and reused across frames. K=2 and K=4 both read smoothA
 * and write smoothB, so they share a single material; only the
 * texture's contents differ between K=2 and K=4 at render time.
 */
export class SmoothingPass {
  private readonly nrfX1FromPass1RT: MeshBasicNodeMaterial;
  private readonly nrfYFromSmoothA: MeshBasicNodeMaterial;
  private readonly nrfX2FromSmoothB: MeshBasicNodeMaterial;
  private readonly nrf2DFromSmoothB: MeshBasicNodeMaterial;
  private readonly mesh: Mesh;
  private readonly scene: Scene;

  constructor(
    private readonly opts: {
      readonly pass1RT: RenderTarget;
      readonly smoothA: RenderTarget;
      readonly smoothB: RenderTarget;
      readonly finalRT: RenderTarget;
      readonly uniforms: SmoothingUniforms;
    },
  ) {
    this.nrfX1FromPass1RT = buildNRF1DMaterial({
      inputTexture: opts.pass1RT.texture,
      axis: 'x',
      uniforms: opts.uniforms,
    });
    this.nrfYFromSmoothA = buildNRF1DMaterial({
      inputTexture: opts.smoothA.texture,
      axis: 'y',
      uniforms: opts.uniforms,
    });
    this.nrfX2FromSmoothB = buildNRF1DMaterial({
      inputTexture: opts.smoothB.texture,
      axis: 'x',
      uniforms: opts.uniforms,
    });
    this.nrf2DFromSmoothB = buildNRF2DCleanupMaterial({
      inputTexture: opts.smoothB.texture,
      uniforms: opts.uniforms,
    });

    this.mesh = new Mesh(new PlaneGeometry(2, 2), this.nrfX1FromPass1RT);
    this.mesh.frustumCulled = false;
    this.scene = new Scene();
    this.scene.add(this.mesh);
  }

  /** The texture the production pipeline reads (= 2D cleanup output). */
  get outputTexture(): Texture {
    return this.opts.finalRT.texture;
  }

  /**
   * Run dispatches K=1..K (clamped to [0, 5]). Caller must have filled
   * `pass1RT` first. The renderer's render-target binding is mutated
   * during the call; the caller restores it afterward.
   */
  run(renderer: WebGPURenderer, K: number, camera: import('three').Camera): void {
    const Kclamped = Math.max(0, Math.min(K, NRF_TOTAL_DISPATCHES));
    if (Kclamped <= 0) return;

    if (Kclamped >= 1) {
      this.mesh.material = this.nrfX1FromPass1RT;
      renderer.setRenderTarget(this.opts.smoothA);
      renderer.render(this.scene, camera);
    }
    if (Kclamped >= 2) {
      this.mesh.material = this.nrfYFromSmoothA;
      renderer.setRenderTarget(this.opts.smoothB);
      renderer.render(this.scene, camera);
    }
    if (Kclamped >= 3) {
      this.mesh.material = this.nrfX2FromSmoothB;
      renderer.setRenderTarget(this.opts.smoothA);
      renderer.render(this.scene, camera);
    }
    if (Kclamped >= 4) {
      this.mesh.material = this.nrfYFromSmoothA;
      renderer.setRenderTarget(this.opts.smoothB);
      renderer.render(this.scene, camera);
    }
    if (Kclamped >= 5) {
      this.mesh.material = this.nrf2DFromSmoothB;
      renderer.setRenderTarget(this.opts.finalRT);
      renderer.render(this.scene, camera);
    }
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.nrfX1FromPass1RT.dispose();
    this.nrfYFromSmoothA.dispose();
    this.nrfX2FromSmoothB.dispose();
    this.nrf2DFromSmoothB.dispose();
  }
}
