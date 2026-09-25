import {
  Color,
  FloatType,
  LinearFilter,
  NearestFilter,
  NoColorSpace,
  RGBAFormat,
  RenderTarget,
  Scene,
  Vector2,
  type Camera,
} from 'three';
import { uniform } from 'three/tsl';
import { type WebGPURenderer } from 'three/webgpu';
import type { FluidSystem } from '../FluidSystem.js';
import {
  buildDefaultParams,
  type FluidSurfaceParamDefaults,
  type FluidSurfaceParams,
  type SmoothingResolution,
} from './params.js';
import {
  buildAnisotropyKernel,
  createAnisotropyKernelUniforms,
  type AnisotropyKernelUniforms,
} from './passes/anisotropy.js';
import { createDepthPassMesh, type DepthPassMesh } from './passes/depth.js';
import { NRF_FAR_SENTINEL, NRF_TOTAL_DISPATCHES, SmoothingPass } from './passes/smoothing.js';
import { createThicknessPassMesh, type ThicknessPassMesh } from './passes/thickness.js';
import { createPhysicalSurfaceMesh, type PhysicalSurfaceMesh } from './surface/PhysicalSurface.js';
import { DebugRenderer } from './debug/DebugRenderer.js';

/**
 * Screen-space fluid surface renderer.
 *
 * Production pipeline (per frame, run via `prepareRender()` before the
 * harness's standard scene render):
 *   1. Pass 1 — sphere imposters → `pass1RT`.
 *   2. Pass 2 — Truong & Yuksel 2018 narrow-range filter (4 separable
 *      1D dispatches + 1 fixed-kernel 2D cleanup) → `finalRT`.
 *   3. Pass 3 — additive Gaussian thickness splat → `thicknessRT`.
 *
 * The harness's standard scene render then rasterises the user's scene
 * including `renderer.mesh` (the PBR `MeshPhysicalNodeMaterial`-backed
 * surface), which samples `finalRT` at vertex stage and `thicknessRT`
 * at fragment stage. three.js handles lighting, IBL, and transmission
 * resolve.
 *
 * Production path is PBR (`MeshPhysicalNodeMaterial`). Surface debug
 * views (#13 viewPos, #14 normal) live in `debug/surfaceViews.ts` and
 * run via `DebugRenderer`. Anisotropic-ellipsoid imposters (Yu & Turk
 * 2010) hook in via the Pass-1 depth material.
 */
export interface FluidSurfaceRendererOptions {
  readonly fluidSystem: FluidSystem;
  readonly renderer: WebGPURenderer;
  /** Harness-owned scene. The user's content + the surface mesh. */
  readonly scene: Scene;
  /** Harness-owned camera. */
  readonly camera: Camera;
  /** Initial parameter values (each handle's `value`). Optional. */
  readonly defaults?: FluidSurfaceParamDefaults;
}

function makeFloatTarget(
  width: number,
  height: number,
  withDepth: boolean,
  name: string,
): RenderTarget {
  const rt = new RenderTarget(width, height, {
    format: RGBAFormat,
    type: FloatType,
    minFilter: NearestFilter,
    magFilter: NearestFilter,
    depthBuffer: withDepth,
  });
  rt.texture.colorSpace = NoColorSpace;
  rt.texture.name = name;
  return rt;
}

function resolutionFactor(res: SmoothingResolution): number {
  return res === 'full' ? 1 : res === 'half' ? 0.5 : 0.25;
}

export class FluidSurfaceRenderer {
  /** Live-tunable parameter handles, grouped by pipeline stage. */
  readonly params: FluidSurfaceParams;

  /**
   * The renderable surface mesh. Add to your scene once; the harness's
   * standard render pipeline picks it up.
   *
   * Visibility is auto-managed by the `params.debug.view` callback —
   * when a debug view is active, `mesh.visible` is false and the spec
   * routes its `customRender` to `renderDebugView()` instead.
   */
  readonly mesh: PhysicalSurfaceMesh['mesh'];

  private readonly opts: FluidSurfaceRendererOptions;
  private readonly fluidScene: Scene;
  private readonly thicknessScene: Scene;
  private readonly anisotropyUniforms: AnisotropyKernelUniforms;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private readonly anisotropyKernel: any;
  /** Last seen anisotropy.enabled — rebuilds the depth material on flip. */
  private lastAnisotropyEnabled: boolean;

  private readonly pass1RT: RenderTarget;
  private readonly smoothA: RenderTarget;
  private readonly smoothB: RenderTarget;
  private readonly finalRT: RenderTarget;
  private readonly thicknessRT: RenderTarget;

  private depthPass: DepthPassMesh;
  private readonly thicknessPass: ThicknessPassMesh;
  private readonly smoothingPass: SmoothingPass;
  private readonly physicalSurface: PhysicalSurfaceMesh;
  private readonly debugRenderer: DebugRenderer;

  private readonly radiusUniform: ReturnType<typeof uniform<'float', number>>;
  private readonly splatRadiusUniform: ReturnType<typeof uniform<'float', number>>;
  private readonly filterSigmaUniform: ReturnType<typeof uniform<'float', number>>;
  private readonly filterDeltaUniform: ReturnType<typeof uniform<'float', number>>;
  private readonly filterMuUniform: ReturnType<typeof uniform<'float', number>>;
  private readonly depthScaleUniform: ReturnType<typeof uniform<'float', number>>;
  private readonly derivativeScaleUniform: ReturnType<typeof uniform<'float', number>>;

  private readonly clearColorScratch = new Color();
  private readonly sentinelColor = new Color();
  private readonly sizeScratch = new Vector2();

  constructor(opts: FluidSurfaceRendererOptions) {
    this.opts = opts;
    this.params = buildDefaultParams(opts.defaults);

    const r = opts.fluidSystem.particles.particleRadius;

    // Uniforms wired to params. The handles' `value` fields drive
    // these at the start of every `prepareRender()` call.
    this.radiusUniform = uniform(this.params.depth.imposterRadius.value * r, 'float');
    this.splatRadiusUniform = uniform(this.params.thickness.splatRadius.value * r, 'float');
    this.filterSigmaUniform = uniform(this.params.smoothing.sigma.value * r, 'float');
    this.filterDeltaUniform = uniform(this.params.smoothing.delta.value * r, 'float');
    this.filterMuUniform = uniform(this.params.smoothing.mu.value * r, 'float');
    this.depthScaleUniform = uniform(this.params.debug.depthScale.value, 'float');
    this.derivativeScaleUniform = uniform(this.params.debug.derivativeScale.value, 'float');

    // RTs sized by smoothing resolution (Pass 1 + NRF) and canvas
    // resolution (thickness + scene-behind, sampled with bilinear so
    // the surface composite doesn't show pixel-grid artifacts).
    // getDrawingBufferSize, not getSize: setPixelRatio multiplies the
    // drawing buffer past CSS px, and the RTs are sampled by drawing-
    // buffer-aligned screenUV — using CSS px here aliases pass1RT to
    // 1/DPR² of true canvas resolution.
    const sizeVec = new Vector2();
    opts.renderer.getDrawingBufferSize(sizeVec);
    const fullW = Math.max(1, Math.floor(sizeVec.x));
    const fullH = Math.max(1, Math.floor(sizeVec.y));
    const factor = resolutionFactor(this.params.smoothing.resolution.value);
    const smoothW = Math.max(1, Math.floor(fullW * factor));
    const smoothH = Math.max(1, Math.floor(fullH * factor));

    this.pass1RT = makeFloatTarget(smoothW, smoothH, true, 'FluidSurface.pass1');
    this.smoothA = makeFloatTarget(smoothW, smoothH, false, 'FluidSurface.smoothA');
    this.smoothB = makeFloatTarget(smoothW, smoothH, false, 'FluidSurface.smoothB');
    this.finalRT = makeFloatTarget(smoothW, smoothH, false, 'FluidSurface.finalRT');
    // finalRT sampled by the surface mesh at canvas resolution via
    // `screenUV`; bilinear avoids row-wise banding when canvas != smooth.
    this.finalRT.texture.minFilter = LinearFilter;
    this.finalRT.texture.magFilter = LinearFilter;

    this.thicknessRT = makeFloatTarget(fullW, fullH, false, 'FluidSurface.thickness');
    this.thicknessRT.texture.minFilter = LinearFilter;
    this.thicknessRT.texture.magFilter = LinearFilter;

    // Phase 14c — allocate anisotropy storage unconditionally so the
    // depth-material rebuild path on `params.anisotropy.enabled` flip
    // doesn't have to lazy-allocate (and risk a toggle racing the
    // rebuild against an in-flight dispatch). Buffers are zero-init,
    // ~5 MB at 100k particles.
    opts.fluidSystem.enableAnisotropyBuffers();
    // Calibrate the initial `k_s` from `h` when the input value is the
    // params-file default (1.0) — meaning the user hasn't tuned and the
    // owning spec didn't supply an override via `defaults.anisotropyKs`.
    // Yu & Turk's `Σ̃ = k_s · σ` wants `||k_s · C|| ≈ 1` (dimensionless),
    // and a typical interior particle's covariance eigenvalues are
    // O(h²) — so `k_s ≈ 1/h²` is the right starting point. Cheap
    // closed-form approximation of U-54's auto-derivation. A user-
    // tuned value (anything ≠ 1.0) survives this gate and persists
    // across scene rebuilds via the spec's `defaults.anisotropyKs`
    // round-trip.
    const ksFromHandle = this.params.anisotropy.ks.value;
    const ksAutoCalibrated = 1.0 / (opts.fluidSystem.h * opts.fluidSystem.h);
    const ksInitial = ksFromHandle === 1.0 ? ksAutoCalibrated : ksFromHandle;
    this.params.anisotropy.ks.value = ksInitial;
    this.anisotropyUniforms = createAnisotropyKernelUniforms({
      kr: this.params.anisotropy.kr.value,
      ks: ksInitial,
      kn: this.params.anisotropy.kn.value,
      nEpsilon: this.params.anisotropy.nEpsilon.value,
      lambda: this.params.anisotropy.lambda.value,
    });
    this.anisotropyKernel = buildAnisotropyKernel({
      fluidSystem: opts.fluidSystem,
      aniso: this.anisotropyUniforms,
    });
    this.lastAnisotropyEnabled = this.params.anisotropy.enabled.value;

    // Pre-pass meshes.
    this.depthPass = createDepthPassMesh({
      fluidSystem: opts.fluidSystem,
      radiusUniform: this.radiusUniform,
      useAnisotropy: this.lastAnisotropyEnabled,
    });
    this.fluidScene = new Scene();
    this.fluidScene.add(this.depthPass.mesh);

    this.thicknessPass = createThicknessPassMesh({
      fluidSystem: opts.fluidSystem,
      splatRadiusUniform: this.splatRadiusUniform,
    });
    this.thicknessScene = new Scene();
    this.thicknessScene.add(this.thicknessPass.mesh);

    this.smoothingPass = new SmoothingPass({
      pass1RT: this.pass1RT,
      smoothA: this.smoothA,
      smoothB: this.smoothB,
      finalRT: this.finalRT,
      uniforms: {
        sigma: this.filterSigmaUniform,
        delta: this.filterDeltaUniform,
        mu: this.filterMuUniform,
      },
    });

    // Surface mesh — PBR `MeshPhysicalNodeMaterial` over a tessellated
    // quad. Add to scene once; visibility is toggled by debug-view
    // changes via the `params.debug.view` reaction below.
    this.physicalSurface = createPhysicalSurfaceMesh({
      smoothedDepthTexture: this.finalRT.texture,
      thicknessTexture: this.thicknessRT.texture,
      fluidColor: this.params.surface.color.value,
      ior: this.params.surface.ior.value,
      attenuationDistance: this.params.surface.attenuationDistance.value,
      roughness: this.params.surface.roughness.value,
      thicknessScale: this.params.surface.thicknessScale.value,
      envIntensity: this.params.surface.envIntensity.value,
      environment: this.opts.scene.environment ?? null,
    });
    this.mesh = this.physicalSurface.mesh;
    this.mesh.visible = this.params.debug.view.value === 'off';

    this.debugRenderer = new DebugRenderer({
      renderer: opts.renderer,
      scene: opts.scene,
      camera: opts.camera,
      fluidSystem: opts.fluidSystem,
      fluidScene: this.fluidScene,
      depthMesh: this.depthPass.mesh,
      depthMaterial: this.depthPass.material,
      smoothingPass: this.smoothingPass,
      thicknessScene: this.thicknessScene,
      pass1RT: this.pass1RT,
      smoothA: this.smoothA,
      smoothB: this.smoothB,
      finalRT: this.finalRT,
      thicknessRT: this.thicknessRT,
      radiusUniform: this.radiusUniform,
      depthScaleUniform: this.depthScaleUniform,
      derivativeScaleUniform: this.derivativeScaleUniform,
      filterSigmaUniform: this.filterSigmaUniform,
      filterDeltaUniform: this.filterDeltaUniform,
      filterMuUniform: this.filterMuUniform,
      farSentinel: NRF_FAR_SENTINEL,
      params: this.params,
      anisotropyKernel: this.anisotropyKernel,
      anisotropyNEpsilonUniform: this.anisotropyUniforms.nEpsilon,
    });
  }

  /**
   * Pre-render hook. Call before rendering the main scene. Pushes
   * current parameter values into uniforms / the PBR material's
   * mutable fields, then runs the production pre-passes (Pass 1 →
   * NRF → thickness) so the surface mesh's textures carry valid
   * data when the harness's standard render reaches them.
   *
   * When a debug view is active, this is a no-op — `renderDebugView()`
   * runs the pre-passes itself (it needs the same RTs but with
   * potentially partial NRF dispatches via `inspectIter`).
   */
  prepareRender(): void {
    this.resizeTargets();
    this.syncParamsToUniforms();
    this.maybeRebuildDepthMaterial();

    if (this.params.debug.view.value !== 'off') {
      // Debug renderer drives everything; PBR mesh hidden anyway.
      this.mesh.visible = false;
      return;
    }

    this.mesh.visible = true;

    const r = this.opts.renderer;

    // Phase 14c — when anisotropy is enabled, dispatch the Yu & Turk
    // §4 compute kernel once per frame before the depth pass reads
    // `anisotropyDiag` / `anisotropyOff` / `smoothedPositions`.
    // Async, but the render path inherently awaits prior compute via
    // WebGPU's command queue; we kick the dispatch and proceed.
    if (this.params.anisotropy.enabled.value) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      void (r as any).computeAsync([this.anisotropyKernel]);
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const savedClearColor = (r as any).getClearColor(this.clearColorScratch).clone();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const savedClearAlpha = (r as any).getClearAlpha();
    const savedAutoClear = r.autoClear;

    // Pass 1 — imposters → pass1RT, cleared to NRF far-sentinel so
    // background pixels bias the smoother's clamp branch.
    this.sentinelColor.setRGB(NRF_FAR_SENTINEL, 0, 0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (r as any).setClearColor(this.sentinelColor, 1.0);
    r.setRenderTarget(this.pass1RT);
    r.autoClear = true;
    this.depthPass.mesh.material = this.depthPass.material;
    r.render(this.fluidScene, this.opts.camera);

    // Pass 2 — full NRF (5 dispatches) → finalRT.
    this.smoothingPass.run(r, NRF_TOTAL_DISPATCHES, this.opts.camera);

    // Pass 3 — thickness splat → thicknessRT.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (r as any).setClearColor(0x000000, 0);
    r.setRenderTarget(this.thicknessRT);
    r.autoClear = true;
    r.render(this.thicknessScene, this.opts.camera);

    // Restore canvas + state for the standard scene render.
    r.setRenderTarget(null);
    r.getSize(this.sizeScratch);
    r.setViewport(0, 0, this.sizeScratch.x, this.sizeScratch.y);
    r.setScissor(0, 0, this.sizeScratch.x, this.sizeScratch.y);
    r.autoClear = savedAutoClear;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (r as any).setClearColor(savedClearColor, savedClearAlpha);
  }

  /**
   * Debug-view render. Call instead of the main scene render when
   * `params.debug.view.value !== 'off'`; the spec is responsible for
   * the conditional wiring (a `live` callback flips it as the artist
   * changes the view dropdown).
   */
  renderDebugView(): void {
    this.resizeTargets();
    this.syncParamsToUniforms();
    this.mesh.visible = false;
    this.debugRenderer.render();
  }

  /** Keep screen-space buffers aligned with the canvas after a resize or DPR change. */
  private resizeTargets(): void {
    this.opts.renderer.getDrawingBufferSize(this.sizeScratch);
    const width = Math.max(1, Math.floor(this.sizeScratch.x));
    const height = Math.max(1, Math.floor(this.sizeScratch.y));
    const factor = resolutionFactor(this.params.smoothing.resolution.value);
    const smoothWidth = Math.max(1, Math.floor(width * factor));
    const smoothHeight = Math.max(1, Math.floor(height * factor));
    for (const target of [this.pass1RT, this.smoothA, this.smoothB, this.finalRT]) {
      if (target.width !== smoothWidth || target.height !== smoothHeight)
        target.setSize(smoothWidth, smoothHeight);
    }
    if (this.thicknessRT.width !== width || this.thicknessRT.height !== height)
      this.thicknessRT.setSize(width, height);
  }

  /**
   * Push parameter handle values into uniforms and the PBR material's
   * mutable fields. Called at the top of `prepareRender` and
   * `renderDebugView` so a single live update is visible on the next
   * frame regardless of which path runs.
   */
  private syncParamsToUniforms(): void {
    const r = this.opts.fluidSystem.particles.particleRadius;
    this.radiusUniform.value = this.params.depth.imposterRadius.value * r;
    this.splatRadiusUniform.value = this.params.thickness.splatRadius.value * r;
    this.filterSigmaUniform.value = this.params.smoothing.sigma.value * r;
    this.filterDeltaUniform.value = this.params.smoothing.delta.value * r;
    this.filterMuUniform.value = this.params.smoothing.mu.value * r;
    this.depthScaleUniform.value = this.params.debug.depthScale.value;
    this.derivativeScaleUniform.value = this.params.debug.derivativeScale.value;
    this.anisotropyUniforms.kr.value = this.params.anisotropy.kr.value;
    this.anisotropyUniforms.ks.value = this.params.anisotropy.ks.value;
    this.anisotropyUniforms.kn.value = this.params.anisotropy.kn.value;
    this.anisotropyUniforms.nEpsilon.value = this.params.anisotropy.nEpsilon.value;
    this.anisotropyUniforms.lambda.value = this.params.anisotropy.lambda.value;

    // Surface material — push live param values onto the PBR material.
    this.physicalSurface.material.attenuationDistance =
      this.params.surface.attenuationDistance.value;
    this.physicalSurface.material.roughness = this.params.surface.roughness.value;
    this.physicalSurface.material.ior = this.params.surface.ior.value;
    this.physicalSurface.material.envMapIntensity = this.params.surface.envIntensity.value;
    // attenuationColor shares the params handle's Color instance, so a
    // setRGB on the handle already updates the material.
    this.physicalSurface.thicknessScaleUniform.value = this.params.surface.thicknessScale.value;
  }

  /**
   * Rebuild the depth-pass mesh's material when `params.anisotropy.enabled`
   * has flipped since the last frame. Sphere-imposter and ellipsoid-
   * imposter live in different TSL node graphs (different `positionNode`,
   * `outputNode`); switching is a material rebuild rather than a uniform
   * branch — same lifecycle as the surface-mode flip (U-FR-7's analogue
   * for the depth stage). The mesh stays the same; we drop the old
   * material's GPU resources and swap in the new one.
   */
  private maybeRebuildDepthMaterial(): void {
    const enabled = this.params.anisotropy.enabled.value;
    if (enabled === this.lastAnisotropyEnabled) return;

    const oldMaterial = this.depthPass.material;
    const rebuilt = createDepthPassMesh({
      fluidSystem: this.opts.fluidSystem,
      radiusUniform: this.radiusUniform,
      useAnisotropy: enabled,
    });
    // Keep the existing InstancedMesh — the renderer's `fluidScene`
    // already points at it. Replace just its material and the
    // DebugRenderer's reference.
    this.depthPass.mesh.material = rebuilt.material;
    this.depthPass = {
      mesh: this.depthPass.mesh,
      material: rebuilt.material,
    };
    // The freshly-built rebuilt.mesh is unused now; dispose its
    // geometry to free the duplicate quad allocation.
    rebuilt.mesh.geometry.dispose();
    oldMaterial.dispose();

    // DebugRenderer holds the depth material reference internally;
    // notify it to re-bind. (See `DebugRenderer.swapDepthMaterial`.)
    this.debugRenderer.swapDepthMaterial(rebuilt.material);

    this.lastAnisotropyEnabled = enabled;
  }

  dispose(): void {
    this.debugRenderer.dispose();
    this.physicalSurface.mesh.geometry.dispose();
    this.physicalSurface.material.dispose();
    this.smoothingPass.dispose();
    this.thicknessPass.mesh.geometry.dispose();
    this.thicknessPass.material.dispose();
    this.depthPass.mesh.geometry.dispose();
    this.depthPass.material.dispose();
    this.pass1RT.dispose();
    this.smoothA.dispose();
    this.smoothB.dispose();
    this.finalRT.dispose();
    this.thicknessRT.dispose();
  }
}
