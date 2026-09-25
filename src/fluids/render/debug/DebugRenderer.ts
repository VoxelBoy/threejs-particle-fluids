import { Color, Mesh, Vector2, type Camera, type RenderTarget, type Scene } from 'three';
import { MeshBasicNodeMaterial, type WebGPURenderer } from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { FluidSystem } from '../../FluidSystem.js';
import type { DebugView, FluidSurfaceParams } from '../params.js';
import { NRF_TOTAL_DISPATCHES, type SmoothingPass } from '../passes/smoothing.js';
import { buildPass1DebugMaterial } from './pass1Views.js';
import {
  buildPass2DebugMaterial,
  buildPass2DebugScene,
  type Pass2DebugMode,
} from './pass2Views.js';
import {
  buildSurfaceDebugMaterial,
  buildSurfaceDebugScene,
  type SurfaceDebugMode,
} from './surfaceViews.js';
import { buildDepthBlitMaterial, buildDepthBlitScene } from './depthBlit.js';
import { buildAnisotropyDebugMaterial, type AnisotropyDebugMode } from './anisotropyViews.js';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';

const PASS1_DIRECT_MODES: Record<string, 'coverage' | 'diskMask' | 'depthGray' | 'depthHeat'> = {
  'pass1.coverage': 'coverage',
  'pass1.diskMask': 'diskMask',
  'pass1.depthGray': 'depthGray',
  'pass1.depthHeat': 'depthHeat',
};

const PASS2_DEBUG_MODES: Partial<Record<DebugView, Pass2DebugMode>> = {
  'pass2.iter0': 'depthGray',
  'pass2.iterK': 'depthGray',
  'pass2.clampActivation': 'clampActivation',
  'pass2.rangeMask': 'rangeMask',
  'pass2.kernelSize': 'kernelSize',
  'pass2.biasDiscardMask': 'biasDiscardMask',
  'pass2.deltaPerIter': 'deltaPerIter',
};

const SURFACE_DEBUG_MODES: Partial<Record<DebugView, SurfaceDebugMode>> = {
  'surface.viewPos': 'viewPos',
  'surface.normal': 'normal',
  'surface.ddx': 'ddxViewPos',
  'surface.ddy': 'ddyViewPos',
};

const ANISO_DEBUG_MODES: Partial<Record<DebugView, AnisotropyDebugMode>> = {
  'aniso.neighborCount': 'neighborCount',
  'aniso.interior': 'interior',
  'aniso.sigma1': 'sigma1',
  'aniso.aniso': 'aniso',
};

interface Pass2DebugBundle {
  readonly fromPass1RT: MeshBasicNodeMaterial;
  readonly fromSmoothA: MeshBasicNodeMaterial;
  readonly fromSmoothB: MeshBasicNodeMaterial;
}

interface Pass2DepthGrayBundle extends Pass2DebugBundle {
  readonly fromFinalRT: MeshBasicNodeMaterial;
}

interface Pass2DeltaBundle {
  readonly K1: MeshBasicNodeMaterial;
  readonly K2or4: MeshBasicNodeMaterial;
  readonly K3: MeshBasicNodeMaterial;
  readonly K5: MeshBasicNodeMaterial;
}

export interface DebugRendererOptions {
  readonly renderer: WebGPURenderer;
  readonly scene: Scene;
  readonly camera: Camera;
  readonly fluidSystem: FluidSystem;
  readonly fluidScene: Scene;
  readonly depthMesh: import('three').InstancedMesh;
  readonly depthMaterial: MeshBasicNodeMaterial;
  readonly smoothingPass: SmoothingPass;
  readonly thicknessScene: Scene;
  readonly pass1RT: RenderTarget;
  readonly smoothA: RenderTarget;
  readonly smoothB: RenderTarget;
  readonly finalRT: RenderTarget;
  readonly thicknessRT: RenderTarget;
  readonly radiusUniform: ReturnType<typeof uniform<'float', number>>;
  readonly depthScaleUniform: ReturnType<typeof uniform<'float', number>>;
  readonly derivativeScaleUniform: ReturnType<typeof uniform<'float', number>>;
  readonly filterSigmaUniform: ReturnType<typeof uniform<'float', number>>;
  readonly filterDeltaUniform: ReturnType<typeof uniform<'float', number>>;
  readonly filterMuUniform: ReturnType<typeof uniform<'float', number>>;
  readonly farSentinel: number;
  readonly params: FluidSurfaceParams;
  /**
   * Phase 14c — anisotropy compute kernel reference, dispatched
   * unconditionally before any `aniso.*` debug view rasterises (so the
   * diagnostic buffer is fresh regardless of `params.anisotropy.enabled`).
   */
  readonly anisotropyKernel: ComputeNode;
  /**
   * Tracks `params.anisotropy.nEpsilon.value` so the debug overlays
   * (the heatmap N_ε ring + the green/red interior split) update
   * live as the artist drags the slider.
   */
  readonly anisotropyNEpsilonUniform: ReturnType<typeof uniform<'float', number>>;
}

/**
 * Orchestrates per-pass debug rendering. When the artist switches
 * `params.debug.view.value` away from `'off'`, the consuming spec
 * routes its `customRender` hook here; the production PBR mesh is
 * hidden in parallel.
 *
 * DebugRenderer takes responsibility for:
 *   - Rendering the user's scene to canvas first so debug overlays
 *     that alpha-discard sentinel pixels (e.g. surface.normal) show
 *     the scene through the gaps.
 *   - Re-running production pre-passes (depth, NRF, thickness) up to
 *     whatever K the requested view inspects.
 *   - Substituting per-mode debug materials onto a full-screen quad
 *     and rendering to canvas.
 *
 * Production rendering does NOT route through here; that's the PBR
 * mesh in `surface/PhysicalSurface.ts` consumed via the harness's
 * standard scene render.
 */
export class DebugRenderer {
  private readonly opts: DebugRendererOptions;
  /**
   * Mutable mirror of `opts.depthMaterial`. The renderer rebuilds the
   * production depth material on `params.anisotropy.enabled` flip and
   * notifies us via {@link swapDepthMaterial}; we read this field
   * (instead of `opts.depthMaterial`) when restoring the production
   * material after a debug-view render.
   */
  private currentDepthMaterial: MeshBasicNodeMaterial;

  private readonly pass1Debug: {
    readonly coverage: MeshBasicNodeMaterial;
    readonly diskMask: MeshBasicNodeMaterial;
    readonly depthGray: MeshBasicNodeMaterial;
    readonly depthHeat: MeshBasicNodeMaterial;
  };

  private readonly pass2DebugMesh: Mesh;
  private readonly pass2DebugScene: Scene;
  private readonly pass2Debug: {
    readonly depthGray: Pass2DepthGrayBundle;
    readonly clampActivation: Pass2DebugBundle;
    readonly rangeMask: Pass2DebugBundle;
    readonly kernelSize: Pass2DebugBundle;
    readonly biasDiscardMask: Pass2DebugBundle;
    readonly delta: Pass2DeltaBundle;
  };

  private readonly surfaceDebug: Record<SurfaceDebugMode, MeshBasicNodeMaterial>;
  private readonly surfaceDebugMesh: Mesh;
  private readonly surfaceDebugScene: Scene;

  /**
   * Per-mode materials for the four `aniso.*` debug views. Built once;
   * swapped onto `opts.depthMesh` for the duration of an aniso render
   * (similar to how `pass1Direct` swaps in pass1Views).
   */
  private readonly anisoDebug: Record<AnisotropyDebugMode, MeshBasicNodeMaterial>;

  private readonly depthBlitMaterial: MeshBasicNodeMaterial;
  private readonly depthBlitMesh: Mesh;
  private readonly depthBlitScene: Scene;

  private readonly clearColorScratch = new Color();
  private readonly sentinelColor = new Color();
  private readonly sizeScratch = new Vector2();

  constructor(opts: DebugRendererOptions) {
    this.opts = opts;
    this.currentDepthMaterial = opts.depthMaterial;

    const pass1Args = {
      fluidSystem: opts.fluidSystem,
      radiusUniform: opts.radiusUniform,
      depthScaleUniform: opts.depthScaleUniform,
    } as const;
    this.pass1Debug = {
      coverage: buildPass1DebugMaterial({ ...pass1Args, mode: 'coverage' }),
      diskMask: buildPass1DebugMaterial({ ...pass1Args, mode: 'diskMask' }),
      depthGray: buildPass1DebugMaterial({ ...pass1Args, mode: 'depthGray' }),
      depthHeat: buildPass1DebugMaterial({ ...pass1Args, mode: 'depthHeat' }),
    };

    const buildBundle = (mode: Pass2DebugMode): Pass2DebugBundle => ({
      fromPass1RT: buildPass2DebugMaterial({
        mode,
        inputTexture: opts.pass1RT.texture,
        depthScaleUniform: opts.depthScaleUniform,
        derivativeScaleUniform: opts.derivativeScaleUniform,
        filterSigmaUniform: opts.filterSigmaUniform,
        filterDeltaUniform: opts.filterDeltaUniform,
        filterMuUniform: opts.filterMuUniform,
      }),
      fromSmoothA: buildPass2DebugMaterial({
        mode,
        inputTexture: opts.smoothA.texture,
        depthScaleUniform: opts.depthScaleUniform,
        derivativeScaleUniform: opts.derivativeScaleUniform,
        filterSigmaUniform: opts.filterSigmaUniform,
        filterDeltaUniform: opts.filterDeltaUniform,
        filterMuUniform: opts.filterMuUniform,
      }),
      fromSmoothB: buildPass2DebugMaterial({
        mode,
        inputTexture: opts.smoothB.texture,
        depthScaleUniform: opts.depthScaleUniform,
        derivativeScaleUniform: opts.derivativeScaleUniform,
        filterSigmaUniform: opts.filterSigmaUniform,
        filterDeltaUniform: opts.filterDeltaUniform,
        filterMuUniform: opts.filterMuUniform,
      }),
    });
    this.pass2Debug = {
      depthGray: {
        ...buildBundle('depthGray'),
        fromFinalRT: buildPass2DebugMaterial({
          mode: 'depthGray',
          inputTexture: opts.finalRT.texture,
          depthScaleUniform: opts.depthScaleUniform,
          derivativeScaleUniform: opts.derivativeScaleUniform,
          filterSigmaUniform: opts.filterSigmaUniform,
          filterDeltaUniform: opts.filterDeltaUniform,
          filterMuUniform: opts.filterMuUniform,
        }),
      },
      clampActivation: buildBundle('clampActivation'),
      rangeMask: buildBundle('rangeMask'),
      kernelSize: buildBundle('kernelSize'),
      biasDiscardMask: buildBundle('biasDiscardMask'),
      delta: {
        K1: buildPass2DebugMaterial({
          mode: 'deltaPerIter',
          inputTexture: opts.smoothA.texture,
          prevTexture: opts.pass1RT.texture,
          depthScaleUniform: opts.depthScaleUniform,
          derivativeScaleUniform: opts.derivativeScaleUniform,
          filterSigmaUniform: opts.filterSigmaUniform,
          filterDeltaUniform: opts.filterDeltaUniform,
          filterMuUniform: opts.filterMuUniform,
        }),
        K2or4: buildPass2DebugMaterial({
          mode: 'deltaPerIter',
          inputTexture: opts.smoothB.texture,
          prevTexture: opts.smoothA.texture,
          depthScaleUniform: opts.depthScaleUniform,
          derivativeScaleUniform: opts.derivativeScaleUniform,
          filterSigmaUniform: opts.filterSigmaUniform,
          filterDeltaUniform: opts.filterDeltaUniform,
          filterMuUniform: opts.filterMuUniform,
        }),
        K3: buildPass2DebugMaterial({
          mode: 'deltaPerIter',
          inputTexture: opts.smoothA.texture,
          prevTexture: opts.smoothB.texture,
          depthScaleUniform: opts.depthScaleUniform,
          derivativeScaleUniform: opts.derivativeScaleUniform,
          filterSigmaUniform: opts.filterSigmaUniform,
          filterDeltaUniform: opts.filterDeltaUniform,
          filterMuUniform: opts.filterMuUniform,
        }),
        K5: buildPass2DebugMaterial({
          mode: 'deltaPerIter',
          inputTexture: opts.finalRT.texture,
          prevTexture: opts.smoothB.texture,
          depthScaleUniform: opts.depthScaleUniform,
          derivativeScaleUniform: opts.derivativeScaleUniform,
          filterSigmaUniform: opts.filterSigmaUniform,
          filterDeltaUniform: opts.filterDeltaUniform,
          filterMuUniform: opts.filterMuUniform,
        }),
      },
    };

    const dbg2 = buildPass2DebugScene(this.pass2Debug.depthGray.fromPass1RT);
    this.pass2DebugMesh = dbg2.mesh;
    this.pass2DebugScene = dbg2.scene;

    const surfaceArgs = {
      smoothedDepthTexture: opts.finalRT.texture,
      depthScaleUniform: opts.depthScaleUniform,
      derivativeScaleUniform: opts.derivativeScaleUniform,
    } as const;
    this.surfaceDebug = {
      viewPos: buildSurfaceDebugMaterial({ ...surfaceArgs, mode: 'viewPos' }),
      normal: buildSurfaceDebugMaterial({ ...surfaceArgs, mode: 'normal' }),
      ddxViewPos: buildSurfaceDebugMaterial({ ...surfaceArgs, mode: 'ddxViewPos' }),
      ddyViewPos: buildSurfaceDebugMaterial({ ...surfaceArgs, mode: 'ddyViewPos' }),
    };
    const surf = buildSurfaceDebugScene(this.surfaceDebug.viewPos);
    this.surfaceDebugMesh = surf.mesh;
    this.surfaceDebugScene = surf.scene;

    // Phase 14c — per-particle anisotropy diagnostic imposters. Reuse
    // the production depth-pass mesh (1 quad per fluid particle); swap
    // its material at render time to one of these four.
    const anisoArgs = {
      fluidSystem: opts.fluidSystem,
      radiusUniform: opts.radiusUniform,
      nEpsilonUniform: opts.anisotropyNEpsilonUniform,
    } as const;
    this.anisoDebug = {
      neighborCount: buildAnisotropyDebugMaterial({ ...anisoArgs, mode: 'neighborCount' }),
      interior: buildAnisotropyDebugMaterial({ ...anisoArgs, mode: 'interior' }),
      sigma1: buildAnisotropyDebugMaterial({ ...anisoArgs, mode: 'sigma1' }),
      aniso: buildAnisotropyDebugMaterial({ ...anisoArgs, mode: 'aniso' }),
    };

    this.depthBlitMaterial = buildDepthBlitMaterial({
      inputTexture: opts.pass1RT.texture,
      depthScaleUniform: opts.depthScaleUniform,
    });
    const blit = buildDepthBlitScene(this.depthBlitMaterial);
    this.depthBlitMesh = blit.mesh;
    this.depthBlitScene = blit.scene;
  }

  /** True when the requested view is anything other than `'off'`. */
  isActive(): boolean {
    return this.opts.params.debug.view.value !== 'off';
  }

  /**
   * Render the active debug view to the canvas. Caller is responsible
   * for hiding the production PBR surface mesh while this runs (the
   * harness's customRender hook replaces the standard scene render so
   * the PBR mesh wouldn't get drawn anyway, but the scene capture into
   * sceneBehindRT could pick it up if it's still visible).
   */
  render(): void {
    const view = this.opts.params.debug.view.value;
    if (view === 'off') return;

    const r = this.opts.renderer;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const savedClearColor = (r as any).getClearColor(this.clearColorScratch).clone();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const savedClearAlpha = (r as any).getClearAlpha();
    const savedAutoClear = r.autoClear;

    // Step 1 — render the user's scene to canvas first so debug
    // overlays that alpha-discard sentinel pixels (e.g.
    // surface.normal) reveal the scene through the gaps.
    this.bindCanvasTarget();
    r.autoClear = true;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (r as any).setClearColor(savedClearColor, savedClearAlpha);
    r.render(this.opts.scene, this.opts.camera);

    if (view in PASS1_DIRECT_MODES) {
      this.renderPass1Direct(
        PASS1_DIRECT_MODES[view] as 'coverage' | 'diskMask' | 'depthGray' | 'depthHeat',
      );
    } else if (view === 'pass1.rtResOverlay') {
      this.renderRtResOverlay();
    } else if (view in PASS2_DEBUG_MODES) {
      this.renderPass2Debug(view);
    } else if (view in SURFACE_DEBUG_MODES) {
      this.renderSurfaceDebug(SURFACE_DEBUG_MODES[view] as SurfaceDebugMode);
    } else if (view in ANISO_DEBUG_MODES) {
      this.renderAnisotropyDebug(ANISO_DEBUG_MODES[view] as AnisotropyDebugMode);
    }

    r.autoClear = savedAutoClear;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (r as any).setClearColor(savedClearColor, savedClearAlpha);
  }

  private bindCanvasTarget(): void {
    const r = this.opts.renderer;
    r.setRenderTarget(null);
    r.getSize(this.sizeScratch);
    r.setViewport(0, 0, this.sizeScratch.x, this.sizeScratch.y);
    r.setScissor(0, 0, this.sizeScratch.x, this.sizeScratch.y);
  }

  private renderPass1ToPass1RT(): void {
    const r = this.opts.renderer;
    this.sentinelColor.setRGB(this.opts.farSentinel, 0, 0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (r as any).setClearColor(this.sentinelColor, 1.0);
    r.setRenderTarget(this.opts.pass1RT);
    r.autoClear = true;
    this.opts.depthMesh.material = this.currentDepthMaterial;
    r.render(this.opts.fluidScene, this.opts.camera);
  }

  private renderThicknessToThicknessRT(): void {
    const r = this.opts.renderer;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (r as any).setClearColor(0x000000, 0);
    r.setRenderTarget(this.opts.thicknessRT);
    r.autoClear = true;
    r.render(this.opts.thicknessScene, this.opts.camera);
  }

  private renderPass1Direct(mode: 'coverage' | 'diskMask' | 'depthGray' | 'depthHeat'): void {
    const r = this.opts.renderer;
    this.opts.depthMesh.material = this.pass1Debug[mode];
    this.bindCanvasTarget();
    r.autoClear = false;
    r.render(this.opts.fluidScene, this.opts.camera);
    // Restore production material.
    this.opts.depthMesh.material = this.currentDepthMaterial;
  }

  /**
   * Phase 14c — anisotropy diagnostic render. Dispatches the
   * anisotropy compute kernel up-front (regardless of
   * `params.anisotropy.enabled` — debug views need fresh diagnostic
   * data even when production rendering uses sphere imposters), then
   * rasterises per-particle imposters with the requested colormap.
   */
  private renderAnisotropyDebug(mode: AnisotropyDebugMode): void {
    const r = this.opts.renderer;
    // Kernel dispatch — async fire-and-forget; the render below
    // submits to the same queue and sees its results in order.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    void (r as any).computeAsync([this.opts.anisotropyKernel]);

    this.opts.depthMesh.material = this.anisoDebug[mode];
    this.bindCanvasTarget();
    r.autoClear = false;
    r.render(this.opts.fluidScene, this.opts.camera);
    this.opts.depthMesh.material = this.currentDepthMaterial;
  }

  private renderRtResOverlay(): void {
    const r = this.opts.renderer;
    this.renderPass1ToPass1RT();
    this.bindCanvasTarget();
    r.autoClear = false;
    r.render(this.depthBlitScene, this.opts.camera);
  }

  private renderPass2Debug(view: DebugView): void {
    const r = this.opts.renderer;
    const K = Math.max(0, Math.min(this.opts.params.debug.inspectIter.value, NRF_TOTAL_DISPATCHES));

    this.renderPass1ToPass1RT();
    const Keffective = view === 'pass2.iter0' ? 0 : K;
    this.opts.smoothingPass.run(r, Keffective, this.opts.camera);

    const dbgMat = this.selectPass2DebugMaterial(view, Keffective);
    if (dbgMat !== null) {
      this.pass2DebugMesh.material = dbgMat;
      this.bindCanvasTarget();
      r.autoClear = false;
      r.render(this.pass2DebugScene, this.opts.camera);
    }
  }

  private renderSurfaceDebug(mode: SurfaceDebugMode): void {
    const r = this.opts.renderer;
    this.renderPass1ToPass1RT();
    this.opts.smoothingPass.run(r, NRF_TOTAL_DISPATCHES, this.opts.camera);
    this.renderThicknessToThicknessRT();

    this.surfaceDebugMesh.material = this.surfaceDebug[mode];
    this.bindCanvasTarget();
    r.autoClear = false;
    r.render(this.surfaceDebugScene, this.opts.camera);
  }

  private selectPass2DebugMaterial(view: DebugView, K: number): MeshBasicNodeMaterial | null {
    const mode = PASS2_DEBUG_MODES[view];
    if (mode === undefined) return null;

    if (view === 'pass2.iter0') {
      return this.pass2Debug.depthGray.fromPass1RT;
    }

    if (mode === 'deltaPerIter') {
      if (K === 1) return this.pass2Debug.delta.K1;
      if (K === 2 || K === 4) return this.pass2Debug.delta.K2or4;
      if (K === 3) return this.pass2Debug.delta.K3;
      if (K === 5) return this.pass2Debug.delta.K5;
      return null;
    }

    if (mode === 'depthGray') {
      if (K === 0) return this.pass2Debug.depthGray.fromPass1RT;
      if (K === 1 || K === 3) return this.pass2Debug.depthGray.fromSmoothA;
      if (K === 2 || K === 4) return this.pass2Debug.depthGray.fromSmoothB;
      return this.pass2Debug.depthGray.fromFinalRT;
    }

    const bundle =
      mode === 'clampActivation'
        ? this.pass2Debug.clampActivation
        : mode === 'rangeMask'
          ? this.pass2Debug.rangeMask
          : mode === 'kernelSize'
            ? this.pass2Debug.kernelSize
            : this.pass2Debug.biasDiscardMask;

    if (K === 0 || K === 1) return bundle.fromPass1RT;
    if (K === 2 || K === 4) return bundle.fromSmoothA;
    return bundle.fromSmoothB;
  }

  /**
   * Phase 14c — notify the debug renderer that the production
   * depth-pass material has been rebuilt (typically because
   * `params.anisotropy.enabled` flipped). The renderer owns disposal
   * of the OLD material; we just update our restore-after-debug-view
   * reference.
   */
  swapDepthMaterial(material: MeshBasicNodeMaterial): void {
    this.currentDepthMaterial = material;
  }

  dispose(): void {
    for (const mat of Object.values(this.pass1Debug)) {
      mat.dispose();
    }
    this.pass2DebugMesh.geometry.dispose();
    for (const bundle of [
      this.pass2Debug.clampActivation,
      this.pass2Debug.rangeMask,
      this.pass2Debug.kernelSize,
      this.pass2Debug.biasDiscardMask,
    ]) {
      bundle.fromPass1RT.dispose();
      bundle.fromSmoothA.dispose();
      bundle.fromSmoothB.dispose();
    }
    this.pass2Debug.depthGray.fromPass1RT.dispose();
    this.pass2Debug.depthGray.fromSmoothA.dispose();
    this.pass2Debug.depthGray.fromSmoothB.dispose();
    this.pass2Debug.depthGray.fromFinalRT.dispose();
    this.pass2Debug.delta.K1.dispose();
    this.pass2Debug.delta.K2or4.dispose();
    this.pass2Debug.delta.K3.dispose();
    this.pass2Debug.delta.K5.dispose();

    this.surfaceDebugMesh.geometry.dispose();
    for (const mat of Object.values(this.surfaceDebug)) {
      mat.dispose();
    }
    this.depthBlitMesh.geometry.dispose();
    this.depthBlitMaterial.dispose();
  }
}
