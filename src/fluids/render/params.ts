import { Color } from 'three';

/**
 * Live-tunable fluid renderer parameter handles.
 *
 * Every parameter on `FluidSurfaceRenderer` is exposed as a typed handle
 * carrying its current `value`, a UI label, a tooltip description, and
 * range/default metadata. The grouped shape (`depth`, `smoothing`,
 * `thickness`, `surface`, `debug`) lets a parameter panel render each group
 * as its own folder and lets future settings-export / preset-load
 * features introspect the schema without a parallel metadata file.
 *
 * Phase 14c adds the `anisotropy` group and extends `surface.mode`
 * to include `'analytic'`. Phase 14b ships PBR-only.
 */

/** A live-tunable scalar parameter. UI sliders bind to `.value`. */
export interface NumberHandle {
  /** Mutable: the live value. */
  value: number;
  readonly label: string;
  /** Tooltip shown on hover. Cite paper section / equation. */
  readonly description: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly default: number;
}

/** A live-tunable boolean parameter. */
export interface BooleanHandle {
  value: boolean;
  readonly label: string;
  readonly description: string;
  readonly default: boolean;
}

/** A live-tunable enum parameter. `T` is the literal-string union. */
export interface SelectHandle<T extends string> {
  value: T;
  readonly label: string;
  readonly description: string;
  readonly options: readonly { readonly label: string; readonly value: T }[];
  readonly default: T;
}

/**
 * A live-tunable colour parameter. `value` is a three.js `Color` so the
 * caller can `set('#aabbcc')` / `setRGB(...)` without us re-broadcasting
 * a numeric channel.
 */
export interface ColorHandle {
  /** Mutable: mutate the colour itself, e.g. `value.set('#aabbcc')`. */
  readonly value: Color;
  readonly label: string;
  readonly description: string;
  /** Default as 0xRRGGBB hex. */
  readonly default: number;
}

/**
 * Pass-2 (NRF) target resolution. Lower res = cheaper smoothing +
 * implicit gap-bridging (multiple particles per smoothed pixel). Truong
 * §4 + van der Laan §3.5.1 measure quarter as the production setting.
 */
export type SmoothingResolution = 'full' | 'half' | 'quarter';

/**
 * Per-pass diagnostic visualisations. `'off'` is the production path;
 * any other value short-circuits the renderer to a single debug view.
 */
export type DebugView =
  | 'off'
  | 'pass1.coverage'
  | 'pass1.diskMask'
  | 'pass1.depthGray'
  | 'pass1.depthHeat'
  | 'pass1.rtResOverlay'
  | 'pass2.iter0'
  | 'pass2.iterK'
  | 'pass2.clampActivation'
  | 'pass2.rangeMask'
  | 'pass2.kernelSize'
  | 'pass2.biasDiscardMask'
  | 'pass2.deltaPerIter'
  | 'surface.viewPos'
  | 'surface.normal'
  | 'surface.ddx'
  | 'surface.ddy'
  | 'aniso.neighborCount'
  | 'aniso.interior'
  | 'aniso.sigma1'
  | 'aniso.aniso';

export interface DepthParams {
  /**
   * Sphere-imposter radius for Pass 1, expressed as a multiple of the
   * scene-global particle radius. 1.0 = paper default; >1.0 inflates
   * each imposter so adjacent particles overlap before NRF runs (helps
   * sparse-density fluids). Rendering-only.
   */
  readonly imposterRadius: NumberHandle;
}

/**
 * Phase 14c — Yu & Turk 2010 anisotropy compute pass parameters. Drives
 * the per-particle covariance / eigendecomposition / G_i^{-1} pipeline
 * that feeds the depth pass's ellipsoid-imposter path. Disabled by
 * default (the sphere-imposter path remains the production fallback).
 */
export interface AnisotropyParams {
  /**
   * Enables the Yu & Turk anisotropic kernel + ellipsoid-imposter
   * depth path. Toggling rebuilds the depth-pass material
   * (sphere ↔ ellipsoid graphs differ) — same lifecycle as
   * `surface.mode`. Rendering-only.
   */
  readonly enabled: BooleanHandle;
  /**
   * Yu & Turk eq. 15 `k_r` — eigenvalue ratio cap. `σ̃_k = max(σ_k,
   * σ_1/k_r)` for k = 2, 3 prevents extreme stretches at thin features.
   * Paper default 4. Higher = more permissive (more elongated
   * imposters); lower = closer to spherical.
   */
  readonly kr: NumberHandle;
  /**
   * Yu & Turk eq. 15 `k_s` — covariance scale factor. Paper hardcodes
   * 1400 for their h; ours is auto-derived to satisfy `||k_s · C|| ≈ 1`
   * for interior particles. UI exposes this for tuning; the renderer
   * runs a one-shot calibration on the first frame and writes the
   * derived value back into this handle. See U-FR-4.
   */
  readonly ks: NumberHandle;
  /**
   * Yu & Turk eq. 15 `k_n` — isolated-particle radius. Particles with
   * fewer than `nEpsilon` neighbours within `r_i = 2·h` use
   * `Σ̃ = k_n · I` (isotropic, small). Paper default 0.5.
   */
  readonly kn: NumberHandle;
  /**
   * Yu & Turk eq. 15 `N_ε` — minimum-neighbour threshold for the
   * interior branch. Below this the kernel falls back to `k_n · I`.
   * Paper default 25.
   */
  readonly nEpsilon: NumberHandle;
  /**
   * Yu & Turk eq. 6 `λ` — Laplacian centre-smoothing strength.
   * `x̄_i = (1−λ)·x_i + λ·x_i^w`. Paper recommends `λ ∈ [0.9, 1.0]`;
   * default 0.95 is the midpoint.
   */
  readonly lambda: NumberHandle;
}

export interface SmoothingParams {
  /**
   * Truong & Yuksel 2018 §3.1 world-space Gaussian σ as a multiple of
   * particle radius. Default 0.7 per Truong §4.
   */
  readonly sigma: NumberHandle;
  /**
   * Truong §3.2 range threshold δ as a multiple of particle radius.
   * Default 10 per Truong §4. Smaller δ ⇒ sharper silhouette.
   */
  readonly delta: NumberHandle;
  /**
   * Truong §3.2 clamp magnitude μ as a multiple of particle radius.
   * Default 1 per Truong §4. Out-of-range neighbours contribute z'_i+μ
   * instead of being dropped — the gap-fill mechanism.
   */
  readonly mu: NumberHandle;
  /** Pass-2 RT resolution. Rebuilds RTs (not material). */
  readonly resolution: SelectHandle<SmoothingResolution>;
}

export interface ThicknessParams {
  /**
   * Pass-3 Gaussian splat radius as a multiple of particle radius.
   * Default 2 per Phase 14a. Larger ⇒ more accumulated thickness ⇒
   * darker fluid via Beer-Lambert.
   */
  readonly splatRadius: NumberHandle;
}

export interface SurfaceParams {
  /** Beer-Lambert absorption tint. Default deep ocean (`#0a2840`). */
  readonly color: ColorHandle;
  /**
   * Distance over which transmitted light decays by 1/e toward
   * `color`. Pool water ≈ 3 m; murky tea ≈ 0.3 m; glass ≈ 100 m.
   */
  readonly attenuationDistance: NumberHandle;
  /**
   * Multiplier mapping accumulated `thicknessRT` alpha to metres for
   * the PBR volume model. Default 0.01.
   */
  readonly thicknessScale: NumberHandle;
  /**
   * PBR microfacet roughness. 0 = mirror water surface; higher =
   * blurrier reflections.
   */
  readonly roughness: NumberHandle;
  /** Index of refraction. Water 1.33, glass 1.5, diamond 2.42. */
  readonly ior: NumberHandle;
  /**
   * Environment-map specular intensity multiplier. 1 = three.js
   * default. Higher boosts the IBL specular response.
   */
  readonly envIntensity: NumberHandle;
}

export interface DebugParams {
  /** Per-pass diagnostic. `'off'` = production rendering. */
  readonly view: SelectHandle<DebugView>;
  /**
   * For `pass2.*` views — index of the NRF dispatch to inspect, K∈[0,5].
   * 0 = raw pass1RT; 5 = final cleanup output.
   */
  readonly inspectIter: NumberHandle;
  /**
   * Depth-normalisation distance (m) for `pass1.depth*` /
   * `pass2.depthGray` / `surface.viewPos`.
   */
  readonly depthScale: NumberHandle;
  /**
   * Multiplier for the `pass2.deltaPerIter` diverging colormap. Tune
   * so per-iter changes are visible without saturating.
   */
  readonly derivativeScale: NumberHandle;
}

/**
 * The full live-tunable parameter handle exposed by
 * `FluidSurfaceRenderer`. Each group is a sub-record so consumers
 * (parameter panels and preset loaders) can iterate
 * groups uniformly.
 */
export interface FluidSurfaceParams {
  readonly depth: DepthParams;
  readonly anisotropy: AnisotropyParams;
  readonly smoothing: SmoothingParams;
  readonly thickness: ThicknessParams;
  readonly surface: SurfaceParams;
  readonly debug: DebugParams;
}

const DEBUG_VIEW_OPTIONS: readonly { readonly label: string; readonly value: DebugView }[] = [
  { label: 'off (production)', value: 'off' },
  { label: '#1 pass1.coverage', value: 'pass1.coverage' },
  { label: '#2 pass1.diskMask', value: 'pass1.diskMask' },
  { label: '#3 pass1.depthGray', value: 'pass1.depthGray' },
  { label: '#4 pass1.depthHeat', value: 'pass1.depthHeat' },
  { label: '#5 pass1.rtResOverlay', value: 'pass1.rtResOverlay' },
  { label: '#6 pass2.iter0', value: 'pass2.iter0' },
  { label: '#7 pass2.iterK', value: 'pass2.iterK' },
  { label: '#8 pass2.clampActivation', value: 'pass2.clampActivation' },
  { label: '#9 pass2.rangeMask', value: 'pass2.rangeMask' },
  { label: '#10 pass2.kernelSize', value: 'pass2.kernelSize' },
  { label: '#11 pass2.biasDiscardMask', value: 'pass2.biasDiscardMask' },
  { label: '#12 pass2.deltaPerIter', value: 'pass2.deltaPerIter' },
  { label: '#13 surface.viewPos', value: 'surface.viewPos' },
  { label: '#14 surface.normal', value: 'surface.normal' },
  { label: '#14a surface.ddx (×derivativeScale)', value: 'surface.ddx' },
  { label: '#14b surface.ddy (×derivativeScale)', value: 'surface.ddy' },
  { label: '#15 aniso.neighborCount (heatmap)', value: 'aniso.neighborCount' },
  { label: '#16 aniso.interior (green/red)', value: 'aniso.interior' },
  { label: '#17 aniso.sigma1 (log10)', value: 'aniso.sigma1' },
  { label: '#18 aniso.aniso (σ_3/σ_1)', value: 'aniso.aniso' },
];

const SMOOTHING_RES_OPTIONS: readonly {
  readonly label: string;
  readonly value: SmoothingResolution;
}[] = [
  { label: 'quarter', value: 'quarter' },
  { label: 'half', value: 'half' },
  { label: 'full', value: 'full' },
];

/** Initial values for each handle. Public so consumers can pass overrides. */
export interface FluidSurfaceParamDefaults {
  readonly imposterRadiusMul?: number;
  readonly anisotropyEnabled?: boolean;
  readonly anisotropyKr?: number;
  readonly anisotropyKs?: number;
  readonly anisotropyKn?: number;
  readonly anisotropyNEpsilon?: number;
  readonly anisotropyLambda?: number;
  readonly sigmaMul?: number;
  readonly deltaMul?: number;
  readonly muMul?: number;
  readonly smoothingResolution?: SmoothingResolution;
  readonly splatRadiusMul?: number;
  readonly fluidColor?: number;
  readonly attenuationDistance?: number;
  readonly thicknessScale?: number;
  readonly roughness?: number;
  readonly ior?: number;
  readonly envIntensity?: number;
  readonly debugView?: DebugView;
  readonly inspectIter?: number;
  readonly depthScale?: number;
  readonly derivativeScale?: number;
}

/**
 * Build a fresh `FluidSurfaceParams` instance with paper-grounded
 * defaults. Each handle's `.value` is independently mutable; the rest
 * (label, description, range) is read-only.
 */
export function buildDefaultParams(defaults: FluidSurfaceParamDefaults = {}): FluidSurfaceParams {
  const imposterRadius = defaults.imposterRadiusMul ?? 1.0;
  const anisotropyEnabled = defaults.anisotropyEnabled ?? false;
  const anisotropyKr = defaults.anisotropyKr ?? 4.0;
  const anisotropyKs = defaults.anisotropyKs ?? 1.0;
  const anisotropyKn = defaults.anisotropyKn ?? 0.5;
  const anisotropyNEpsilon = defaults.anisotropyNEpsilon ?? 25;
  const anisotropyLambda = defaults.anisotropyLambda ?? 0.95;
  const sigma = defaults.sigmaMul ?? 0.7;
  const delta = defaults.deltaMul ?? 10.0;
  const mu = defaults.muMul ?? 1.0;
  const smoothingResolution = defaults.smoothingResolution ?? 'quarter';
  const splatRadius = defaults.splatRadiusMul ?? 2.0;
  const fluidColorHex = defaults.fluidColor ?? 0x0a2840;
  const attenuationDistance = defaults.attenuationDistance ?? 3.0;
  const thicknessScale = defaults.thicknessScale ?? 0.01;
  const roughness = defaults.roughness ?? 0.0;
  const ior = defaults.ior ?? 1.33;
  const envIntensity = defaults.envIntensity ?? 1.0;
  const debugView = defaults.debugView ?? 'off';
  const inspectIter = defaults.inspectIter ?? 5;
  const depthScale = defaults.depthScale ?? 5.0;
  const derivativeScale = defaults.derivativeScale ?? 20.0;

  return {
    depth: {
      imposterRadius: {
        value: imposterRadius,
        label: 'Imposter Radius (×r)',
        description:
          'Pass 1 sphere-imposter radius as a multiple of particle radius r. >1 inflates the imposter so adjacent particles overlap before NRF runs — smooths sparse silhouettes at the cost of slightly larger fluid extent. Rendering-only.',
        min: 0.5,
        max: 3.0,
        step: 0.1,
        default: 1.0,
      },
    },
    anisotropy: {
      enabled: {
        value: anisotropyEnabled,
        label: 'Anisotropic Kernels',
        description:
          'Yu & Turk 2010 §4 — per-particle ellipsoid imposters with PCA-derived orientation and scale. Smooths sparse-particle silhouettes that the radius slider only patches. Rebuilds the depth-pass material on flip.',
        default: false,
      },
      kr: {
        value: anisotropyKr,
        label: 'Eigenvalue Ratio Cap (k_r)',
        description:
          'Yu & Turk eq. 15 — caps the worst-case stretch ratio between principal axes (σ̃_k = max(σ_k, σ_1/k_r) for k=2,3). Paper default 4. Higher = more elongated; lower = closer to spherical.',
        min: 1.0,
        max: 16.0,
        step: 0.5,
        default: 4.0,
      },
      ks: {
        value: anisotropyKs,
        label: 'Covariance Scale (k_s)',
        description:
          'Yu & Turk eq. 15 — covariance scaling factor. Has units of 1/m² so `k_s · C` is dimensionless. The renderer auto-calibrates the initial value to `1/h²` (~the right magnitude for `||k_s · C|| ≈ 1` over interior particles); the slider is for manual tuning around that. See U-FR-4.',
        min: 0.001,
        max: 1000000.0,
        step: 0.1,
        default: 1.0,
      },
      kn: {
        value: anisotropyKn,
        label: 'Isolated-Particle Radius (k_n)',
        description:
          'Yu & Turk eq. 15 — isotropic Σ̃ = k_n · I for particles with fewer than N_ε neighbours. Paper default 0.5.',
        min: 0.05,
        max: 5.0,
        step: 0.05,
        default: 0.5,
      },
      nEpsilon: {
        value: anisotropyNEpsilon,
        label: 'Min Neighbours (N_ε)',
        description:
          'Yu & Turk eq. 15 — minimum-neighbour threshold for the interior branch. Particles with N ≤ N_ε fall back to k_n · I. Paper default 25.',
        min: 1,
        max: 200,
        step: 1,
        default: 25,
      },
      lambda: {
        value: anisotropyLambda,
        label: 'Centre Smoothing (λ)',
        description:
          'Yu & Turk eq. 6 Laplacian centre smoothing strength: x̄_i = (1−λ)·x_i + λ·x_i^w. Paper recommends λ ∈ [0.9, 1.0]; default 0.95 is the midpoint.',
        min: 0.0,
        max: 1.0,
        step: 0.01,
        default: 0.95,
      },
    },
    smoothing: {
      sigma: {
        value: sigma,
        label: 'Filter σ (×r)',
        description:
          'Truong & Yuksel 2018 §3.1 world-space Gaussian σ. Default 0.7 per §4. Larger ⇒ wider screen-space kernel ⇒ better gap-bridging but more silhouette flattening.',
        min: 0.1,
        max: 5.0,
        step: 0.1,
        default: 0.7,
      },
      delta: {
        value: delta,
        label: 'Range Threshold δ (×r)',
        description:
          'Truong §3.2 narrow-range cutoff. Default 10 per §4. Neighbours within δ contribute as-is; beyond δ are clamped (eq. 2) or ignored (eq. 3). Smaller δ ⇒ sharper silhouette.',
        min: 0.5,
        max: 50.0,
        step: 0.5,
        default: 10.0,
      },
      mu: {
        value: mu,
        label: 'Clamp μ (×r)',
        description:
          "Truong §3.2 clamp magnitude. Default 1 per §4. Out-of-range neighbours contribute z'_i + μ instead of being dropped.",
        min: 0.1,
        max: 10.0,
        step: 0.1,
        default: 1.0,
      },
      resolution: {
        value: smoothingResolution,
        label: 'Smoothing Resolution',
        description:
          "RT resolution for the NRF passes. 'quarter' = paper's measured production setting; lower res naturally averages multiple particles per pixel. Rebuilds RTs.",
        options: SMOOTHING_RES_OPTIONS,
        default: 'quarter',
      },
    },
    thickness: {
      splatRadius: {
        value: splatRadius,
        label: 'Splat Radius (×r)',
        description:
          'Pass 3 Gaussian splat radius. Larger ⇒ more accumulated thickness ⇒ darker fluid via Beer-Lambert. Affects only surface tint, not silhouette.',
        min: 0.5,
        max: 8.0,
        step: 0.1,
        default: 2.0,
      },
    },
    surface: {
      color: {
        value: new Color(fluidColorHex),
        label: 'Absorption Tint',
        description:
          'Beer-Lambert absorption colour. Thicker fluid attenuates the refracted scene-behind toward this tint. Default deep ocean.',
        default: 0x0a2840,
      },
      attenuationDistance: {
        value: attenuationDistance,
        label: 'Attenuation Distance (m)',
        description:
          'PBR Beer-Lambert: distance over which transmitted light decays by 1/e toward the absorption tint. Pool water ≈ 3 m; murky tea ≈ 0.3 m; glass ≈ 100 m.',
        min: 0.05,
        max: 20.0,
        step: 0.05,
        default: 3.0,
      },
      thicknessScale: {
        value: thicknessScale,
        label: 'Thickness → Metres',
        description:
          'Multiplier mapping accumulated thickness texture (sum of Gaussian splats) to "metres" for the PBR volume model. Default 0.01.',
        min: 0.001,
        max: 0.2,
        step: 0.001,
        default: 0.01,
      },
      roughness: {
        value: roughness,
        label: 'Roughness',
        description: 'PBR microfacet roughness. 0 = mirror water; higher = blurrier reflections.',
        min: 0.0,
        max: 1.0,
        step: 0.01,
        default: 0.0,
      },
      ior: {
        value: ior,
        label: 'Index of Refraction',
        description: 'Water 1.33; ice 1.31; glass 1.5; diamond 2.42.',
        min: 1.0,
        max: 2.5,
        step: 0.01,
        default: 1.33,
      },
      envIntensity: {
        value: envIntensity,
        label: 'Env Reflection Intensity',
        description:
          'PBR env-map specular intensity multiplier. Pulls IBL highlights from `scene.environment`.',
        min: 0.0,
        max: 5.0,
        step: 0.05,
        default: 1.0,
      },
    },
    debug: {
      view: {
        value: debugView,
        label: 'Debug View',
        description:
          'Per-pass diagnostic. Off = production. Pass 1 (#1–#5) verifies sphere imposters; Pass 2 (#6–#12) inspects NRF dispatch K; Surface (#13–#19) inspects shading intermediates.',
        options: DEBUG_VIEW_OPTIONS,
        default: 'off',
      },
      inspectIter: {
        value: inspectIter,
        label: 'Inspect Iter K',
        description:
          'NRF dispatch index for Pass 2 debug views (#7–#12). 0 = raw pass1RT; 5 = final cleanup output.',
        min: 0,
        max: 5,
        step: 1,
        default: 5,
      },
      depthScale: {
        value: depthScale,
        label: 'Debug Depth Scale (m)',
        description:
          'Normalisation distance for depth-based debug views. viewSurfaceZ ÷ this scale → [0, 1].',
        min: 0.5,
        max: 30.0,
        step: 0.5,
        default: 5.0,
      },
      derivativeScale: {
        value: derivativeScale,
        label: 'Derivative Scale',
        description:
          'Multiplier for the #12 deltaPerIter diverging colormap. Tune so per-iter changes are visible without saturating.',
        min: 0.1,
        max: 500.0,
        step: 0.1,
        default: 20.0,
      },
    },
  };
}
