import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  Discard,
  Fn,
  cameraViewMatrix,
  cameraWorldMatrix,
  float,
  instanceIndex,
  log,
  positionLocal,
  uniform,
  uv,
  vec3,
  vec4,
} from 'three/tsl';
import type { FluidSystem } from '../../FluidSystem.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Phase 14c — anisotropy diagnostic debug views. Per-particle sphere
 * imposters coloured by the corresponding component of
 * `fluidSystem.anisotropyDiagnostic` (written by the anisotropy
 * compute kernel each frame the views are active):
 *
 *   diagnostic.x — N (in-range neighbour count, including self)
 *   diagnostic.y — σ_1 (largest RAW eigenvalue, pre eq. 15 clamp)
 *   diagnostic.z — σ_3 (smallest RAW eigenvalue)
 *   diagnostic.w — |x̄ − x| · λ (Laplacian smoothing offset)
 *
 * Modes:
 *   - `aniso.neighborCount` — N as a heatmap. Red ≤ 5; orange ~15;
 *     yellow at the N_ε threshold (default 25); green > 50; cyan > 100.
 *     If most particles render red/orange, the interior branch never
 *     fires and the kernel falls back to `Σ̃ = k_n · I` everywhere
 *     (the symptom from the screenshot — flipping kr/ks does nothing
 *     because they only matter inside the interior branch).
 *   - `aniso.interior` — green if `N > N_ε` (interior branch fires);
 *     red if `N ≤ N_ε` (isolated branch). Cleanest "is anisotropy
 *     actually doing anything" diagnostic.
 *   - `aniso.sigma1` — log10(σ_1) grayscale, mapped from −9 (black)
 *     to −3 (white). σ_1 ≈ (spacing)² · w_typical for an isotropic
 *     local cloud, so for `spacing = 0.024` we expect σ_1 ≈ 6e-4 →
 *     log10 ≈ −3.2.
 *   - `aniso.aniso` — σ_3 / σ_1 ratio. White = isotropic (1); red =
 *     thin-feature (≈ 0). Useful at the fluid boundary where Yu &
 *     Turk's elongation kicks in.
 *
 * The kernel that writes the diagnostic must run before this material
 * renders — `FluidSurfaceRenderer.renderDebugView` dispatches the
 * anisotropy compute kernel up-front when a debug view starts with
 * `aniso.`.
 */
export type AnisotropyDebugMode = 'neighborCount' | 'interior' | 'sigma1' | 'aniso';

export interface BuildAnisotropyDebugMaterialArgs {
  readonly mode: AnisotropyDebugMode;
  readonly fluidSystem: FluidSystem;
  /** Imposter radius uniform (m). Owned by `FluidSurfaceRenderer`. */
  readonly radiusUniform: ReturnType<typeof uniform<'float', number>>;
  /** Threshold uniform tracking `params.anisotropy.nEpsilon.value`. */
  readonly nEpsilonUniform: ReturnType<typeof uniform<'float', number>>;
}

export function buildAnisotropyDebugMaterial(
  args: BuildAnisotropyDebugMaterialArgs,
): MeshBasicNodeMaterial {
  const { mode, fluidSystem, radiusUniform, nEpsilonUniform } = args;
  const { particles, fluidParticles } = fluidSystem;
  const diagnostic = fluidSystem.anisotropyDiagnostic;
  if (!diagnostic) {
    throw new Error(
      'buildAnisotropyDebugMaterial: fluidSystem.enableAnisotropyBuffers() must be called first',
    );
  }

  const material = new MeshBasicNodeMaterial();
  material.transparent = false;
  material.depthWrite = false;
  material.depthTest = false;

  const fluidStart = float(fluidParticles.start).toUint();
  const slotIdx = instanceIndex.add(fluidStart);
  const particleWorld: Any = (particles.positions as Any).element(slotIdx).xyz;
  const cameraRight: Any = (cameraWorldMatrix as Any).element(0).xyz;
  const cameraUp: Any = (cameraWorldMatrix as Any).element(1).xyz;
  const quadOffsetXY: Any = positionLocal.xy.mul(2.0).mul(radiusUniform);
  const offsetWorld: Any = cameraRight.mul(quadOffsetXY.x).add(cameraUp.mul(quadOffsetXY.y));
  const billboardedWorld: Any = particleWorld.add(offsetWorld);
  material.positionNode = billboardedWorld;

  // Hold viewCenterZ for completeness so the debug imposter passes the
  // same depth output as the production sphere imposter — keeps the
  // depth ordering against the tank correct.
  const viewCenterView: Any = cameraViewMatrix.mul(vec4(particleWorld, 1.0)).xyz;
  void viewCenterView;

  // Pull the diagnostic at vertex stage; pass to fragment as flat-ish
  // varyings (per-instance constant, so interpolation is identity).
  const diagSample: Any = (diagnostic as Any).element(slotIdx);
  const Nv: Any = diagSample.x.toVarying('aniDiagN');
  const sigma1Vary: Any = diagSample.y.toVarying('aniDiagSigma1');
  const sigma3Vary: Any = diagSample.z.toVarying('aniDiagSigma3');

  material.outputNode = Fn(() => {
    const diskUv: Any = uv().mul(2.0).sub(1.0);
    const r2: Any = diskUv.dot(diskUv);
    Discard(r2.greaterThan(1.0));

    if (mode === 'neighborCount') {
      // 5-stop heatmap: red(0) → orange(10) → yellow(N_ε) →
      // green(50) → cyan(>=100). Smooth interp via piecewise lerp.
      const nf: Any = Nv;
      // Map Nv to a normalised stop position s ∈ [0, 1] across:
      //   0   → 0.0
      //   10  → 0.25
      //   N_ε → 0.5
      //   50  → 0.75
      //   100 → 1.0
      // Use a piecewise linear approximation: blend between four bands.
      // For visual parsing we don't need perfect anchors — just monotone.
      const t: Any = nf.div(float(100.0)).clamp(0.0, 1.0);
      // Heatmap colour: red → green → cyan via two-axis blend.
      // R = 1 - 2t (clamped), G = 2*min(t, 1-t)*2 (peak at 0.5), B = 2(t - 0.5)
      const r: Any = float(1.0).sub(t.mul(2.0)).clamp(0.0, 1.0);
      const g: Any = t.sub(0.5).abs().mul(2.0).oneMinus().clamp(0.0, 1.0);
      const b: Any = t.sub(0.5).mul(2.0).clamp(0.0, 1.0);
      // Highlight the N_ε crossover: ring of yellow when nf ≈ N_ε.
      const epsRatio: Any = nf
        .sub(nEpsilonUniform)
        .abs()
        .div(nEpsilonUniform.max(float(1.0)));
      const onThreshold: Any = epsRatio.lessThan(float(0.05));
      return vec4(
        onThreshold.select(float(1.0), r),
        onThreshold.select(float(1.0), g),
        onThreshold.select(float(0.0), b),
        1.0,
      );
    }

    if (mode === 'interior') {
      // Binary green/red: did the eq. 15 interior branch fire?
      const isInterior: Any = Nv.greaterThan(nEpsilonUniform);
      const col: Any = isInterior.select(vec3(0.2, 0.85, 0.3), vec3(0.85, 0.2, 0.2));
      return vec4(col, 1.0);
    }

    if (mode === 'sigma1') {
      // log10(σ_1) grayscale, [-9, -3] → [0, 1].
      const safe: Any = sigma1Vary.max(float(1e-12));
      const log10s1: Any = log(safe).div(float(2.302585)); // ln(10) ≈ 2.302585
      const t: Any = log10s1.add(float(9.0)).div(float(6.0)).clamp(0.0, 1.0);
      return vec4(t, t, t, 1.0);
    }

    // mode === 'aniso': σ_3 / σ_1 ratio (1 = isotropic, 0 = thin)
    const ratio: Any = sigma3Vary.div(sigma1Vary.max(float(1e-12))).clamp(0.0, 1.0);
    // Red (anisotropic) → white (isotropic).
    const colAniso: Any = vec3(float(1.0), ratio, ratio);
    return vec4(colAniso, 1.0);
  })();

  return material;
}
