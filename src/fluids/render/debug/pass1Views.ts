import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  Discard,
  Fn,
  abs,
  cameraViewMatrix,
  cameraWorldMatrix,
  float,
  instanceIndex,
  positionLocal,
  uniform,
  uv,
  vec4,
} from 'three/tsl';
import type { FluidSystem } from '../../FluidSystem.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Pass-1 debug variants for views #1–#4. Each variant shares the depth
 * pass's vertex pipeline (instanced billboard quads, world-space
 * imposter centred on the particle) and changes only the fragment
 * output:
 *
 *   - `coverage` (#1): no Discard, solid white. Verifies imposter
 *     quads rasterise (instancing + billboard math).
 *   - `diskMask` (#2): Discard outside unit disk, solid white inside.
 *     Verifies the disk-discard predicate.
 *   - `depthGray` (#3): Discard outside disk, grayscale of
 *     `viewSurfaceZ / depthScale`. Verifies depth-output math.
 *   - `depthHeat` (#4): Discard outside disk, jet-style colormap (blue
 *     → green → red) of `viewSurfaceZ / depthScale`. Same data as
 *     `depthGray`, easier to read sub-cm depth differences.
 *
 * View #5 (`pass1.rtResOverlay`) is a full-screen blit of the
 * production `pass1RT` and lives in `debug/depthBlit.ts`.
 */
export type Pass1DebugMode = 'coverage' | 'diskMask' | 'depthGray' | 'depthHeat';

export interface BuildPass1DebugMaterialArgs {
  readonly mode: Pass1DebugMode;
  readonly fluidSystem: FluidSystem;
  readonly radiusUniform: ReturnType<typeof uniform<'float', number>>;
  readonly depthScaleUniform: ReturnType<typeof uniform<'float', number>>;
}

export function buildPass1DebugMaterial(args: BuildPass1DebugMaterialArgs): MeshBasicNodeMaterial {
  const { mode, fluidSystem, radiusUniform, depthScaleUniform } = args;
  const { particles, fluidParticles } = fluidSystem;

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

  const viewCenterView: Any = cameraViewMatrix.mul(vec4(particleWorld, 1.0)).xyz;
  const viewCenterZ: Any = viewCenterView.z.toVarying(`pass1Debug_${mode}_viewCenterZ`);

  material.outputNode = Fn(() => {
    const diskUv: Any = uv().mul(2.0).sub(1.0);
    const r2: Any = diskUv.dot(diskUv);

    if (mode === 'coverage') {
      return vec4(1.0, 1.0, 1.0, 1.0);
    }

    Discard(r2.greaterThan(1.0));

    if (mode === 'diskMask') {
      return vec4(1.0, 1.0, 1.0, 1.0);
    }

    const zOffset: Any = r2.oneMinus().sqrt().mul(radiusUniform);
    const viewSurfaceZ: Any = viewCenterZ.add(zOffset);
    const distance: Any = viewSurfaceZ.negate();
    const t: Any = distance.div(depthScaleUniform).clamp(0.0, 1.0);

    if (mode === 'depthGray') {
      return vec4(t, t, t, 1.0);
    }

    // depthHeat — jet-style colormap blue → green → red.
    const r: Any = t.sub(0.5).mul(2.0).clamp(0.0, 1.0);
    const b: Any = float(0.5).sub(t).mul(2.0).clamp(0.0, 1.0);
    const g: Any = float(1.0).sub(abs(t.sub(0.5)).mul(2.0));
    return vec4(r, g, b, 1.0);
  })();

  return material;
}
