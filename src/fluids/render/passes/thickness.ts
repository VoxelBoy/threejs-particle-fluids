import { AdditiveBlending, InstancedMesh, Matrix4, PlaneGeometry } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  Discard,
  Fn,
  cameraWorldMatrix,
  exp,
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
 * Pass 3 — Thickness (van der Laan, Green, Sainz 2009 §3.3, eq. 9):
 *
 *   T(x, y) = Σ_i d((x − x_i) / σ_i, (y − y_i) / σ_i)
 *
 * Each fluid particle splatted as an additive-blended Gaussian-alpha
 * point sprite. Accumulated thickness drives Beer-Lambert attenuation
 * in the surface composite (paper §3.5 eq. 14): thicker fluid absorbs
 * more of the refracted scene-behind sample.
 *
 * Paper §3.3 verbatim: *"Strictly speaking this measure of thickness is
 * only correct if the particles do not overlap, but this is a
 * reasonable assumption in SPH due to repulsive inter-particle
 * forces"*. We accept the approximation per the paper.
 *
 * Implementation matches the depth pass billboard pattern. Differences:
 *   - Larger billboard (`splatRadius · r` instead of `r`): the
 *     Gaussian's tail extends well past the imposter's "core".
 *   - No `Discard(r² > 1)` outside the inscribed disk; the Gaussian's
 *     natural falloff would alias if hard-clipped at the disk edge.
 *     We do hard-clip outside the inscribed disk so corners don't
 *     contribute Gaussian tail past the quad.
 *   - Output: scalar Gaussian alpha additively blended into the bound
 *     RT's red channel.
 */

export interface CreateThicknessPassOptions {
  readonly fluidSystem: FluidSystem;
  /**
   * Effective splat radius (m). Owner derives this from
   * `params.thickness.splatRadius.value · particleRadius` and writes it
   * to the uniform live; the pass does not own the conversion.
   */
  readonly splatRadiusUniform: ReturnType<typeof uniform<'float', number>>;
}

export interface ThicknessPassMesh {
  readonly mesh: InstancedMesh;
  readonly material: MeshBasicNodeMaterial;
}

export function createThicknessPassMesh(options: CreateThicknessPassOptions): ThicknessPassMesh {
  const { fluidSystem, splatRadiusUniform } = options;
  const { particles, fluidParticles } = fluidSystem;

  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthWrite = false;
  material.depthTest = false;
  material.blending = AdditiveBlending;

  const fluidStart = float(fluidParticles.start).toUint();
  const slotIdx = instanceIndex.add(fluidStart);
  const particleWorld: Any = (particles.positions as Any).element(slotIdx).xyz;

  const cameraRight: Any = (cameraWorldMatrix as Any).element(0).xyz;
  const cameraUp: Any = (cameraWorldMatrix as Any).element(1).xyz;

  const quadOffsetXY: Any = positionLocal.xy.mul(2.0).mul(splatRadiusUniform);
  const offsetWorld: Any = cameraRight.mul(quadOffsetXY.x).add(cameraUp.mul(quadOffsetXY.y));

  const billboardedWorld: Any = particleWorld.add(offsetWorld);
  material.positionNode = billboardedWorld;

  // Gaussian d(u, v) = exp(−(u² + v²) · k); k=5 keeps falloff inside the
  // inscribed disk so neighbour quads don't leak Gaussian tail past
  // their bounds.
  material.outputNode = Fn(() => {
    const diskUv: Any = uv().mul(2.0).sub(1.0);
    const r2: Any = diskUv.dot(diskUv);
    Discard(r2.greaterThan(1.0));
    const gauss: Any = exp(r2.mul(-5.0));
    return vec4(gauss, 0.0, 0.0, 1.0);
  })();

  const geometry = new PlaneGeometry(1, 1);
  const mesh = new InstancedMesh(geometry, material, fluidParticles.count);
  mesh.name = 'FluidThicknessPassMesh';
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;

  const identity = new Matrix4();
  for (let i = 0; i < fluidParticles.count; i++) {
    mesh.setMatrixAt(i, identity);
  }
  mesh.instanceMatrix.needsUpdate = true;

  return { mesh, material };
}
