import {
  BackSide,
  BoxGeometry,
  Data3DTexture,
  HalfFloatType,
  LinearFilter,
  Mesh,
  RedFormat,
  RepeatWrapping,
  Vector3,
} from 'three';
import { MeshBasicNodeMaterial, Storage3DTexture, type WebGPURenderer } from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicLoad,
  atomicStore,
  cameraPosition,
  cameraProjectionMatrixInverse,
  color,
  float,
  getViewPosition,
  instanceIndex,
  instancedArray,
  ivec3,
  mix,
  mx_noise_float,
  positionWorld,
  screenCoordinate,
  screenUV,
  texture3D,
  textureStore,
  uint,
  uniform,
  vec3,
  vec4,
  viewportDepthTexture,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type { SmokeTracers } from './types.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Small, seamless density-detail texture; generated once and advected during rendering. */
function createDetailTexture(): Data3DTexture {
  const size = 64;
  const data = new Uint8Array(size ** 3);
  const hash = (x: number, y: number, z: number, period: number) => {
    let n =
      Math.imul(x % period, 374761393) +
      Math.imul(y % period, 668265263) +
      Math.imul(z % period, 2147483647);
    n = Math.imul(n ^ (n >>> 13), 1274126177);
    return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
  };
  for (let z = 0; z < size; z++)
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        let value = 0;
        for (const [period, weight] of [
          [8, 0.55],
          [16, 0.3],
          [32, 0.15],
        ]) {
          const p = [x, y, z].map((v) => (v / size) * period!);
          const base = p.map(Math.floor);
          const f = p.map((v) => {
            const t = v - Math.floor(v);
            return t * t * (3 - 2 * t);
          });
          for (let dz = 0; dz < 2; dz++)
            for (let dy = 0; dy < 2; dy++)
              for (let dx = 0; dx < 2; dx++) {
                value +=
                  weight! *
                  hash(base[0]! + dx, base[1]! + dy, base[2]! + dz, period!) *
                  (dx ? f[0]! : 1 - f[0]!) *
                  (dy ? f[1]! : 1 - f[1]!) *
                  (dz ? f[2]! : 1 - f[2]!);
              }
        }
        data[x + y * size + z * size * size] = Math.round(value * 255);
      }
  const texture = new Data3DTexture(data, size, size, size);
  texture.format = RedFormat;
  texture.minFilter = texture.magFilter = LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = RepeatWrapping;
  texture.needsUpdate = true;
  return texture;
}

export interface VolumetricGasRendererOptions {
  readonly gas: SmokeTracers;
  readonly min: Vector3;
  readonly max: Vector3;
  readonly resolution?: readonly [number, number, number];
  readonly steps?: number;
  readonly density?: number;
}

/** GPU density splatting, separable filtering, and bounded single-scattering ray marching. */
export class VolumetricGasRenderer {
  readonly object: Mesh;
  readonly density: ReturnType<typeof uniform<'float', number>>;
  readonly time = uniform(0, 'float');
  private readonly kernels: ComputeNode[];
  private readonly textures: (Storage3DTexture | Data3DTexture)[];

  constructor(options: VolumetricGasRendererOptions) {
    const { gas, min, max, resolution = [32, 56, 32], steps = 56 } = options;
    const [nx, ny, nz] = resolution;
    if (
      resolution.some((n) => !Number.isInteger(n) || n < 4 || n > 128) ||
      steps < 8 ||
      steps > 128
    )
      throw new Error('Volume resolution must be 4–128 voxels per axis and steps must be 8–128.');
    const size = max.clone().sub(min);
    if (Math.min(size.x, size.y, size.z) <= 0)
      throw new Error('Volume bounds must have positive extent.');
    const count = nx * ny * nz;
    const ticks = (instancedArray(count, 'uint') as Any).setAtomic(true);
    const a = instancedArray(count, 'float');
    const b = instancedArray(count, 'float');
    const minimum = uniform(min.clone());
    const extent = uniform(size);
    this.density = uniform(options.density ?? 1, 'float');
    const scale = vec3(nx - 1, ny - 1, nz - 1);
    const indexOf = (p: Any): Any =>
      p.z
        .mul(nx * ny)
        .add(p.y.mul(nx))
        .add(p.x)
        .toUint();
    const cellOf = (i: Any): Any =>
      vec3(i.mod(nx).toFloat(), i.div(nx).mod(ny).toFloat(), i.div(nx * ny).toFloat());
    const clear = Fn(() => {
      atomicStore(ticks.element(instanceIndex), uint(0));
    })().compute(count);
    const splat = Fn(() => {
      const i: Any = instanceIndex;
      If(gas.smokeAlive.element(i).greaterThan(uint(0)), () => {
        const p: Any = gas.smokePositions
          .element(i)
          .xyz.sub(minimum)
          .div(extent)
          .mul(scale)
          .toVar();
        If(
          p
            .greaterThanEqual(vec3(1))
            .all()
            .and(p.lessThan(scale.sub(1)).all()),
          () => {
            const base = p.floor(),
              f = p.fract();
            const fade: Any = float(1).sub(gas.smokeAge.element(i).div(gas.lifetime)).max(0);
            for (let z = 0; z < 2; z++)
              for (let y = 0; y < 2; y++)
                for (let x = 0; x < 2; x++) {
                  const w: Any = (x ? f.x : float(1).sub(f.x))
                    .mul(y ? f.y : float(1).sub(f.y))
                    .mul(z ? f.z : float(1).sub(f.z))
                    .mul(fade);
                  atomicAdd(ticks.element(indexOf(base.add(vec3(x, y, z)))), w.mul(65536).toUint());
                }
          },
        );
      });
    })().compute(gas.capacity);
    const blur = (axis: number, source: Any, target: Any, atomic = false): ComputeNode =>
      Fn(() => {
        const cell: Any = cellOf(instanceIndex).toVar();
        const sum: Any = float(0).toVar();
        const weights = [0.0366, 0.1113, 0.2167, 0.2708, 0.2167, 0.1113, 0.0366];
        for (let k = -3; k <= 3; k++) {
          const offset = [0, 0, 0];
          offset[axis] = k;
          const neighbor: Any = cell
            .add(vec3(...(offset as [number, number, number])))
            .clamp(vec3(0), scale);
          const value: Any = atomic
            ? (atomicLoad(source.element(indexOf(neighbor))) as Any).toFloat().div(65536)
            : source.element(indexOf(neighbor));
          sum.addAssign(value.mul(weights[k + 3]!));
        }
        target.element(instanceIndex).assign(sum);
      })().compute(count);
    const extinction = 0.00015 / ((size.x * size.y * size.z) / count);
    const sampleParticles = Fn(([world]: Any[]) => {
      const unit: Any = world.sub(minimum).div(extent).toVar();
      const edge: Any = unit.min(vec3(1).sub(unit));
      const fade: Any = edge.x
        .min(edge.z)
        .smoothstep(0, 0.07)
        .mul(unit.y.smoothstep(0, 0.025))
        .mul(float(1).sub(unit.y.smoothstep(0.76, 1)));
      const p: Any = unit.mul(scale).clamp(vec3(0), scale.sub(0.001)).toVar();
      const cell = p.floor(),
        f = p.fract();
      const sum: Any = float(0).toVar();
      for (let z = 0; z < 2; z++)
        for (let y = 0; y < 2; y++)
          for (let x = 0; x < 2; x++) {
            const w: Any = (x ? f.x : float(1).sub(f.x))
              .mul(y ? f.y : float(1).sub(f.y))
              .mul(z ? f.z : float(1).sub(f.z));
            sum.addAssign(a.element(indexOf(cell.add(vec3(x, y, z)))).mul(w));
          }
      return sum.mul(extinction).mul(this.density).mul(fade);
    });
    const densityTexture = new Storage3DTexture(nx, ny, nz);
    const volumeTexture = new Storage3DTexture(nx, ny, nz);
    const detailTexture = createDetailTexture();
    for (const texture of [densityTexture, volumeTexture]) texture.type = HalfFloatType;
    this.textures = [densityTexture, volumeTexture, detailTexture];
    const densityAt = (world: Any): Any =>
      (texture3D(densityTexture, world.sub(minimum).div(extent)) as Any).level(0).r;
    // Build detail and incident light once per voxel. The ray marcher then uses
    // one hardware-filtered 3D sample per step rather than dozens of buffer reads.
    const detail = Fn(() => {
      const cell: Any = cellOf(instanceIndex);
      const p: Any = minimum.add(
        cell
          .add(0.5)
          .div(vec3(nx, ny, nz))
          .mul(extent),
      );
      const flow: Any = p.add(vec3(0, this.time.mul(-0.16), 0));
      const coarse: Any = mx_noise_float(flow.mul(8));
      const fine: Any = mx_noise_float(flow.mul(19).add(17));
      const modulation: Any = coarse.mul(0.95).add(fine.mul(0.35)).add(0.65).max(0.08);
      const density: Any = sampleParticles(p).mul(modulation.mul(modulation).mul(1.8));
      textureStore(densityTexture, ivec3(cell), vec4(density, 0, 0, 1)).toWriteOnly();
    })().compute(count);
    const sun = vec3(-0.35, 0.8, 0.4).normalize();
    const lighting = Fn(() => {
      const cell: Any = cellOf(instanceIndex);
      const p: Any = minimum.add(
        cell
          .add(0.5)
          .div(vec3(nx, ny, nz))
          .mul(extent),
      );
      const shadow: Any = densityAt(p.add(sun.mul(0.06)))
        .mul(0.08)
        .add(densityAt(p.add(sun.mul(0.18))).mul(0.14))
        .add(densityAt(p.add(sun.mul(0.36))).mul(0.22))
        .add(densityAt(p.add(sun.mul(0.62))).mul(0.28));
      const light: Any = shadow.mul(-1.6).exp();
      textureStore(volumeTexture, ivec3(cell), vec4(densityAt(p), light, 0, 1)).toWriteOnly();
    })().compute(count);
    this.kernels = [
      clear,
      splat,
      blur(0, ticks, a, true),
      blur(1, a, b),
      blur(2, b, a),
      detail,
      lighting,
    ];
    const material = new MeshBasicNodeMaterial({
      transparent: true,
      side: BackSide,
      depthWrite: false,
      // Depth-testing the box rejects smoke in front of opaque geometry whenever
      // its exit face is behind that geometry. Clip each ray to scene depth instead.
      depthTest: false,
      fog: false,
    });
    material.fragmentNode = Fn(() => {
      const direction: Any = positionWorld.sub(cameraPosition).normalize().toVar();
      const inv: Any = vec3(1).div(direction.add(0.000001));
      const near: Any = minimum.sub(cameraPosition).mul(inv);
      const far: Any = minimum.add(extent).sub(cameraPosition).mul(inv);
      const lower = near.min(far),
        upper = near.max(far);
      const start: Any = lower.x.max(lower.y).max(lower.z).max(0).toVar();
      const end: Any = upper.x.min(upper.y).min(upper.z).toVar();
      const sceneDepth: Any = viewportDepthTexture();
      const opaqueDistance: Any = getViewPosition(
        screenUV,
        sceneDepth.r,
        cameraProjectionMatrixInverse,
      ).length();
      end.assign(end.min(opaqueDistance));
      const step: Any = end.sub(start).max(0).div(steps).toVar();
      const jitter: Any = screenCoordinate
        .dot(vec3(12.9898, 78.233, 0).xy)
        .sin()
        .mul(43758.5453)
        .fract();
      const t: Any = start.add(step.mul(jitter)).toVar();
      const transmittance: Any = float(1).toVar();
      const radiance: Any = vec3(0).toVar();
      Loop(steps, () => {
        const p: Any = cameraPosition.add(direction.mul(t));
        const field: Any = (texture3D(volumeTexture, p.sub(minimum).div(extent)) as Any)
          .level(0)
          .toVar();
        const detailUV: Any = p.mul(1.2).add(vec3(0, this.time.mul(-0.12), 0));
        const grain: Any = (texture3D(detailTexture, detailUV) as Any).level(0).r;
        const density: Any = field.r.mul(grain.mul(2.8).sub(0.7).max(0.08).pow(1.5).mul(1.6));
        If(density.greaterThan(0.005), () => {
          const shade: Any = mix(color(0x3b4758), color(0xd8dfe6), field.g.mul(0.82).add(0.18));
          const alpha: Any = float(1).sub(density.mul(step).negate().exp());
          radiance.addAssign(shade.mul(alpha).mul(transmittance));
          transmittance.mulAssign(float(1).sub(alpha));
        });
        If(transmittance.lessThan(0.015), () => {
          Break();
        });
        t.addAssign(step);
      });
      const alpha: Any = float(1).sub(transmittance);
      return vec4(radiance.div(alpha.max(0.0001)), alpha);
    })();
    this.object = new Mesh(new BoxGeometry(size.x, size.y, size.z), material);
    this.object.position.copy(min).addScaledVector(size, 0.5);
    this.object.name = 'GasVolume';
    this.object.frustumCulled = false;
  }

  async update(renderer: WebGPURenderer): Promise<void> {
    if (this.object.visible) await renderer.computeAsync(this.kernels);
  }

  dispose(): void {
    this.object.geometry.dispose();
    (this.object.material as MeshBasicNodeMaterial).dispose();
    for (const texture of this.textures) texture.dispose();
  }
}
