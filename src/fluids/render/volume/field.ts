import { HalfFloatType, Vector3 } from 'three';
import { Storage3DTexture } from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicLoad,
  atomicStore,
  float,
  instanceIndex,
  instancedArray,
  int,
  ivec3,
  textureStore,
  vec2,
  uint,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import {
  emitColliderSdf,
  emitSampleSdf,
  type ParticleRange,
  type PrimitiveSet,
  type SDFCollider,
} from '../../../core/index.js';
import type { FluidSystem } from '../../FluidSystem.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Fixed-point scale for atomic splatting. 16 fractional bits keep CIC weights exact to ~1e-5. */
const FIXED = 65536;
/** Field value (voxels) stored where no particle is within the filter footprint. */
export const FIELD_BAND = 6;

export interface SurfaceFieldOptions {
  readonly fluidSystem: FluidSystem;
  /** World-space region the liquid can occupy. Particles outside are ignored. */
  readonly min: Vector3;
  readonly max: Vector3;
  /** Upper bound on voxel count; the voxel grows past the particle radius to respect it. */
  readonly voxelBudget: number;
  /** Colliders the liquid wets: the surface climbs them to form a meniscus. */
  readonly colliders?: PrimitiveSet | undefined;
  /**
   * Moving colliders carved out of the liquid every frame (e.g. bubbles).
   * Unlike `colliders`, these are not wetted.
   */
  readonly carve?: PrimitiveSet | undefined;
  /** Mesh colliders the liquid wets. */
  readonly sdfColliders?: readonly SDFCollider[] | undefined;
  /** Non-fluid particles (floating bodies) the liquid also wets. */
  readonly solids?: ParticleRange | undefined;
  /**
   * Smear each particle back along its velocity over this many seconds
   * (capped at three spacings). Joins fast, stretched streams that the
   * solver resolves as a chain of separated particles. 0 disables it.
   */
  readonly motionStretch?: number | undefined;
}

/**
 * Surface field uniforms. Distances are world metres; the kernels convert
 * to voxels internally so the look is independent of grid resolution.
 */
export interface SurfaceFieldUniforms {
  /** Surface radius around an isolated particle (m). */
  readonly radius: ReturnType<typeof uniform<'float', number>>;
  /** How high the liquid climbs a wetted wall (m). 0 disables the meniscus. */
  readonly meniscusHeight: ReturnType<typeof uniform<'float', number>>;
  /** Horizontal reach of the meniscus from the wall (m). */
  readonly meniscusWidth: ReturnType<typeof uniform<'float', number>>;
}

/**
 * GPU surface field for a particle liquid.
 *
 * Each frame:
 *   1. Particles are splatted into a voxel grid with trilinear (cloud-in-cell)
 *      weights: mass and the mass-weighted offset from each voxel centre.
 *   2. A separable Gaussian (three 1D passes) spreads both sums. Offsets are
 *      re-based onto each destination voxel as they move, so the result is
 *      the exact kernel-weighted mean particle position x̄ around every voxel.
 *   3. The surface is `φ = |p − x̄| − R` (Zhu & Bridson 2005): an isolated
 *      particle becomes a sphere of radius R, and flat regions stay flat
 *      instead of showing a bump per particle. Near wetted solids φ is
 *      lowered so the surface rises along the wall into a meniscus.
 *
 * The result is written to a half-float 3D texture (R = φ in voxels,
 * G = liquid volume fraction, B = wall proximity) for hardware-filtered
 * ray marching. All passes are O(voxels) or O(particles); nothing depends
 * on screen resolution.
 */
export class SurfaceField {
  readonly texture: Storage3DTexture;
  readonly uniforms: SurfaceFieldUniforms;
  /** Grid origin (world). */
  readonly origin: Vector3;
  /** World size of the whole grid; a multiple of `voxel` on each axis. */
  readonly extent: Vector3;
  readonly voxel: number;
  readonly resolution: readonly [number, number, number];
  readonly kernels: readonly ComputeNode[];
  /** Recomputes wetted-collider proximity. Run once, and again after colliders move. */
  readonly wallKernel: ComputeNode;

  constructor(options: SurfaceFieldOptions) {
    const { fluidSystem, min, max, voxelBudget, colliders, sdfColliders = [], solids } = options;
    const stretch = options.motionStretch ?? 0;
    const carve = options.carve;
    const { particles, fluidParticles } = fluidSystem;
    const r = particles.particleRadius;
    const spacing = fluidSystem.particleSpacing;
    const size = max.clone().sub(min);
    if (Math.min(size.x, size.y, size.z) <= 0)
      throw new Error('Liquid bounds must have positive extent.');

    const voxel = Math.max(r, Math.cbrt((size.x * size.y * size.z) / voxelBudget));
    const nx = Math.ceil(size.x / voxel),
      ny = Math.ceil(size.y / voxel),
      nz = Math.ceil(size.z / voxel);
    const count = nx * ny * nz;
    this.voxel = voxel;
    this.resolution = [nx, ny, nz];
    this.origin = min.clone();
    this.extent = new Vector3(nx, ny, nz).multiplyScalar(voxel);

    this.uniforms = {
      radius: uniform(r * 1.25, 'float'),
      meniscusHeight: uniform(r * 1.6, 'float'),
      meniscusWidth: uniform(r * 2.5, 'float'),
    };

    const origin = uniform(this.origin);
    const invVoxel = float(1 / voxel);
    const dims = vec3(nx, ny, nz);
    const indexOf = (c: Any): Any =>
      c.z
        .mul(nx * ny)
        .add(c.y.mul(nx))
        .add(c.x)
        .toUint();
    const cellOf = (i: Any): Any =>
      ivec3(i.mod(nx).toInt(), i.div(nx).mod(ny).toInt(), i.div(nx * ny).toInt());

    const mass = (instancedArray(count, 'int') as Any).setAtomic(true);
    const offX = (instancedArray(count, 'int') as Any).setAtomic(true);
    const offY = (instancedArray(count, 'int') as Any).setAtomic(true);
    const offZ = (instancedArray(count, 'int') as Any).setAtomic(true);
    const solid = (instancedArray(count, 'int') as Any).setAtomic(true);
    const a = instancedArray(count, 'vec4');
    const b = instancedArray(count, 'vec4');
    const sa = instancedArray(count, 'float');
    const sb = instancedArray(count, 'float');

    const clear = Fn(() => {
      const i: Any = instanceIndex;
      atomicStore(mass.element(i), int(0));
      atomicStore(offX.element(i), int(0));
      atomicStore(offY.element(i), int(0));
      atomicStore(offZ.element(i), int(0));
      atomicStore(solid.element(i), int(0));
    })().compute(count);

    // Continuous grid coordinate with voxel centres on integers.
    const gridCoord = (world: Any): Any => world.sub(origin).mul(invVoxel).sub(0.5);

    const splat = (range: ParticleRange, fluid: boolean): ComputeNode =>
      Fn(() => {
        const i: Any = instanceIndex.add(uint(range.start));
        const position: Any = particles.positions.element(i).xyz.toVar();
        // Sub-samples spread back along the velocity; each carries an equal share.
        const samples = fluid && stretch > 0 ? 3 : 1;
        const trail: Any =
          samples > 1
            ? (() => {
                const v: Any = particles.velocities.element(i).xyz.mul(stretch);
                return v.mul(float(3 * spacing).div(v.length().max(3 * spacing)));
              })()
            : undefined;
        for (let sample = 0; sample < samples; sample++) {
          const at: Any = sample === 0 ? position : position.sub(trail.mul(sample / (samples - 1)));
          const g: Any = gridCoord(at).toVar();
          If(
            g
              .greaterThanEqual(vec3(0))
              .all()
              .and(g.lessThan(dims.sub(1)).all()),
            () => {
              const base: Any = g.floor().toVar();
              const f: Any = g.sub(base).toVar();
              for (let z = 0; z < 2; z++)
                for (let y = 0; y < 2; y++)
                  for (let x = 0; x < 2; x++) {
                    const w: Any = (x ? f.x : f.x.oneMinus())
                      .mul(y ? f.y : f.y.oneMinus())
                      .mul(z ? f.z : f.z.oneMinus())
                      .mul(1 / samples)
                      .toVar();
                    const cell: Any = indexOf(base.add(vec3(x, y, z)));
                    if (fluid) {
                      atomicAdd(mass.element(cell), w.mul(FIXED).round().toInt());
                      atomicAdd(offX.element(cell), w.mul(f.x.sub(x)).mul(FIXED).round().toInt());
                      atomicAdd(offY.element(cell), w.mul(f.y.sub(y)).mul(FIXED).round().toInt());
                      atomicAdd(offZ.element(cell), w.mul(f.z.sub(z)).mul(FIXED).round().toInt());
                    } else {
                      atomicAdd(solid.element(cell), w.mul(FIXED).round().toInt());
                    }
                  }
            },
          );
        }
      })().compute(range.count);

    // Gaussian over ±TAPS voxels. σ ≈ 0.75 particle spacings keeps the lattice
    // ripple below 1e-4 while leaving splash detail intact.
    const sigma = (0.75 * spacing) / voxel;
    const taps = Math.min(6, Math.max(2, Math.ceil(2.5 * sigma)));
    const weights = Array.from({ length: taps * 2 + 1 }, (_, k) =>
      Math.exp(-((k - taps) ** 2) / (2 * sigma * sigma)),
    );
    const weightSum = weights.reduce((sum, w) => sum + w, 0);
    for (let k = 0; k < weights.length; k++) weights[k]! /= weightSum;

    const volumeScale = (spacing / voxel) ** 3;
    const texture = new Storage3DTexture(nx, ny, nz);
    texture.type = HalfFloatType;
    texture.name = 'FluidSurfaceField';
    this.texture = texture;

    // Distance to wetted colliders changes only when they move, so it lives
    // in its own buffer that `wallKernel` refreshes on demand:
    // x = wetting proximity, y = signed distance to the nearest collider (m).
    const walls = instancedArray(count, 'vec2');
    // `toFloat()` on an ivec3 yields a scalar, so convert each component.
    const voxelCentre = (cell: Any): Any =>
      origin.add(vec3(cell.x.toFloat(), cell.y.toFloat(), cell.z.toFloat()).add(0.5).mul(voxel));
    const nearestCollider = (set: PrimitiveSet, p: Any): Any => {
      const nearest: Any = float(1e3).toVar();
      Loop(
        { start: uint(0), end: set.numColliders as Any, type: 'uint', condition: '<' },
        ({ i }: { i: Any }) => {
          const phi: Any = float(0).toVar();
          const grad: Any = vec3(0).toVar();
          emitColliderSdf(set, i, p, phi, grad);
          nearest.assign(nearest.min(phi));
        },
      );
      return nearest;
    };
    this.wallKernel = Fn(() => {
      const cell: Any = cellOf(instanceIndex);
      const proximity: Any = float(0).toVar();
      const nearest: Any = float(1e3).toVar();
      if (colliders || sdfColliders.length > 0) {
        const p: Any = voxelCentre(cell);
        if (colliders) nearest.assign(nearestCollider(colliders, p));
        for (const collider of sdfColliders) {
          const phi: Any = float(0).toVar();
          const grad: Any = vec3(0).toVar();
          emitSampleSdf(collider.fields, p, phi, grad);
          // Outside its baked grid the field clamps to the edge value; the
          // distance to the grid's bounding sphere is a valid lower bound there.
          const f: Any = collider.fields;
          const half: Any = f.resolution.mul(f.voxelSize).mul(0.5);
          const centre: Any = f.position.add(f.rotation.mul(f.bakedOrigin.add(half)).mul(f.scale));
          const bound: Any = p.sub(centre).length().sub(half.length().mul(f.scale));
          nearest.assign(nearest.min(phi.max(bound)));
        }
        const width: Any = this.uniforms.meniscusWidth.max(1e-5);
        proximity.assign(nearest.max(0).div(width).negate().exp());
      }
      walls.element(instanceIndex).assign(vec2(proximity, nearest));
    })().compute(count);
    const wallProximity = (solidMass: Any): Any =>
      solidMass
        .mul(volumeScale * 1.5)
        .clamp(0, 1)
        .max(walls.element(instanceIndex).x);

    const blur = (axis: 0 | 1 | 2, final: boolean): ComputeNode =>
      Fn(() => {
        const cell: Any = cellOf(instanceIndex).toVar();
        const sum: Any = vec4(0).toVar();
        const solidSum: Any = float(0).toVar();
        for (let k = -taps; k <= taps; k++) {
          const step = [0, 0, 0];
          step[axis] = k;
          const neighbor: Any = cell.add(ivec3(step[0]!, step[1]!, step[2]!));
          const n: Any = neighbor[(['x', 'y', 'z'] as const)[axis]];
          If(n.greaterThanEqual(0).and(n.lessThan([nx, ny, nz][axis]!)), () => {
            const j: Any = indexOf(neighbor);
            let sample: Any, solidSample: Any;
            if (axis === 0) {
              const load = (buffer: Any): Any => (atomicLoad(buffer.element(j)) as Any).toFloat();
              sample = vec4(load(mass), load(offX), load(offY), load(offZ)).div(FIXED);
              solidSample = (atomicLoad(solid.element(j)) as Any).toFloat().div(FIXED);
            } else {
              sample = (axis === 1 ? a : b).element(j);
              solidSample = (axis === 1 ? sa : sb).element(j);
            }
            const s: Any = sample.toVar();
            // Re-base the neighbour's offsets onto this voxel's centre.
            const shift = [0, 0, 0];
            shift[axis] = k;
            const rebased: Any = vec4(
              s.x,
              s.yzw.add(vec3(shift[0]!, shift[1]!, shift[2]!).mul(s.x)),
            );
            sum.addAssign(rebased.mul(weights[k + taps]!));
            solidSum.addAssign(solidSample.mul(weights[k + taps]!));
          });
        }
        if (!final) {
          (axis === 0 ? a : b).element(instanceIndex).assign(sum);
          (axis === 0 ? sa : sb).element(instanceIndex).assign(solidSum);
          return;
        }
        const w: Any = sum.x;
        const mean: Any = sum.yzw.div(w.max(1e-8));
        const radius: Any = this.uniforms.radius.mul(invVoxel);
        const phi: Any = w
          .greaterThan(1e-5)
          .select(mean.length().sub(radius), float(FIELD_BAND))
          .min(FIELD_BAND)
          .toVar();
        const proximity: Any = wallProximity(solidSum).toVar();
        // Wetting: lower φ near walls, but only within reach of the surface so
        // dry walls above the liquid stay dry.
        const lift: Any = this.uniforms.meniscusHeight.mul(invVoxel).toVar();
        const reach: Any = phi.div(lift.mul(1.5).max(1e-5)).oneMinus().clamp(0, 1);
        phi.subAssign(lift.mul(proximity).mul(reach.mul(reach)));
        // No liquid inside solids: carve colliders out, which also cuts the
        // surface cleanly where it meets a wall.
        phi.assign(phi.max(walls.element(instanceIndex).y.negate().mul(invVoxel)));
        if (carve) {
          const p: Any = voxelCentre(cell);
          phi.assign(phi.max(nearestCollider(carve, p).negate().mul(invVoxel)));
        }
        textureStore(
          texture,
          cell,
          vec4(phi, w.mul(volumeScale).min(4), proximity, 1),
        ).toWriteOnly();
      })().compute(count);

    const kernels: ComputeNode[] = [clear, splat(fluidParticles, true)];
    if (solids && solids.count > 0) kernels.push(splat(solids, false));
    kernels.push(blur(0, false), blur(1, false), blur(2, true));
    this.kernels = kernels;
  }

  dispose(): void {
    this.texture.dispose();
  }
}
