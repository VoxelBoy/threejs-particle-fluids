import {
  Fn,
  If,
  Loop,
  float,
  instanceIndex,
  instancedArray,
  uint,
  uniform,
  uniformArray,
  vec3,
} from 'three/tsl';
import { Quaternion, Vector3, type Object3D } from 'three';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { ParticleSystem } from '../particles.js';
import {
  emitColliderContact,
  emitColliderFriction,
  type Collider,
  type ColliderContext,
  type ColliderKernels,
} from './collider.js';
import {
  FLAG_INVERT,
  KIND_BOX,
  KIND_CAPSULE,
  KIND_PLANE,
  KIND_SPHERE,
  emitColliderSdf,
} from './primitives.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** Friction and motion shared by every primitive. */
export interface PrimitiveOptions {
  /** Static friction coefficient. Default 0.5. */
  readonly muS?: number;
  /** Kinetic friction coefficient. Default 0.4. */
  readonly muK?: number;
  /**
   * Surface velocity in m/s, used for friction (a conveyor belt, for example).
   * Primitives attached to an object derive this from the object's motion.
   */
  readonly velocity?: Vector3;
}

export interface SolidPrimitiveOptions extends PrimitiveOptions {
  /** Keep particles inside the shape instead of outside it. Default `false`. */
  readonly invert?: boolean;
}

export interface BoxOptions extends SolidPrimitiveOptions {
  /** Orientation. Default identity. */
  readonly rotation?: Quaternion;
}

interface Primitive {
  kind: number;
  flags: number;
  /** Plane: normal. Sphere and box: center (w = radius). Capsule: a (w = radius). */
  data0: [number, number, number, number];
  /** Plane: point. Box: half extents. Capsule: b. w = μ_s. */
  data1: [number, number, number, number];
  /** Velocity, w = μ_k. */
  velocity: [number, number, number, number];
  rotation: Quaternion;
}

interface Attachment {
  readonly slot: number;
  readonly object: Object3D;
  readonly previous: Vector3;
}

/** GPU copies of the primitives, allocated the first time a kernel needs them. */
interface PrimitiveBuffers {
  readonly capacity: number;
  /** `(kind << 16) | flags` per slot. */
  readonly packed: StorageBufferNode<'uint'>;
  readonly data0: StorageBufferNode<'vec4'>;
  readonly data1: StorageBufferNode<'vec4'>;
  readonly velocity: StorageBufferNode<'vec4'>;
  /** Orientation per slot, stored in uniform memory to save storage bindings. */
  readonly rotation: Any;
  readonly rotations: Quaternion[];
  readonly count: UniformNode<'uint', number>;
}

/**
 * Planes, spheres, boxes, and capsules that particles collide with.
 *
 * Add shapes, then pass the set to {@link SimLoop}'s `colliders`. To move a
 * shape, {@link attach} it to an `Object3D` and move the object; the set
 * follows it every step and derives its velocity for friction.
 *
 * ```ts
 * const colliders = new PrimitiveSet(particles);
 * colliders.addPlane(new Vector3(0, 1, 0), new Vector3());
 * const paddle = colliders.addBox(new Vector3(0, 0.5, 0), new Vector3(0.2, 0.05, 0.01));
 * colliders.attach(paddle, paddleMesh);
 * ```
 */
export class PrimitiveSet implements Collider {
  readonly particles: ParticleSystem;
  /** Increments whenever primitives change, so renderers can refresh cached shapes. */
  version = 0;

  private readonly primitives: Primitive[] = [];
  private readonly attachments: Attachment[] = [];
  private readonly reserved: number | undefined;
  private buffers: PrimitiveBuffers | undefined;
  private dirty = true;
  /**
   * Seconds left in the current frame. The solver rewinds moving primitives by
   * this much, so a once-per-frame update becomes a smooth sweep.
   */
  private readonly motionClock = instancedArray(1, 'float');

  /**
   * @param options.capacity Slots to reserve for primitives added after the
   *   simulation starts. Defaults to the number added before then.
   */
  constructor(particles: ParticleSystem, options: { readonly capacity?: number } = {}) {
    if (
      options.capacity !== undefined &&
      !(Number.isInteger(options.capacity) && options.capacity > 0)
    ) {
      throw new Error(`PrimitiveSet: capacity must be a positive integer, got ${options.capacity}`);
    }
    this.particles = particles;
    this.reserved = options.capacity;
  }

  /** Number of primitives. */
  get count(): number {
    return this.primitives.length;
  }

  /** Add a plane through `point`; particles stay on the side `normal` points to. */
  addPlane(normal: Vector3, point: Vector3, options: PrimitiveOptions = {}): number {
    if (!(normal.lengthSq() > 0)) throw new Error('PrimitiveSet.addPlane: normal must be non-zero');
    const n = normal.clone().normalize();
    return this.add(KIND_PLANE, 0, [n.x, n.y, n.z, 0], [point.x, point.y, point.z, 0], options);
  }

  addSphere(center: Vector3, radius: number, options: SolidPrimitiveOptions = {}): number {
    assertPositive(radius, 'PrimitiveSet.addSphere: radius');
    return this.add(
      KIND_SPHERE,
      options.invert ? FLAG_INVERT : 0,
      [center.x, center.y, center.z, radius],
      [0, 0, 0, 0],
      options,
    );
  }

  addBox(center: Vector3, halfExtents: Vector3, options: BoxOptions = {}): number {
    if (!(halfExtents.x > 0 && halfExtents.y > 0 && halfExtents.z > 0)) {
      throw new Error('PrimitiveSet.addBox: halfExtents must be positive');
    }
    return this.add(
      KIND_BOX,
      options.invert ? FLAG_INVERT : 0,
      [center.x, center.y, center.z, 0],
      [halfExtents.x, halfExtents.y, halfExtents.z, 0],
      options,
      options.rotation,
    );
  }

  /** Add a capsule between `a` and `b`. */
  addCapsule(a: Vector3, b: Vector3, radius: number, options: PrimitiveOptions = {}): number {
    assertPositive(radius, 'PrimitiveSet.addCapsule: radius');
    return this.add(KIND_CAPSULE, 0, [a.x, a.y, a.z, radius], [b.x, b.y, b.z, 0], options);
  }

  /**
   * Make a primitive follow `object`'s world position and orientation. The
   * primitive keeps its own shape; only its placement comes from the object.
   */
  attach(slot: number, object: Object3D): void {
    this.primitiveAt(slot);
    const index = this.attachments.findIndex((a) => a.slot === slot);
    if (index >= 0) this.attachments.splice(index, 1);
    const position = object.getWorldPosition(new Vector3());
    this.attachments.push({ slot, object, previous: position.clone() });
    this.moveTo(slot, position);
    object.getWorldQuaternion(this.primitives[slot]!.rotation);
    this.dirty = true;
  }

  /** Move and resize a sphere. `velocity` feeds friction. */
  setSphere(slot: number, center: Vector3, radius: number, velocity?: Vector3): void {
    const primitive = this.primitiveAt(slot);
    if (primitive.kind !== KIND_SPHERE)
      throw new Error(`PrimitiveSet.setSphere: slot ${slot} is not a sphere`);
    assertPositive(radius, 'PrimitiveSet.setSphere: radius');
    primitive.data0 = [center.x, center.y, center.z, radius];
    primitive.velocity = [
      velocity?.x ?? 0,
      velocity?.y ?? 0,
      velocity?.z ?? 0,
      primitive.velocity[3],
    ];
    this.dirty = true;
  }

  /** Follow attached objects and upload any changes. {@link SimLoop.step} calls this. */
  update(dt: number): void {
    if (this.attachments.length > 0 && dt > 0) {
      const position = new Vector3();
      for (const { slot, object, previous } of this.attachments) {
        object.getWorldPosition(position);
        const primitive = this.primitives[slot]!;
        object.getWorldQuaternion(primitive.rotation);
        const v = position.clone().sub(previous).divideScalar(dt);
        primitive.velocity = [v.x, v.y, v.z, primitive.velocity[3]];
        this.moveTo(slot, position);
        previous.copy(position);
      }
      this.dirty = true;
    }
    if (this.dirty && this.buffers) this.upload(this.buffers);
  }

  /** @internal GPU buffers read by collider kernels and the fluid surface renderer. */
  get gpu(): PrimitiveBuffers {
    if (!this.buffers) {
      const capacity = Math.max(1, this.reserved ?? this.primitives.length);
      if (capacity < this.primitives.length) {
        throw new Error(
          `PrimitiveSet: ${this.primitives.length} primitives exceed capacity ${capacity}`,
        );
      }
      const rotations = Array.from({ length: capacity }, () => new Quaternion());
      this.buffers = {
        capacity,
        packed: instancedArray(capacity, 'uint'),
        data0: instancedArray(capacity, 'vec4'),
        data1: instancedArray(capacity, 'vec4'),
        velocity: instancedArray(capacity, 'vec4'),
        rotation: uniformArray(rotations, 'vec4'),
        rotations,
        // r184's typings only accept 'float' here; 'uint' works at runtime.
        count: uniform(0, 'uint' as 'float') as unknown as UniformNode<'uint', number>,
      };
      this.upload(this.buffers);
    }
    return this.buffers;
  }

  /** @internal */
  buildKernels({
    particles,
    dt,
    substeps,
    positions,
    velocities,
  }: ColliderContext): ColliderKernels {
    const gpu = this.gpu;
    const radius = particles.particleRadius;
    // (λ_n, λ_t) per particle–primitive pair.
    const lambda = instancedArray(2 * particles.capacity * gpu.capacity, 'float');
    const clock: Any = this.motionClock.element(0);

    const resetLambda = Fn(() => {
      lambda.element(instanceIndex).assign(0);
    })().compute(2 * particles.capacity * gpu.capacity);

    const forEachPrimitive = (body: (slot: Any) => void): void => {
      Loop(
        { start: uint(0), end: gpu.count as Any, type: 'uint', condition: '<' },
        ({ i }: { i: Any }) => body(i),
      );
    };

    const solve = Fn(() => {
      const p: Any = instanceIndex;
      const w: Any = particles.invMass.element(p).toVar();
      If(w.greaterThan(0), () => {
        const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
        const displacement: Any = xStar.sub(particles.positions.element(p).xyz).toVar();
        forEachPrimitive((slot) => {
          const phi: Any = float(0).toVar();
          const normal: Any = vec3(0).toVar();
          emitColliderSdf(gpu, slot, xStar, phi, normal, clock);
          If(phi.lessThan(radius), () => {
            emitColliderContact({
              particle: p,
              invMass: w,
              displacement,
              colliderDisplacement: gpu.velocity.element(slot).xyz.mul(dt),
              normal,
              depth: float(radius).sub(phi),
              muS: gpu.data1.element(slot).w,
              lambda,
              lambdaIndex: p.mul(uint(gpu.capacity)).add(slot).mul(uint(2)),
              accumulator: positions,
            });
          });
        });
      });
    })().compute(particles.capacity);

    const friction = Fn(() => {
      const p: Any = instanceIndex;
      const w: Any = particles.invMass.element(p).toVar();
      If(w.greaterThan(0), () => {
        const xStar: Any = particles.predictedPositions.element(p).xyz.toVar();
        const velocity: Any = particles.velocities.element(p).xyz.toVar();
        forEachPrimitive((slot) => {
          const lambdaN: Any = lambda
            .element(p.mul(uint(gpu.capacity)).add(slot).mul(uint(2)))
            .toVar();
          If(lambdaN.greaterThan(0), () => {
            const phi: Any = float(0).toVar();
            const gradient: Any = vec3(0).toVar();
            emitColliderSdf(gpu, slot, xStar, phi, gradient, clock);
            const length: Any = gradient.length().toVar();
            If(length.greaterThan(1e-6), () => {
              const surface: Any = gpu.velocity.element(slot);
              emitColliderFriction({
                particle: p,
                invMass: w,
                velocity,
                colliderVelocity: surface.xyz,
                normal: gradient.div(length),
                lambdaN,
                muK: surface.w,
                dt,
                accumulator: velocities,
              });
            });
          });
        });
      });
    })().compute(particles.capacity);

    return {
      // The clock starts at the time left after the first substep and ticks
      // down once per substep.
      frameStart: [
        Fn(() => {
          clock.assign(dt.mul(substeps.sub(1)));
        })().compute(1),
      ],
      preSolve: [resetLambda],
      solve: [solve],
      postSolve: [friction],
      substepEnd: [
        Fn(() => {
          clock.assign(clock.sub(dt).max(0));
        })().compute(1),
      ],
    };
  }

  dispose(): void {
    this.attachments.length = 0;
  }

  private add(
    kind: number,
    flags: number,
    data0: [number, number, number, number],
    data1: [number, number, number, number],
    options: PrimitiveOptions,
    rotation?: Quaternion,
  ): number {
    if (this.buffers && this.primitives.length >= this.buffers.capacity) {
      throw new Error(
        `PrimitiveSet: capacity ${this.buffers.capacity} is full. Pass a larger \`capacity\` to reserve slots for primitives added after the simulation starts.`,
      );
    }
    const { muS, muK } = resolveFriction(options);
    const v = options.velocity;
    this.primitives.push({
      kind,
      flags,
      data0,
      data1: [data1[0], data1[1], data1[2], muS],
      velocity: [v?.x ?? 0, v?.y ?? 0, v?.z ?? 0, muK],
      rotation: rotation?.clone() ?? new Quaternion(),
    });
    this.dirty = true;
    return this.primitives.length - 1;
  }

  private primitiveAt(slot: number): Primitive {
    const primitive = this.primitives[slot];
    if (!primitive) throw new Error(`PrimitiveSet: no primitive in slot ${slot}`);
    return primitive;
  }

  /** Move a primitive's reference point: the plane point, the center, or a capsule's midpoint. */
  private moveTo(slot: number, position: Vector3): void {
    const primitive = this.primitives[slot]!;
    if (primitive.kind === KIND_PLANE) {
      primitive.data1 = [position.x, position.y, position.z, primitive.data1[3]];
    } else if (primitive.kind === KIND_CAPSULE) {
      const [ax, ay, az, radius] = primitive.data0;
      const [bx, by, bz, muS] = primitive.data1;
      const dx = position.x - (ax + bx) / 2;
      const dy = position.y - (ay + by) / 2;
      const dz = position.z - (az + bz) / 2;
      primitive.data0 = [ax + dx, ay + dy, az + dz, radius];
      primitive.data1 = [bx + dx, by + dy, bz + dz, muS];
    } else {
      primitive.data0 = [position.x, position.y, position.z, primitive.data0[3]];
    }
  }

  private upload(gpu: PrimitiveBuffers): void {
    const packed = gpu.packed.value.array as Uint32Array;
    const data0 = gpu.data0.value.array as Float32Array;
    const data1 = gpu.data1.value.array as Float32Array;
    const velocity = gpu.velocity.value.array as Float32Array;
    this.primitives.forEach((primitive, slot) => {
      packed[slot] = (primitive.kind << 16) | primitive.flags;
      data0.set(primitive.data0, slot * 4);
      data1.set(primitive.data1, slot * 4);
      velocity.set(primitive.velocity, slot * 4);
      gpu.rotations[slot]!.copy(primitive.rotation);
    });
    gpu.packed.value.needsUpdate = true;
    gpu.data0.value.needsUpdate = true;
    gpu.data1.value.needsUpdate = true;
    gpu.velocity.value.needsUpdate = true;
    gpu.count.value = this.primitives.length;
    this.dirty = false;
    this.version++;
  }
}

function assertPositive(value: number, name: string): void {
  if (!(value > 0) || !Number.isFinite(value))
    throw new Error(`${name} must be positive, got ${value}`);
}

/** Validate friction options against the defaults μ_s = 0.5, μ_k = 0.4. */
export function resolveFriction(options: { readonly muS?: number; readonly muK?: number }): {
  readonly muS: number;
  readonly muK: number;
} {
  const muS = options.muS ?? 0.5;
  const muK = options.muK ?? 0.4;
  if (!(muS >= 0) || !(muK >= 0)) {
    throw new Error(`Friction coefficients must be non-negative (got muS=${muS}, muK=${muK})`);
  }
  return { muS, muK };
}
