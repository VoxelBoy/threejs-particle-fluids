[Docs](../README.md) › [API](../README.md#api-reference) › Colliders

# Colliders

Shapes that particles cannot enter. `PrimitiveSet` holds analytic planes, spheres, boxes, and capsules; `SDFCollider` collides with any closed mesh through a baked signed distance field. Pass colliders to [`SimLoop`](./core.md#simloop)'s `colliders` option.

```ts
import {
  PrimitiveSet,
  SDFCollider,
  bakeMeshToSdf,
  sampleSdf,
  sampleSdfGradient,
  encodeSdfBinary,
  decodeSdfBinary,
} from 'threejs-particle-fluids';
```

- [`Collider`](#collider)
- [`PrimitiveSet`](#primitiveset)
- [`SDFCollider`](#sdfcollider)
- [`SDFData`](#sdfdata)
- [`bakeMeshToSdf`](#bakemeshtosdf)
- [`sampleSdf`](#samplesdf)
- [`sampleSdfGradient`](#samplesdfgradient)
- [`encodeSdfBinary`](#encodesdfbinary)
- [`decodeSdfBinary`](#decodesdfbinary)
- [Limitations](#limitations)

## Collider

Interface implemented by `PrimitiveSet` and `SDFCollider`, and only by them. It is exported as a type for typing collider arrays; `SimLoop` throws `SimLoop: every collider must be built for the same ParticleSystem` if `particles` differs from its own.

```ts
interface Collider {
  readonly particles: ParticleSystem;
  update(dt: number): void;
  dispose(): void;
}
```

| Member       | Type                                         | Description                                                           |
| ------------ | -------------------------------------------- | --------------------------------------------------------------------- |
| `particles`  | [`ParticleSystem`](./core.md#particlesystem) | Particles this collider acts on.                                      |
| `update(dt)` | `(dt: number) => void`                       | Called by `SimLoop.step` before every step with the step length in s. |
| `dispose()`  | `() => void`                                 | Releases the collider's resources.                                    |

The interface also has an internal `buildKernels` method whose parameter and return types are not exported. Custom colliders are not supported.

## PrimitiveSet

Planes, spheres, boxes, and capsules in one collider. Each shape occupies a slot; the `add*` methods return the slot index. A particle is in contact when its center is closer than `particleRadius` to a shape's surface.

```ts
const walls = new PrimitiveSet(particles);
walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
const paddle = walls.addBox(new Vector3(), new Vector3(0.2, 0.05, 0.01));
walls.attach(paddle, paddleMesh); // paddle now follows paddleMesh
const loop = new SimLoop(particles, { colliders: [walls] }); // GPU slots allocated here
```

### Constructor

```ts
new PrimitiveSet(particles: ParticleSystem, options?: { capacity?: number })
```

| Parameter          | Type                                         | Default                                                          | Description                                                                                                              |
| ------------------ | -------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `particles`        | [`ParticleSystem`](./core.md#particlesystem) | —                                                                | Particles to collide.                                                                                                    |
| `options.capacity` | `number`                                     | primitives added before the GPU buffers are allocated, minimum 1 | GPU slots. Buffers are allocated when a `SimLoop` is built with this set; `add*` throws once this many primitives exist. |

| Throws                                              | When                                               |
| --------------------------------------------------- | -------------------------------------------------- |
| `PrimitiveSet: capacity must be a positive integer` | `capacity` is given and is not a positive integer. |

### PrimitiveOptions

Accepted by every `add*` method.

| Option     | Type      | Default     | Description                                                                                                                 |
| ---------- | --------- | ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| `muS`      | `number`  | `0.5`       | Static friction coefficient.                                                                                                |
| `muK`      | `number`  | `0.4`       | Kinetic friction coefficient.                                                                                               |
| `velocity` | `Vector3` | `(0, 0, 0)` | Surface velocity for friction, m/s (a conveyor belt). The primitive stays put. Replaced every step for attached primitives. |

### SolidPrimitiveOptions

Extends [`PrimitiveOptions`](#primitiveoptions). Accepted by `addSphere`.

| Option   | Type      | Default | Description                                         |
| -------- | --------- | ------- | --------------------------------------------------- |
| `invert` | `boolean` | `false` | Keep particles inside the shape instead of outside. |

### BoxOptions

Extends [`SolidPrimitiveOptions`](#solidprimitiveoptions). Accepted by `addBox`.

| Option     | Type         | Default  | Description                             |
| ---------- | ------------ | -------- | --------------------------------------- |
| `rotation` | `Quaternion` | identity | Box orientation. Copied and normalized. |

### Properties

| Property    | Type                                         | Access    | Description                                                             |
| ----------- | -------------------------------------------- | --------- | ----------------------------------------------------------------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | read-only | Particles this set acts on.                                             |
| `count`     | `number`                                     | read-only | Number of primitives added.                                             |
| `version`   | `number`                                     | read-only | Incremented on every GPU upload so renderers can refresh cached shapes. |

### Methods

#### `addPlane(normal, point, options?)`

```ts
addPlane(normal: Vector3, point: Vector3, options?: PrimitiveOptions): number
```

Adds an infinite plane through `point`. Particles stay on the side `normal` points to. `normal` is normalized. Returns the slot.

#### `addSphere(center, radius, options?)`

```ts
addSphere(center: Vector3, radius: number, options?: SolidPrimitiveOptions): number
```

Adds a sphere. `radius` in m. Returns the slot.

#### `addBox(center, halfExtents, options?)`

```ts
addBox(center: Vector3, halfExtents: Vector3, options?: BoxOptions): number
```

Adds an oriented box. `halfExtents` in m. Returns the slot.

#### `addCapsule(a, b, radius, options?)`

```ts
addCapsule(a: Vector3, b: Vector3, radius: number, options?: PrimitiveOptions): number
```

Adds a capsule around segment `a`–`b`. `radius` in m. Returns the slot. Cannot be inverted.

Errors thrown by the `add*` methods:

| Throws                                                                | When                                                                                                 |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `PrimitiveSet.addPlane: normal must be non-zero`                      | `normal` has zero length.                                                                            |
| `PrimitiveSet.addSphere: radius must be positive`                     | `radius` is not a finite number > 0.                                                                 |
| `PrimitiveSet.addBox: halfExtents must be positive`                   | Any component of `halfExtents` is not > 0.                                                           |
| `PrimitiveSet.addBox: rotation must be a finite, non-zero quaternion` | `rotation` has zero length or is not finite.                                                         |
| `PrimitiveSet.addCapsule: radius must be positive`                    | `radius` is not a finite number > 0.                                                                 |
| `PrimitiveSet: friction coefficients must be non-negative`            | `muS` or `muK` is negative or `NaN`.                                                                 |
| `PrimitiveSet: capacity N is full`                                    | Every slot is used: `capacity` primitives were added, or, without `capacity`, the GPU buffers exist. |
| `PrimitiveSet has been disposed`                                      | Called after `dispose()`.                                                                            |

#### `attach(slot, object)`

```ts
attach(slot: number, object: Object3D): void
```

Makes the primitive follow `object`'s world position and orientation. Immediately moves the primitive's reference point (plane point, sphere or box center, capsule midpoint) to the object's world position, discarding the one passed to `add*`. A box takes the object's world orientation, replacing its `rotation` option. A plane's normal and a capsule's axis keep their current direction and turn with the object from then on. Object scale is ignored. Attaching an already-attached slot replaces the previous object.

On every `update`, the primitive's velocity is set to its displacement divided by `dt`, and its angular velocity (about the reference point) to its turn since the last update divided by `dt`. Friction uses the surface velocity `v + ω × r`. Within a frame, the solver sweeps the primitive from its previous placement to the new one at that constant velocity and angular velocity.

| Throws                               | When                      |
| ------------------------------------ | ------------------------- |
| `PrimitiveSet: no primitive in slot` | `slot` does not exist.    |
| `PrimitiveSet has been disposed`     | Called after `dispose()`. |

#### `setSphere(slot, center, radius, velocity?)`

```ts
setSphere(slot: number, center: Vector3, radius: number, velocity?: Vector3): void
```

Moves and resizes a sphere. `velocity` (m/s, default `(0, 0, 0)`) is used for friction and for sweeping the sphere across the frame's substeps. Friction coefficients and `invert` are kept. Uploaded on the next `update`.

| Throws                                            | When                                 |
| ------------------------------------------------- | ------------------------------------ |
| `PrimitiveSet: no primitive in slot`              | `slot` does not exist.               |
| `PrimitiveSet.setSphere: slot N is not a sphere`  | The slot holds another kind.         |
| `PrimitiveSet.setSphere: radius must be positive` | `radius` is not a finite number > 0. |
| `PrimitiveSet has been disposed`                  | Called after `dispose()`.            |

#### `update(dt)`

```ts
update(dt: number): void
```

Reads attached objects' world transforms (skipped when `dt ≤ 0`) and uploads changed primitives to the GPU. `SimLoop.step` calls it with the step length in s. Does nothing after `dispose()`.

#### `dispose()`

```ts
dispose(): void
```

Removes all attachments, releases the kernels' pipelines and bind groups, and drops the GPU buffers. three.js has no call to free a storage buffer, so their memory is reclaimed when they are garbage-collected, after the `SimLoop` using the set is gone. Later calls other than `update` and `dispose` throw `PrimitiveSet has been disposed`.

## SDFCollider

Collision with a mesh through a signed distance field ([`SDFData`](#sdfdata)). The field is uploaded as a half-float 3D texture and sampled with trilinear filtering. A particle is in contact when its center is closer than `particleRadius + thickness` to the surface.

```ts
const sdf = bakeMeshToSdf(mesh.geometry, { resolution: 64, padding: 0.05 }); // geometry-local space
const collider = new SDFCollider(particles, sdf, { muS: 0.6, muK: 0.5 });
collider.setTransform(mesh.matrixWorld); // place in the world
```

### Constructor

```ts
new SDFCollider(particles: ParticleSystem, sdf: SDFData, options?: SDFColliderOptions)
```

| Parameter   | Type                                         | Description                                                              |
| ----------- | -------------------------------------------- | ------------------------------------------------------------------------ |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particles to collide.                                                    |
| `sdf`       | [`SDFData`](#sdfdata)                        | Field in the mesh's local space. Converted to half floats; not retained. |
| `options`   | [`SDFColliderOptions`](#sdfcollideroptions)  | See below.                                                               |

| Throws                                                                    | When                                                  |
| ------------------------------------------------------------------------- | ----------------------------------------------------- |
| `SDFCollider: resolution must be integers ≥ 4`                            | Any `sdf.resolution` component is not an integer ≥ 4. |
| `SDFCollider: N values do not match resolution`                           | `sdf.data.length` ≠ `resX × resY × resZ`.             |
| `SDFCollider: thickness must be non-negative`                             | `thickness` is negative or not finite.                |
| `SDFCollider: friction coefficients must be non-negative`                 | `muS` or `muK` is negative or `NaN`.                  |
| `SDFCollider.setRotation: rotation must be a finite, non-zero quaternion` | `rotation` has zero length or is not finite.          |
| `SDFCollider.setScale: scale must be positive`                            | `scale` is not a finite number > 0.                   |

### SDFColliderOptions

| Option      | Type         | Default     | Description                                                           |
| ----------- | ------------ | ----------- | --------------------------------------------------------------------- |
| `muS`       | `number`     | `0.5`       | Static friction coefficient.                                          |
| `muK`       | `number`     | `0.4`       | Kinetic friction coefficient.                                         |
| `position`  | `Vector3`    | `(0, 0, 0)` | World position of the field's local origin, m.                        |
| `rotation`  | `Quaternion` | identity    | Orientation about the local origin. Normalized.                       |
| `scale`     | `number`     | `1`         | Uniform scale about the local origin.                                 |
| `thickness` | `number`     | `0`         | Contact distance added to `particleRadius`, m. Not scaled by `scale`. |

### Properties

| Property    | Type                                         | Access     | Description                                                                                                                                                                              |
| ----------- | -------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | read-only  | Particles this collider acts on.                                                                                                                                                         |
| `texture`   | `Data3DTexture`                              | read-only  | The field: `RedFormat`, `HalfFloatType`, linear filtering, clamp-to-edge wrapping.                                                                                                       |
| `muS`       | `number`                                     | read/write | Static friction coefficient. Setter throws `SDFCollider: friction coefficients must be non-negative`.                                                                                    |
| `muK`       | `number`                                     | read/write | Kinetic friction coefficient. Setter throws `SDFCollider: friction coefficients must be non-negative`.                                                                                   |
| `thickness` | `number`                                     | read/write | Contact distance added to `particleRadius`, m. Takes effect on the next step. Not scaled by `scale`. Setter throws `SDFCollider: thickness must be non-negative`.                        |
| `version`   | `number`                                     | read-only  | Incremented whenever position, rotation, or scale changes value, so renderers can refresh anything cached against the placement. Setting the same placement again does not increment it. |
| `position`  | `Vector3`                                    | read-only  | World position of the local origin, m. Returns a copy; use `setPosition`.                                                                                                                |
| `rotation`  | `Quaternion`                                 | read-only  | Orientation. Returns a copy; use `setRotation`.                                                                                                                                          |
| `scale`     | `number`                                     | read-only  | Uniform scale. Use `setScale`.                                                                                                                                                           |

### Methods

#### `setPosition(position)`

```ts
setPosition(position: Vector3): void
```

Sets the world position of the local origin, m. Takes effect on the next step.

#### `setRotation(rotation)`

```ts
setRotation(rotation: Quaternion): void
```

Sets the orientation. `rotation` is normalized.

| Throws                                                                    | When                                         |
| ------------------------------------------------------------------------- | -------------------------------------------- |
| `SDFCollider.setRotation: rotation must be a finite, non-zero quaternion` | `rotation` has zero length or is not finite. |

#### `setScale(scale)`

```ts
setScale(scale: number): void
```

Sets the uniform scale. Distances scale with it; `thickness` does not, since it belongs to the particles.

| Throws                                         | When                                |
| ---------------------------------------------- | ----------------------------------- |
| `SDFCollider.setScale: scale must be positive` | `scale` is not a finite number > 0. |

#### `setTransform(transform)`

```ts
setTransform(transform: Matrix4): void
```

Sets position, rotation, and scale from a matrix, e.g. `mesh.matrixWorld`.

| Throws                                                                    | When                                                     |
| ------------------------------------------------------------------------- | -------------------------------------------------------- |
| `SDFCollider.setTransform: scale must be uniform`                         | Scale axes differ by more than 1e-4 relative to x.       |
| `SDFCollider.setScale: scale must be positive`                            | The decomposed scale is not > 0, e.g. a mirrored matrix. |
| `SDFCollider.setRotation: rotation must be a finite, non-zero quaternion` | The matrix is degenerate (zero scale).                   |

#### `update()`

```ts
update(): void
```

No-op. Placement changes are written straight to uniforms.

#### `dispose()`

```ts
dispose(): void
```

Disposes `texture` and releases the kernels' pipelines and bind groups.

## SDFData

A signed distance field on a regular grid, in the mesh's local space. Negative inside. Produced by [`bakeMeshToSdf`](#bakemeshtosdf) and [`decodeSdfBinary`](#decodesdfbinary); consumed by `SDFCollider`, `sampleSdf`, `encodeSdfBinary`, and [`voxelize`](./softbody-system.md#voxelize).

```ts
interface SDFData {
  readonly data: Float32Array;
  readonly resolution: readonly [number, number, number];
  readonly origin: readonly [number, number, number];
  readonly voxelSize: readonly [number, number, number];
}
```

| Field        | Type                       | Description                                                                                                                             |
| ------------ | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `data`       | `Float32Array`             | Signed distance per voxel, m. Index `x + y·resX + z·resX·resY`.                                                                         |
| `resolution` | `[number, number, number]` | Voxel count per axis.                                                                                                                   |
| `origin`     | `[number, number, number]` | Outer corner of voxel `(0, 0, 0)`, local space, m. Voxel `(i, j, k)` is centered at `origin + (i + 0.5, j + 0.5, k + 0.5) · voxelSize`. |
| `voxelSize`  | `[number, number, number]` | Voxel edge length per axis, m.                                                                                                          |

## bakeMeshToSdf

```ts
bakeMeshToSdf(mesh: BufferGeometry | TriangleMesh, options: BakeOptions): SDFData
```

Bakes a closed triangle mesh into an [`SDFData`](#sdfdata) on the CPU, synchronously. Uses the geometry's local vertex positions; object transforms are not applied. A `BufferGeometry` is read from its `position` attribute and index (non-indexed geometry is a triangle soup; `drawRange` and groups are ignored).

The grid is cubic: `resolution³` voxels, side length equal to the longest axis of the mesh bounds plus `2 × padding`, centered on the bounds. Each voxel stores the distance from its center to the nearest triangle, signed by a majority vote of ray casts along +X, +Y, and +Z (started a 10⁻⁴-voxel sideways offset from the center, so they don't run along the edges of meshes symmetric about the grid).

| Parameter | Type                                                              | Description                     |
| --------- | ----------------------------------------------------------------- | ------------------------------- |
| `mesh`    | `BufferGeometry` \| [`TriangleMesh`](./extending.md#trianglemesh) | Closed, manifold triangle mesh. |
| `options` | [`BakeOptions`](#bakeoptions)                                     | See below.                      |

### BakeOptions

| Option       | Type     | Default                             | Description                                                                                                                                                                      |
| ------------ | -------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resolution` | `number` | required                            | Voxels per axis. Integer ≥ 4.                                                                                                                                                    |
| `padding`    | `number` | two voxels (one below resolution 8) | Space added around the mesh bounds on every side, m. Outside the grid the field reads its edge voxels, so keep this above the contact distance (`particleRadius` + `thickness`). |

| Throws                                                        | When                                                                    |
| ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `bakeMeshToSdf: resolution must be an integer ≥ 4`            | Invalid `resolution`.                                                   |
| `bakeMeshToSdf: padding must be a non-negative finite number` | Invalid `padding`.                                                      |
| `bakeMeshToSdf: indices.length must be a multiple of 3`       | Non-indexed geometry whose vertex count is not a multiple of 3.         |
| `bakeMeshToSdf: mesh has zero triangles`                      | No triangles.                                                           |
| `bakeMeshToSdf: index N out of range`                         | A geometry index exceeds the vertex count.                              |
| `TriangleMesh: ...`                                           | A `TriangleMesh` argument has malformed arrays or out-of-range indices. |
| `bakeMeshToSdf: mesh appears non-watertight`                  | More than 1% of voxels got disagreeing axis votes.                      |

## sampleSdf

```ts
sampleSdf(sdf: SDFData, x: number, y: number, z: number): number
```

Trilinearly interpolated distance at `(x, y, z)` in the field's local space, m. Uses the same voxel-center convention as the GPU. Points outside the grid read the nearest edge values. Reads the `Float32Array` directly, so results differ from the GPU's half-float texture by rounding.

| Throws                                       | When                                                  |
| -------------------------------------------- | ----------------------------------------------------- |
| `sampleSdf: resolution must be integers ≥ 2` | Any `sdf.resolution` component is not an integer ≥ 2. |

## sampleSdfGradient

```ts
sampleSdfGradient(sdf: SDFData, x: number, y: number, z: number): [number, number, number]
```

Central-difference gradient at `(x, y, z)` in local space, with a step of one voxel per axis (six `sampleSdf` calls, the GPU's pattern). Not normalized; `|∇φ| ≈ 1` away from the medial axis and grid edges.

## encodeSdfBinary

```ts
encodeSdfBinary(sdf: SDFData): ArrayBuffer
```

Serializes a field to the `.sdf.bin` format (little-endian, 48-byte header followed by `resX·resY·resZ` float32 values in `data` order):

| Offset | Size       | Field                 |
| ------ | ---------- | --------------------- |
| 0      | 4          | Magic `"PSDF"`        |
| 4      | 4          | Version, u32 = `1`    |
| 8      | 12         | `resolution`, 3 × u32 |
| 20     | 4          | Reserved, u32 = `0`   |
| 24     | 12         | `origin`, 3 × f32     |
| 36     | 12         | `voxelSize`, 3 × f32  |
| 48     | 4 × voxels | `data`, f32           |

| Throws                                                       | When                                  |
| ------------------------------------------------------------ | ------------------------------------- |
| `encodeSdfBinary: resolution ... does not match data length` | `data.length` ≠ `resX × resY × resZ`. |

## decodeSdfBinary

```ts
decodeSdfBinary(buffer: ArrayBuffer): SDFData
```

Parses a `.sdf.bin` buffer. Voxel data is copied out of `buffer`. Trailing bytes are ignored. `SDFCollider` additionally requires each resolution ≥ 4.

```ts
const sdf = decodeSdfBinary(await(await fetch('/bunny.sdf.bin')).arrayBuffer());
```

| Throws                                         | When                                                   |
| ---------------------------------------------- | ------------------------------------------------------ |
| `decodeSdfBinary: buffer too small for header` | Fewer than 48 bytes.                                   |
| `decodeSdfBinary: bad magic`                   | First 4 bytes are not `"PSDF"`.                        |
| `decodeSdfBinary: unsupported version`         | Version is not `1`.                                    |
| `decodeSdfBinary: resolution must be ≥ 2`      | Any resolution component is below 2.                   |
| `decodeSdfBinary: voxelSize must be positive`  | Any voxel size is not a finite number > 0.             |
| `decodeSdfBinary: buffer too small (`          | Fewer bytes than the header plus `4 × resX·resY·resZ`. |

## Limitations

- Every collider acts on every particle in its `ParticleSystem` with inverse mass > 0; there is no per-range or per-material filtering.
- Colliders are fixed when the `SimLoop` is constructed; none can be added or removed afterwards.
- `PrimitiveSet` tests every particle against every primitive in each solver iteration and again in the friction pass.
- `PrimitiveSet` capacity is fixed when the `SimLoop` is built; `add*` throws once it is full. No other maximum is enforced in code.
- `PrimitiveSet`'s contact buffer is `8 × particles.capacity × capacity` bytes, and its orientations and angular velocities take `16 × capacity` bytes of uniform memory each, bounded by the device's `maxStorageBufferBindingSize` and `maxUniformBufferBindingSize`.
- `PrimitiveSet` has no remove, detach, or friction setter; only spheres can be reshaped (`setSphere`); planes, boxes, and capsules move only through `attach`.
- `attach` ignores object scale. Between updates an attached primitive is assumed to move at constant velocity and turn at constant angular velocity; a turn of more than half a revolution per step is read as the shorter turn the other way.
- Only spheres and boxes can be inverted.
- `SDFCollider` supports uniform scale only; `setTransform` throws on non-uniform or mirrored scale.
- `SDFCollider` friction treats the collider as static, and placement changes are applied instantly, not swept; particles left inside are pushed toward the nearest surface, which for thin parts can be the far side.
- After construction an `SDFCollider`'s field, resolution, origin, and voxel size are fixed; position, rotation, scale, `thickness`, `muS`, and `muK` can change every step.
- Each `SDFCollider` builds its own kernels and takes 7 texture samples per particle per solver iteration, plus 7 per contacting particle in the friction pass.
- Fields are stored on the GPU as half floats: about 3 significant digits, magnitudes clamped to 65504.
- Outside its grid the field reads the edge voxels, so edge voxels closer than the contact distance cause contacts beyond the grid. The default `padding` of two voxels keeps edge voxels about 1.5 voxels out; pass a larger `padding` when the contact distance is larger than that.
- `bakeMeshToSdf` is brute force: time is O(`resolution³` × triangles), runs on the calling thread, and allocates `4 × resolution³` bytes.
- `bakeMeshToSdf` always produces a cubic grid, so short axes of an elongated mesh get the same voxel size as the longest.
- The bake's sign test requires a closed, manifold mesh; up to 1% of voxels with disagreeing votes are accepted silently.
