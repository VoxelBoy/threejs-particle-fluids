[Docs](../README.md) › [API](../README.md#api-reference) › Colliders

# Colliders

Colliders are shapes that particles can't enter. Use `PrimitiveSet` for planes, spheres, boxes, and capsules, and `SDFCollider` for any closed mesh. Pass them to [`SimLoop`](./core.md#simloop)'s `colliders` option.

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

The interface that `PrimitiveSet` and `SDFCollider` implement. It's exported so you can type an array of colliders. Every collider must be built for the same `ParticleSystem` as the `SimLoop` you pass it to. Otherwise the `SimLoop` constructor throws `SimLoop: every collider must be built for the same ParticleSystem`.

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
| `update(dt)` | `(dt: number) => void`                       | Called by `SimLoop.step` before each step, with the step length in s. |
| `dispose()`  | `() => void`                                 | Releases the collider's resources.                                    |

You can't write your own collider. The interface also has an internal `buildKernels` method, and its types aren't exported.

## PrimitiveSet

A collider made of simple shapes: planes, spheres, boxes, and capsules. Use it for floors, walls, containers, and paddles. Each shape takes a slot, and each `add*` method returns its slot index. A particle touches a shape when its center is closer than `particleRadius` to the shape's surface.

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

| Parameter          | Type                                         | Default                                                    | Description                                                                                                       |
| ------------------ | -------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `particles`        | [`ParticleSystem`](./core.md#particlesystem) | —                                                          | Particles to collide.                                                                                             |
| `options.capacity` | `number`                                     | the number added before the `SimLoop` is built, at least 1 | Slots to reserve. They're allocated when a `SimLoop` is built with this set, and `add*` throws once they're full. |

| Throws                                              | When                                                   |
| --------------------------------------------------- | ------------------------------------------------------ |
| `PrimitiveSet: capacity must be a positive integer` | You passed a `capacity` that isn't a positive integer. |

### PrimitiveOptions

Accepted by every `add*` method.

| Option     | Type      | Default     | Description                                                                                                                                |
| ---------- | --------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `muS`      | `number`  | `0.5`       | Static friction coefficient.                                                                                                               |
| `muK`      | `number`  | `0.4`       | Kinetic friction coefficient.                                                                                                              |
| `velocity` | `Vector3` | `(0, 0, 0)` | Surface velocity for friction, m/s, as on a conveyor belt. The primitive itself doesn't move. Attached primitives overwrite it every step. |

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

Adds an infinite plane through `point`. Particles stay on the side that `normal` points to. `normal` doesn't need to be normalized. Returns the slot.

#### `addSphere(center, radius, options?)`

```ts
addSphere(center: Vector3, radius: number, options?: SolidPrimitiveOptions): number
```

Adds a sphere with `radius` in metres. Returns the slot.

#### `addBox(center, halfExtents, options?)`

```ts
addBox(center: Vector3, halfExtents: Vector3, options?: BoxOptions): number
```

Adds a box. `halfExtents` is half its size along each axis, in metres. Returns the slot.

#### `addCapsule(a, b, radius, options?)`

```ts
addCapsule(a: Vector3, b: Vector3, radius: number, options?: PrimitiveOptions): number
```

Adds a capsule around the segment from `a` to `b`, with `radius` in metres. Returns the slot. Capsules can't be inverted.

Errors thrown by the `add*` methods:

| Throws                                                                | When                                                                                                                    |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `PrimitiveSet.addPlane: normal must be non-zero`                      | You passed a zero-length `normal`.                                                                                      |
| `PrimitiveSet.addSphere: radius must be positive`                     | `radius` is zero, negative, or not finite.                                                                              |
| `PrimitiveSet.addBox: halfExtents must be positive`                   | A component of `halfExtents` is zero or negative.                                                                       |
| `PrimitiveSet.addBox: rotation must be a finite, non-zero quaternion` | `rotation` is all zeros or has a non-finite component.                                                                  |
| `PrimitiveSet.addCapsule: radius must be positive`                    | `radius` is zero, negative, or not finite.                                                                              |
| `PrimitiveSet: friction coefficients must be non-negative`            | `muS` or `muK` is negative or `NaN`.                                                                                    |
| `PrimitiveSet: capacity N is full`                                    | Every slot is taken. You added `capacity` primitives, or you didn't pass `capacity` and the `SimLoop` is already built. |
| `PrimitiveSet has been disposed`                                      | You called it after `dispose()`.                                                                                        |

#### `attach(slot, object)`

```ts
attach(slot: number, object: Object3D): void
```

Makes the primitive follow `object` as it moves and turns.

The primitive's reference point (the plane's point, the sphere's or box's center, or the capsule's midpoint) jumps to the object's world position right away, replacing the one you passed to `add*`. A box takes the object's world orientation, which replaces its `rotation` option. A plane's normal and a capsule's axis keep their current direction, then turn with the object from there. Object scale is ignored. If the slot is already attached, the new object replaces the old one.

On each `update`, the primitive's velocity and angular velocity are how far it moved and turned since the last update, divided by `dt`. Friction uses the surface's velocity at each contact, including the spin, so a turning paddle drags particles with it. Within a frame, the solver sweeps the primitive from its old placement to its new one at those rates.

| Throws                               | When                                |
| ------------------------------------ | ----------------------------------- |
| `PrimitiveSet: no primitive in slot` | No primitive has that `slot` index. |
| `PrimitiveSet has been disposed`     | You called it after `dispose()`.    |

#### `setSphere(slot, center, radius, velocity?)`

```ts
setSphere(slot: number, center: Vector3, radius: number, velocity?: Vector3): void
```

Moves and resizes a sphere. `velocity` is in m/s and defaults to `(0, 0, 0)`. Friction uses it, and so does the solver when it sweeps the sphere across the frame's substeps. The friction coefficients and `invert` stay as they were. The change is uploaded on the next `update`.

| Throws                                            | When                                       |
| ------------------------------------------------- | ------------------------------------------ |
| `PrimitiveSet: no primitive in slot`              | No primitive has that `slot` index.        |
| `PrimitiveSet.setSphere: slot N is not a sphere`  | The slot holds a different kind of shape.  |
| `PrimitiveSet.setSphere: radius must be positive` | `radius` is zero, negative, or not finite. |
| `PrimitiveSet has been disposed`                  | You called it after `dispose()`.           |

#### `update(dt)`

```ts
update(dt: number): void
```

Moves attached primitives to their objects, then uploads any changed primitives to the GPU. `SimLoop.step` calls it for you. When `dt` is 0 or negative, attached objects aren't read. After `dispose()`, it does nothing.

#### `dispose()`

```ts
dispose(): void
```

Removes all attachments, releases the kernels' pipelines and bind groups, and drops the GPU buffers. three.js has no call to free a storage buffer, so their memory comes back when they're garbage-collected, after the `SimLoop` using the set is gone. After this, every method except `update` and `dispose` throws `PrimitiveSet has been disposed`.

## SDFCollider

A collider shaped like any closed mesh. You first bake the mesh into a signed distance field ([`SDFData`](#sdfdata)), which the collider uploads to the GPU as a half-float 3D texture. A particle touches the mesh when its center is closer than `particleRadius + thickness` to the surface.

```ts
const sdf = bakeMeshToSdf(mesh.geometry, { resolution: 64, padding: 0.05 }); // geometry-local space
const collider = new SDFCollider(particles, sdf, { muS: 0.6, muK: 0.5 });
collider.setTransform(mesh.matrixWorld); // place in the world
```

### Constructor

```ts
new SDFCollider(particles: ParticleSystem, sdf: SDFData, options?: SDFColliderOptions)
```

| Parameter   | Type                                         | Description                                                               |
| ----------- | -------------------------------------------- | ------------------------------------------------------------------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particles to collide.                                                     |
| `sdf`       | [`SDFData`](#sdfdata)                        | Field in the mesh's local space. It's copied as half floats and not kept. |
| `options`   | [`SDFColliderOptions`](#sdfcollideroptions)  | See below.                                                                |

| Throws                                                                    | When                                                              |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `SDFCollider: resolution must be integers ≥ 4`                            | A component of `sdf.resolution` isn't an integer of at least 4.   |
| `SDFCollider: N values do not match resolution`                           | `sdf.data` doesn't hold exactly one value per voxel.              |
| `SDFCollider: thickness must be non-negative`                             | `thickness` is negative or not finite.                            |
| `SDFCollider: friction coefficients must be non-negative`                 | `muS` or `muK` is negative or `NaN`.                              |
| `SDFCollider.setRotation: rotation must be a finite, non-zero quaternion` | The `rotation` option is all zeros or has a non-finite component. |
| `SDFCollider.setScale: scale must be positive`                            | The `scale` option is zero, negative, or not finite.              |

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

| Property    | Type                                         | Access     | Description                                                                                                                                                                        |
| ----------- | -------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | read-only  | Particles this collider acts on.                                                                                                                                                   |
| `texture`   | `Data3DTexture`                              | read-only  | The field: `RedFormat`, `HalfFloatType`, linear filtering, clamp-to-edge wrapping.                                                                                                 |
| `muS`       | `number`                                     | read/write | Static friction coefficient. Setter throws `SDFCollider: friction coefficients must be non-negative`.                                                                              |
| `muK`       | `number`                                     | read/write | Kinetic friction coefficient. Setter throws `SDFCollider: friction coefficients must be non-negative`.                                                                             |
| `thickness` | `number`                                     | read/write | Contact distance added to `particleRadius`, m. Takes effect on the next step. Not scaled by `scale`. Setter throws `SDFCollider: thickness must be non-negative`.                  |
| `version`   | `number`                                     | read-only  | Goes up whenever the position, rotation, or scale changes, so renderers know to refresh anything cached against the placement. Setting the same placement again doesn't change it. |
| `position`  | `Vector3`                                    | read-only  | World position of the local origin, m. Returns a copy, so use `setPosition` to change it.                                                                                          |
| `rotation`  | `Quaternion`                                 | read-only  | Orientation. Returns a copy, so use `setRotation` to change it.                                                                                                                    |
| `scale`     | `number`                                     | read-only  | Uniform scale. Use `setScale` to change it.                                                                                                                                        |

### Methods

#### `setPosition(position)`

```ts
setPosition(position: Vector3): void
```

Sets the world position of the local origin, in metres. It takes effect on the next step.

#### `setRotation(rotation)`

```ts
setRotation(rotation: Quaternion): void
```

Sets the orientation. You don't need to normalize `rotation`.

| Throws                                                                    | When                                                   |
| ------------------------------------------------------------------------- | ------------------------------------------------------ |
| `SDFCollider.setRotation: rotation must be a finite, non-zero quaternion` | `rotation` is all zeros or has a non-finite component. |

#### `setScale(scale)`

```ts
setScale(scale: number): void
```

Sets the uniform scale. The field's distances scale with it. `thickness` doesn't, because it belongs to the particles rather than the mesh.

| Throws                                         | When                                      |
| ---------------------------------------------- | ----------------------------------------- |
| `SDFCollider.setScale: scale must be positive` | `scale` is zero, negative, or not finite. |

#### `setTransform(transform)`

```ts
setTransform(transform: Matrix4): void
```

Sets position, rotation, and scale from a matrix, such as `mesh.matrixWorld`.

| Throws                                                            | When                                                                            |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `SDFCollider.setTransform: scale must be uniform`                 | The matrix scales the axes by different amounts (more than 1e-4 relative to x). |
| `SDFCollider.setTransform: scale must be non-zero`                | The matrix scales the mesh to nothing.                                          |
| `SDFCollider.setTransform: mirrored transforms aren’t supported`  | The matrix flips the mesh, for example with a negative scale.                   |
| `SDFCollider.setTransform: the matrix has NaN or infinite values` | The matrix contains `NaN` or infinite values.                                   |

#### `update()`

```ts
update(): void
```

Does nothing, because placement changes go straight to the GPU.

#### `dispose()`

```ts
dispose(): void
```

Disposes `texture` and releases the kernels' pipelines and bind groups.

## SDFData

A grid of distances to a mesh's surface, in the mesh's local space. Distances are negative inside the mesh. You get one from [`bakeMeshToSdf`](#bakemeshtosdf) or [`decodeSdfBinary`](#decodesdfbinary). `SDFCollider`, `sampleSdf`, `encodeSdfBinary`, and [`voxelize`](./softbody-system.md#voxelize) take one.

```ts
interface SDFData {
  readonly data: Float32Array;
  readonly resolution: readonly [number, number, number];
  readonly origin: readonly [number, number, number];
  readonly voxelSize: readonly [number, number, number];
}
```

| Field        | Type                       | Description                                                                                                                               |
| ------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `data`       | `Float32Array`             | Signed distance per voxel, m. Voxel `(x, y, z)` is at index `x + y·resX + z·resX·resY`.                                                   |
| `resolution` | `[number, number, number]` | Voxel count per axis.                                                                                                                     |
| `origin`     | `[number, number, number]` | Outer corner of voxel `(0, 0, 0)` in local space, m. Voxel `(i, j, k)` is centered at `origin + (i + 0.5, j + 0.5, k + 0.5) · voxelSize`. |
| `voxelSize`  | `[number, number, number]` | Voxel edge length per axis, m.                                                                                                            |

## bakeMeshToSdf

```ts
bakeMeshToSdf(mesh: BufferGeometry | TriangleMesh, options: BakeOptions): SDFData
```

Turns a closed triangle mesh into an [`SDFData`](#sdfdata) on the CPU, blocking until it's done. It uses the geometry's own vertex positions, so the object's transform isn't applied. A `BufferGeometry` is read from its `position` attribute and index. Without an index, every three vertices make a triangle. `drawRange` and groups are ignored.

The grid is a cube with `resolution` voxels along each side. It's centered on the mesh's bounding box, and each side is the box's longest axis plus `padding` at both ends. Each voxel stores the distance from its center to the nearest triangle. Whether it's inside is decided by a majority vote of rays cast along +X, +Y, and +Z. The rays start 10⁻⁴ voxel to the side of the center, so they don't run along the edges of meshes that are symmetric about the grid.

| Parameter | Type                                                              | Description                     |
| --------- | ----------------------------------------------------------------- | ------------------------------- |
| `mesh`    | `BufferGeometry` \| [`TriangleMesh`](./extending.md#trianglemesh) | Closed, manifold triangle mesh. |
| `options` | [`BakeOptions`](#bakeoptions)                                     | See below.                      |

### BakeOptions

| Option       | Type     | Default                             | Description                                                                                                                                                                            |
| ------------ | -------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resolution` | `number` | required                            | Voxels per axis. Integer ≥ 4.                                                                                                                                                          |
| `padding`    | `number` | two voxels (one below resolution 8) | Space added around the mesh bounds on every side, m. Outside the grid the field reads its edge voxels, so keep this larger than the contact distance (`particleRadius` + `thickness`). |

| Throws                                                        | When                                                                                                                    |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `bakeMeshToSdf: resolution must be an integer ≥ 4`            | `resolution` isn't an integer of at least 4.                                                                            |
| `bakeMeshToSdf: padding must be a non-negative finite number` | `padding` is negative or not finite.                                                                                    |
| `bakeMeshToSdf: mesh has zero triangles`                      | The mesh has no triangles.                                                                                              |
| `toTriangleMesh: ...`                                         | `mesh` isn't a `BufferGeometry` or `TriangleMesh`, or the geometry has no `position` attribute with 3 components.       |
| `TriangleMesh: ...`                                           | An array length isn't a multiple of 3, or an index points past the last vertex. This applies to a `BufferGeometry` too. |
| `bakeMeshToSdf: mesh appears non-watertight`                  | More than 1% of voxels got conflicting votes, usually because the mesh has holes.                                       |

## sampleSdf

```ts
sampleSdf(sdf: SDFData, x: number, y: number, z: number): number
```

Returns the trilinearly interpolated distance at `(x, y, z)` in the field's local space, in metres. It places voxel centers the same way the GPU does, and points outside the grid read the nearest edge values. Results differ slightly from the GPU, which reads half floats instead of the `Float32Array`.

| Throws                                       | When                                                            |
| -------------------------------------------- | --------------------------------------------------------------- |
| `sampleSdf: resolution must be integers ≥ 2` | A component of `sdf.resolution` isn't an integer of at least 2. |

## sampleSdfGradient

```ts
sampleSdfGradient(sdf: SDFData, x: number, y: number, z: number): [number, number, number]
```

Returns the field's gradient at `(x, y, z)` in local space. It compares samples one voxel to either side on each axis, which takes six `sampleSdf` calls, as on the GPU. The result isn't normalized. Its length is close to 1, except near the grid edges and at points equally close to two parts of the surface.

## encodeSdfBinary

```ts
encodeSdfBinary(sdf: SDFData): ArrayBuffer
```

Serializes a field to the `.sdf.bin` format. The format is little-endian, with a 48-byte header followed by the `resX·resY·resZ` float32 values in `data` order.

| Offset | Size       | Field                 |
| ------ | ---------- | --------------------- |
| 0      | 4          | Magic `"PSDF"`        |
| 4      | 4          | Version, u32 = `1`    |
| 8      | 12         | `resolution`, 3 × u32 |
| 20     | 4          | Reserved, u32 = `0`   |
| 24     | 12         | `origin`, 3 × f32     |
| 36     | 12         | `voxelSize`, 3 × f32  |
| 48     | 4 × voxels | `data`, f32           |

| Throws                                                       | When                                                 |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| `encodeSdfBinary: resolution ... does not match data length` | `sdf.data` doesn't hold exactly one value per voxel. |

## decodeSdfBinary

```ts
decodeSdfBinary(buffer: ArrayBuffer): SDFData
```

Parses a `.sdf.bin` buffer. The voxel data is copied out of `buffer`, and trailing bytes are ignored. This accepts resolutions as low as 2, but `SDFCollider` needs at least 4.

```ts
const sdf = decodeSdfBinary(await(await fetch('/bunny.sdf.bin')).arrayBuffer());
```

| Throws                                         | When                                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| `decodeSdfBinary: buffer too small for header` | The buffer is shorter than the 48-byte header.                                       |
| `decodeSdfBinary: bad magic`                   | The first 4 bytes aren't `"PSDF"`.                                                   |
| `decodeSdfBinary: unsupported version`         | The version isn't `1`.                                                               |
| `decodeSdfBinary: resolution must be ≥ 2`      | A resolution component is below 2.                                                   |
| `decodeSdfBinary: voxelSize must be positive`  | A voxel size is zero, negative, or not finite.                                       |
| `decodeSdfBinary: buffer too small (`          | The buffer is shorter than the header plus `4 × resX·resY·resZ` bytes of voxel data. |

## Limitations

- A collider acts on every particle in its `ParticleSystem` with an inverse mass above 0. You can't limit it to some objects or materials.
- You can't add or remove colliders after the `SimLoop` is constructed. To take an obstacle out of play, move it out of the way.
- `PrimitiveSet` tests every particle against every primitive in each solver iteration, and again in the friction pass.
- A `PrimitiveSet`'s capacity can't grow after the `SimLoop` is built. Beyond that, only the device limits it. The contact buffer takes `8 × particles.capacity × capacity` bytes and must fit in `maxStorageBufferBindingSize`. Orientations and angular velocities each take `16 × capacity` bytes and must fit in `maxUniformBufferBindingSize`.
- You can't remove a primitive, detach it, or change its friction. Only spheres can be reshaped, with `setSphere`. Planes, boxes, and capsules can only move through `attach`.
- Between updates, an attached primitive is assumed to move and turn at a steady rate. A turn of more than half a revolution in one step is read as the shorter turn the other way.
- You can't invert a plane or a capsule, only spheres and boxes. To keep particles on the other side of a plane, flip its normal.
- `SDFCollider` supports only uniform, positive scale, so `setTransform` throws on a stretched, flattened, or mirrored matrix.
- `SDFCollider` friction treats the mesh as still, and a moved mesh jumps to its new placement instead of sweeping across substeps. Particles left inside are pushed toward the nearest surface, which for a thin part can be the far side. Keep each step's movement small compared with the thinnest part.
- You can't change an `SDFCollider`'s field after construction, including its resolution, origin, and voxel size. Position, rotation, scale, `thickness`, `muS`, and `muK` can change every step.
- Each `SDFCollider` builds its own kernels and takes 7 texture samples per particle per solver iteration, plus 7 for each touching particle in the friction pass.
- Fields are stored on the GPU as half floats, which keep about 3 significant digits and clamp magnitudes to 65504.
- Outside its grid, the field reads the edge voxels. If they're closer to the surface than the contact distance, particles beyond the grid hit an invisible wall. The default `padding` of two voxels puts edge voxel centers about 1.5 voxels from the mesh bounds, so pass a larger `padding` if the contact distance is bigger.
- `bakeMeshToSdf` checks every triangle for every voxel, so its time grows with `resolution³` times the triangle count. It runs on the calling thread and allocates `4 × resolution³` bytes. To skip it at load time, bake ahead with `encodeSdfBinary` and load the file with `decodeSdfBinary`.
- `bakeMeshToSdf` always makes a cubic grid, so an elongated mesh gets the same voxel size on its short axes as on its longest.
- The bake's inside test needs a closed, manifold mesh. If up to 1% of voxels get conflicting votes, the bake still succeeds without an error.
