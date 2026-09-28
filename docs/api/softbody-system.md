[Docs](../README.md) › [API](../README.md#api-reference) › SoftbodySystem

# SoftbodySystem

Soft and rigid bodies made of particles, solved with shape matching (Müller et al. 2005; Müller & Chentanez 2011) as XPBD constraints. `voxelize` fills a closed shape with particles; `SoftbodyMesh` draws a mesh that deforms with them.

```ts
import { SoftbodySystem, SoftbodyMesh, voxelize } from 'threejs-particle-fluids';
```

- [`SoftbodySystem`](#softbodysystem-1)
- [`SoftbodyMesh`](#softbodymesh)
- [`voxelize`](#voxelize)

## SoftbodySystem

A [`Material`](./extending.md#material) that holds one or more bodies over the same [`ParticleSystem`](./core.md#particlesystem). Add it to a [`SimLoop`](./core.md#simloop)'s `materials`; bodies touch each other only when the loop's `contact` option is on. There is no separate rigid body class: a rigid body is a body with `compliance: 0` under `'global'` shape matching.

```ts
const shape = voxelize(geometry, { particleRadius: 0.015 });
const particles = new ParticleSystem(renderer, shape.count, 0.015);
particles.uploadParticles(
  Array.from({ length: shape.count }, (_, i) => ({
    position: [shape.positions[i * 3]!, shape.positions[i * 3 + 1]!, shape.positions[i * 3 + 2]!],
  })),
); // before the system: rest shape defaults to these positions
const bodies = new SoftbodySystem(particles, {
  bodies: [
    { range: { start: 0, count: shape.count }, surfaceCount: shape.surfaceCount, compliance: 1e-6 },
  ],
});
const loop = new SimLoop(particles, { materials: [bodies], contact: true });
```

### Constructor

```ts
new SoftbodySystem(particles: ParticleSystem, options: SoftbodySystemOptions)
```

| Parameter   | Type                                              | Description                                                         |
| ----------- | ------------------------------------------------- | ------------------------------------------------------------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem)      | Particle storage. Positions and `invMass` are read at construction. |
| `options`   | [`SoftbodySystemOptions`](#softbodysystemoptions) | See below.                                                          |

### SoftbodySystemOptions

| Option          | Type                                       | Default    | Description                                                                                         |
| --------------- | ------------------------------------------ | ---------- | --------------------------------------------------------------------------------------------------- |
| `bodies`        | readonly [`SoftbodyDef`](#softbodydef)`[]` | required   | The bodies. At least one.                                                                           |
| `shapeMatching` | `'global' \| 'local'`                      | `'global'` | Shape-matching mode; see [Shape matching](#shape-matching).                                         |
| `selfCollision` | `boolean`                                  | `false`    | Let a body's particles contact each other. When `false`, see [Collision groups](#collision-groups). |

### SoftbodyDef

| Field           | Type                                       | Default                       | Description                                                                                                                                                             |
| --------------- | ------------------------------------------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `range`         | [`ParticleRange`](./core.md#particlerange) | required                      | The body's particles, surface particles first. Must not overlap another body.                                                                                           |
| `surfaceCount`  | `number`                                   | `range.count`                 | Number of leading particles on the surface. Returned by [`surfaceRange`](#surfacerangeindex).                                                                           |
| `compliance`    | `number`                                   | `0`                           | Shape-matching compliance, s²/kg. 0 is rigid; see [Compliance](#compliance).                                                                                            |
| `restPositions` | `Float32Array`                             | positions uploaded to `range` | Rest shape, xyz per particle (`3 × range.count` values), m.                                                                                                             |
| `edges`         | `Uint32Array`                              | `undefined`                   | Neighbor pairs `[i0, j0, i1, j1, …]`, indices local to the body. Required for `'local'`; also used by `SoftbodyMesh` binding. Checked by the constructor in both modes. |

### SoftbodyBody

A body's settings after defaults are applied, as stored in [`bodies`](#properties).

| Field           | Type                                       | Description                                |
| --------------- | ------------------------------------------ | ------------------------------------------ |
| `range`         | [`ParticleRange`](./core.md#particlerange) | The body's particles.                      |
| `surfaceCount`  | `number`                                   | Number of leading surface particles.       |
| `restPositions` | `Float32Array`                             | Rest shape, xyz per particle, m.           |
| `restCenter`    | `readonly [number, number, number]`        | Mass-weighted center of the rest shape, m. |
| `edges`         | `Uint32Array \| undefined`                 | Neighbor pairs as given in the def.        |

### Shape matching

- `'global'`: each body fits one rotation and center to all its particles and pulls them toward the rotated rest shape. Particles are weighted by their mass at construction. Stiff; bodies wobble but bend little.
- `'local'`: each particle fits a rotation to itself and its `edges` neighbors (Müller & Chentanez 2011, §5.1), so bodies bend and fold. Tracks a per-particle orientation in `particles.rotation`. Every body needs `edges`.

Both modes write `bodyCenters` and `bodyRotations` every substep. Under `'local'` they are the best fit to the whole body, computed after the solve and not used by it.

Mass weights and rest centers are read from `invMass` at construction. A later [`setInvMass`](./core.md#setinvmassrange-invmass) changes how particles move, but not the weights used by `'global'` matching or `bodyCenters`.

### Compliance

Each particle carries its own constraint, so at a fixed compliance a body with more particles is stiffer. Scale compliance with the particle count to keep the same feel.

- `'global'`: stays nearly rigid at any compliance; compliance mostly sets how much the body wobbles. Try `1e-7` to `1e-5`.
- `'local'`: compliance sets the material. [`Simulation.addSoftbody`](./simulation.md) uses:

```ts
compliance = 10 ** (-6 + 3 * softness) * (count / 200); // softness in [0, 1]: 0 firm rubber, 1 loose jelly
```

### Collision groups

With `selfCollision: false`, [`build`](#buildcontext) keeps each body's particles from contacting each other through collision groups ([`ParticleInit.collisionGroup`](./core.md#particleinit)):

- All of the body's particles in group 0 (the default): the body gets a new group of its own.
- All in one non-zero group: kept, so bodies you put in the same group also skip each other.
- Mixed groups: throws.

With `selfCollision: true`, groups are left as uploaded.

### Properties

| Property        | Type                                         | Access    | Description                                                                                                                                      |
| --------------- | -------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `particles`     | [`ParticleSystem`](./core.md#particlesystem) | read-only | Particle storage passed to the constructor.                                                                                                      |
| `bodies`        | readonly [`SoftbodyBody`](#softbodybody)`[]` | read-only | Resolved bodies, in `options.bodies` order.                                                                                                      |
| `shapeMatching` | `'global' \| 'local'`                        | read-only | Shape-matching mode.                                                                                                                             |
| `bodyCenters`   | `StorageBufferNode<'vec4'>`                  | read-only | Current mass-weighted center of body `b` in `.xyz` of element `b`, m. Zero until the first step.                                                 |
| `bodyRotations` | `StorageBufferNode<'vec4'>`                  | read-only | Best-fit rotation of body `b` from its rest shape, as row vectors in `.xyz` of elements `3b`, `3b + 1`, `3b + 2`. Identity until the first step. |

### Methods

#### `particleRange(index)`

```ts
particleRange(index: number): ParticleRange
```

The particles of body `index`.

#### `surfaceRange(index)`

```ts
surfaceRange(index: number): ParticleRange
```

The first `surfaceCount` particles of body `index`, e.g. for [`FluidSystem.addBoundary`](./fluid-system.md#addboundaryrange-options).

#### `setCompliance(index, compliance)`

```ts
setCompliance(index: number, compliance: number): void
```

Set body `index`'s compliance, s²/kg. Takes effect on the next step.

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

[`Material`](./extending.md#material) hook, called once by the `SimLoop` constructor. Assigns [collision groups](#collision-groups) and compiles the shape-matching kernels.

### Errors

| Throws                                                          | When                                                                                        |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `SoftbodySystem: at least one body is required`                 | `bodies` is empty.                                                                          |
| `SoftbodySystem: body <b> has no particles`                     | `range.count` is 0, e.g. from a `voxelize` result with `count` 0.                           |
| `SoftbodySystem body <b>: invalid particle range`               | `range` is not integer, has `count` < 0, or exceeds capacity.                               |
| `SoftbodySystem: particle <i> belongs to two bodies`            | Two ranges overlap.                                                                         |
| `SoftbodySystem: body <b> needs <n> rest coordinates`           | `restPositions.length` ≠ `3 × range.count`.                                                 |
| `SoftbodySystem: body <b> surfaceCount <n> is out of range`     | `surfaceCount` is not an integer in `[0, range.count]`.                                     |
| `SoftbodySystem: body <b> is flat`                              | Rest shape is planar or collinear.                                                          |
| `SoftbodySystem: compliance must be ≥ 0`                        | Compliance is negative, `NaN`, or infinite (constructor or `setCompliance`).                |
| `SoftbodySystem: no body <index>`                               | `particleRange`, `surfaceRange`, or `setCompliance` with an index out of range.             |
| `SoftbodySystem: body <b> needs edges for local shape matching` | `'local'` and a body has no `edges`.                                                        |
| `SoftbodySystem: body <b> has an odd number of edge indices`    | `edges.length` is odd.                                                                      |
| `SoftbodySystem: body <b> has an invalid edge`                  | An edge index ≥ `range.count`, or `i === j`.                                                |
| `SoftbodySystem: body <b> has mixed collision groups`           | `selfCollision: false` and the body's particles are in different groups; thrown by `build`. |

## SoftbodyMesh

A `Mesh` whose vertices follow one body. Each vertex binds to its four nearest rest particles with inverse-distance weights; the vertex shader blends their rotations by dual quaternions.

```ts
import { SoftbodyMesh } from 'threejs-particle-fluids';
```

### Constructor

```ts
new SoftbodyMesh(softbody: SoftbodySystem, bodyIndex: number, geometry: BufferGeometry, material?: MeshStandardMaterial)
```

| Parameter   | Type                                  | Description                                                                                                                                                                                        |
| ----------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `softbody`  | [`SoftbodySystem`](#softbodysystem-1) | System that owns the body.                                                                                                                                                                         |
| `bodyIndex` | `number`                              | Body to follow.                                                                                                                                                                                    |
| `geometry`  | `BufferGeometry`                      | World-space geometry positioned exactly on the body's rest shape. Gains `influences` and `weights` attributes.                                                                                     |
| `material`  | `MeshStandardMaterial`                | Source of `color`, `emissive`, `roughness`, `metalness`, `side`, texture maps, and `normalScale`, copied onto a `MeshStandardNodeMaterial`. Default: color `0xd07030`, roughness 0.6, metalness 0. |

Sets `frustumCulled = false`, `castShadow = true`, `receiveShadow = true`.

### Properties

| Property    | Type                                  | Access    | Description                |
| ----------- | ------------------------------------- | --------- | -------------------------- |
| `softbody`  | [`SoftbodySystem`](#softbodysystem-1) | read-only | System that owns the body. |
| `bodyIndex` | `number`                              | read-only | Body this mesh follows.    |

### Errors

| Throws                                             | When                          |
| -------------------------------------------------- | ----------------------------- |
| `SoftbodyMesh: no body <index>`                    | `bodyIndex` is out of range.  |
| `SoftbodyMesh: geometry has no position attribute` | `geometry` has no `position`. |

## voxelize

```ts
voxelize(shape: BufferGeometry | TriangleMesh | SDFData, options: VoxelizeOptions): VoxelizeResult
```

Fill a closed shape with particles on a cubic grid of spacing `2 × particleRadius`, surface particles first. The grid spans the shape's bounding box (mesh) or the field's full extent (`SDFData`). A point is inside a mesh by ray-cast parity, and inside a field where the sampled distance is below `dilation`. Surface particles are those with fewer than six face-adjacent occupied neighbors.

Positions come out in the shape's own space, not world space: a mesh's vertex coordinates (no object transform applied), or the field's coordinates (the space of its `origin`). Transform them before uploading. A shape smaller than the grid gives `count` 0 without throwing; check it before building a body.

```ts
import { voxelize } from 'threejs-particle-fluids';
```

| Parameter | Type                                                                                                     | Description                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `shape`   | `BufferGeometry \| `[`TriangleMesh`](./extending.md#trianglemesh)`\|`[`SDFData`](./colliders.md#sdfdata) | Closed mesh or baked signed distance field. Read in its own space; see above. |
| `options` | [`VoxelizeOptions`](#voxelizeoptions)                                                                    | See below.                                                                    |

### VoxelizeOptions

| Option           | Type      | Default  | Description                                                                                     |
| ---------------- | --------- | -------- | ----------------------------------------------------------------------------------------------- |
| `particleRadius` | `number`  | required | Particle radius, m. Grid spacing is `2 × particleRadius`.                                       |
| `largestPiece`   | `boolean` | `false`  | Keep only the largest face-connected piece.                                                     |
| `dilation`       | `number`  | `0`      | `SDFData` only: also fill points up to this distance outside the surface, m. Throws for meshes. |

### VoxelizeResult

| Field          | Type           | Description                                                             |
| -------------- | -------------- | ----------------------------------------------------------------------- |
| `positions`    | `Float32Array` | xyz per particle, m, surface particles first, in the shape's own space. |
| `count`        | `number`       | Number of particles. Can be 0.                                          |
| `surfaceCount` | `number`       | Number of leading surface particles.                                    |
| `edges`        | `Uint32Array`  | Face-adjacent pairs `[i0, j0, i1, j1, …]` with `i < j`.                 |

### Errors

| Throws                                             | When                                                        |
| -------------------------------------------------- | ----------------------------------------------------------- |
| `voxelize: particleRadius must be positive`        | `particleRadius` is ≤ 0, `NaN`, or infinite.                |
| `voxelize: mesh has no vertices`                   | Mesh input is empty.                                        |
| `voxelize: dilation applies only to SDFData input` | `dilation` given with mesh input.                           |
| `voxelize: dilation must be a finite number`       | `dilation` is `NaN` or infinite.                            |
| `TriangleMesh: …`                                  | `TriangleMesh` input has malformed `vertices` or `indices`. |

## Limitations

- Upload particle positions and `invMass` before constructing `SoftbodySystem`; the rest shape, rest centers, and mass weights are read then and not refreshed by `setInvMass`.
- Bodies, rest shapes, edges, and `shapeMatching` are fixed after construction; only compliance can change.
- Rest shapes must be 3D; flat or single-line particle sets are rejected.
- `'local'` ignores mass differences within a body: its neighborhood fits assume uniform mass.
- `'local'` damps rigid spin: a body set spinning freely loses most of its angular velocity within a fraction of a second. `'global'` keeps it.
- `SoftbodyMesh` requires an identity transform; the skinning writes world positions.
- `SoftbodyMesh` binding is a CPU search costing vertices × body particles.
- Mesh input to `voxelize` must be closed; the inside test costs grid cells × triangles on the CPU.
- `voxelize` output for `SDFData` is in the field's space, not world space.
- All particles share one radius, so a body's resolution is set by `ParticleSystem.particleRadius`.
