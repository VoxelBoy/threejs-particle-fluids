[Docs](../README.md) › [API](../README.md#api-reference) › SoftbodySystem

# SoftbodySystem

`SoftbodySystem` simulates soft and rigid bodies made of particles. It uses shape matching (Müller et al. 2005; Müller & Chentanez 2011), solved as XPBD constraints. `voxelize` fills a closed shape with particles, and `SoftbodyMesh` draws a mesh that bends with them.

```ts
import { SoftbodySystem, SoftbodyMesh, voxelize } from 'threejs-particle-fluids';
```

- [`SoftbodySystem`](#softbodysystem-1)
- [`SoftbodyMesh`](#softbodymesh)
- [`voxelize`](#voxelize)

## SoftbodySystem

A [`Material`](./extending.md#material) that holds one or more bodies in the same [`ParticleSystem`](./core.md#particlesystem). Add it to a [`SimLoop`](./core.md#simloop)'s `materials`. Bodies only touch each other when the loop's `contact` option is on. For a rigid body, use `compliance: 0` with `'global'` shape matching.

```ts
const shape = voxelize(geometry, { particleRadius: 0.015 });
const particles = new ParticleSystem(renderer, shape.count, 0.015);
// Upload first, because the rest shape defaults to these positions.
particles.uploadParticles(
  Array.from({ length: shape.count }, (_, i) => ({
    position: [shape.positions[i * 3]!, shape.positions[i * 3 + 1]!, shape.positions[i * 3 + 2]!],
  })),
);
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
| `shapeMatching` | `'global' \| 'local'`                      | `'global'` | Shape-matching mode. See [Shape matching](#shape-matching).                                         |
| `selfCollision` | `boolean`                                  | `false`    | Let a body's particles contact each other. When `false`, see [Collision groups](#collision-groups). |

### SoftbodyDef

| Field           | Type                                       | Default                       | Description                                                                                                                                          |
| --------------- | ------------------------------------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `range`         | [`ParticleRange`](./core.md#particlerange) | required                      | The body's particles, surface particles first. Must not overlap another body.                                                                        |
| `surfaceCount`  | `number`                                   | `range.count`                 | Number of leading particles on the surface. Returned by [`surfaceRange`](#surfacerangeindex).                                                        |
| `compliance`    | `number`                                   | `0`                           | Shape-matching compliance, s²/kg. `0` is rigid. See [Compliance](#compliance).                                                                       |
| `restPositions` | `Float32Array`                             | positions uploaded to `range` | Rest shape, xyz per particle (`3 × range.count` values), m.                                                                                          |
| `edges`         | `Uint32Array`                              | `undefined`                   | Neighbor pairs `[i0, j0, i1, j1, …]`, indexed within the body. Required for `'local'`. `SoftbodyMesh` also uses them when binding. Always validated. |

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

- `'global'` fits one mass-weighted rotation and center to the whole body and pulls every particle toward the rotated rest shape. Bodies wobble but bend very little.
- `'local'` fits a rotation to each particle and its `edges` neighbors, so bodies can bend and fold. Every body needs `edges`. Each particle's orientation is stored in `particles.rotation`.

Both modes update `bodyCenters` and `bodyRotations` every substep. Under `'local'`, they're a best fit to the whole body and don't affect the solve.

The mass weights and rest centers come from `invMass` at construction. A later [`setInvMass`](./core.md#setinvmassrange-invmass) changes how particles move, but `'global'` matching and `bodyCenters` keep the original weights.

### Compliance

Every particle carries its own constraint, so at the same compliance a body with more particles is stiffer. To keep the same feel at a different particle count, scale compliance with the count.

- `'global'`: the body stays nearly rigid at any compliance, which mostly sets how much it wobbles. Try values from `1e-7` to `1e-5`.
- `'local'`: compliance sets what the material feels like. [`Simulation.addSoftbody`](./simulation.md) uses this formula:

```ts
compliance = 10 ** (-6 + 3 * softness) * (count / 200); // softness from 0 (firm rubber) to 1 (loose jelly)
```

### Collision groups

When `selfCollision` is `false`, [`build`](#buildcontext) uses collision groups ([`ParticleInit.collisionGroup`](./core.md#particleinit)) to keep a body's particles from contacting each other:

- If all of a body's particles are in group 0, the default, the body gets a new group of its own.
- If they share one non-zero group, it's kept, so bodies you put in the same group also skip each other.
- If they're in different groups, `build` throws.

When `selfCollision` is `true`, groups are left as you uploaded them.

### Properties

| Property        | Type                                         | Access    | Description                                                                                                                                        |
| --------------- | -------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `particles`     | [`ParticleSystem`](./core.md#particlesystem) | read-only | Particle storage passed to the constructor.                                                                                                        |
| `bodies`        | readonly [`SoftbodyBody`](#softbodybody)`[]` | read-only | Resolved bodies, in `options.bodies` order.                                                                                                        |
| `shapeMatching` | `'global' \| 'local'`                        | read-only | Shape-matching mode.                                                                                                                               |
| `bodyCenters`   | `StorageBufferNode<'vec4'>`                  | read-only | Current mass-weighted center of body `b` in `.xyz` of element `b`, m. Zero until the first step.                                                   |
| `bodyRotations` | `StorageBufferNode<'vec4'>`                  | read-only | Best-fit rotation of body `b` from its rest shape. Its rows are in `.xyz` of elements `3b`, `3b + 1`, and `3b + 2`. Identity until the first step. |

### Methods

#### `particleRange(index)`

```ts
particleRange(index: number): ParticleRange
```

Returns the particles of body `index`.

#### `surfaceRange(index)`

```ts
surfaceRange(index: number): ParticleRange
```

Returns the first `surfaceCount` particles of body `index`, for example to pass to [`FluidSystem.addBoundary`](./fluid-system.md#addboundaryrange-options).

#### `setCompliance(index, compliance)`

```ts
setCompliance(index: number, compliance: number): void
```

Sets body `index`'s compliance, in s²/kg. The change takes effect on the next step.

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

The [`Material`](./extending.md#material) hook, which the `SimLoop` constructor calls once. It assigns [collision groups](#collision-groups) and compiles the shape-matching kernels.

### Errors

| Throws                                                          | When                                                                                                            |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `SoftbodySystem: at least one body is required`                 | `bodies` is empty.                                                                                              |
| `SoftbodySystem: body <b> has no particles`                     | A body's `range.count` is 0, for example from a `voxelize` result with no particles.                            |
| `SoftbodySystem body <b>: invalid particle range`               | A range's `start` or `count` isn't an integer, `start` or `count` is negative, or the range runs past capacity. |
| `SoftbodySystem: particle <i> belongs to two bodies`            | Two bodies' ranges overlap.                                                                                     |
| `SoftbodySystem: body <b> needs <n> rest coordinates`           | `restPositions` doesn't hold 3 values per particle in the range.                                                |
| `SoftbodySystem: body <b> surfaceCount <n> is out of range`     | `surfaceCount` isn't a whole number from 0 to `range.count`.                                                    |
| `SoftbodySystem: body <b> is flat`                              | The rest shape is flat or a single line.                                                                        |
| `SoftbodySystem: compliance must be ≥ 0`                        | A compliance passed to the constructor or `setCompliance` is negative, `NaN`, or infinite.                      |
| `SoftbodySystem: no body <index>`                               | `particleRange`, `surfaceRange`, or `setCompliance` got an index with no body.                                  |
| `SoftbodySystem: body <b> needs edges for local shape matching` | `shapeMatching` is `'local'` and a body has no `edges`.                                                         |
| `SoftbodySystem: body <b> has an odd number of edge indices`    | `edges` has an odd length.                                                                                      |
| `SoftbodySystem: body <b> has an invalid edge`                  | An edge refers to a particle outside the body, or joins a particle to itself.                                   |
| `SoftbodySystem: body <b> has mixed collision groups`           | `selfCollision` is `false` and a body's particles are in different collision groups. `build` throws this one.   |

## SoftbodyMesh

A `Mesh` whose vertices follow one body as it bends. Each vertex follows its four nearest rest particles, weighted by inverse distance and blended with dual quaternions. If the body has `edges`, particles on one-particle-thick chains are skipped so thin parts don't tear.

```ts
import { SoftbodyMesh } from 'threejs-particle-fluids';
```

### Constructor

```ts
new SoftbodyMesh(softbody: SoftbodySystem, bodyIndex: number, geometry: BufferGeometry, material?: MeshStandardMaterial)
```

| Parameter   | Type                                  | Description                                                                                                                                                                              |
| ----------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `softbody`  | [`SoftbodySystem`](#softbodysystem-1) | System that owns the body.                                                                                                                                                               |
| `bodyIndex` | `number`                              | Body to follow.                                                                                                                                                                          |
| `geometry`  | `BufferGeometry`                      | Geometry in world space, placed exactly over the body's rest shape. The constructor adds `influences` and `weights` attributes to it.                                                    |
| `material`  | `MeshStandardMaterial`                | Copied onto a `MeshStandardNodeMaterial`: `color`, `emissive`, `roughness`, `metalness`, `side`, texture maps, and `normalScale`. Default: color `0xd07030`, roughness 0.6, metalness 0. |

The constructor sets `frustumCulled` to `false`, and `castShadow` and `receiveShadow` to `true`.

### Properties

| Property    | Type                                  | Access    | Description                |
| ----------- | ------------------------------------- | --------- | -------------------------- |
| `softbody`  | [`SoftbodySystem`](#softbodysystem-1) | read-only | System that owns the body. |
| `bodyIndex` | `number`                              | read-only | Body this mesh follows.    |

### Errors

| Throws                                             | When                                           |
| -------------------------------------------------- | ---------------------------------------------- |
| `SoftbodyMesh: no body <index>`                    | `bodyIndex` doesn't name a body in the system. |
| `SoftbodyMesh: geometry has no position attribute` | `geometry` has no `position` attribute.        |

## voxelize

```ts
voxelize(shape: BufferGeometry | TriangleMesh | SDFData, options: VoxelizeOptions): VoxelizeResult
```

Fills a closed shape with particles on a cubic grid with spacing `2 × particleRadius`, ready to become a soft body. The grid covers a mesh's bounding box or a field's full extent. A field point is inside where its distance is below `dilation`. Surface particles, those missing at least one of their six grid neighbors, come first.

Positions are in the shape's own space, not world space. That means a mesh's vertex coordinates without its object transform, or the space of a field's `origin`. Transform them before uploading. A shape smaller than the grid gives `count` 0 without throwing, so check `count` first.

```ts
import { voxelize } from 'threejs-particle-fluids';
```

| Parameter | Type                                                                                                     | Description                                                            |
| --------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `shape`   | `BufferGeometry \| `[`TriangleMesh`](./extending.md#trianglemesh)`\|`[`SDFData`](./colliders.md#sdfdata) | A closed mesh or a baked signed distance field, read in its own space. |
| `options` | [`VoxelizeOptions`](#voxelizeoptions)                                                                    | See below.                                                             |

### VoxelizeOptions

| Option           | Type      | Default  | Description                                                                                       |
| ---------------- | --------- | -------- | ------------------------------------------------------------------------------------------------- |
| `particleRadius` | `number`  | required | Particle radius, m. Grid spacing is `2 × particleRadius`.                                         |
| `largestPiece`   | `boolean` | `false`  | Keep only the largest face-connected piece.                                                       |
| `dilation`       | `number`  | `0`      | `SDFData` only. Also fills grid points up to this far outside the surface, m. Throws with a mesh. |

### VoxelizeResult

| Field          | Type           | Description                                                             |
| -------------- | -------------- | ----------------------------------------------------------------------- |
| `positions`    | `Float32Array` | xyz per particle, m, surface particles first, in the shape's own space. |
| `count`        | `number`       | Number of particles. Can be 0.                                          |
| `surfaceCount` | `number`       | Number of leading surface particles.                                    |
| `edges`        | `Uint32Array`  | Face-adjacent pairs `[i0, j0, i1, j1, …]` with `i < j`.                 |

### Errors

| Throws                                             | When                                                          |
| -------------------------------------------------- | ------------------------------------------------------------- |
| `voxelize: particleRadius must be positive`        | `particleRadius` isn't a positive finite number.              |
| `voxelize: mesh has no vertices`                   | The mesh is empty.                                            |
| `voxelize: dilation applies only to SDFData input` | You passed `dilation` with a mesh.                            |
| `voxelize: dilation must be a finite number`       | `dilation` is `NaN` or infinite.                              |
| `TriangleMesh: …`                                  | A `TriangleMesh` input has malformed `vertices` or `indices`. |

## Limitations

- Upload particle positions and `invMass` before you construct `SoftbodySystem`, because they're only read then.
- You can't change bodies, rest shapes, edges, or `shapeMatching` after construction. Only compliance can change.
- A body's rest shape must be 3D. It can't be flat or a single line.
- `'local'` matching treats every particle in a body as the same mass.
- You can't transform a `SoftbodyMesh`, because it writes world positions. Keep it at the identity.
- Binding a `SoftbodyMesh` runs on the CPU and costs vertices times body particles.
- `voxelize` can't fill a mesh with holes, because it tests each point by ray-cast parity. The test runs on the CPU and costs grid cells times triangles.
- All particles share one radius, so you can't give one body finer particles than another.
