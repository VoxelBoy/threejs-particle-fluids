[Docs](../README.md) › [API](../README.md#api-reference) › ClothSystem

# ClothSystem

The low-level pieces behind [`Simulation.addCloth`](./cloth.md), which builds a rectangle from them. Use them directly for other shapes, custom pins, or settings `Simulation` doesn't expose. A cloth is a triangle mesh of particles held together by stretch, bending, and long-range tether constraints. Wind pushes on each triangle with drag and lift.

```ts
import {
  ClothSystem,
  createClothGraph,
  createClothSurface,
  type ClothGraph,
  type ClothGraphOptions,
  type ClothSurfaceOptions,
  type ClothSystemOptions,
  type TetherConstraint,
} from 'threejs-particle-fluids';
```

- [`createClothGraph`](#createclothgraph) — [`ClothGraphOptions`](#clothgraphoptions), [`ClothGraph`](#clothgraph)
- [`ClothSystem`](#clothsystem-1) — [`ClothSystemOptions`](#clothsystemoptions)
- [`createClothSurface`](#createclothsurface) — [`ClothSurfaceOptions`](#clothsurfaceoptions)
- [Pinning](#pinning)
- [Bend compliance and resolution](#bend-compliance-and-resolution)
- [Particle spacing and self-collision](#particle-spacing-and-self-collision)

```ts
const segments = 40;
const geometry = new PlaneGeometry(1.2, 1.2, segments, segments).translate(0, 1, 0);
const graph = createClothGraph(geometry, {
  pinnedIndices: Array.from({ length: segments + 1 }, (_, i) => i), // top row
});

const spacing = 1.2 / segments;
const particles = new ParticleSystem(renderer, graph.positions.length, spacing / 2.2);
const cloth = new ClothSystem(particles, { graph, wind: new Vector3(0, 0, 1.5) });
const loop = new SimLoop(particles, { substeps: 8, materials: [cloth], contact: true });

scene.add(createClothSurface(cloth, { columns: segments + 1, rows: segments + 1 }));

await loop.step(1 / 60); // each frame
```

## createClothGraph

```ts
createClothGraph(geom: BufferGeometry, options?: ClothGraphOptions): ClothGraph
```

Builds the cloth's structure from an indexed triangle `BufferGeometry`, on the CPU. Positions are read as they are, with no transform applied. Any vertex within 1e-6 m of an earlier vertex is welded into it, so seams join into one piece of cloth. After welding, every vertex must belong to at least one triangle with non-zero area.

| Parameter | Type                                      | Description                                                                        |
| --------- | ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `geom`    | `BufferGeometry`                          | Indexed triangle geometry with a 3-component `position` attribute, in world space. |
| `options` | [`ClothGraphOptions`](#clothgraphoptions) | See below.                                                                         |

| Throws                                                              | When                                                                                                   |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `createClothGraph: surfaceDensity must be a positive finite number` | `surfaceDensity` isn't a positive finite number.                                                       |
| `createClothGraph: input BufferGeometry must be indexed`            | `geom` has no index. Convert it with `mergeVertices` from `three/addons/utils/BufferGeometryUtils.js`. |
| `createClothGraph: input BufferGeometry has no position attribute`  | `geom` has no `position` attribute.                                                                    |
| `createClothGraph: position attribute itemSize must be 3`           | The `position` attribute doesn't have 3 components.                                                    |
| `createClothGraph: index count … is not a multiple of 3`            | The index count isn't a multiple of 3.                                                                 |
| `createClothGraph: pinnedIndices[k] is …, not a vertex index`       | A pinned index isn't an integer from 0 to `positions.length − 1` in the welded vertex list.            |
| `createClothGraph: vertex … is in no non-degenerate triangle`       | A welded vertex isn't in any triangle, or only in zero-area ones, so it would have no mass.            |

### ClothGraphOptions

| Option           | Type                | Default | Description                                                                                                                                         |
| ---------------- | ------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `surfaceDensity` | `number`            | `0.2`   | Mass per area, kg/m².                                                                                                                               |
| `pinnedIndices`  | `readonly number[]` | `[]`    | Vertices to hold in place (`invMass = 0`), as indices into the welded vertex list ([`ClothGraph.positions`](#clothgraph)). See [Pinning](#pinning). |

### ClothGraph

`createClothGraph` returns this. All indices are local to the cloth, from 0 to `positions.length − 1`.

| Property              | Type                                                                                             | Description                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `positions`           | `readonly (readonly [number, number, number])[]`                                                 | Welded vertex positions, m, one per particle, in the order each first appears.                                                                                          |
| `invMass`             | `readonly number[]`                                                                              | Inverse mass, kg⁻¹. `0` for pinned vertices. Otherwise a vertex's mass is `surfaceDensity` times a third of the total area of its triangles.                            |
| `distancePairs`       | `readonly (readonly [number, number])[]`                                                         | Every unique triangle edge `[i, j]`, `i < j`. One stretch constraint each.                                                                                              |
| `distanceRestLengths` | `readonly number[]`                                                                              | Rest length of each `distancePairs` entry, m.                                                                                                                           |
| `triangles`           | `readonly (readonly [number, number, number])[]`                                                 | Non-degenerate triangles, in input order. Used for wind forces.                                                                                                         |
| `bendingTuples`       | `readonly (readonly [number, number, number, number])[]`                                         | `[p1, p2, p3, p4]` for each edge shared by exactly two triangles. `p1` and `p2` are the edge, and `p3` and `p4` are the opposite vertices. One bending constraint each. |
| `bendingRestAngles`   | `readonly number[]`                                                                              | Signed rest angle between the two triangles of each tuple, in radians, in (−π, π]. `0` is flat.                                                                         |
| `diagnostics`         | `{ boundaryEdges: number; nonManifoldEdgesSkipped: number; degenerateTrianglesSkipped: number }` | Counts of edges with one triangle, edges with three or more triangles (which get stretch but no bending), and triangles skipped for repeated vertices or zero area.     |

## ClothSystem

A [`Material`](./extending.md#material) that simulates a `ClothGraph` in a [`SimLoop`](./core.md#simloop). The constructor writes the graph's positions and inverse masses into the particle system, so you don't upload the cloth's particles yourself. It also sets their velocities to zero, and it keeps their collision groups.

### Constructor

```ts
new ClothSystem(particles: ParticleSystem, options: ClothSystemOptions)
```

| Parameter   | Type                                         | Description                                                                       |
| ----------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particle storage. Must have room for `offset + graph.positions.length` particles. |
| `options`   | [`ClothSystemOptions`](#clothsystemoptions)  | See below.                                                                        |

| Throws                                                                                                                           | When                                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `ClothSystem: invalid particle range`                                                                                            | `offset` isn't a non-negative integer, the graph has no vertices, or `offset + graph.positions.length` is more than `particles.capacity`. |
| `ClothSystem: stretchCompliance must be ≥ 0` (likewise `bendCompliance`, `tetherCompliance`, `stretchTolerance`, `drag`, `lift`) | You passed a negative, `NaN`, or infinite value.                                                                                          |
| `ClothSystem: damping must be between 0 and 1`                                                                                   | `damping` is outside 0 to 1, or `NaN`.                                                                                                    |

### ClothSystemOptions

| Option              | Type                        | Default     | Description                                                                                                                                                                         |
| ------------------- | --------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graph`             | [`ClothGraph`](#clothgraph) | required    | Shape, masses, and pins.                                                                                                                                                            |
| `offset`            | `number`                    | `0`         | First particle slot the cloth occupies.                                                                                                                                             |
| `stretchCompliance` | `number`                    | `1e-7`      | Compliance of the stretch constraint on each edge, s²/kg. `0` doesn't stretch at all.                                                                                               |
| `bendCompliance`    | `number`                    | `1e-5`      | Compliance of the bending constraints, rad²/(N·m). Higher values drape more loosely. See [Bend compliance and resolution](#bend-compliance-and-resolution).                         |
| `tetherCompliance`  | `number`                    | `0`         | Compliance of the tethers, s²/kg. `0` is rigid.                                                                                                                                     |
| `stretchTolerance`  | `number`                    | `0`         | Slack in each tether, as a fraction of its rest length along the cloth (`0.06` = 6%).                                                                                               |
| `wind`              | `Vector3`                   | `(0, 0, 0)` | Wind velocity, m/s. The vector is copied.                                                                                                                                           |
| `drag`              | `number`                    | `0.6125`    | Air drag coefficient `½ · C_D · ρ_air`, kg/m³, ≥ 0.                                                                                                                                 |
| `lift`              | `number`                    | `0.3`       | Air lift coefficient `½ · C_L · ρ_air`, kg/m³, ≥ 0.                                                                                                                                 |
| `damping`           | `number`                    | none (off)  | 0–1. After every substep, blends each free particle's velocity toward the average of its edge neighbors. Leave it out to skip this pass. Pass `0` to enable it and change it later. |

### Properties

| Property         | Type                                         | Access              | Description                                                                                          |
| ---------------- | -------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| `particles`      | [`ParticleSystem`](./core.md#particlesystem) | read-only           | The particle system passed to the constructor.                                                       |
| `graph`          | [`ClothGraph`](#clothgraph)                  | read-only           | The graph passed in `options`.                                                                       |
| `range`          | [`ParticleRange`](./core.md#particlerange)   | read-only           | `{ start: offset, count: graph.positions.length }`.                                                  |
| `tethers`        | `readonly TetherConstraint[]`                | read-only           | Tethers built from the pins. Empty when nothing is pinned. See below.                                |
| `wind`           | `Vector3`                                    | read-only reference | Wind velocity, m/s. Change the vector to change the wind.                                            |
| `drag`           | `number`                                     | read/write          | Air drag, kg/m³, ≥ 0.                                                                                |
| `lift`           | `number`                                     | read/write          | Air lift, kg/m³, ≥ 0.                                                                                |
| `damping`        | `number`                                     | read/write          | Velocity smoothing, 0–1. Reads `0` if the `damping` option wasn't given, and setting it then throws. |
| `bendCompliance` | `number`                                     | read/write          | Bending compliance. Changes take effect on the next step.                                            |

Each `TetherConstraint` has these fields:

| Field        | Type     | Description                                                                                                |
| ------------ | -------- | ---------------------------------------------------------------------------------------------------------- |
| `particle`   | `number` | Cloth-local index of a free particle.                                                                      |
| `anchor`     | `number` | Cloth-local index of the nearest pinned vertex in one pinned island. The tether uses its current position. |
| `restRadius` | `number` | Distance along the rest mesh from `anchor` to `particle`, times `1 + stretchTolerance`, m.                 |

| Throws                                                                         | When                                                                       |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `ClothSystem: bendCompliance must be ≥ 0` (likewise `drag`, `lift`)            | You set it to a negative, `NaN`, or infinite value.                        |
| `ClothSystem: damping must be between 0 and 1`                                 | You set `damping` outside 0 to 1.                                          |
| `ClothSystem: pass \`damping\` in the options to enable it before changing it` | You set `damping` without passing the `damping` option to the constructor. |

### Methods

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

Implements [`Material.build`](./extending.md#material), which `SimLoop` calls once when it's constructed.

## createClothSurface

```ts
createClothSurface(cloth: ClothSystem, options: ClothSurfaceOptions): Mesh
```

Returns a `Mesh` that draws a grid-shaped cloth as a smooth surface. The GPU places its vertices on a bicubic surface over the particles and computes normals from that surface. The cloth must be a row-major grid, like a `PlaneGeometry`. Particle `cloth.range.start + row × columns + column` is grid node (`column`, `row`). The mesh is a `PlaneGeometry(1, 1, (columns − 1) × subdivisions, (rows − 1) × subdivisions)` with `frustumCulled` set to `false`, and `castShadow` and `receiveShadow` set to `true`.

Particle positions become the mesh's local positions, as with `createParticleMesh`, so the mesh's world transform (its own and its parents') is applied on top. The physics never sees this transform, so keep it at the identity to draw the cloth where it's simulated.

| Parameter | Type                                          | Description                  |
| --------- | --------------------------------------------- | ---------------------------- |
| `cloth`   | [`ClothSystem`](#clothsystem-1)               | Grid-shaped cloth to follow. |
| `options` | [`ClothSurfaceOptions`](#clothsurfaceoptions) | See below.                   |

| Throws                                                                  | When                                                                                        |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `createClothSurface: a C×R grid does not match the cloth's N particles` | `columns` or `rows` is less than 2, or `columns × rows` isn't equal to `cloth.range.count`. |

### ClothSurfaceOptions

| Option         | Type                       | Default                                                 | Description                                                                                                                             |
| -------------- | -------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `columns`      | `number`                   | required                                                | Particles per row. `widthSegments + 1` for a `PlaneGeometry`.                                                                           |
| `rows`         | `number`                   | required                                                | Number of rows. `heightSegments + 1` for a `PlaneGeometry`.                                                                             |
| `subdivisions` | `number`                   | `3`                                                     | Mesh vertices per grid cell along each axis.                                                                                            |
| `material`     | `MeshPhysicalNodeMaterial` | new double-sided material, `roughness: 0.4`, `sheen: 1` | Material to draw with. Its `positionNode` and `normalNode` are overwritten, so don't share it with another surface.                     |
| `smooth`       | `boolean`                  | `true`                                                  | `true` fits a cubic B-spline near the particles, which smooths out grid-scale buckling. `false` fits a Catmull-Rom spline through them. |

## Pinning

A pinned vertex has `invMass = 0`, so the solver never moves it. Set pins with [`ClothGraphOptions.pinnedIndices`](#clothgraphoptions).

Pin indices refer to the welded vertex list. `PlaneGeometry` has no duplicate vertices, so welding doesn't change its indices. For `PlaneGeometry(w, h, sx, sy)`, vertex `i` is at row `floor(i / (sx + 1))` and column `i % (sx + 1)`, with row 0 at the top (+y). The top row is indices `0` to `sx`.

Pinned vertices joined to each other by edges form an island. Each free particle gets a tether to the nearest pinned vertex in each of its closest islands, up to 4, measured along the mesh. A tether only pulls when the particle is farther from its anchor than `restRadius`. This keeps pinned cloth from stretching under its own weight.

Tethers read their anchor's current position, so they follow pins you move at runtime. To move a pinned particle, write both `particles.positions` and `particles.predictedPositions`, because the solver reads the predicted position of pinned particles.

`Simulation`'s `pin` option chooses pins for you. `'top'` pins the whole top row, `'top-corners'` the two top corners, `'corners'` all four corners, and `'none'` nothing. See [Cloth](./cloth.md).

## Bend compliance and resolution

At the same `bendCompliance`, a finer grid or a lighter cloth resists folding more. `Simulation` keeps the drape the same at any resolution with this formula:

```ts
bendCompliance = 10 ** (-1 + 5 * softness) * (segments / 30) ** 4 * (0.35 / surfaceDensity);
```

| Term             | Meaning                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| `softness`       | `Simulation` cloth softness, 0 (stiff) to 1 (silk). Default `0.75`.              |
| `segments`       | Grid segments across the cloth's width (`widthSegments` of the `PlaneGeometry`). |
| `surfaceDensity` | Mass per area in kg/m² (`Simulation`'s `weight`, default `0.1`).                 |

The library doesn't export this formula. If you tune a `ClothSystem` at one resolution and build it at another, scale `bendCompliance` the same way.

## Particle spacing and self-collision

`ClothSystem` has no self-collision of its own. Cloth collides with itself and with other particles through the `SimLoop`'s `contact` option, which pushes apart any two particles closer than `2 × particleRadius`.

By default, cloth particles are in collision group `0`, so neighbors in the graph also contact each other. For self-collision to work, every rest edge must be longer than `2 × particleRadius`. Otherwise, neighbors are always in contact and push the sheet apart.

`Simulation` spaces cloth particles `2.2 × particleRadius` apart, with `segments = round(width / (2.2 × particleRadius))`. Neighbors never touch at rest, so folds can collide.

A spacing below `2 × particleRadius` closes gaps that colliders could slip through. In that case, leave `contact` off, or put the cloth in its own group with `particles.setCollisionGroup(cloth.range, group)`. You can set the group before or after constructing the `ClothSystem`. The cloth then doesn't collide with itself.

## Limitations

- The `ClothSystem` constructor zeroes the velocities in its range, so upload any starting velocity after constructing it.
- You can't change the topology, rest lengths, rest angles, masses, tethers, `stretchCompliance`, `tetherCompliance`, or `stretchTolerance` after construction.
- Tether lengths come from the rest mesh, so you can't move pins farther apart than the rest shape allows. If you do, the cloth stretches against its tethers.
- Wind is one velocity for the whole cloth, so it can't vary across it.
- `createClothGraph` can't take non-indexed geometry. It also gives no bending constraint to an edge shared by three or more triangles.
- `createClothSurface` can only draw row-major grid cloth. Draw other shapes with `createParticleMesh`, or with your own nodes that read `particles.positions`.
- With `smooth: true`, the surface doesn't pass exactly through the particles, including those on pinned edges. Use `smooth: false` if the surface must touch them.
