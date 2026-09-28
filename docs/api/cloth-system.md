[Docs](../README.md) › [API](../README.md#api-reference) › ClothSystem

# ClothSystem

Low-level cloth: a triangle mesh of particles joined by stretch, bending, and long-range tether constraints, with per-triangle wind drag and lift. [`Simulation.addCloth`](./cloth.md) builds a rectangle with these pieces; use them directly for other shapes, custom pins, or settings `Simulation` does not expose.

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

Builds the cloth topology from an indexed triangle `BufferGeometry` on the CPU. Positions are read verbatim from the `position` attribute; no transform is applied. A vertex within 1e-6 m of an earlier vertex is welded into it, so seams become connected cloth. After welding, every vertex must be in at least one non-degenerate triangle.

| Parameter | Type                                      | Description                                                                        |
| --------- | ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `geom`    | `BufferGeometry`                          | Indexed triangle geometry with a 3-component `position` attribute, in world space. |
| `options` | [`ClothGraphOptions`](#clothgraphoptions) | See below.                                                                         |

| Throws                                                              | When                                                                                                        |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `createClothGraph: surfaceDensity must be a positive finite number` | `surfaceDensity` ≤ 0, `NaN`, or infinite.                                                                   |
| `createClothGraph: input BufferGeometry must be indexed`            | `geom.getIndex()` is `null`. Convert with `mergeVertices` from `three/addons/utils/BufferGeometryUtils.js`. |
| `createClothGraph: input BufferGeometry has no position attribute`  | No `position` attribute.                                                                                    |
| `createClothGraph: position attribute itemSize must be 3`           | `position.itemSize` ≠ 3.                                                                                    |
| `createClothGraph: index count … is not a multiple of 3`            | Index count is not a multiple of 3.                                                                         |
| `createClothGraph: pinnedIndices[k] is …, not a vertex index`       | A pinned index is not an integer in `[0, positions.length)` of the welded vertex list.                      |
| `createClothGraph: vertex … is in no non-degenerate triangle`       | A welded vertex is unused or only in zero-area triangles. It would have no mass.                            |

### ClothGraphOptions

| Option           | Type                | Default | Description                                                                                                                         |
| ---------------- | ------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `surfaceDensity` | `number`            | `0.2`   | Mass per area in kg/m².                                                                                                             |
| `pinnedIndices`  | `readonly number[]` | `[]`    | Vertices given `invMass = 0`. Indices into the welded vertex list ([`ClothGraph.positions`](#clothgraph)). See [Pinning](#pinning). |

### ClothGraph

Returned by `createClothGraph`. All indices are cloth-local (0 to `positions.length − 1`).

| Property              | Type                                                                                             | Description                                                                                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `positions`           | `readonly (readonly [number, number, number])[]`                                                 | Welded vertex positions in m, one per particle, in first-occurrence order.                                                                                  |
| `invMass`             | `readonly number[]`                                                                              | Inverse mass in kg⁻¹. `0` for pinned vertices; otherwise `1 / (surfaceDensity × ⅓ Σ incident triangle areas)`.                                              |
| `distancePairs`       | `readonly (readonly [number, number])[]`                                                         | Every unique triangle edge `[i, j]`, `i < j`. One stretch constraint each.                                                                                  |
| `distanceRestLengths` | `readonly number[]`                                                                              | Rest length of each `distancePairs` entry in m.                                                                                                             |
| `triangles`           | `readonly (readonly [number, number, number])[]`                                                 | Non-degenerate triangles, in input order. Used for wind forces.                                                                                             |
| `bendingTuples`       | `readonly (readonly [number, number, number, number])[]`                                         | `[p1, p2, p3, p4]` per edge shared by exactly two triangles: `p1, p2` the edge, `p3, p4` the opposite vertices. One bending constraint each.                |
| `bendingRestAngles`   | `readonly number[]`                                                                              | Signed rest dihedral angle of each tuple in radians, in (−π, π]. `0` for flat.                                                                              |
| `diagnostics`         | `{ boundaryEdges: number; nonManifoldEdgesSkipped: number; degenerateTrianglesSkipped: number }` | Counts of edges with one triangle, edges with three or more triangles (stretch only, no bending), and triangles skipped for repeated vertices or zero area. |

## ClothSystem

A [`Material`](./extending.md#material) that runs a `ClothGraph` in a [`SimLoop`](./core.md#simloop). The constructor writes the graph's positions and inverse masses into the particle system, so the cloth's particles need no separate upload. It sets their velocities to zero and keeps their collision groups.

### Constructor

```ts
new ClothSystem(particles: ParticleSystem, options: ClothSystemOptions)
```

| Parameter   | Type                                         | Description                                                                       |
| ----------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particle storage. Must have room for `offset + graph.positions.length` particles. |
| `options`   | [`ClothSystemOptions`](#clothsystemoptions)  | See below.                                                                        |

| Throws                                                                                                                           | When                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `ClothSystem: invalid particle range`                                                                                            | `offset` is not a non-negative integer, the graph has no vertices, or `offset + graph.positions.length` > `particles.capacity`. |
| `ClothSystem: stretchCompliance must be ≥ 0` (likewise `bendCompliance`, `tetherCompliance`, `stretchTolerance`, `drag`, `lift`) | Value is negative, `NaN`, or infinite.                                                                                          |
| `ClothSystem: damping must be between 0 and 1`                                                                                   | `damping` is outside [0, 1] or `NaN`.                                                                                           |

### ClothSystemOptions

| Option              | Type                        | Default     | Description                                                                                                                                                                                 |
| ------------------- | --------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graph`             | [`ClothGraph`](#clothgraph) | required    | Shape, masses, and pins.                                                                                                                                                                    |
| `offset`            | `number`                    | `0`         | First particle slot the cloth occupies.                                                                                                                                                     |
| `stretchCompliance` | `number`                    | `1e-7`      | XPBD compliance of the edge constraints in s²/kg. `0` is inextensible.                                                                                                                      |
| `bendCompliance`    | `number`                    | `1e-5`      | XPBD compliance of the dihedral-angle constraints in rad²/(N·m). Higher drapes more loosely. See [Bend compliance and resolution](#bend-compliance-and-resolution).                         |
| `tetherCompliance`  | `number`                    | `0`         | XPBD compliance of the tethers in s²/kg. `0` is rigid.                                                                                                                                      |
| `stretchTolerance`  | `number`                    | `0`         | Tether slack as a fraction of the rest geodesic distance (`0.06` = 6%).                                                                                                                     |
| `wind`              | `Vector3`                   | `(0, 0, 0)` | Wind velocity in m/s. Copied.                                                                                                                                                               |
| `drag`              | `number`                    | `0.6125`    | Air drag coefficient `½ · C_D · ρ_air` in kg/m³, ≥ 0.                                                                                                                                       |
| `lift`              | `number`                    | `0.3`       | Air lift coefficient `½ · C_L · ρ_air` in kg/m³, ≥ 0.                                                                                                                                       |
| `damping`           | `number`                    | none (off)  | 0–1. Blends each free particle's velocity toward the mean of its edge neighbors' after every substep. The pass is built only when this is given; pass `0` to enable it and change it later. |

### Properties

| Property         | Type                                         | Access              | Description                                                                                   |
| ---------------- | -------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------- |
| `particles`      | [`ParticleSystem`](./core.md#particlesystem) | read-only           | The particle system passed to the constructor.                                                |
| `graph`          | [`ClothGraph`](#clothgraph)                  | read-only           | The graph passed in `options`.                                                                |
| `range`          | [`ParticleRange`](./core.md#particlerange)   | read-only           | `{ start: offset, count: graph.positions.length }`.                                           |
| `tethers`        | `readonly TetherConstraint[]`                | read-only           | Tethers built from the pins; empty when nothing is pinned. See below.                         |
| `wind`           | `Vector3`                                    | read-only reference | Wind velocity in m/s. Mutate it to change the wind.                                           |
| `drag`           | `number`                                     | read/write          | Air drag in kg/m³, ≥ 0.                                                                       |
| `lift`           | `number`                                     | read/write          | Air lift in kg/m³, ≥ 0.                                                                       |
| `damping`        | `number`                                     | read/write          | Velocity smoothing, 0–1. `0` when the `damping` option was not given; setting it then throws. |
| `bendCompliance` | `number`                                     | read/write          | Bending compliance. Takes effect on the next step.                                            |

Each `TetherConstraint` has:

| Field        | Type     | Description                                                                                          |
| ------------ | -------- | ---------------------------------------------------------------------------------------------------- |
| `particle`   | `number` | Cloth-local index of a free particle.                                                                |
| `anchor`     | `number` | Cloth-local index of the nearest pinned vertex in one pinned island. Its current position is used.   |
| `restRadius` | `number` | Geodesic distance along the rest mesh from `anchor` to `particle`, × `(1 + stretchTolerance)`, in m. |

| Throws                                                                         | When                                                                       |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `ClothSystem: bendCompliance must be ≥ 0` (likewise `drag`, `lift`)            | Setting it to a negative, `NaN`, or infinite value.                        |
| `ClothSystem: damping must be between 0 and 1`                                 | Setting `damping` outside [0, 1].                                          |
| `ClothSystem: pass \`damping\` in the options to enable it before changing it` | Setting `damping` when the `damping` option was not given at construction. |

### Methods

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

Implements [`Material.build`](./extending.md#material). Called once by `SimLoop` at construction; not called directly.

## createClothSurface

```ts
createClothSurface(cloth: ClothSystem, options: ClothSurfaceOptions): Mesh
```

Returns a `Mesh` whose vertices are placed on the GPU on a bicubic surface over the cloth's particles, with analytic normals. The cloth must be a grid in row-major order: particle `cloth.range.start + row × columns + column` is grid node (`column`, `row`), as produced by `PlaneGeometry`. The mesh is a `PlaneGeometry(1, 1, (columns − 1) × subdivisions, (rows − 1) × subdivisions)` with `frustumCulled = false` and `castShadow` and `receiveShadow` set to `true`.

Particle positions become the mesh's local positions, as with `createParticleMesh`, so the mesh's world transform (its own and its parents') is applied on top of them. Keep it at the identity to draw the cloth where it is simulated; physics never sees the transform.

| Parameter | Type                                          | Description                  |
| --------- | --------------------------------------------- | ---------------------------- |
| `cloth`   | [`ClothSystem`](#clothsystem-1)               | Grid-shaped cloth to follow. |
| `options` | [`ClothSurfaceOptions`](#clothsurfaceoptions) | See below.                   |

| Throws                                                                  | When                                                                  |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `createClothSurface: a C×R grid does not match the cloth's N particles` | `columns` < 2, `rows` < 2, or `columns × rows` ≠ `cloth.range.count`. |

### ClothSurfaceOptions

| Option         | Type                       | Default                                                 | Description                                                                                                                              |
| -------------- | -------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `columns`      | `number`                   | required                                                | Particles per row. `widthSegments + 1` for a `PlaneGeometry`.                                                                            |
| `rows`         | `number`                   | required                                                | Number of rows. `heightSegments + 1` for a `PlaneGeometry`.                                                                              |
| `subdivisions` | `number`                   | `3`                                                     | Mesh vertices per grid cell along each axis.                                                                                             |
| `material`     | `MeshPhysicalNodeMaterial` | new double-sided material, `roughness: 0.4`, `sheen: 1` | Material to draw with. Its `positionNode` and `normalNode` are overwritten, so don't share it with another surface.                      |
| `smooth`       | `boolean`                  | `true`                                                  | `true`: uniform cubic B-spline near the particles, which filters grid-scale buckling. `false`: Catmull-Rom spline through the particles. |

## Pinning

- A pinned vertex has `invMass = 0` and is not moved by the solver. Set pins with [`ClothGraphOptions.pinnedIndices`](#clothgraphoptions).
- Indices refer to the welded vertex list. `PlaneGeometry` has no duplicate vertices, so for `PlaneGeometry(w, h, sx, sy)` vertex `i` is at row `floor(i / (sx + 1))` and column `i % (sx + 1)`, with row 0 at +y. Top row: indices `0 … sx`.
- Pinned vertices connected to each other by edges form an island. Each free particle gets one tether to the nearest pinned vertex in each of its up to 4 closest islands, measured along the mesh.
- A tether only acts when the particle is farther from its anchor than `restRadius`; it stops pinned cloth from stretching under its own weight.
- Tethers read their anchor particle's current position, so they follow pins moved at runtime. To move a pinned particle, write both `particles.positions` and `particles.predictedPositions`; the solver reads the predicted position of pinned particles.
- `Simulation`'s `pin` option (`'top'`, `'top-corners'`, `'corners'`, `'none'`) pins the whole top row, the two top corners, the four corners, or nothing. See [Cloth](./cloth.md).

## Bend compliance and resolution

At a fixed `bendCompliance`, a finer grid or a lighter cloth resists folding more. `Simulation` keeps the drape constant across resolutions with:

```ts
bendCompliance = 10 ** (-1 + 5 * softness) * (segments / 30) ** 4 * (0.35 / surfaceDensity);
```

| Term             | Meaning                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| `softness`       | `Simulation` cloth softness, 0 (stiff) to 1 (silk). Default `0.75`.              |
| `segments`       | Grid segments across the cloth's width (`widthSegments` of the `PlaneGeometry`). |
| `surfaceDensity` | Mass per area in kg/m² (`Simulation`'s `weight`, default `0.1`).                 |

This function is internal to `Simulation`; apply the same scaling when building a `ClothSystem` at a different resolution from a tuned one.

## Particle spacing and self-collision

- `ClothSystem` has no self-collision of its own. Cloth collides with itself and with other particles through `SimLoop` particle contacts (`contact` option), which push apart any two particles closer than `2 × particleRadius`.
- Cloth particles are in collision group `0`, so graph neighbors also contact each other. For self-collision, every rest edge length must exceed `2 × particleRadius`; otherwise neighbors are permanently in contact and the sheet is pushed apart.
- `Simulation` spaces cloth particles `2.2 × particleRadius` apart (`segments = round(width / (2.2 × particleRadius))`), so neighbors never touch at rest and folds can collide.
- A spacing below `2 × particleRadius` closes gaps colliders could pass through, but then leave `contact` off or put the cloth in its own group with `particles.setCollisionGroup(cloth.range, group)`, before or after constructing the `ClothSystem`. The cloth then does not collide with itself.

## Limitations

- The `ClothSystem` constructor overwrites positions, inverse masses, and velocities (to zero) for its range. Upload initial velocities after constructing it.
- Topology, rest lengths, rest angles, masses, and tethers are fixed at construction.
- `stretchCompliance`, `tetherCompliance`, and `stretchTolerance` cannot be changed after construction.
- `damping` can be changed only if the `damping` option was given at construction.
- Tether rest radii come from the rest mesh. Moving pins farther apart than the rest shape allows stretches the cloth against its tethers.
- Wind is one uniform velocity for the whole cloth.
- `createClothGraph` requires indexed geometry; edges shared by three or more triangles get no bending constraint.
- `createClothSurface` supports only row-major grid cloth; draw other shapes with `createParticleMesh` or custom nodes on `particles.positions`.
- The surface mesh treats particle positions as local coordinates; a transform on it or its parents moves only the drawing.
- `createClothSurface` mutates the material passed in; do not share one material between two surfaces.
- With `smooth: true` the surface does not pass exactly through the particles, including pinned edges.
