[Docs](../README.md) › [Advanced](../README.md#advanced) › `ClothSystem`

# `ClothSystem`

`ClothSystem` simulates cloth as a sheet of particles joined by rules: neighbors keep their distance, the sheet resists folding, and pinned cloth doesn't sag. Wind pushes on every triangle. `createClothGraph` builds the cloth from any indexed geometry, and `createClothSurface` draws a grid cloth as a smooth surface.

`sim.addCloth` builds a rectangle this way; the [Cloth guide](../cloth.md) covers it. Build cloth yourself for shapes other than a rectangle, custom pins, or settings such as `drag`, `lift`, and `stretchCompliance`. For one setting on a `Simulation` cloth, `cloth.clothSystem` gives you the `ClothSystem` it made; see [The objects underneath](../simulation.md#the-objects-underneath).

## A hanging curtain

```ts
import { PlaneGeometry, Vector3 } from 'three';
import {
  ClothSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createClothGraph,
  createClothSurface,
} from 'threejs-particle-fluids';

// `renderer`, `scene`, and `camera` are as in Getting started.

const segments = 40;
const geometry = new PlaneGeometry(1.2, 1.2, segments, segments).translate(0, 1, 0);

// Pin the whole top row.
const graph = createClothGraph(geometry, {
  surfaceDensity: 0.1,
  pinnedIndices: Array.from({ length: segments + 1 }, (_, i) => i),
});

const spacing = 1.2 / segments;
const particles = new ParticleSystem(renderer, graph.positions.length, 0.7 * spacing);
const cloth = new ClothSystem(particles, { graph, wind: new Vector3(0, 0, 1.5) });

// A ball for the curtain to hang against.
const ball = new PrimitiveSet(particles);
ball.addSphere(new Vector3(0, 0.7, 0.3), 0.2);

// Neighbors overlap at this radius, so leave particle contacts off.
const loop = new SimLoop(particles, { substeps: 8, materials: [cloth], colliders: [ball] });

const surface = createClothSurface(cloth, { columns: segments + 1, rows: segments + 1 });
scene.add(surface);

// Each frame. The surface updates itself on the GPU.
await loop.step(1 / 60);
renderer.render(scene, camera);
```

The cloth writes its particles' positions and masses into the particle system from the graph, so you don't upload them.

## `createClothGraph`

`createClothGraph(geometry, options)` reads an **indexed** triangle geometry and returns a `ClothGraph`. A _constraint_ is a rule the solver enforces, such as "these two particles stay this far apart". The graph holds each particle's position and inverse mass, the edges that become stretch constraints, the pairs of neighboring triangles that become bending constraints, and the triangles the wind pushes on.

Positions are taken as they are, so transform the geometry into place first. Non-indexed geometry is rejected; convert it with `mergeVertices` from `three/addons/utils/BufferGeometryUtils.js`.

| Option           | Default | What it does                                                                   |
| ---------------- | ------- | ------------------------------------------------------------------------------ |
| `surfaceDensity` | 0.2     | Mass per area in kg/m². 0.2 is about light cotton.                             |
| `pinnedIndices`  | none    | Vertices held in place. Indices refer to the graph's deduplicated vertex list. |

`graph.diagnostics` counts problems in your geometry: edges on the cloth's border (`boundaryEdges`), edges shared by more than two triangles, which get no bending (`nonManifoldEdgesSkipped`), and triangles with zero area, which are skipped (`degenerateTrianglesSkipped`). Check it if the cloth folds strangely. For a `PlaneGeometry`, vertex `i` is at row `floor(i / (segments + 1))`, column `i % (segments + 1)`, with row 0 at the top.

## `ClothSystem` options

Pinned cloth also gets _tethers_: hidden limits that stop each particle getting farther from the pins, measured along the cloth, than it started. They keep the cloth from sagging under its own weight.

| Option              | Default | What it does                                                                                                               |
| ------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------- |
| `graph`             | —       | The cloth from `createClothGraph`.                                                                                         |
| `offset`            | 0       | First particle slot the cloth occupies, when it shares the system with other materials.                                    |
| `stretchCompliance` | 1e-7    | How much the cloth stretches (0 is rigid, higher is softer). The default barely stretches.                                 |
| `bendCompliance`    | 1e-5    | How easily the cloth folds. 0 resists folding, and higher values drape more loosely. Live: `cloth.bendCompliance`.         |
| `tetherCompliance`  | 0       | How much the tethers can stretch (0 is rigid, higher is softer).                                                           |
| `stretchTolerance`  | 0       | Extra slack in the tethers, as a fraction. 0.06 lets each particle get 6% farther from the pins than it started.           |
| `wind`              | none    | Wind velocity in m/s. Mutate `cloth.wind` to change it.                                                                    |
| `drag`              | 0.6125  | How strongly air resists the cloth moving through it. Live: `cloth.drag`.                                                  |
| `lift`              | 0.3     | How strongly air moving across the cloth pushes it sideways. Live: `cloth.lift`.                                           |
| `damping`           | 0       | 0–1. Blends velocities toward neighbors' after each substep, which calms ripples against colliders. Live: `cloth.damping`. |

### Keeping the look as resolution changes

With the same `bendCompliance`, a finer grid resists folding more, and so does lighter cloth. The velvet presets and `Simulation` correct for both: they multiply `bendCompliance` by `(segments / 30)⁴` and divide it by the surface density, so the drape looks the same at every particle count:

```ts
const bendCompliance = base * (segments / 30) ** 4 * (0.35 / surfaceDensity);
```

### Particle radius

Pick the radius from the grid spacing:

- **About 0.7 × spacing**, as in Velvet Curtain: neighboring particles overlap and leave no gaps for colliders to poke through. Leave particle contacts off, or put the cloth in its own [collision group](combining-materials.md#collision-groups), as Tarp Runoff does, so the overlaps aren't treated as collisions.
- **Under 0.5 × spacing**, as in Velvet Drape: neighbors never touch, so contacts only fire between separate folds, and the cloth can collide with itself. Self-collision needs `contact: true` on the `SimLoop`.

## Collisions

- **Thin collider features** such as ears can slip between cloth particles. Give an `SDFCollider` a `thickness`; see [Colliders](colliders.md#sdfcollider-any-mesh).
- **Liquid on cloth**: `liquid.addBoundary(cloth.range)`, with a heavy cloth (high `surfaceDensity`) so pooled liquid can't push its particles apart and leak through.

Boundaries, contacts, and collision groups are covered in [Combining materials](combining-materials.md).

## `createClothSurface`

```ts
createClothSurface(cloth, { columns, rows, subdivisions: 3, smooth: true, material });
```

Draws a smooth curved surface through the cloth's particles, on the GPU. The cloth must come from a grid (such as a `PlaneGeometry`) whose vertices are in row-major order; `columns` and `rows` are particles per row and number of rows.

- `subdivisions`: mesh vertices per grid cell along each axis. Default 3.
- `smooth`: the surface passes near the particles instead of exactly through them (default `true`). This hides small wrinkles at the grid spacing but keeps larger folds.
- `material`: a `MeshPhysicalNodeMaterial` to draw with; its position and normal nodes are replaced. The default is a double-sided sheen material.

For a cloth that isn't a grid, draw it with `createParticleMesh` or write your own surface from `particles.positions`.

Background reading: bending (Bergou et al. 2006) and long-range attachments for pinned cloth (Kim et al. 2012).

---

Previous: [`SoftbodySystem`](softbody-system.md) · Next: [`GasSystem`](gas-system.md)
