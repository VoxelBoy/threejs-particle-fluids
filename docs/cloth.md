[Docs](README.md) › Cloth

# Cloth

`ClothSystem` simulates cloth as particles joined by stretch constraints, bending constraints (Bergou et al. 2006), and long-range attachments that stop pinned cloth from sagging (Kim et al. 2012), with aerodynamic drag and lift on every triangle. `createClothGraph` builds the cloth from any indexed geometry, and `createClothSurface` draws a grid cloth as a smooth surface.

## A hanging curtain

```ts
import { PlaneGeometry, Vector3 } from 'three';
import {
  ClothSystem,
  ParticleSystem,
  SimLoop,
  createClothGraph,
  createClothSurface,
} from 'threejs-particle-fluids';

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

// Neighbors overlap at this radius, so leave particle contacts off.
const loop = new SimLoop(particles, { substeps: 8, materials: [cloth], colliders: [ball] });

const surface = createClothSurface(cloth, { columns: segments + 1, rows: segments + 1 });
scene.add(surface);
```

The cloth writes its particles' positions and masses into the particle system from the graph, so you don't upload them.

## `createClothGraph`

`createClothGraph(geometry, options)` reads an **indexed** triangle geometry and returns a `ClothGraph`: per-particle positions and inverse masses, the edges that become stretch constraints, the triangle pairs that become bending constraints, and each triangle for wind. Positions are taken as they are, so transform the geometry into place first. Non-indexed geometry is rejected; convert it with `mergeVertices` from `three/addons/utils/BufferGeometryUtils.js`.

| Option           | Default | What it does                                                                   |
| ---------------- | ------- | ------------------------------------------------------------------------------ |
| `surfaceDensity` | 0.2     | Mass per area in kg/m². 0.2 is about light cotton.                             |
| `pinnedIndices`  | none    | Vertices held in place. Indices refer to the graph's deduplicated vertex list. |

`graph.diagnostics` reports boundary edges, non-manifold edges skipped for bending, and degenerate triangles skipped. For a `PlaneGeometry`, vertex `i` is at row `floor(i / (segments + 1))`, column `i % (segments + 1)`, with row 0 at the top.

## `ClothSystem` options

| Option              | Default | What it does                                                                                                               |
| ------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------- |
| `graph`             | —       | The cloth from `createClothGraph`.                                                                                         |
| `offset`            | 0       | First particle slot the cloth occupies, when it shares the system with other materials.                                    |
| `stretchCompliance` | 1e-7    | Stretch compliance in s²/kg. The default barely stretches.                                                                 |
| `bendCompliance`    | 1e-5    | Bending compliance. Higher values drape more loosely. Live: `cloth.bendCompliance`.                                        |
| `tetherCompliance`  | 0       | Compliance of the long-range attachments to the pins.                                                                      |
| `stretchTolerance`  | 0       | How far past its rest distance a particle may drift from its pins, as a fraction.                                          |
| `wind`              | none    | Wind velocity in m/s. Mutate `cloth.wind` to change it.                                                                    |
| `drag`              | 0.6125  | Air drag, `½ · C_D · ρ_air`. Live: `cloth.drag`.                                                                           |
| `lift`              | 0.3     | Air lift, `½ · C_L · ρ_air`. Live: `cloth.lift`.                                                                           |
| `damping`           | 0       | 0–1. Blends velocities toward neighbors' after each substep, which calms ripples against colliders. Live: `cloth.damping`. |

### Keeping the look as resolution changes

Bending stiffness depends on the grid: as you refine it, the bending gradients grow and each particle's mass shrinks. The velvet presets scale `bendCompliance` by `(segments / 30)⁴` and divide it by the surface density, so the drape looks the same at every particle count:

```ts
const bendCompliance = base * (segments / 30) ** 4 * (0.35 / surfaceDensity);
```

### Particle radius

Pick the radius from the grid spacing:

- **About 0.7 × spacing**, as in Velvet Curtain: neighboring particles overlap and leave no gaps for colliders to poke through. Leave particle contacts off, or put the cloth in its own collision group, as Tarp Runoff does, so the overlaps aren't treated as collisions.
- **Under 0.5 × spacing**, as in Velvet Drape: neighbors never touch, so cloth-on-cloth contacts only fire between separate folds, and the cloth can collide with itself.

## Collisions

- **Colliders** work as for any particle. For thin features such as ears, give an `SDFCollider` a `thickness` so they can't slip between particles. See [Colliders](colliders.md).
- **Self-collision** needs `contact: true` on the `SimLoop` and a radius below half the spacing.
- **Overlapping particles**: when neighbors overlap on purpose, stop them colliding with each other:

  ```ts
  particles.setCollisionGroup(cloth.range, 1);
  ```

- **Liquid on cloth**: add the cloth as a boundary of the fluid (`liquid.addBoundary(cloth.range)`). A heavy cloth (high `surfaceDensity`) keeps pooled liquid from pushing its particles apart and leaking through. See [Combining materials](combining-materials.md).

## `createClothSurface`

```ts
createClothSurface(cloth, { columns, rows, subdivisions: 3, smooth: true, material });
```

Draws a grid cloth as a bicubic surface through its particles, with analytic normals, entirely on the GPU. The cloth must come from a grid (such as a `PlaneGeometry`) whose vertices are in row-major order; `columns` and `rows` are particles per row and number of rows.

- `subdivisions`: mesh vertices per grid cell along each axis. Default 3.
- `smooth`: fit a cubic B-spline near the particles (default) instead of a Catmull-Rom spline through them. The B-spline filters out grid-scale buckling while keeping larger folds.
- `material`: a `MeshPhysicalNodeMaterial` to draw with; its position and normal nodes are replaced. The default is a double-sided sheen material.

For a cloth that isn't a grid, draw it with `createParticleMesh` or write your own surface from `particles.positions`.

---

Previous: [Soft bodies](soft-bodies.md) · Next: [Smoke](smoke.md)
