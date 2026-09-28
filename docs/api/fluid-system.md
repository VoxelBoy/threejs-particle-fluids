[Docs](../README.md) › [API](../README.md#api-reference) › FluidSystem

# FluidSystem

Low-level liquid classes: the Position Based Fluids solver (Macklin & Müller 2013), an implicit viscosity pass for thick liquids, and a ray-marched surface renderer. [`Simulation.addFluid`](./fluid.md) builds these for you (`ViscositySolver` only when `thickness` is given); this page is for building them directly.

```ts
import { FluidSystem, ViscositySolver, FluidSurfaceRenderer } from 'threejs-particle-fluids';
```

**Contents:** [FluidSystem](#fluidsystem-1) · [ViscositySolver](#viscositysolver) · [FluidSurfaceRenderer](#fluidsurfacerenderer) · [FluidAppearance](#fluidappearance) · [Emitting](#emitting) · [Limitations](#limitations)

## FluidSystem

A [`Material`](./extending.md#material) that keeps the particles in `range` at rest density. Add it to a [`SimLoop`](./core.md#simloop)'s `materials`. Optional effects (`viscosity`, `vorticity`, `surfaceTension`, `adhesion`) are compiled into the solver only when given at construction.

```ts
particles.uploadParticles(points); // spaced particleSpacing apart
const water = new FluidSystem(particles, { range: waterRange, viscosity: 0.01, surfaceTension: 0 });
water.addBoundary(duckRange);
const loop = new SimLoop(particles, { materials: [water], colliders: [walls] });
water.surfaceTension = 0.1; // allowed: surfaceTension was given, even as 0
```

### Constructor

```ts
new FluidSystem(particles: ParticleSystem, options?: FluidSystemOptions)
```

| Parameter   | Type                                         | Description              |
| ----------- | -------------------------------------------- | ------------------------ |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.        |
| `options`   | [`FluidSystemOptions`](#fluidsystemoptions)  | See below. Default `{}`. |

Sets the inverse mass of every particle in `range` to `1 / mass`, and fills `density` over `range` with `restDensity`. `ParticleSystem.uploadParticles` sets inverse masses too (default `1`), so upload before constructing the fluid, or call `particles.setInvMass(fluid.range, 1 / fluid.mass)` after.

#### FluidSystemOptions

| Option            | Type                                       | Default                                   | Description                                                                                                          |
| ----------------- | ------------------------------------------ | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `range`           | [`ParticleRange`](./core.md#particlerange) | `{ start: 0, count: particles.capacity }` | Particles that make up the fluid.                                                                                    |
| `restDensity`     | `number`                                   | `1000`                                    | Rest density, kg/m³.                                                                                                 |
| `particleSpacing` | `number`                                   | `2 × particles.particleRadius`            | Distance between particles at rest, m. Sets `mass`.                                                                  |
| `smoothingRadius` | `number`                                   | `2 × particleSpacing`                     | SPH smoothing length, m.                                                                                             |
| `compliance`      | `number`                                   | `1e-4`                                    | XPBD compliance of the density constraint. `0` is incompressible. Fixed after construction.                          |
| `viscosity`       | `number`                                   | off                                       | XSPH velocity blending coefficient.                                                                                  |
| `vorticity`       | `number`                                   | off                                       | Vorticity confinement strength.                                                                                      |
| `surfaceTension`  | `number`                                   | off                                       | Surface tension coefficient (Akinci et al. 2013). Acts between fluid particles only.                                 |
| `adhesion`        | `number`                                   | off                                       | Attraction toward boundary particles (see [`addBoundary`](#addboundaryrange-options)). No effect without boundaries. |

#### Errors

| Throws                                      | When                                                                            |
| ------------------------------------------- | ------------------------------------------------------------------------------- |
| `FluidSystem: invalid particle range`       | `range` is not integer, is empty, or exceeds `particles.capacity`.              |
| `FluidSystem: restDensity must be positive` | `restDensity` ≤ 0 or not finite. Same for `particleSpacing`, `smoothingRadius`. |
| `FluidSystem: compliance must be ≥ 0`       | `compliance` < 0 or `NaN`.                                                      |
| `FluidSystem: <option> must be finite`      | `viscosity`, `vorticity`, `surfaceTension`, or `adhesion` is `NaN` or infinite. |

### Properties

| Property          | Type                                         | Access     | Description                                                                                                                 |
| ----------------- | -------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| `particles`       | [`ParticleSystem`](./core.md#particlesystem) | read-only  | Particle storage.                                                                                                           |
| `range`           | [`ParticleRange`](./core.md#particlerange)   | read-only  | Particles that make up the fluid.                                                                                           |
| `restDensity`     | `number`                                     | read-only  | Rest density, kg/m³.                                                                                                        |
| `particleSpacing` | `number`                                     | read-only  | Rest spacing, m.                                                                                                            |
| `smoothingRadius` | `number`                                     | read-only  | SPH smoothing length, m.                                                                                                    |
| `neighborRadius`  | `number`                                     | read-only  | Equals `smoothingRadius`. `SimLoop` sizes its hash grid from the largest `neighborRadius`.                                  |
| `mass`            | `number`                                     | read-only  | Mass of one particle, kg: `restDensity × particleSpacing³`.                                                                 |
| `density`         | `StorageBufferNode<'float'>`                 | read-only  | Per-particle density, kg/m³, one entry per particle in the system. Written every solver iteration for particles in `range`. |
| `viscosity`       | `number`                                     | read/write | XSPH coefficient. Requires `viscosity` at construction.                                                                     |
| `vorticity`       | `number`                                     | read/write | Vorticity confinement strength. Requires `vorticity` at construction.                                                       |
| `surfaceTension`  | `number`                                     | read/write | Surface tension coefficient. Requires `surfaceTension` at construction.                                                     |
| `adhesion`        | `number`                                     | read/write | Adhesion coefficient. Requires `adhesion` at construction.                                                                  |

| Throws                                                                        | When                                                                                                         |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| ``FluidSystem: pass `<name>` in the options to enable it before changing it`` | Reading or writing `viscosity`, `vorticity`, `surfaceTension`, or `adhesion` when that option was not given. |
| `FluidSystem: <name> must be finite`                                          | Writing `NaN` or an infinite value to one of those properties.                                               |

### Methods

#### `addBoundary(range, options?)`

```ts
addBoundary(range: ParticleRange, options?: { readonly dynamic?: boolean }): void
```

Treat the particles in `range` as a solid boundary (Akinci et al. 2012): the fluid cannot pass through them, pushes them (buoyancy), and is attracted to them when `adhesion` is set. Intended for the surface particles of a soft body or cloth. Call before creating the `SimLoop`.

| Parameter         | Type                                       | Description                                                                                                                                                   |
| ----------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `range`           | [`ParticleRange`](./core.md#particlerange) | Boundary particles. Must not overlap the fluid's `range`.                                                                                                     |
| `options.dynamic` | `boolean`                                  | Recompute boundary volumes every substep from predicted positions. Default `true`. `false` computes them once, before the first step, from current positions. |

- Each boundary particle gets a volume `1 / Σ W` summed over particles of the same `addBoundary` range only.
- The fluid's push is applied only to particles in its own boundary ranges with `invMass > 0`.
- Boundary volumes are stored in the shared `ParticleSystem.boundaryVolume`. See [Limitations](#limitations) for what that means with several fluids.

| Throws                                                                | When                                                   |
| --------------------------------------------------------------------- | ------------------------------------------------------ |
| `FluidSystem.addBoundary: add boundaries before creating the SimLoop` | The fluid was already built by a `SimLoop`.            |
| `FluidSystem.addBoundary: invalid particle range`                     | `range` is not integer, is empty, or exceeds capacity. |
| `FluidSystem.addBoundary: a boundary cannot overlap the fluid`        | `range` intersects the fluid's `range`.                |
| `FluidSystem.addBoundary: boundaries cannot overlap each other`       | `range` intersects a range added earlier.              |

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<boolean>
```

`true` if, at the last neighbor rebuild, some fluid particle had more than 64 neighbors within `smoothingRadius` and the extras were dropped. Reads back from the GPU; for debugging and tests. [`SimLoop.readbackOverflow()`](./core.md#simloop) does not include it.

| Throws                                          | When                              |
| ----------------------------------------------- | --------------------------------- |
| `FluidSystem: add the fluid to a SimLoop first` | The fluid has not been built yet. |

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

Called once by `SimLoop`. Returns the fluid's kernels. See [`Material`](./extending.md#material).

## ViscositySolver

Implicit viscosity for thick liquids. Each substep solves `(I − dt·ν·L) v = v₀` for the fluid's velocities with Jacobi sweeps, which stays stable at viscosities where `FluidSystem`'s XSPH `viscosity` does not. A [`Material`](./extending.md#material); list it after its fluid in `materials`.

```ts
const honey = new FluidSystem(particles, { viscosity: 0.03 });
const thick = new ViscositySolver(honey, { viscosity: 20 });
const loop = new SimLoop(particles, { materials: [honey, thick] }); // order matters
```

### Constructor

```ts
new ViscositySolver(fluid: FluidSystem, options: ViscositySolverOptions)
```

| Parameter | Type                                                | Description                        |
| --------- | --------------------------------------------------- | ---------------------------------- |
| `fluid`   | [`FluidSystem`](#fluidsystem-1)                     | Fluid whose velocities are solved. |
| `options` | [`ViscositySolverOptions`](#viscositysolveroptions) | Required.                          |

#### ViscositySolverOptions

| Option       | Type     | Default  | Description                                                                                                                                                                                                                                                               |
| ------------ | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `viscosity`  | `number` | required | Kinematic viscosity ν, m²/s, as the discrete solve sees it: `L` is the difference from the neighbors' weighted mean velocity times `10 / smoothingRadius²`, so tune by eye (honey ≈ 20). Must be ≥ 0. Not on the same scale as `FluidSystem`'s unitless XSPH `viscosity`. |
| `iterations` | `number` | `12`     | Jacobi sweeps per substep. Integer, 1–64. Fixed after construction.                                                                                                                                                                                                       |

#### Errors

| Throws                                                             | When                                                                                                     |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `ViscositySolver: viscosity must be ≥ 0`                           | `viscosity` < 0 or not finite, at construction or when set later.                                        |
| `ViscositySolver: iterations must be an integer from 1 to 64`      | `iterations` out of range or not an integer.                                                             |
| ``ViscositySolver: list it after its FluidSystem in `materials` `` | Thrown by `SimLoop` construction when `fluid` was not built first (listed after the solver, or missing). |

### Properties

| Property    | Type                            | Access     | Description                        |
| ----------- | ------------------------------- | ---------- | ---------------------------------- |
| `fluid`     | [`FluidSystem`](#fluidsystem-1) | read-only  | Fluid whose velocities are solved. |
| `viscosity` | `number`                        | read/write | ν, m²/s. Must be ≥ 0 and finite.   |

### Methods

#### `build()`

```ts
build(): MaterialKernels
```

Called once by `SimLoop`. Returns `iterations + 2` post-solve dispatches over the fluid's range.

## FluidSurfaceRenderer

Draws a `FluidSystem` as a smooth liquid surface. Each `update()` splats the particles into a voxel field; the mesh's fragment shader sphere-traces that field, refracts the opaque scene behind it with Beer–Lambert absorption, and adds Fresnel-weighted environment reflection, screen-space reflections, and a GGX highlight from the scene's key light. It writes depth.

```ts
const surface = new FluidSurfaceRenderer(water, {
  renderer,
  scene,
  camera,
  bounds,
  colliders: [walls],
});
scene.add(surface.mesh); // not added automatically

// each frame
await loop.step(dt);
await surface.update();
renderer.render(scene, camera);
```

### Constructor

```ts
new FluidSurfaceRenderer(fluid: FluidSystem, options: FluidSurfaceRendererOptions)
```

| Parameter | Type                                                          | Description    |
| --------- | ------------------------------------------------------------- | -------------- |
| `fluid`   | [`FluidSystem`](#fluidsystem-1)                               | Fluid to draw. |
| `options` | [`FluidSurfaceRendererOptions`](#fluidsurfacerendereroptions) | Required.      |

#### FluidSurfaceRendererOptions

Every option is fixed after construction unless a property below says otherwise.

| Option          | Type                                               | Default                                                      | Description                                                                                                                                                                              |
| --------------- | -------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer`      | `WebGPURenderer`                                   | required                                                     | Runs the field kernels and `pick()`.                                                                                                                                                     |
| `scene`         | `Scene`                                            | required                                                     | `scene.environment`, `scene.environmentIntensity`, and the key light are read every `update()`.                                                                                          |
| `camera`        | `Camera`                                           | required                                                     | Used only by `pick()`, which casts rays from it. Drawing uses whichever camera renders the mesh.                                                                                         |
| `bounds`        | `Box3`                                             | required                                                     | World box the liquid can occupy, m. Particles outside are ignored.                                                                                                                       |
| `colliders`     | `readonly (PrimitiveSet \| SDFCollider)[]`         | `[]`                                                         | Colliders the surface is cut against and climbs into a meniscus. See [`PrimitiveSet`](./colliders.md#primitiveset), [`SDFCollider`](./colliders.md#sdfcollider).                         |
| `carve`         | [`PrimitiveSet`](./colliders.md#primitiveset)      | none                                                         | Shapes cut out of the surface every frame. Not wetted.                                                                                                                                   |
| `solids`        | [`ParticleRange`](./core.md#particlerange)         | none                                                         | Non-fluid particles (floating bodies) the surface climbs.                                                                                                                                |
| `motionStretch` | `number`                                           | `0`                                                          | Smear each fluid particle back along its velocity over this time, s, capped at `3 × particleSpacing`. `0` disables it. Must be ≥ 0.                                                      |
| `voxelBudget`   | `number`                                           | `FluidSurfaceRenderer.defaultVoxelBudget(fluid.range.count)` | Upper bound on voxels in the field, about 76 bytes each. Must be ≥ 1.                                                                                                                    |
| `appearance`    | `Partial<`[`FluidAppearance`](#fluidappearance)`>` | all defaults                                                 | Initial look.                                                                                                                                                                            |
| `cavities`      | `{ smokeColor: number; smokeDensity: number }`     | none                                                         | Draw pockets cut by `carve` with a reflective rim and a smoke fill. `smokeColor` is sRGB hex; `smokeDensity` is extinction per metre, 1/m, ≥ 0. Adds 16 steps to the transmission march. |
| `refraction`    | `boolean`                                          | `true`                                                       | Bend transmitted light. `false` samples the scene behind unbent.                                                                                                                         |

Voxel edge length is `max(particles.particleRadius, ∛(bounds volume / voxelBudget))`.

#### Errors

| Throws                                                    | When                                            |
| --------------------------------------------------------- | ----------------------------------------------- |
| `FluidSurfaceRenderer: bounds must have positive extent`  | `bounds` has zero or negative size on any axis. |
| `FluidSurfaceRenderer: motionStretch must be ≥ 0`         | `motionStretch` < 0 or not finite.              |
| `FluidSurfaceRenderer: voxelBudget must be ≥ 1`           | `voxelBudget` < 1 or not finite.                |
| `FluidSurfaceRenderer: smokeDensity must be ≥ 0`          | `cavities.smokeDensity` < 0 or not finite.      |
| `FluidSurfaceRenderer: appearance.<field> must be finite` | A field of `appearance` is `NaN` or infinite.   |

### Properties

| Property       | Type                            | Access     | Description                                                                                                              |
| -------------- | ------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| `mesh`         | `Mesh`                          | read-only  | Back-faced box over the field, named `'FluidSurface'`, `renderOrder = -1`, `frustumCulled = false`. Add it to the scene. |
| `fluid`        | [`FluidSystem`](#fluidsystem-1) | read-only  | Fluid being drawn.                                                                                                       |
| `reflections`  | `boolean`                       | read/write | Screen-space reflections. Default `true`. When `false`, reflects only the environment.                                   |
| `smokeDensity` | `number`                        | read/write | Cavity smoke extinction, 1/m, ≥ 0. Initial value `cavities.smokeDensity`. Requires `cavities` at construction.           |

| Throws                                                                                             | When                                                  |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| ``FluidSurfaceRenderer: pass `cavities` in the options to enable smokeDensity before changing it`` | Reading or writing `smokeDensity` without `cavities`. |
| `FluidSurfaceRenderer: smokeDensity must be ≥ 0`                                                   | Writing a negative, `NaN`, or infinite value.         |

### Methods

#### `update()`

```ts
update(): Promise<void>
```

Rebuild the surface field from the current particle positions. Call once per frame, after stepping and before rendering. Also:

- re-reads the key light and `scene.environmentIntensity`;
- follows `scene.environment`: a new map replaces the old one in place, and gaining or losing a map rebuilds the surface shader once;
- computes collider wetting on the first call, and again whenever a `PrimitiveSet` or `SDFCollider` in `colliders` has changed `version` (they move or change).

Throws `FluidSurfaceRenderer: already disposed` after `dispose()`.

#### `setAppearance(appearance)`

```ts
setAppearance(appearance: Partial<FluidAppearance>): void
```

Change any [`FluidAppearance`](#fluidappearance) fields. Omitted or `undefined` fields keep their value. Throws `FluidSurfaceRenderer: appearance.<field> must be finite` for a `NaN` or infinite field, before changing anything.

#### `pick(uv)`

```ts
pick(uv: Vector2): Promise<Vector3 | null>
```

World-space point where the ray through viewport coordinate `uv` (`[0, 1]`, y down) from `options.camera` meets the liquid, or `null`. Traces the field from the last `update()`. Reads back from the GPU. Throws `FluidSurfaceRenderer: already disposed` after `dispose()`.

#### `dispose()`

```ts
dispose(): void
```

Remove `mesh` from its parent and free its geometry and material, the field texture, the field's GPU buffers, and the compiled kernels. Safe to call twice; `update()` and `pick()` throw afterwards.

#### `FluidSurfaceRenderer.defaultVoxelBudget(particleCount)`

```ts
static defaultVoxelBudget(particleCount: number): number
```

Default `voxelBudget`: `1_200_000` for ≥ 50 000 particles, `900_000` for ≥ 20 000, else `600_000`.

### Key light

The key light is a `DirectionalLight` in `scene`: the last one in traversal order with `castShadow`, else the first one found. It is looked up at construction and on each `update()` until one is found, then kept. Its direction and `color × intensity` are read every `update()`. With no directional light there is no specular highlight.

## FluidAppearance

Look of the liquid. Pass as `options.appearance` or to `setAppearance()`. All fields are live and must be finite.

| Field                 | Type     | Default    | Description                                                                                                        |
| --------------------- | -------- | ---------- | ------------------------------------------------------------------------------------------------------------------ |
| `color`               | `number` | `0x2a8fb0` | sRGB hex. Colour white light takes on after crossing `attenuationDistance` of liquid. Channels clamped to ≥ 0.001. |
| `attenuationDistance` | `number` | `0.6`      | Path length at which transmitted light reaches `color`, m. Clamped to ≥ 1e-4.                                      |
| `scattering`          | `number` | `0.08`     | Light scattered back out of the body. `0` clear, `1` milky.                                                        |
| `ior`                 | `number` | `1.333`    | Index of refraction. Also sets Fresnel reflectance.                                                                |
| `roughness`           | `number` | `0.04`     | Blurs reflections and widens the highlight. Clamped to 0.02–1.                                                     |
| `envIntensity`        | `number` | `1`        | Multiplier on environment reflection.                                                                              |
| `metalness`           | `number` | `0`        | Blend from dielectric (`0`) to metal (`1`).                                                                        |
| `metalColor`          | `number` | `0xc8d2da` | sRGB hex. Reflectance tint at `metalness: 1`.                                                                      |

## Emitting

`FluidSystem` has no emit method; the particle count is fixed by `range`. To pour, park the fluid's particles pinned out of view, then release a batch per frame from a compute kernel:

1. After constructing the fluid, call `particles.setInvMass(fluid.range, 0)`. The constructor sets masses, so this must come after it.
2. For each released particle `i`, write `particles.positions[i]` and `particles.predictedPositions[i]` (nozzle position), `particles.velocities[i]`, `particles.invMass[i] = 1 / fluid.mass`, and `fluid.density[i] = fluid.restDensity`.
3. Release the next batch only after the stream has advanced one `particleSpacing`, so new particles do not overlap.

[`demo/presets/honey.ts`](../../demo/presets/honey.ts) implements this.

## Limitations

- `viscosity`, `vorticity`, `surfaceTension`, and `adhesion` can be changed later only if given at construction; otherwise their properties throw.
- `compliance`, `restDensity`, `particleSpacing`, `smoothingRadius`, and `range` are fixed after construction.
- `ParticleSystem.uploadParticles` resets `invMass` (default `1`); upload before constructing the fluid, or call `setInvMass(fluid.range, 1 / fluid.mass)` after.
- `addBoundary` must be called before the `SimLoop` is created; boundaries cannot overlap the fluid or each other.
- Each dynamic boundary adds one dispatch over its particles per substep; static ones run once, before the first step, and are not updated if they move.
- Any boundary adds a reaction accumulator (3 × `capacity` int32) with a reset per substep and an apply every solver iteration over the span from the first boundary particle to the last.
- Boundary volumes live in the shared `ParticleSystem`. With several fluids on one particle system, give every fluid the same boundaries, as `Simulation` does. Otherwise:
  - every `FluidSystem` treats a boundary added to any of them as a boundary, but pushes back only on its own;
  - a range added to fluids with different `smoothingRadius` gets the volume of whichever fluid ran last.
- Non-fluid particles within `smoothingRadius` that are not boundaries (another fluid's particles, a soft body's interior) enter the density sum with their own mass and push the fluid in the pressure solve without receiving a reaction. This is what keeps two fluids on one particle system apart. Surface tension ignores them.
- The neighbor list stores at most 64 neighbors per fluid particle; extras are dropped. Check with [`readbackOverflow()`](#readbackoverflow).
- `ViscositySolver` must be listed after its `FluidSystem` in `materials`; it runs `iterations + 2` dispatches per substep.
- `FluidSurfaceRenderer` options are fixed after construction; only `appearance`, `reflections`, and `smokeDensity` change later.
- Gaining or losing `scene.environment` recompiles the surface shader on the next `update()`.
- `update()` cost scales with voxel count (about 76 bytes each; four full-grid dispatches per call) and with fluid particle count (splatted 3 times with `motionStretch > 0`).
- `carve` is evaluated for every primitive at every voxel on every `update()`.
