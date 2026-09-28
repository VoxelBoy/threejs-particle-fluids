[Docs](../README.md) › [API](../README.md#api-reference) › GasSystem

# GasSystem

Smoke from passive tracers. A [`FluidSystem`](./fluid-system.md) of air particles fills the volume and does the flow simulation; `GasSystem` optionally heats that air so it rises, and moves massless tracers with the air's velocity. The renderers draw the tracers.

```ts
import {
  GasSystem,
  GasVolumeRenderer,
  GasSpriteRenderer,
  type GasSystemOptions,
  type GasVolumeRendererOptions,
  type GasSpriteRendererOptions,
  type HeatSource,
  type SmokeTracers,
} from 'threejs-particle-fluids';
```

- [`GasSystem`](#gassystem-1)
- [`HeatSource`](#heatsource)
- [`SmokeTracers`](#smoketracers)
- [`GasVolumeRenderer`](#gasvolumerenderer)
- [`GasSpriteRenderer`](#gasspriterenderer)

## GasSystem

A [`Material`](./extending.md#material) that owns the smoke tracers and, with `heatSources`, a temperature per air particle. Each substep it moves every live tracer by the kernel-weighted average velocity of the air particles within the fluid's `smoothingRadius` (Macklin et al. 2014, §7.2.1), then applies Boussinesq buoyancy `buoyancy · (T − T̄)` upward to the air, where `T̄` is the mean air temperature.

```ts
const air = new FluidSystem(particles, { viscosity: 0.02, vorticity: 0.06 });
const smoke = new GasSystem(air, {
  capacity: 30000,
  lifetime: 6,
  heatSources: [{ position: new Vector3(0, 0, 0), radius: 0.15 }],
});
// The gas goes before its fluid. Low gravity keeps the air settled.
const loop = new SimLoop(particles, { gravity: new Vector3(0, -1, 0), materials: [smoke, air] });

// Each frame:
smoke.emit([0, 0.05, 0]);
await loop.step(1 / 60);
```

### Constructor

```ts
new GasSystem(fluid: FluidSystem, options: GasSystemOptions)
```

| Parameter | Type                                    | Description                                                                |
| --------- | --------------------------------------- | -------------------------------------------------------------------------- |
| `fluid`   | [`FluidSystem`](./fluid-system.md)      | The air. Tracers follow the velocities of particles in `fluid.range` only. |
| `options` | [`GasSystemOptions`](#gassystemoptions) | See below. Required, for `capacity`.                                       |

| Throws                                             | When                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| `GasSystem: capacity must be a positive integer`   | `capacity` is not an integer > 0.                                   |
| `GasSystem: lifetime must be positive`             | `lifetime` is ≤ 0, `NaN`, or infinite.                              |
| `GasSystem: buoyancy must be finite`               | `buoyancy` is not finite.                                           |
| `GasSystem: cooling must be ≥ 0`                   | `cooling` is negative, `NaN`, or infinite.                          |
| `GasSystem: buoyancy and cooling need heatSources` | `buoyancy` or `cooling` is given without a non-empty `heatSources`. |

#### GasSystemOptions

| Option        | Type                                     | Default  | Description                                                                                    |
| ------------- | ---------------------------------------- | -------- | ---------------------------------------------------------------------------------------------- |
| `capacity`    | `number`                                 | required | Most tracers alive at once. Fixes the size of every tracer buffer.                             |
| `lifetime`    | `number`                                 | `5`      | Tracer lifetime, in s.                                                                         |
| `bounds`      | `Box3`                                   | none     | Tracers outside this box are retired early. Cloned at construction.                            |
| `heatSources` | readonly [`HeatSource`](#heatsource)`[]` | none     | Spheres that set air temperature to 1. A non-empty array enables air temperature and buoyancy. |
| `buoyancy`    | `number`                                 | `3`      | Upward acceleration of air at temperature 1 above the mean, in m/s². Requires `heatSources`.   |
| `cooling`     | `number`                                 | `0.8`    | Exponential cooling rate of air temperature, per s. Requires `heatSources`.                    |

### Properties

| Property          | Type                                      | Access     | Description                                                                                                                                                 |
| ----------------- | ----------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fluid`           | [`FluidSystem`](./fluid-system.md)        | read-only  | The air.                                                                                                                                                    |
| `capacity`        | `number`                                  | read-only  | Tracer slots.                                                                                                                                               |
| `lifetime`        | `number`                                  | read-only  | Tracer lifetime, in s.                                                                                                                                      |
| `bounds`          | `Box3 \| undefined`                       | read-only  | Copy of `options.bounds`. Mutating it has no effect on the solver.                                                                                          |
| `smokePositions`  | `StorageBufferNode<'vec4'>`               | read-only  | Tracer positions in `xyz`, in m. `capacity` entries.                                                                                                        |
| `smokeVelocities` | `StorageBufferNode<'vec4'>`               | read-only  | Velocity each tracer moved with in the last substep, in m/s, in `xyz`.                                                                                      |
| `smokeAge`        | `StorageBufferNode<'float'>`              | read-only  | Seconds since each tracer was released.                                                                                                                     |
| `smokeAlive`      | `StorageBufferNode<'uint'>`               | read-only  | `1` for live tracers, `0` for free or retired slots.                                                                                                        |
| `temperature`     | `StorageBufferNode<'float'> \| undefined` | read-only  | Temperature per air particle, indexed from `fluid.range.start`. Starts at 0. Present only with `heatSources`.                                               |
| `buoyancy`        | `number`                                  | read/write | See `GasSystemOptions.buoyancy`, in m/s². Requires `heatSources`.                                                                                           |
| `cooling`         | `number`                                  | read/write | See `GasSystemOptions.cooling`, per s. Requires `heatSources`.                                                                                              |
| `aliveCount`      | `number`                                  | read-only  | Live tracers, counted on the CPU from release times plus pending emits. No GPU readback. Tracers retired by `bounds` still count until their lifetime ends. |
| `neighborRadius`  | `number`                                  | read-only  | `fluid.smoothingRadius`, in m. Read by `SimLoop` to size the neighbor grid.                                                                                 |

| Throws                                               | When                                                              |
| ---------------------------------------------------- | ----------------------------------------------------------------- |
| `GasSystem: give heatSources to use air temperature` | `buoyancy` or `cooling` is read or written without `heatSources`. |
| `GasSystem: buoyancy must be finite`                 | `buoyancy` is set to a non-finite value.                          |
| `GasSystem: cooling must be ≥ 0`                     | `cooling` is set to a negative, `NaN`, or infinite value.         |

### Methods

#### `emit(position)`

```ts
emit(position: Vector3 | readonly [number, number, number]): boolean
```

Queue one tracer at `position` (m), released at the start of the next `SimLoop.step`. Slots are reused in ring order, oldest first. Returns `false`, and drops the tracer, when the next slot's tracer is younger than `lifetime` or `capacity` tracers are already queued this step. A steady stream needs `capacity ≥ emission rate × lifetime`. Throws `GasSystem: already disposed` after `dispose()`.

#### `update(dt)`

```ts
update(dt: number): void
```

Called by `SimLoop.step` with the step length in s. Uploads queued tracers and advances the CPU clock used by `emit` and `aliveCount`.

| Throws                                                                    | When                                                                  |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| ``GasSystem: its FluidSystem must be in the same SimLoop's `materials` `` | First step after `build`, when `fluid` wasn't built by the same loop. |
| `GasSystem: already disposed`                                             | After `dispose()`.                                                    |

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

Called once by the `SimLoop` constructor. Returns the spawn kernel (`beforeStep`) and the advection and heat kernels (`postSolve`). A gas and its fluid can be built again by a later loop.

| Throws                                                        | When                                        |
| ------------------------------------------------------------- | ------------------------------------------- |
| ``GasSystem: list it before its FluidSystem in `materials` `` | `fluid` was already built by the same loop. |

#### `dispose()`

```ts
dispose(): void
```

Free the tracer, temperature, and spawn buffers and the kernels built by `build`. Dispose the `SimLoop` first. Later `emit` and `update` calls throw. Renderers drawing the tracers are disposed separately.

## HeatSource

```ts
interface HeatSource {
  readonly position: Vector3;
  readonly radius: number;
}
```

A sphere that sets the temperature of air particles inside it to 1 every substep. Overlapping sources don't add.

| Property   | Type      | Description                                          |
| ---------- | --------- | ---------------------------------------------------- |
| `position` | `Vector3` | Center, in m. Mutate in place to move the source.    |
| `radius`   | `number`  | Radius, in m. Read once when the `SimLoop` is built. |

## SmokeTracers

```ts
interface SmokeTracers {
  readonly capacity: number;
  readonly lifetime: number;
  readonly smokePositions: StorageBufferNode<'vec4'>;
  readonly smokeAge: StorageBufferNode<'float'>;
  readonly smokeAlive: StorageBufferNode<'uint'>;
  readonly smokeVelocities?: StorageBufferNode<'vec4'>;
}
```

The tracer buffers both renderers read. [`GasSystem`](#gassystem-1) implements it; a custom emitter that fills the same buffers can be drawn too.

| Property          | Type                         | Description                                                                                         |
| ----------------- | ---------------------------- | --------------------------------------------------------------------------------------------------- |
| `capacity`        | `number`                     | Entries in each buffer.                                                                             |
| `lifetime`        | `number`                     | Tracer lifetime, in s. Renderers fade tracers by `age / lifetime`.                                  |
| `smokePositions`  | `StorageBufferNode<'vec4'>`  | Positions in `xyz`, in m.                                                                           |
| `smokeAge`        | `StorageBufferNode<'float'>` | Age, in s.                                                                                          |
| `smokeAlive`      | `StorageBufferNode<'uint'>`  | `1` live, `0` free. Only live tracers are drawn.                                                    |
| `smokeVelocities` | `StorageBufferNode<'vec4'>`  | Optional. Velocity in `xyz`, in m/s. Passed to `GasSpriteRenderer`'s `colorNode`; zero when absent. |

## GasVolumeRenderer

Lit volumetric smoke. Each `update()` splats live tracers into a density grid, blurs it, computes single-scattering light toward one direction, and writes two 3D textures; `object` ray marches them inside a box, clipped to scene depth.

```ts
const volume = new GasVolumeRenderer(smoke, {
  renderer,
  min: new Vector3(-0.5, 0, -0.5),
  max: new Vector3(0.5, 1.7, 0.5),
});
scene.add(volume.object);

// Each frame, after loop.step():
await volume.update();
renderer.render(scene, camera);
```

### Constructor

```ts
new GasVolumeRenderer(gas: SmokeTracers, options: GasVolumeRendererOptions)
```

| Parameter | Type                                                    | Description                             |
| --------- | ------------------------------------------------------- | --------------------------------------- |
| `gas`     | [`SmokeTracers`](#smoketracers)                         | Tracers to draw, usually a `GasSystem`. |
| `options` | [`GasVolumeRendererOptions`](#gasvolumerendereroptions) | See below.                              |

| Throws                                                    | When                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------- |
| `GasVolumeRenderer: resolution must be integers in 4–128` | A `resolution` entry is not an integer in [4, 128].           |
| `GasVolumeRenderer: steps must be an integer in 8–128`    | `steps` is not an integer in [8, 128].                        |
| `GasVolumeRenderer: max must exceed min on every axis`    | `max − min` is ≤ 0 or `NaN` on any axis.                      |
| `GasVolumeRenderer: edgeFade.sides must be in [0, 0.5]`   | `edgeFade.sides` is outside [0, 0.5] or `NaN`.                |
| `GasVolumeRenderer: edgeFade.bottom must be in [0, 1]`    | `edgeFade.bottom` is outside [0, 1] or `NaN`. Same for `top`. |

#### GasVolumeRendererOptions

| Option           | Type                                | Default                                     | Description                                                                                                                                                                                               |
| ---------------- | ----------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer`       | `WebGPURenderer`                    | required                                    | Runs the grid compute passes.                                                                                                                                                                             |
| `min`            | `Vector3`                           | required                                    | Minimum corner of the volume box, in m.                                                                                                                                                                   |
| `max`            | `Vector3`                           | required                                    | Maximum corner of the volume box, in m.                                                                                                                                                                   |
| `resolution`     | `readonly [number, number, number]` | `[32, 56, 32]`                              | Density grid voxels per axis, integers 4–128.                                                                                                                                                             |
| `steps`          | `number`                            | `56`                                        | Ray-march samples per pixel, 8–128.                                                                                                                                                                       |
| `density`        | `number`                            | `1`                                         | Opacity multiplier.                                                                                                                                                                                       |
| `color`          | `number`                            | `0xd8dfe6`                                  | Color of fully lit smoke.                                                                                                                                                                                 |
| `shadowColor`    | `number`                            | `0x3b4758`                                  | Color of fully self-shadowed smoke.                                                                                                                                                                       |
| `lightDirection` | `Vector3`                           | `(-0.35, 0.8, 0.4)`                         | Direction toward the light. Normalized and copied.                                                                                                                                                        |
| `edgeFade`       | `{ sides?, bottom?, top? }`         | `{ sides: 0.07, bottom: 0.025, top: 0.24 }` | Width of the smooth fade of density to zero near the box's walls, as a fraction of the box size: `sides` for the four vertical walls, `bottom` and `top` for the floor and ceiling. `0` turns a fade off. |

### Properties

| Property  | Type     | Access     | Description                                                                                                     |
| --------- | -------- | ---------- | --------------------------------------------------------------------------------------------------------------- |
| `object`  | `Mesh`   | read-only  | Box mesh to add to the scene. Back faces, no depth test or write, `frustumCulled = false`, named `'GasVolume'`. |
| `density` | `number` | read/write | Opacity multiplier.                                                                                             |

### Methods

#### `update()`

```ts
update(): Promise<void>
```

Rebuild the density and light textures from the tracers. Call once per frame, after the simulation step and before rendering. Does nothing while `object.visible` is `false`.

#### `dispose()`

```ts
dispose(): void
```

Dispose the box geometry, material, and both 3D textures.

## GasSpriteRenderer

Draws each tracer as a camera-facing soft disk whose opacity decays with age: `initialOpacity · e^(−age / opacityTau)`. Reads the tracer buffers directly; no per-frame call.

### Constructor

```ts
new GasSpriteRenderer(gas: SmokeTracers, options?: GasSpriteRendererOptions)
```

| Parameter | Type                                                    | Description      |
| --------- | ------------------------------------------------------- | ---------------- |
| `gas`     | [`SmokeTracers`](#smoketracers)                         | Tracers to draw. |
| `options` | [`GasSpriteRendererOptions`](#gasspriterendereroptions) | See below.       |

| Throws                                                | When                             |
| ----------------------------------------------------- | -------------------------------- |
| `GasSpriteRenderer: size must be positive`            | `size` ≤ 0, `NaN`, or infinite.  |
| `GasSpriteRenderer: initialOpacity must be in (0, 1]` | `initialOpacity` outside (0, 1]. |
| `GasSpriteRenderer: opacityTau must be positive`      | `opacityTau` ≤ 0 or `NaN`.       |

#### GasSpriteRendererOptions

| Option           | Type                           | Default            | Description                                                                      |
| ---------------- | ------------------------------ | ------------------ | -------------------------------------------------------------------------------- |
| `color`          | `number \| string`             | `0xeeeeee`         | Sprite color. Ignored when `colorNode` is given.                                 |
| `size`           | `number`                       | `0.08`             | Sprite width, in m.                                                              |
| `initialOpacity` | `number`                       | `0.6`              | Opacity of a new tracer, in (0, 1].                                              |
| `opacityTau`     | `number`                       | `gas.lifetime / 3` | Time for opacity to fall by a factor of e, in s.                                 |
| `colorNode`      | `(position, velocity) => Node` | none               | Per-sprite color as a TSL `vec3`, from the tracer's position and velocity nodes. |

### Properties

| Property | Type            | Access     | Description                                                                            |
| -------- | --------------- | ---------- | -------------------------------------------------------------------------------------- |
| `object` | `InstancedMesh` | read-only  | `capacity` instanced quads. Transparent, no depth write, `frustumCulled = false`.      |
| `size`   | `number`        | read/write | Sprite width, in m. Throws `GasSpriteRenderer: size must be positive` like the option. |

### Methods

#### `dispose()`

```ts
dispose(): void
```

Dispose the quad geometry and material.

## Limitations

- `GasSystem` must come before its `FluidSystem` in the same loop's `materials`. The reverse throws when the `SimLoop` is built; a missing fluid throws on the first `step`.
- Air and liquid don't mix in one `ParticleSystem`: a `SimLoop` has one gravity, and a fluid's density sum counts every particle within `smoothingRadius`, whatever its range. [`Simulation.addSmoke`](./simulation.md) refuses liquids, soft bodies, and cloth; the low-level API doesn't check.
- Tracers with no air particle within `smoothingRadius` get zero velocity and stop. The air must fill the whole region smoke moves through.
- The density solve only resists compression, so rising air can leave gaps; a small gravity on the air (around 1 m/s²) keeps it settled.
- `capacity`, `lifetime`, `bounds`, and the list of heat sources are fixed after construction; source radii are fixed once the `SimLoop` is built.
- A tracer retired early by `bounds` frees its slot only when its lifetime ends.
- Mean air temperature is summed as 32-bit fixed point, 1/4096 per step. Above about 1.05 × 10⁶ air particles the step grows to keep the sum from overflowing, so the mean gets coarser (about 1/430 at 10⁷ particles).
- `GasVolumeRenderer`: box, `resolution`, `steps`, `color`, `shadowColor`, `lightDirection`, and `edgeFade` are fixed at construction. Moving `object` doesn't move the volume.
- `GasVolumeRenderer` fades density to zero near the walls (by default within 7% of the side walls, the bottom 2.5%, and the top 24% of the box); set `edgeFade` to change it.
- `GasVolumeRenderer` ignores tracers outside the box and in its outermost voxel layer.
- `GasVolumeRenderer.update()` cost scales with voxel count and `capacity`; fragment cost scales with `steps` and the box's screen area.
- `GasSpriteRenderer` sprites aren't depth sorted, so overlapping sprites can blend in the wrong order.
- `GasSpriteRenderer` draws all `capacity` instances every frame, live or not.
