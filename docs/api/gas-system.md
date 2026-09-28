[Docs](../README.md) › [API](../README.md#api-reference) › GasSystem

# GasSystem

These are the low-level smoke classes. A [`FluidSystem`](./fluid-system.md) of air particles fills the volume and simulates the flow. `GasSystem` carries massless tracer particles along with that air, and can heat the air so it rises. `GasVolumeRenderer` and `GasSpriteRenderer` draw the tracers.

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

A [`Material`](./extending.md#material) that moves smoke tracers with the air (Macklin et al. 2014, section 7.2.1). Each substep, every live tracer moves at the average velocity of the air particles within the fluid's `smoothingRadius`, with closer particles counting more. With `heatSources`, it also tracks each air particle's temperature and applies Boussinesq buoyancy, which pushes air up or down in proportion to how far it is from the average temperature.

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
| `options` | [`GasSystemOptions`](#gassystemoptions) | Required, because `capacity` has no default.                               |

| Throws                                             | When                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| `GasSystem: capacity must be a positive integer`   | `capacity` isn't a positive integer.                                |
| `GasSystem: lifetime must be positive`             | `lifetime` is zero, negative, `NaN`, or infinite.                   |
| `GasSystem: buoyancy must be finite`               | `buoyancy` is `NaN` or infinite.                                    |
| `GasSystem: cooling must be ≥ 0`                   | `cooling` is negative, `NaN`, or infinite.                          |
| `GasSystem: buoyancy and cooling need heatSources` | You gave `buoyancy` or `cooling` without a non-empty `heatSources`. |

#### GasSystemOptions

| Option        | Type                                     | Default  | Description                                                                                        |
| ------------- | ---------------------------------------- | -------- | -------------------------------------------------------------------------------------------------- |
| `capacity`    | `number`                                 | required | Most tracers alive at once. Fixes the size of every tracer buffer.                                 |
| `lifetime`    | `number`                                 | `5`      | Tracer lifetime, in s.                                                                             |
| `bounds`      | `Box3`                                   | none     | Tracers outside this box are retired early. Cloned at construction.                                |
| `heatSources` | readonly [`HeatSource`](#heatsource)`[]` | none     | Spheres that set air temperature to 1. A non-empty array turns on air temperature and buoyancy.    |
| `buoyancy`    | `number`                                 | `3`      | Upward acceleration of air whose temperature is 1 above the mean, in m/s². Requires `heatSources`. |
| `cooling`     | `number`                                 | `0.8`    | Exponential cooling rate of air temperature, per s. Requires `heatSources`.                        |

### Properties

| Property          | Type                                      | Access     | Description                                                                                                                                                                                                  |
| ----------------- | ----------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fluid`           | [`FluidSystem`](./fluid-system.md)        | read-only  | The air.                                                                                                                                                                                                     |
| `capacity`        | `number`                                  | read-only  | Tracer slots.                                                                                                                                                                                                |
| `lifetime`        | `number`                                  | read-only  | Tracer lifetime, in s.                                                                                                                                                                                       |
| `bounds`          | `Box3 \| undefined`                       | read-only  | A copy of `options.bounds`. Changing it has no effect.                                                                                                                                                       |
| `smokePositions`  | `StorageBufferNode<'vec4'>`               | read-only  | Tracer positions in `xyz`, in m. `capacity` entries.                                                                                                                                                         |
| `smokeVelocities` | `StorageBufferNode<'vec4'>`               | read-only  | Velocity each tracer moved with in the last substep, in m/s, in `xyz`.                                                                                                                                       |
| `smokeAge`        | `StorageBufferNode<'float'>`              | read-only  | Seconds since each tracer was released.                                                                                                                                                                      |
| `smokeAlive`      | `StorageBufferNode<'uint'>`               | read-only  | `1` for live tracers, `0` for free or retired slots.                                                                                                                                                         |
| `temperature`     | `StorageBufferNode<'float'> \| undefined` | read-only  | Temperature of each air particle, indexed from `fluid.range.start`. Starts at 0. Present only with `heatSources`.                                                                                            |
| `buoyancy`        | `number`                                  | read/write | See `GasSystemOptions.buoyancy`, in m/s². Requires `heatSources`.                                                                                                                                            |
| `cooling`         | `number`                                  | read/write | See `GasSystemOptions.cooling`, per s. Requires `heatSources`.                                                                                                                                               |
| `aliveCount`      | `number`                                  | read-only  | Number of live tracers, including ones queued by `emit`. It's counted on the CPU from release times, so reading it doesn't touch the GPU. Tracers retired by `bounds` still count until their lifetime ends. |
| `neighborRadius`  | `number`                                  | read-only  | `fluid.smoothingRadius`, in m. `SimLoop` reads it to size the neighbor grid.                                                                                                                                 |

| Throws                                               | When                                                           |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| `GasSystem: give heatSources to use air temperature` | You read or set `buoyancy` or `cooling` without `heatSources`. |
| `GasSystem: buoyancy must be finite`                 | You set `buoyancy` to `NaN` or an infinite value.              |
| `GasSystem: cooling must be ≥ 0`                     | You set `cooling` to a negative, `NaN`, or infinite value.     |

### Methods

#### `emit(position)`

```ts
emit(position: Vector3 | readonly [number, number, number]): boolean
```

Queues one tracer at `position`, in m. It's released at the start of the next `SimLoop.step`. Slots are reused in ring order, oldest first. `emit` returns `false` and drops the tracer if the next slot's tracer is younger than `lifetime`, or if `capacity` tracers are already queued this step. For a steady stream, make `capacity` at least the emission rate times `lifetime`. Throws `GasSystem: already disposed` after `dispose()`.

#### `update(dt)`

```ts
update(dt: number): void
```

`SimLoop.step` calls this with the step length in seconds. It uploads queued tracers and advances the clock that `emit` and `aliveCount` use.

| Throws                                                                    | When                                                                                    |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| ``GasSystem: its FluidSystem must be in the same SimLoop's `materials` `` | `fluid` wasn't built by the same loop. This is checked on the first step after `build`. |
| `GasSystem: already disposed`                                             | You called it after `dispose()`.                                                        |

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

The `SimLoop` constructor calls this once. It returns the spawn kernel as `beforeStep`, and the advection and heat kernels as `postSolve`. A gas and its fluid can be built again by a later loop.

| Throws                                                        | When                                                 |
| ------------------------------------------------------------- | ---------------------------------------------------- |
| ``GasSystem: list it before its FluidSystem in `materials` `` | You listed the gas after its fluid in the same loop. |

#### `dispose()`

```ts
dispose(): void
```

Frees the tracer, temperature, and spawn buffers and the kernels built by `build`. Dispose the `SimLoop` first. After this, `emit` and `update` throw. Renderers that draw the tracers are disposed separately.

## HeatSource

```ts
interface HeatSource {
  readonly position: Vector3;
  readonly radius: number;
}
```

A sphere that sets the temperature of air particles inside it to 1 every substep. Overlapping sources don't add together.

| Property   | Type      | Description                                          |
| ---------- | --------- | ---------------------------------------------------- |
| `position` | `Vector3` | Center, in m. Change it in place to move the source. |
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

The tracer buffers that both renderers read. [`GasSystem`](#gassystem-1) implements it. You can also draw tracers from your own emitter, as long as it fills the same buffers.

| Property          | Type                         | Description                                                                                                 |
| ----------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `capacity`        | `number`                     | Entries in each buffer.                                                                                     |
| `lifetime`        | `number`                     | Tracer lifetime, in s. Renderers fade tracers by `age / lifetime`.                                          |
| `smokePositions`  | `StorageBufferNode<'vec4'>`  | Positions in `xyz`, in m.                                                                                   |
| `smokeAge`        | `StorageBufferNode<'float'>` | Age, in s.                                                                                                  |
| `smokeAlive`      | `StorageBufferNode<'uint'>`  | `1` for live, `0` for free. Only live tracers are drawn.                                                    |
| `smokeVelocities` | `StorageBufferNode<'vec4'>`  | Optional. Velocity in `xyz`, in m/s. `GasSpriteRenderer` passes it to `colorNode`, or zero if it's missing. |

## GasVolumeRenderer

Draws the tracers as lit, volumetric smoke inside a box. Each `update()` turns the live tracers into a blurred density grid, lights it from one direction, and writes the result to two 3D textures. `object` ray-marches those textures and stops at the scene's depth, so solid objects hide the smoke behind them.

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

| Throws                                                    | When                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------ |
| `GasVolumeRenderer: resolution must be integers in 4–128` | A `resolution` entry isn't an integer from 4 to 128.                     |
| `GasVolumeRenderer: steps must be an integer in 8–128`    | `steps` isn't an integer from 8 to 128.                                  |
| `GasVolumeRenderer: max must exceed min on every axis`    | `max` isn't greater than `min` on every axis.                            |
| `GasVolumeRenderer: edgeFade.sides must be in [0, 0.5]`   | `edgeFade.sides` is outside [0, 0.5] or `NaN`.                           |
| `GasVolumeRenderer: edgeFade.bottom must be in [0, 1]`    | `edgeFade.bottom` is outside [0, 1] or `NaN`. `top` throws the same way. |

#### GasVolumeRendererOptions

| Option           | Type                                | Default                                     | Description                                                                                                                                                                                 |
| ---------------- | ----------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer`       | `WebGPURenderer`                    | required                                    | Runs the grid compute passes.                                                                                                                                                               |
| `min`            | `Vector3`                           | required                                    | Minimum corner of the volume box, in m.                                                                                                                                                     |
| `max`            | `Vector3`                           | required                                    | Maximum corner of the volume box, in m.                                                                                                                                                     |
| `resolution`     | `readonly [number, number, number]` | `[32, 56, 32]`                              | Density grid voxels per axis, integers 4–128.                                                                                                                                               |
| `steps`          | `number`                            | `56`                                        | Ray-march samples per pixel, 8–128.                                                                                                                                                         |
| `density`        | `number`                            | `1`                                         | Opacity multiplier.                                                                                                                                                                         |
| `color`          | `number`                            | `0xd8dfe6`                                  | Color of fully lit smoke.                                                                                                                                                                   |
| `shadowColor`    | `number`                            | `0x3b4758`                                  | Color of fully self-shadowed smoke.                                                                                                                                                         |
| `lightDirection` | `Vector3`                           | `(-0.35, 0.8, 0.4)`                         | Direction toward the light. Normalized and copied.                                                                                                                                          |
| `edgeFade`       | `{ sides?, bottom?, top? }`         | `{ sides: 0.07, bottom: 0.025, top: 0.24 }` | How far density fades to zero near the box's walls, as a fraction of the box size. `sides` covers the four vertical walls, `bottom` the floor, and `top` the ceiling. `0` turns a fade off. |

### Properties

| Property  | Type     | Access     | Description                                                                                                                          |
| --------- | -------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `object`  | `Mesh`   | read-only  | Box mesh to add to the scene. Draws back faces with no depth test or depth write. Named `'GasVolume'`, with `frustumCulled = false`. |
| `density` | `number` | read/write | Opacity multiplier.                                                                                                                  |

### Methods

#### `update()`

```ts
update(): Promise<void>
```

Rebuilds the density and light textures from the tracers. Call it once per frame, after the simulation step and before rendering. It does nothing while `object.visible` is `false`.

#### `dispose()`

```ts
dispose(): void
```

Disposes the box geometry, the material, and both 3D textures.

## GasSpriteRenderer

Draws each tracer as a soft disk that faces the camera and fades with age. A new tracer starts at `initialOpacity`, and its opacity falls by a factor of e every `opacityTau` seconds. The renderer reads the tracer buffers directly, so it needs no per-frame call.

### Constructor

```ts
new GasSpriteRenderer(gas: SmokeTracers, options?: GasSpriteRendererOptions)
```

| Parameter | Type                                                    | Description      |
| --------- | ------------------------------------------------------- | ---------------- |
| `gas`     | [`SmokeTracers`](#smoketracers)                         | Tracers to draw. |
| `options` | [`GasSpriteRendererOptions`](#gasspriterendereroptions) | See below.       |

| Throws                                                | When                                          |
| ----------------------------------------------------- | --------------------------------------------- |
| `GasSpriteRenderer: size must be positive`            | `size` is zero, negative, `NaN`, or infinite. |
| `GasSpriteRenderer: initialOpacity must be in (0, 1]` | `initialOpacity` is outside (0, 1].           |
| `GasSpriteRenderer: opacityTau must be positive`      | `opacityTau` is zero, negative, or `NaN`.     |

#### GasSpriteRendererOptions

| Option           | Type                           | Default            | Description                                                                      |
| ---------------- | ------------------------------ | ------------------ | -------------------------------------------------------------------------------- |
| `color`          | `number \| string`             | `0xeeeeee`         | Sprite color. Ignored when `colorNode` is given.                                 |
| `size`           | `number`                       | `0.08`             | Sprite width, in m.                                                              |
| `initialOpacity` | `number`                       | `0.6`              | Opacity of a new tracer, in (0, 1].                                              |
| `opacityTau`     | `number`                       | `gas.lifetime / 3` | Time for opacity to fall by a factor of e, in s.                                 |
| `colorNode`      | `(position, velocity) => Node` | none               | Per-sprite color as a TSL `vec3`, from the tracer's position and velocity nodes. |

### Properties

| Property | Type            | Access     | Description                                                                       |
| -------- | --------------- | ---------- | --------------------------------------------------------------------------------- |
| `object` | `InstancedMesh` | read-only  | `capacity` instanced quads. Transparent, no depth write, `frustumCulled = false`. |
| `size`   | `number`        | read/write | Sprite width, in m. A bad value throws the same error as the option.              |

### Methods

#### `dispose()`

```ts
dispose(): void
```

Disposes the quad geometry and material.

## Limitations

- You must list `GasSystem` before its `FluidSystem` in the same loop's `materials`.
- You can't put air and liquid in one `ParticleSystem`. A `SimLoop` has only one gravity, and a fluid's density counts every particle within `smoothingRadius`, whatever its range. [`Simulation.addSmoke`](./simulation.md) refuses liquids, soft bodies, and cloth, but the low-level API doesn't check.
- Tracers can't move where there's no air. A tracer with no air particle within `smoothingRadius` gets zero velocity and stops, so fill the whole region the smoke moves through with air.
- Rising air can leave gaps, because the density solve only resists compression. A small gravity on the air, around 1 m/s², keeps it settled.
- You can't change `capacity`, `lifetime`, `bounds`, or the list of heat sources after construction. A source's `radius` is fixed once the `SimLoop` is built.
- A tracer retired early by `bounds` frees its slot only when its lifetime ends.
- The mean air temperature is summed in 32-bit fixed point, in steps of 1/4096. Above about 1.05 million air particles, the step grows to keep the sum from overflowing, so the mean gets coarser. At 10 million particles the step is about 1/430.
- You can't change the `GasVolumeRenderer` box, `resolution`, `steps`, `color`, `shadowColor`, `lightDirection`, or `edgeFade` after construction. Moving `object` doesn't move the volume.
- `GasVolumeRenderer` fades smoke out near the box's walls, over the top 24% by default. Set `edgeFade` to change it.
- `GasVolumeRenderer` ignores tracers outside the box and in its outermost layer of voxels.
- The cost of `GasVolumeRenderer.update()` grows with voxel count and `capacity`. Drawing cost grows with `steps` and with how much of the screen the box covers.
- `GasSpriteRenderer` doesn't sort sprites by depth, so overlapping sprites can blend in the wrong order.
- `GasSpriteRenderer` draws all `capacity` instances every frame, including dead tracers.
