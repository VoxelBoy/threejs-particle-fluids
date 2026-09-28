[Docs](../README.md) › [API](../README.md#api-reference) › Simulation

# Simulation

High-level entry point. Collects fluids, soft bodies, cloth, smoke, and obstacles, then builds and couples the underlying systems on the first `step()`.

```ts
import { Simulation, createParticleRenderer } from 'threejs-particle-fluids';
```

- [`createParticleRenderer`](#createparticlerenderer)
- [`Simulation`](#simulation-1)
  - [Constructor](#constructor) · [`SimulationOptions`](#simulationoptions)
  - [Properties](#properties)
  - [Methods](#methods): [`addFluid`](#addfluidoptions) · [`addSoftbody`](#addsoftbodyoptions) · [`addCloth`](#addclothoptions) · [`addSmoke`](#addsmokeoptions) · [`addFloor`](#addflooroptions) · [`addSphere`](#addsphereoptions) · [`addBox`](#addboxoptions) · [`addCapsule`](#addcapsuleoptions) · [`addMesh`](#addmeshoptions) · [`start`](#start) · [`step`](#stepdt) · [`dispose`](#dispose)
- [Errors](#errors)
- [Limitations](#limitations)

---

## createParticleRenderer

```ts
createParticleRenderer(options?: Partial<WebGPURendererParameters>): Promise<WebGPURenderer>
```

Creates and initializes a `WebGPURenderer` that requests the device limits the solver needs. Use it in place of `new WebGPURenderer()`.

| Parameter | Type                                | Description                                                                                 |
| --------- | ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `options` | `Partial<WebGPURendererParameters>` | Passed to `WebGPURenderer`. `requiredLimits` is merged with the limits below, not replaced. |

Required device limits:

| Limit                               | Value |
| ----------------------------------- | ----- |
| `maxComputeInvocationsPerWorkgroup` | 1024  |
| `maxComputeWorkgroupSizeX`          | 1024  |
| `maxStorageBuffersPerShaderStage`   | 10    |

| Throws                                             | When                                                     |
| -------------------------------------------------- | -------------------------------------------------------- |
| `createParticleRenderer: WebGPU is unavailable, …` | The renderer fell back to WebGL. There is no WebGL path. |

---

## Simulation

### Constructor

```ts
new Simulation(options: SimulationOptions)
```

### SimulationOptions

| Option           | Type             | Default                                  | Description                                                                        |
| ---------------- | ---------------- | ---------------------------------------- | ---------------------------------------------------------------------------------- |
| `renderer`       | `WebGPURenderer` | required                                 | From [`createParticleRenderer`](#createparticlerenderer).                          |
| `scene`          | `Scene`          | required                                 | Scene the surfaces are added to. Liquid reflects `scene.environment`.              |
| `camera`         | `Camera`         | required                                 | Camera the liquid surface is ray-marched from.                                     |
| `particleRadius` | `number`         | required                                 | Radius of every particle, m. Particles are placed `2 × particleRadius` apart.      |
| `maxParticles`   | `number`         | required                                 | Upper limit on the total particle count. The build throws if the scene needs more. |
| `container`      | `Box3`           | none                                     | Walls on the floor and four sides. Required for smoke.                             |
| `closed`         | `boolean`        | `false`                                  | Adds a lid to `container`. Always on with smoke.                                   |
| `gravity`        | `Vector3`        | `(0, -9.81, 0)`; `(0, -1, 0)` with smoke | m/s². Copied; change it later through [`gravity`](#properties).                    |
| `substeps`       | `number`         | computed                                 | Solver substeps per 1/60 s step. See [Substeps](#substeps).                        |

#### Particle count

`particleRadius` and `maxParticles` are both required. Nothing is sized automatically. The count at a given radius is:

| Content                 | Particles                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| Fluid box               | `volume / (2r)³`, after clipping to the container and removing space taken by obstacles and solids |
| Fluid or soft body mesh | `volume / (2r)³` (voxelized)                                                                       |
| Cloth `w × h`           | `(round(w / 2.2r) + 1) × (round(h / 2.2r) + 1)`                                                    |
| Smoke                   | `containerVolume / (2r)³` (air fills the container)                                                |

Example: a 0.4 × 0.5 × 0.6 m block of water at `r = 0.014` is about 4,800 particles.

GPU cost grows with particle count and substeps. Test on the slowest hardware you target. [`particleCount`](#properties) reports the actual count after `start()`.

#### Substeps

When `substeps` is omitted:

```
substeps = min(24, ceil(base × max(1, 0.018 / particleRadius)))
base     = max(fluid ? 4 : 0, smoke ? 2 : 0, softbody ? 6 : 0, cloth ? 8 : 0)
```

Each substep costs about as much as the first. Raise it for fast or thin obstacles and stiff solids; lower it to save time.

### Properties

| Property         | Type                                         | Access     | Description                                                                                    |
| ---------------- | -------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------- |
| `gravity`        | `Vector3`                                    | read       | Live gravity vector, m/s². Mutate it in place; applied on the next step.                       |
| `particleRadius` | `number`                                     | read       | The `particleRadius` option.                                                                   |
| `particleCount`  | `number`                                     | read       | Particles in use. `0` until the simulation starts.                                             |
| `showParticles`  | `boolean`                                    | read/write | Hide the rendered surfaces and draw raw particles. Default `false`.                            |
| `particleSystem` | [`ParticleSystem`](./core.md#particlesystem) | read       | The underlying particle storage. Throws before the simulation starts.                          |
| `loop`           | [`SimLoop`](./core.md#simloop)               | read       | The underlying solver loop. `loop.substeps` can be changed while running. Throws before start. |

### Methods

All `add*` methods must be called before the first `start()` or `step()`. Positions are world space, in metres.

#### `addFluid(options)`

```ts
addFluid(options: FluidOptions): Fluid
```

Fills a box or a closed mesh with liquid. See [Fluid](./fluid.md).

#### `addSoftbody(options)`

```ts
addSoftbody(options: SoftbodyOptions): Softbody
```

Replaces a closed mesh with a deformable copy. See [Softbody](./softbody.md).

#### `addCloth(options)`

```ts
addCloth(options: ClothOptions): Cloth
```

Adds a rectangular cloth. See [Cloth](./cloth.md).

#### `addSmoke(options?)`

```ts
addSmoke(options?: SmokeOptions): Smoke
```

Adds a heated smoke source. Requires `container`. See [Smoke](./smoke.md).

#### `addFloor(options?)`

```ts
addFloor(options?: { height?: number; friction?: number }): void
```

| Option     | Type     | Default | Description                    |
| ---------- | -------- | ------- | ------------------------------ |
| `height`   | `number` | `0`     | Floor height, m.               |
| `friction` | `number` | `0.5`   | 0 (slick) to about 1 (sticky). |

#### `addSphere(options)`

```ts
addSphere(options: { radius: number; center?: Vector3; follow?: Object3D; friction?: number }): void
```

| Option     | Type       | Default     | Description                                           |
| ---------- | ---------- | ----------- | ----------------------------------------------------- |
| `radius`   | `number`   | required    | m.                                                    |
| `center`   | `Vector3`  | `(0, 0, 0)` | Ignored when `follow` is set.                         |
| `follow`   | `Object3D` | none        | Center tracks the object's world position every step. |
| `friction` | `number`   | `0.5`       |                                                       |

#### `addBox(options)`

```ts
addBox(options: { size: Vector3; center?: Vector3; rotation?: Euler; follow?: Object3D; friction?: number }): void
```

| Option     | Type       | Default     | Description                                                        |
| ---------- | ---------- | ----------- | ------------------------------------------------------------------ |
| `size`     | `Vector3`  | required    | Full edge lengths, m.                                              |
| `center`   | `Vector3`  | `(0, 0, 0)` | Ignored when `follow` is set.                                      |
| `rotation` | `Euler`    | none        | Ignored when `follow` is set.                                      |
| `follow`   | `Object3D` | none        | Center and rotation track the object's world transform every step. |
| `friction` | `number`   | `0.5`       |                                                                    |

#### `addCapsule(options)`

```ts
addCapsule(options: { start: Vector3; end: Vector3; radius: number; follow?: Object3D; friction?: number }): void
```

| Option     | Type       | Default  | Description                                                                                 |
| ---------- | ---------- | -------- | ------------------------------------------------------------------------------------------- |
| `start`    | `Vector3`  | required | First end point, m.                                                                         |
| `end`      | `Vector3`  | required | Second end point, m.                                                                        |
| `radius`   | `number`   | required | m.                                                                                          |
| `follow`   | `Object3D` | none     | Segment midpoint tracks the object's world position; the capsule turns as the object turns. |
| `friction` | `number`   | `0.5`    |                                                                                             |

#### `addMesh(options)`

```ts
addMesh(options: { mesh: Mesh; resolution?: number; friction?: number }): void
```

Makes a closed mesh solid by baking a signed distance field on the CPU when the simulation starts. The collider follows the mesh's world transform every step.

| Option       | Type     | Default  | Description                                  |
| ------------ | -------- | -------- | -------------------------------------------- |
| `mesh`       | `Mesh`   | required | Closed, watertight geometry.                 |
| `resolution` | `number` | `64`     | Distance field cells along the longest axis. |
| `friction`   | `number` | `0.5`    |                                              |

#### `start()`

```ts
start(): Promise<void>
```

Builds particles, systems, colliders, and renderers. Called by the first `step()` if not called earlier. Runs synchronously on the main thread (voxelizing and SDF baking), so show any loading UI before calling it. If it throws, source meshes are shown again and `start()` can be retried.

#### `step(dt?)`

```ts
step(dt?: number): Promise<void>
```

Advances the simulation and updates the rendered surfaces. Await it before `renderer.render()`.

| `dt`     | Behavior                                                                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| omitted  | Runs fixed 1/60 s steps to match wall-clock time: at most 4 per call, excess time dropped. The first call only starts the clock. After more than 250 ms without a call, the clock restarts instead of catching up. |
| `number` | Advances exactly `dt` seconds in one step. Use a constant, e.g. `1 / 60`, not the frame delta.                                                                                                                     |

Calling `step()` while a step is running returns the running step's promise.

#### `dispose()`

```ts
dispose(): void
```

Removes everything the simulation added to the scene, disposes its renderers, geometries, and default materials, and shows source meshes again. A `material` passed to `addCloth` is not disposed. The simulation can't be used afterwards.

---

## Errors

| Message                                                                                  | Cause                                                                                    |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `Simulation: particleRadius must be a positive number of metres, …`                      | Missing or invalid `particleRadius`.                                                     |
| `Simulation: maxParticles must be a positive integer, …`                                 | Missing or invalid `maxParticles`.                                                       |
| `Simulation: the scene needs N particles at particleRadius r, above maxParticles (M). …` | Raise `maxParticles`, raise `particleRadius`, or add less. Thrown by `start()`/`step()`. |
| `Simulation: <call> must happen before the first step() or start()`                      | An `add*` call after the simulation started.                                             |
| `Simulation: add a fluid, smoke, soft body, or cloth before stepping`                    | Only obstacles were added.                                                               |
| `addFluid: give either `box`or`mesh``                                                    | Both or neither given.                                                                   |
| `addFluid: the fluid has no room; check its box and the container`                       | The box is outside the container or fully occupied.                                      |
| `addSoftbody: the mesh is too small for the particle size`                               | The mesh voxelizes to zero particles. Lower `particleRadius` or scale the mesh up.       |
| `addSmoke: smoke needs a `container` for the air to fill`                                | No `container`.                                                                          |
| `addSmoke: a simulation can have one smoke source`                                       | `addSmoke` called twice.                                                                 |
| `<call>: gas and liquid can’t be simulated together, …`                                  | Smoke mixed with fluid, soft bodies, or cloth.                                           |
| `Simulation.particleSystem is created on the first step` (also `Simulation.loop`)        | Accessed before `start()`.                                                               |
| `bakeMeshToSdf: mesh appears non-watertight — …`                                         | `addMesh` mesh has holes.                                                                |
| `SDFCollider.setTransform: scale must be uniform`                                        | `addMesh` mesh has non-uniform scale.                                                    |

Errors thrown during the build reject the promise returned by `start()` or `step()`.

---

## Limitations

- Everything must be added before the first `start()` or `step()`. Nothing can be added or removed afterwards; build a new `Simulation` instead.
- One particle radius for the whole simulation. Small soft bodies in a large pool get few particles.
- Smoke can't share a simulation with fluid, soft bodies, or cloth. Use a second `Simulation`.
- One smoke source per simulation.
- The container can't move. Tilt `gravity` or use moving `addBox` obstacles instead.
- `follow` ignores the followed object's scale.
- `addMesh` shapes are rigid and baked once. The mesh can move and rotate, but deforming it has no effect.
- Custom [materials](./extending.md) can't be added to a `Simulation`. Build the scene with the [low-level API](./core.md) instead.
- Solid-to-solid contact friction is fixed at `muS 0.3, muK 0.2`. Cloth drag, lift, and damping are fixed at construction; change them on [`cloth.clothSystem`](./cloth-system.md) after `start()`.
- Without a `container`, the liquid surface is only drawn inside a box around the starting particles, and a warning is logged.
