[Docs](../README.md) › [API](../README.md#api-reference) › Core

# Core

The classes that `Simulation` is built on. Use them when you want to assemble a scene by hand. `ParticleSystem` holds the particles, `SimLoop` runs the solver, `FrameStepper` keeps the solver in step with real time, and `createParticleMesh` draws the particles.

```ts
import {
  ParticleSystem,
  SimLoop,
  FrameStepper,
  createParticleMesh,
  assertRange,
} from 'threejs-particle-fluids';
```

- [`ParticleSystem`](#particlesystem) with [`ParticleInit`](#particleinit), [`ParticleRange`](#particlerange), [`ParticleSnapshot`](#particlesnapshot)
- [`assertRange`](#assertrange)
- [`SimLoop`](#simloop) with [`SimLoopOptions`](#simloopoptions), [`ContactOptions`](#contactoptions), [`SimLoopOverflow`](#simloopoverflow), [step order](#step-order)
- [`FrameStepper`](#framestepper) with [`FrameStepperOptions`](#framestepperoptions), [`FrameStepperResult`](#framestepperresult)
- [`createParticleMesh`](#createparticlemesh) with [`ParticleMeshOptions`](#particlemeshoptions)

## ParticleSystem

Holds every particle in a simulation in GPU buffers. All particles have the same radius, so a single grid can find their neighbors. Each material owns a [`ParticleRange`](#particlerange) of the slots, and a [`SimLoop`](#simloop) advances all of them together.

### Constructor

```ts
new ParticleSystem(renderer: WebGPURenderer, capacity: number, particleRadius: number)
```

| Parameter        | Type             | Description                                              |
| ---------------- | ---------------- | -------------------------------------------------------- |
| `renderer`       | `WebGPURenderer` | Renderer that runs the compute kernels.                  |
| `capacity`       | `number`         | Number of particle slots. Positive integer.              |
| `particleRadius` | `number`         | Radius shared by every particle, in m. Positive, finite. |

Every slot starts at the origin, at rest, pinned (`invMass = 0`), in collision group 0, with no rotation.

| Throws                                                            | When                                                    |
| ----------------------------------------------------------------- | ------------------------------------------------------- |
| `ParticleSystem: capacity must be a positive integer`             | `capacity` is zero, negative, or not a whole number.    |
| `ParticleSystem: particleRadius must be a positive finite number` | `particleRadius` is zero, negative, `NaN`, or infinite. |

### Properties

Every buffer is a TSL storage node (`StorageBufferNode`) with one element per slot. After changing a buffer's `.value.array` on the CPU, set `.value.needsUpdate = true`.

| Property             | Type                         | Access    | Description                                                                                             |
| -------------------- | ---------------------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| `renderer`           | `WebGPURenderer`             | read-only | Renderer passed to the constructor.                                                                     |
| `capacity`           | `number`                     | read-only | Number of slots.                                                                                        |
| `particleRadius`     | `number`                     | read-only | Shared radius, in m.                                                                                    |
| `positions`          | `StorageBufferNode<'vec4'>`  | read-only | Position, in m, in `xyz`. `w` unused.                                                                   |
| `predictedPositions` | `StorageBufferNode<'vec4'>`  | read-only | Positions the solver works on during a substep, in m.                                                   |
| `velocities`         | `StorageBufferNode<'vec4'>`  | read-only | Velocity, in m/s, in `xyz`.                                                                             |
| `invMass`            | `StorageBufferNode<'float'>` | read-only | Inverse mass, in 1/kg. `0` pins the particle.                                                           |
| `collisionGroup`     | `StorageBufferNode<'uint'>`  | read-only | See [`ParticleInit.collisionGroup`](#particleinit).                                                     |
| `boundaryVolume`     | `StorageBufferNode<'float'>` | read-only | Boundary volume, in m³. Non-zero for particles a `FluidSystem` treats as solid boundary.                |
| `rotation`           | `StorageBufferNode<'vec4'>`  | read-only | Orientation as a unit quaternion `(x, y, z, w)`. Only changed by soft bodies with local shape matching. |
| `predictedRotation`  | `StorageBufferNode<'vec4'>`  | read-only | Orientation being solved during a substep.                                                              |
| `angularVelocity`    | `StorageBufferNode<'vec4'>`  | read-only | Angular velocity, in rad/s, in `xyz`.                                                                   |
| `disposed`           | `boolean`                    | read-only | `true` after `dispose()`.                                                                               |

### Methods

#### `uploadParticles(data, start?)`

```ts
uploadParticles(data: readonly ParticleInit[], start?: number): void
```

Write initial state into the slots starting at `start`, which defaults to 0. This sets `positions`, `predictedPositions`, `velocities`, `invMass`, and `collisionGroup`. An empty `data` writes nothing, but `start` still has to be a whole number from 0 to `capacity`.

| Throws                                                     | When                                                                       |
| ---------------------------------------------------------- | -------------------------------------------------------------------------- |
| `ParticleSystem.uploadParticles: invalid particle range`   | `start` is negative or not a whole number, or `data` runs past `capacity`. |
| `ParticleSystem.uploadParticles: invMass must be …`        | An `invMass` is negative, `NaN`, or infinite.                              |
| `ParticleSystem.uploadParticles: collisionGroup must be …` | A `collisionGroup` isn't a whole number from 0 to 2³² − 1.                 |
| `ParticleSystem has been disposed`                         | Called after `dispose()`.                                                  |

#### `setInvMass(range, invMass)`

```ts
setInvMass(range: ParticleRange, invMass: number): void
```

Set the inverse mass, in 1/kg, of every particle in `range`.

| Throws                                              | When                                       |
| --------------------------------------------------- | ------------------------------------------ |
| `ParticleSystem.setInvMass: invalid particle range` | See [`assertRange`](#assertrange).         |
| `ParticleSystem.setInvMass: invMass must be …`      | `invMass` is negative, `NaN`, or infinite. |
| `ParticleSystem has been disposed`                  | Called after `dispose()`.                  |

#### `setCollisionGroup(range, group)`

```ts
setCollisionGroup(range: ParticleRange, group: number): void
```

Set the collision group of every particle in `range`.

| Throws                                                       | When                                            |
| ------------------------------------------------------------ | ----------------------------------------------- |
| `ParticleSystem.setCollisionGroup: invalid particle range`   | See [`assertRange`](#assertrange).              |
| `ParticleSystem.setCollisionGroup: collisionGroup must be …` | `group` isn't a whole number from 0 to 2³² − 1. |
| `ParticleSystem has been disposed`                           | Called after `dispose()`.                       |

#### `readback()`

```ts
readback(): Promise<ParticleSnapshot>
```

Copy the particle state from the GPU. This waits for the GPU. Before the first step, it returns what you uploaded.

| Throws                             | When                      |
| ---------------------------------- | ------------------------- |
| `ParticleSystem has been disposed` | Called after `dispose()`. |

#### `dispose()`

```ts
dispose(): void
```

Free the particle buffers on the GPU. Dispose every loop, material, collider, and mesh that reads these buffers first, because their kernels can't run afterwards. Afterwards, `uploadParticles`, `setInvMass`, `setCollisionGroup`, and `readback` throw, and so do `SimLoop.step` and new `SimLoop`s on these particles. Calling `dispose()` again does nothing.

### ParticleInit

Initial state for one particle, passed to [`uploadParticles`](#uploadparticlesdata-start).

| Field            | Type                                | Default     | Description                                                                                     |
| ---------------- | ----------------------------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| `position`       | `readonly [number, number, number]` | required    | Position, in m.                                                                                 |
| `velocity`       | `readonly [number, number, number]` | `[0, 0, 0]` | Velocity, in m/s.                                                                               |
| `invMass`        | `number`                            | `1`         | Inverse mass, in 1/kg. `0` pins the particle.                                                   |
| `collisionGroup` | `number`                            | `0`         | Particles sharing a non-zero group never collide with each other. `0` collides with everything. |

### ParticleRange

A block of `count` consecutive slots, beginning at `start`.

| Field   | Type     | Description      |
| ------- | -------- | ---------------- |
| `start` | `number` | First slot.      |
| `count` | `number` | Number of slots. |

### ParticleSnapshot

A CPU copy of the particle state, returned by [`readback`](#readback). Vector fields hold four floats per particle, and `invMass` holds one.

| Field                | Type           | Description                                  |
| -------------------- | -------------- | -------------------------------------------- |
| `capacity`           | `number`       | Number of slots.                             |
| `positions`          | `Float32Array` | `xyzw` per particle, in m.                   |
| `predictedPositions` | `Float32Array` | `xyzw` per particle, in m.                   |
| `velocities`         | `Float32Array` | `xyzw` per particle, in m/s.                 |
| `invMass`            | `Float32Array` | One value per particle, in 1/kg.             |
| `rotation`           | `Float32Array` | Unit quaternion `(x, y, z, w)` per particle. |
| `predictedRotation`  | `Float32Array` | Unit quaternion `(x, y, z, w)` per particle. |
| `angularVelocity`    | `Float32Array` | `xyzw` per particle, in rad/s.               |

`collisionGroup` and `boundaryVolume` are not included.

## assertRange

```ts
assertRange(
  particles: { readonly capacity: number },
  range: ParticleRange,
  context: string,
): void
```

Throw unless `range` is a non-empty block of whole slots inside `particles`. Use it to check ranges in your own materials.

| Parameter   | Type                              | Description                                    |
| ----------- | --------------------------------- | ---------------------------------------------- |
| `particles` | `{ capacity: number }`            | Usually a [`ParticleSystem`](#particlesystem). |
| `range`     | [`ParticleRange`](#particlerange) | Range to check.                                |
| `context`   | `string`                          | Prefix for the error message.                  |

| Throws                                                           | When                                                                                                                          |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `<context>: invalid particle range start=… count=… (capacity …)` | `start` or `count` isn't a whole number, `start` is negative, `count` is zero or negative, or the range runs past `capacity`. |

## SimLoop

Moves a [`ParticleSystem`](#particlesystem) and every material and collider attached to it forward in time. It uses extended position-based dynamics (XPBD) with substeps. Each step is sent to the GPU in a single `computeAsync` call.

### Constructor

```ts
new SimLoop(particles: ParticleSystem, options?: SimLoopOptions)
```

| Parameter   | Type                                | Description           |
| ----------- | ----------------------------------- | --------------------- |
| `particles` | [`ParticleSystem`](#particlesystem) | Particles to advance. |
| `options`   | [`SimLoopOptions`](#simloopoptions) | See below.            |

The constructor calls each material's `build` in list order. It creates a neighbor grid if any material declares a `neighborRadius` or contacts are on.

| Throws                                                                          | When                                                                                                             |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `SimLoop: substeps must be an integer ≥ 1`                                      | `substeps` is less than 1 or not a whole number.                                                                 |
| `SimLoop: iterations must be an integer ≥ 0`                                    | `iterations` is negative or not a whole number.                                                                  |
| `SimLoop: the ParticleSystem has been disposed`                                 | You passed particles that were already disposed.                                                                 |
| `SimLoop: every collider must be built for the same ParticleSystem`             | A collider was built for a different `ParticleSystem`.                                                           |
| `SimLoop: every material must be built for the same ParticleSystem`             | A material has a `particles` property that points to a different `ParticleSystem`.                               |
| `SimLoop: a material's neighborRadius must be finite and ≥ 0`                   | A material's `neighborRadius` is negative, `NaN`, or infinite.                                                   |
| `SimLoop: a material used the neighbor grid without declaring a neighborRadius` | A material used `context.hashGrid` in `build`, but no material declares a `neighborRadius` and contacts are off. |
| `SimLoop: friction coefficients must be non-negative`                           | `contact.muS` or `contact.muK` is negative or `NaN`.                                                             |
| `ContactBuffer: maxContacts must be a positive integer`                         | `contact.maxContacts` isn't a positive whole number.                                                             |

### SimLoopOptions

| Option       | Type                                                 | Default         | Description                                                                                                                                                                                                                                                     |
| ------------ | ---------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `substeps`   | `number`                                             | `4`             | Substeps per `step`. Integer ≥ 1.                                                                                                                                                                                                                               |
| `iterations` | `number`                                             | `2`             | Constraint iterations per substep. Integer ≥ 0.                                                                                                                                                                                                                 |
| `gravity`    | `Vector3`                                            | `(0, -9.81, 0)` | Gravity, in m/s². Copied.                                                                                                                                                                                                                                       |
| `materials`  | `readonly` [`Material`](./extending.md#material)`[]` | `[]`            | Physics to run. Kernels run in list order.                                                                                                                                                                                                                      |
| `colliders`  | `readonly` [`Collider`](./colliders.md#collider)`[]` | `[]`            | Shapes particles collide with. Must share `particles`.                                                                                                                                                                                                          |
| `contact`    | `boolean \|` [`ContactOptions`](#contactoptions)     | `false`         | Particle–particle contacts. `true` uses the `ContactOptions` defaults. Soft bodies and cloth need contacts to touch each other. Fluids keep their own particles apart without them.                                                                             |
| `hashOrigin` | `Vector3`                                            | `(0, 0, 0)`     | Origin of the neighbor grid's cells, in m. Copied. Neighbor queries slow down for particles more than 512 cells from it on any axis. If your scene is far from the world origin, set this near its middle. You can move it later through `hashGrid.hashOrigin`. |

### ContactOptions

| Option        | Type     | Default        | Description                                                   |
| ------------- | -------- | -------------- | ------------------------------------------------------------- |
| `muS`         | `number` | `0.5`          | Static friction coefficient between particles. ≥ 0.           |
| `muK`         | `number` | `0.4`          | Kinetic friction coefficient between particles. ≥ 0.          |
| `maxContacts` | `number` | `8 × capacity` | Most contact pairs kept per substep. Extra pairs are dropped. |

Contacts look for neighbors within `2.2 × particleRadius`.

### Properties

| Property     | Type                                                 | Access              | Description                                                                                                                                                                                                                          |
| ------------ | ---------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `particles`  | [`ParticleSystem`](#particlesystem)                  | read-only           | Particles passed to the constructor.                                                                                                                                                                                                 |
| `iterations` | `number`                                             | read-only           | Constraint iterations per substep.                                                                                                                                                                                                   |
| `substeps`   | `number`                                             | read/write          | Substeps per `step`. A new value takes effect on the next step, without recompiling. Setting it throws `SimLoop: substeps must be an integer ≥ 1` for an invalid value, and `SimLoop: the loop has been disposed` after `dispose()`. |
| `gravity`    | `Vector3`                                            | read-only reference | Gravity, in m/s². Mutate in place to change it.                                                                                                                                                                                      |
| `dt`         | `UniformNode<'float', number>`                       | read-only           | Substep length, in s, shared by every kernel. `step` sets it to `dt / substeps`.                                                                                                                                                     |
| `hashGrid`   | [`HashGrid`](./extending.md#hashgrid) `\| undefined` | read-only           | Neighbor grid. Present when a material declares a `neighborRadius` or contacts are on. Its cell size is the largest of those radii. Its table has the default size for `capacity`, at most 1,048,576 buckets.                        |
| `contacts`   | `ContactBuffer \| undefined`                         | read-only           | Storage for contact pairs. Present when contacts are on. `readbackCount()` gives the number of pairs found in the last substep.                                                                                                      |

### Methods

#### `step(dt)`

```ts
step(dt: number): Promise<void>
```

Advance the simulation by `dt` seconds. It calls `update(dt)` on every material and collider, then runs the step on the GPU. The first call also runs the one-time `init` kernels. The promise resolves once the GPU work is complete.

| Throws                                              | When                                        |
| --------------------------------------------------- | ------------------------------------------- |
| `SimLoop: step dt must be a positive finite number` | `dt` is zero, negative, `NaN`, or infinite. |
| `SimLoop: the loop has been disposed`               | Called after `dispose()`.                   |
| `SimLoop: the ParticleSystem has been disposed`     | Called after `particles.dispose()`.         |

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<SimLoopOverflow>
```

Report whether the last `step` lost any corrections or contacts because something overflowed. This waits for the GPU, and throws `SimLoop: the loop has been disposed` after `dispose()`. A material's own buffers, such as a `NeighborList`, aren't covered, so check those on the material.

#### `dispose()`

```ts
dispose(): void
```

Free the GPU buffers the loop created, which are its neighbor grid, contact storage, and correction sums. Afterwards, `step`, `readbackOverflow`, and the `substeps` setter throw. Materials, colliders, and the `ParticleSystem` have to be disposed separately. Calling it again does nothing.

### SimLoopOverflow

Returned by [`readbackOverflow`](#readbackoverflow). A field is always `false` when the loop doesn't have that kind of buffer.

| Field        | Type      | Description                                                                                                  |
| ------------ | --------- | ------------------------------------------------------------------------------------------------------------ |
| `positions`  | `boolean` | A position correction from a collider or contact hit the fixed-point limit of 10 m per axis in some substep. |
| `velocities` | `boolean` | A friction correction hit the fixed-point limit of 50 m/s per axis in some substep.                          |
| `grid`       | `boolean` | At the last grid rebuild, a particle was more than 512 cells from the grid's `hashOrigin` on an axis.        |
| `contacts`   | `boolean` | The last substep found more than `maxContacts` contact pairs and dropped the rest.                           |

### Step order

What one `step(dt)` runs, in order. Materials run in the order of `options.materials`, and colliders in the order of `options.colliders`.

On the CPU:

1. Each material's `update(dt)`, then each collider's `update(dt)`.
2. `loop.dt` is set to `dt / substeps`.

On the GPU, first step only:

3. A grid rebuild, if there is a grid and any material has `init` kernels. Then each material's `init` kernels.

On the GPU, once per step:

4. The correction sums' overflow flags are cleared, if the sums exist. Then each collider's `frameStart` kernels, then each material's `beforeStep` kernels.

On the GPU, once per substep:

5. **Predict.** Every particle with `invMass > 0` gets gravity added to its velocity. Its predicted position is where that velocity carries it in `dt`.
6. **Pre-solve.** The position sums are reset and the grid is rebuilt. With contacts on, contact pairs are found and overlapping pairs are pushed apart in both `positions` and `predictedPositions`. Then each collider's `preSolve`, then each material's `preSolve`.
7. **Solve**, repeated `iterations` times. Each material's `solve` kernels, then the contact solve, then each collider's `solve`. The summed position corrections are then added to `predictedPositions`.
8. **Advect.** Every particle with `invMass > 0` gets a new velocity from how far it moved this substep, and moves to its predicted position.
9. **Post-solve.** Each material's `postSolve` kernels.
10. **Friction.** The velocity sums are reset, contact friction and each collider's `postSolve` run, and the summed velocity corrections are added to `velocities`.
11. Each collider's `substepEnd` kernels.

The sums in steps 6, 7, and 10 exist only when contacts are on or the loop has at least one collider.

### Example

Create the particles first, then the material and colliders, and the loop last.

```ts
import { Vector3 } from 'three';
import {
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleMesh,
  createParticleRenderer,
  type ParticleInit,
} from 'threejs-particle-fluids';

const renderer = await createParticleRenderer();

// Particles, spaced one diameter apart.
const radius = 0.012;
const water: ParticleInit[] = [];
for (let x = -0.48; x < -0.1; x += radius * 2)
  for (let y = radius; y < 0.5; y += radius * 2)
    for (let z = -0.28; z < 0.28; z += radius * 2) water.push({ position: [x, y, z] });
const particles = new ParticleSystem(renderer, water.length, radius);
particles.uploadParticles(water);

// Material and colliders, built on the same ParticleSystem.
const fluid = new FluidSystem(particles, { viscosity: 0.02 });
const walls = new PrimitiveSet(particles);
walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
walls.addPlane(new Vector3(1, 0, 0), new Vector3(-0.5, 0, 0));
walls.addPlane(new Vector3(-1, 0, 0), new Vector3(0.5, 0, 0));

// The loop, created last.
const loop = new SimLoop(particles, { substeps: 3, materials: [fluid], colliders: [walls] });
scene.add(createParticleMesh(particles));

async function frame() {
  await loop.step(1 / 60);
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

## FrameStepper

Runs a simulation at real-time speed in steps of one fixed length. Each `pump` adds the elapsed time to a running total and runs as many whole `fixedDt` steps as fit. The solver sees the same timestep at any frame rate.

```ts
const stepper = new FrameStepper({ fixedDt: 1 / 60 });
async function frame(now: number) {
  await stepper.pump(now, (dt) => loop.step(dt));
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

### Constructor

```ts
new FrameStepper(options: FrameStepperOptions)
```

| Parameter | Type                                          | Description |
| --------- | --------------------------------------------- | ----------- |
| `options` | [`FrameStepperOptions`](#framestepperoptions) | See below.  |

| Throws                                                      | When                                              |
| ----------------------------------------------------------- | ------------------------------------------------- |
| `FrameStepper: fixedDt must be positive`                    | `fixedDt` is zero, negative, `NaN`, or infinite.  |
| `FrameStepper: maxStepsPerFrame must be a positive integer` | `maxStepsPerFrame` isn't a positive whole number. |

### FrameStepperOptions

| Option             | Type     | Default  | Description                                                                                |
| ------------------ | -------- | -------- | ------------------------------------------------------------------------------------------ |
| `fixedDt`          | `number` | required | Simulation time per step, in s.                                                            |
| `maxStepsPerFrame` | `number` | `4`      | Most steps run in one `pump`. Elapsed time beyond `maxStepsPerFrame × fixedDt` is dropped. |

### FrameStepperResult

| Field              | Type      | Description                                   |
| ------------------ | --------- | --------------------------------------------- |
| `steps`            | `number`  | Steps run by this call.                       |
| `truncated`        | `boolean` | `true` if elapsed time was dropped this call. |
| `remainderSeconds` | `number`  | Time carried to the next call, in s.          |

### Properties

| Property           | Type     | Access    | Description                     |
| ------------------ | -------- | --------- | ------------------------------- |
| `fixedDt`          | `number` | read-only | Simulation time per step, in s. |
| `maxStepsPerFrame` | `number` | read-only | Most steps per `pump`.          |

### Methods

#### `pump(nowMs, step)`

```ts
pump(nowMs: number, step: (dt: number) => Promise<void>): Promise<FrameStepperResult>
```

Run as many `fixedDt` steps as the time since the previous call allows. Each `step` is awaited before the next one starts. `nowMs` is the current time in ms, such as `performance.now()`. The first call only starts the clock and returns `{ steps: 0, truncated: false, remainderSeconds: 0 }`, and so does the first call after `reset()`. If the clock goes backwards, the elapsed time counts as 0.

#### `reset()`

```ts
reset(): void
```

Clear the running total and the clock after a pause, so the next `pump` doesn't try to catch up.

## createParticleMesh

```ts
createParticleMesh(particles: ParticleSystem, options?: ParticleMeshOptions): InstancedMesh
```

Draw particles as instanced spheres. Each sphere reads its center straight from `particles.positions` on the GPU, so there's no per-frame CPU work. It returns an `InstancedMesh` with a `MeshPhongNodeMaterial`, `count` set to `range.count`, and `frustumCulled` set to `false`.

| Parameter   | Type                                          | Description        |
| ----------- | --------------------------------------------- | ------------------ |
| `particles` | [`ParticleSystem`](#particlesystem)           | Particles to draw. |
| `options`   | [`ParticleMeshOptions`](#particlemeshoptions) | See below.         |

| Throws                                       | When                                                                    |
| -------------------------------------------- | ----------------------------------------------------------------------- |
| `createParticleMesh: invalid particle range` | `options.range` isn't a valid range. See [`assertRange`](#assertrange). |

### ParticleMeshOptions

| Option           | Type                              | Default                | Description                                                                      |
| ---------------- | --------------------------------- | ---------------------- | -------------------------------------------------------------------------------- |
| `range`          | [`ParticleRange`](#particlerange) | all particles          | Particles to draw.                                                               |
| `radius`         | `number`                          | `0.9 × particleRadius` | Sphere radius, in m.                                                             |
| `color`          | `number \| string`                | `0x5fb9ff`             | Sphere color. Ignored when `colorNode` is given.                                 |
| `colorNode`      | `(position) => node`              | none                   | Takes the particle's center (a TSL `vec3`, in m) and returns a TSL `vec3` color. |
| `widthSegments`  | `number`                          | `8`                    | Sphere segments around.                                                          |
| `heightSegments` | `number`                          | `6`                    | Sphere segments top to bottom.                                                   |

## Limitations

- You can't change `capacity` or `particleRadius` after a `ParticleSystem` is constructed.
- `readback()` can't return `collisionGroup` or `boundaryVolume`.
- You can't change a `SimLoop`'s materials, colliders, contacts, or `iterations` after it's constructed. Only `substeps` and `gravity` can change.
- The loop hands out collision groups above the highest one in the CPU copy of `collisionGroup` when it's constructed. Groups you set later, or write only on the GPU, can clash with them, so upload your particles before you create the loop.
- The loop can only check a material's particles if it has a `particles` property. `GasSystem` and `ViscositySolver` report their `FluidSystem`'s particles.
- The grid's table can't grow past 1,048,576 buckets, which it reaches at 524,288 particles. Beyond that, neighbor queries slow down.
- Neighbor lookups stay fast only within `512 × cellSize` of `hashOrigin` on each axis. Particles farther out still find their neighbors, more slowly, and `readbackOverflow().grid` reports it.
- A substep can't keep more than `maxContacts` contact pairs. The rest are dropped, and `readbackOverflow().contacts` reports it.
- Collider and contact corrections can't exceed 10 m per axis per iteration for positions, or 50 m/s per axis per substep for velocities, because they're summed in fixed point. `readbackOverflow()` reports it.
- Unused slots still cost time, because the predict, advect, and sum kernels run over every slot up to `capacity`.
- The GPU work in each step grows with `substeps × iterations` times the number of `solve` kernels.
- `FrameStepper` can't run more than `maxStepsPerFrame` steps per call. When steps take too long, the extra time is dropped and the simulation runs slower than real time.
- The mesh from `createParticleMesh` applies its own transform on top of the simulated positions.
