[Docs](../README.md) › [API](../README.md#api-reference) › Core

# Core

Particle storage, the XPBD solver loop, a fixed-timestep driver, and a debug particle mesh. These are the classes `Simulation` builds on; use them directly to assemble a scene by hand.

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

GPU storage for every particle in a simulation. All particles share one radius, so one uniform grid can find their neighbors. Materials each own a [`ParticleRange`](#particlerange) of the buffers, and a [`SimLoop`](#simloop) advances all of them together.

### Constructor

```ts
new ParticleSystem(renderer: WebGPURenderer, capacity: number, particleRadius: number)
```

| Parameter        | Type             | Description                                              |
| ---------------- | ---------------- | -------------------------------------------------------- |
| `renderer`       | `WebGPURenderer` | Renderer that runs the compute kernels.                  |
| `capacity`       | `number`         | Number of particle slots. Positive integer.              |
| `particleRadius` | `number`         | Radius shared by every particle, in m. Positive, finite. |

Every slot starts at the origin with zero velocity, `invMass = 0`, collision group `0`, and identity rotation.

| Throws                                                            | When                                         |
| ----------------------------------------------------------------- | -------------------------------------------- |
| `ParticleSystem: capacity must be a positive integer`             | `capacity` is not an integer or is ≤ 0.      |
| `ParticleSystem: particleRadius must be a positive finite number` | `particleRadius` is ≤ 0, `NaN`, or infinite. |

### Properties

All buffers are TSL storage nodes (`StorageBufferNode`) with one element per slot. After writing a buffer's `.value.array` on the CPU, set `.value.needsUpdate = true`.

| Property             | Type                         | Access    | Description                                                                                             |
| -------------------- | ---------------------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| `renderer`           | `WebGPURenderer`             | read-only | Renderer passed to the constructor.                                                                     |
| `capacity`           | `number`                     | read-only | Number of slots.                                                                                        |
| `particleRadius`     | `number`                     | read-only | Shared radius, in m.                                                                                    |
| `positions`          | `StorageBufferNode<'vec4'>`  | read-only | Position, in m, in `xyz`. `w` unused.                                                                   |
| `predictedPositions` | `StorageBufferNode<'vec4'>`  | read-only | Positions being solved during a substep (`x*`), in m.                                                   |
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

Write initial state into slots `[start, start + data.length)`. `start` defaults to `0`. Sets `positions`, `predictedPositions`, `velocities`, `invMass`, and `collisionGroup`. An empty `data` writes nothing; `start` must still be an integer in `[0, capacity]`.

| Throws                                                     | When                                                                 |
| ---------------------------------------------------------- | -------------------------------------------------------------------- |
| `ParticleSystem.uploadParticles: invalid particle range`   | The slots fall outside `[0, capacity)` or `start` is not an integer. |
| `ParticleSystem.uploadParticles: invMass must be …`        | An `invMass` is negative, `NaN`, or infinite.                        |
| `ParticleSystem.uploadParticles: collisionGroup must be …` | A `collisionGroup` is not an integer in `[0, 2³² − 1]`.              |
| `ParticleSystem has been disposed`                         | Called after `dispose()`.                                            |

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

Set the collision group of every particle in `range`. Stored as `uint32`.

| Throws                                                       | When                                         |
| ------------------------------------------------------------ | -------------------------------------------- |
| `ParticleSystem.setCollisionGroup: invalid particle range`   | See [`assertRange`](#assertrange).           |
| `ParticleSystem.setCollisionGroup: collisionGroup must be …` | `group` is not an integer in `[0, 2³² − 1]`. |
| `ParticleSystem has been disposed`                           | Called after `dispose()`.                    |

#### `readback()`

```ts
readback(): Promise<ParticleSnapshot>
```

Copy the particle state from the GPU. Waits for the GPU. A buffer no kernel has used yet is copied from its CPU array.

| Throws                             | When                      |
| ---------------------------------- | ------------------------- |
| `ParticleSystem has been disposed` | Called after `dispose()`. |

#### `dispose()`

```ts
dispose(): void
```

Free the particle buffers on the GPU. Later calls to `uploadParticles`, `setInvMass`, `setCollisionGroup`, and `readback` throw, as do `SimLoop.step` and new `SimLoop`s on these particles. Dispose loops, materials, colliders, and meshes that read the buffers first; their kernels can't run afterwards. Calling it again does nothing.

### ParticleInit

Initial state for one particle, passed to [`uploadParticles`](#uploadparticlesdata-start).

| Field            | Type                                | Default     | Description                                                                                     |
| ---------------- | ----------------------------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| `position`       | `readonly [number, number, number]` | required    | Position, in m.                                                                                 |
| `velocity`       | `readonly [number, number, number]` | `[0, 0, 0]` | Velocity, in m/s.                                                                               |
| `invMass`        | `number`                            | `1`         | Inverse mass, in 1/kg. `0` pins the particle.                                                   |
| `collisionGroup` | `number`                            | `0`         | Particles sharing a non-zero group never collide with each other. `0` collides with everything. |

### ParticleRange

A contiguous block of slots, `[start, start + count)`.

| Field   | Type     | Description      |
| ------- | -------- | ---------------- |
| `start` | `number` | First slot.      |
| `count` | `number` | Number of slots. |

### ParticleSnapshot

CPU copy of the particle state, returned by [`readback`](#readback). Vector fields hold 4 floats per particle; `invMass` holds 1.

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

Throw unless `range` is a non-empty block of whole slots inside `particles`. For validating ranges in custom materials.

| Parameter   | Type                              | Description                                    |
| ----------- | --------------------------------- | ---------------------------------------------- |
| `particles` | `{ capacity: number }`            | Usually a [`ParticleSystem`](#particlesystem). |
| `range`     | [`ParticleRange`](#particlerange) | Range to check.                                |
| `context`   | `string`                          | Prefix for the error message.                  |

| Throws                                                           | When                                                                                           |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `<context>: invalid particle range start=… count=… (capacity …)` | `start` or `count` is not an integer, `start < 0`, `count ≤ 0`, or `start + count > capacity`. |

## SimLoop

Advances a [`ParticleSystem`](#particlesystem) and every material and collider attached to it with extended position-based dynamics (XPBD) and substepping. All kernels for one step are submitted to the GPU in one `computeAsync` call.

### Constructor

```ts
new SimLoop(particles: ParticleSystem, options?: SimLoopOptions)
```

| Parameter   | Type                                | Description           |
| ----------- | ----------------------------------- | --------------------- |
| `particles` | [`ParticleSystem`](#particlesystem) | Particles to advance. |
| `options`   | [`SimLoopOptions`](#simloopoptions) | See below.            |

The constructor calls each material's `build` in list order, builds a neighbor grid if any material declares a `neighborRadius` or contacts are on, and compiles the kernel list.

| Throws                                                                          | When                                                                                |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `SimLoop: substeps must be an integer ≥ 1`                                      | `substeps` is not an integer or is < 1.                                             |
| `SimLoop: iterations must be an integer ≥ 0`                                    | `iterations` is not an integer or is < 0.                                           |
| `SimLoop: the ParticleSystem has been disposed`                                 | `particles.dispose()` was called.                                                   |
| `SimLoop: every collider must be built for the same ParticleSystem`             | A collider's `particles` is not `particles`.                                        |
| `SimLoop: every material must be built for the same ParticleSystem`             | A material's `particles` is set and is not `particles`.                             |
| `SimLoop: a material's neighborRadius must be finite and ≥ 0`                   | A material's `neighborRadius` is negative, `NaN`, or infinite.                      |
| `SimLoop: a material used the neighbor grid without declaring a neighborRadius` | A material read a property of `context.hashGrid` during `build` and no grid exists. |
| `SimLoop: friction coefficients must be non-negative`                           | `contact.muS` or `contact.muK` is negative or `NaN`.                                |
| `ContactBuffer: maxContacts must be a positive integer`                         | `contact.maxContacts` is not a positive integer.                                    |

### SimLoopOptions

| Option       | Type                                                 | Default         | Description                                                                                                                                                                                                                                      |
| ------------ | ---------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `substeps`   | `number`                                             | `4`             | Substeps per `step`. Integer ≥ 1.                                                                                                                                                                                                                |
| `iterations` | `number`                                             | `2`             | Constraint iterations per substep. Integer ≥ 0.                                                                                                                                                                                                  |
| `gravity`    | `Vector3`                                            | `(0, -9.81, 0)` | Gravity, in m/s². Copied.                                                                                                                                                                                                                        |
| `materials`  | `readonly` [`Material`](./extending.md#material)`[]` | `[]`            | Physics to run. Kernels run in list order.                                                                                                                                                                                                       |
| `colliders`  | `readonly` [`Collider`](./colliders.md#collider)`[]` | `[]`            | Shapes particles collide with. Must share `particles`.                                                                                                                                                                                           |
| `contact`    | `boolean \|` [`ContactOptions`](#contactoptions)     | `false`         | Particle–particle contacts. `true` uses the `ContactOptions` defaults. Needed for soft bodies and cloth to touch each other; fluids keep their own particles apart without it.                                                                   |
| `hashOrigin` | `Vector3`                                            | `(0, 0, 0)`     | Origin of the neighbor grid's cells, in m. Copied. Neighbor queries slow down for particles more than 512 cells from it on any axis, so set it near the middle of scenes far from the world origin. Move it later through `hashGrid.hashOrigin`. |

### ContactOptions

| Option        | Type     | Default        | Description                                                   |
| ------------- | -------- | -------------- | ------------------------------------------------------------- |
| `muS`         | `number` | `0.5`          | Static friction coefficient between particles. ≥ 0.           |
| `muK`         | `number` | `0.4`          | Kinetic friction coefficient between particles. ≥ 0.          |
| `maxContacts` | `number` | `8 × capacity` | Most contact pairs kept per substep. Extra pairs are dropped. |

Contacts search a radius of `2 × particleRadius × 1.1`.

### Properties

| Property     | Type                                                 | Access              | Description                                                                                                                                                                                                          |
| ------------ | ---------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `particles`  | [`ParticleSystem`](#particlesystem)                  | read-only           | Particles passed to the constructor.                                                                                                                                                                                 |
| `iterations` | `number`                                             | read-only           | Constraint iterations per substep.                                                                                                                                                                                   |
| `substeps`   | `number`                                             | read/write          | Substeps per `step`. Setting it takes effect on the next step without recompiling. Throws `SimLoop: substeps must be an integer ≥ 1` on invalid values, and `SimLoop: the loop has been disposed` after `dispose()`. |
| `gravity`    | `Vector3`                                            | read-only reference | Gravity, in m/s². Mutate in place to change it.                                                                                                                                                                      |
| `dt`         | `UniformNode<'float', number>`                       | read-only           | Substep length, in s, shared by every kernel. Set by `step` to `dt / substeps`.                                                                                                                                      |
| `hashGrid`   | [`HashGrid`](./extending.md#hashgrid) `\| undefined` | read-only           | Neighbor grid. Present when a material declares a `neighborRadius` or contacts are on. Cell size is the largest of those radii. Its table has the default size for `capacity`, at most 1,048,576 buckets.            |
| `contacts`   | `ContactBuffer \| undefined`                         | read-only           | Contact pair storage. Present when contacts are on. `readbackCount()` gives the pairs found in the last substep.                                                                                                     |

### Methods

#### `step(dt)`

```ts
step(dt: number): Promise<void>
```

Advance the simulation by `dt` seconds. Calls every material's `update(dt)` and collider's `update(dt)` on the CPU, sets the substep length, then dispatches the step. The first call also runs the one-time init kernels. Resolves when the GPU work is submitted and complete.

| Throws                                              | When                                |
| --------------------------------------------------- | ----------------------------------- |
| `SimLoop: step dt must be a positive finite number` | `dt` is ≤ 0, `NaN`, or infinite.    |
| `SimLoop: the loop has been disposed`               | Called after `dispose()`.           |
| `SimLoop: the ParticleSystem has been disposed`     | Called after `particles.dispose()`. |

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<SimLoopOverflow>
```

Report the fixed-point and capacity overflows of the last `step`. Any `true` means corrections or contacts were lost that step. Waits for the GPU. Throws `SimLoop: the loop has been disposed` after `dispose()`. Overflow in a material's own buffers, such as a `NeighborList`, is read from the material.

#### `dispose()`

```ts
dispose(): void
```

Free the GPU buffers the loop created: its neighbor grid, contact storage, and correction sums. Later calls to `step`, `readbackOverflow`, and the `substeps` setter throw. Materials, colliders, and the `ParticleSystem` are disposed separately. Calling it again does nothing.

### SimLoopOverflow

Returned by [`readbackOverflow`](#readbackoverflow). Each field is `false` when the loop has nothing of that kind.

| Field        | Type      | Description                                                                                              |
| ------------ | --------- | -------------------------------------------------------------------------------------------------------- |
| `positions`  | `boolean` | A collider or contact position correction reached the fixed-point limit (10 m per axis) in some substep. |
| `velocities` | `boolean` | A friction correction reached the fixed-point limit (50 m/s per axis) in some substep.                   |
| `grid`       | `boolean` | At the last grid rebuild, a particle was more than 512 cells from the grid's `hashOrigin` on an axis.    |
| `contacts`   | `boolean` | The last substep found more than `maxContacts` contact pairs; the rest were dropped.                     |

### Step order

What one `step(dt)` runs, in order. Materials run in the order of `options.materials`; colliders in the order of `options.colliders`.

On the CPU:

1. `material.update(dt)` for each material, then `collider.update(dt)` for each collider.
2. `loop.dt` is set to `dt / substeps`.

On the GPU, first step only:

3. Grid rebuild (if a grid exists and any material has `init` kernels), then each material's `init` kernels.

On the GPU, once per step:

4. Clear the correction sums' overflow flags (if the sums exist), each collider's `frameStart` kernels, then each material's `beforeStep` kernels.

On the GPU, once per substep:

5. **Predict.** For every particle with `invMass > 0`: `v += gravity · dt`, `x* = x + v · dt`.
6. **Pre-solve.** Reset the position accumulator; rebuild the grid; with contacts on, find contact pairs and push apart overlapping pairs in both `positions` and `predictedPositions`; each collider's `preSolve`; each material's `preSolve`.
7. **Solve**, repeated `iterations` times: each material's `solve` kernels; the contact solve; each collider's `solve`; then add the accumulated position corrections to `predictedPositions`.
8. **Advect.** For every particle with `invMass > 0`: `v = (x* − x) / dt`, `x = x*`.
9. **Post-solve.** Each material's `postSolve` kernels.
10. **Friction.** Reset the velocity accumulator; contact friction; each collider's `postSolve`; add the accumulated velocity corrections to `velocities`.
11. Each collider's `substepEnd` kernels.

The accumulator steps in 6, 7, and 10 exist only when contacts are on or at least one collider is given.

### Example

Build a scene by hand: particles, then a material, then colliders, then the loop.

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

Runs a fixed-timestep simulation at wall-clock speed. Elapsed time is added to an accumulator and whole steps of `fixedDt` are taken from it, so the solver sees the same timestep at any display frame rate.

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

| Throws                                                      | When                                            |
| ----------------------------------------------------------- | ----------------------------------------------- |
| `FrameStepper: fixedDt must be positive`                    | `fixedDt` is ≤ 0, `NaN`, or infinite.           |
| `FrameStepper: maxStepsPerFrame must be a positive integer` | `maxStepsPerFrame` is not an integer or is ≤ 0. |

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

Run as many `fixedDt` steps as the time since the previous call allows, awaiting each `step` in sequence. `nowMs` is the current time in ms, e.g. `performance.now()`. The first call, and the first call after `reset()`, only starts the clock and returns `{ steps: 0, truncated: false, remainderSeconds: 0 }`. A negative elapsed time counts as `0`.

#### `reset()`

```ts
reset(): void
```

Clear the accumulator and the clock, e.g. after a pause, so the next `pump` does not catch up.

## createParticleMesh

```ts
createParticleMesh(particles: ParticleSystem, options?: ParticleMeshOptions): InstancedMesh
```

Draw particles as instanced spheres. The vertex shader reads each sphere's center from `particles.positions`, so there is no per-frame CPU work. Returns an `InstancedMesh` with a `MeshPhongNodeMaterial`, `count = range.count`, and `frustumCulled = false`.

| Parameter   | Type                                          | Description        |
| ----------- | --------------------------------------------- | ------------------ |
| `particles` | [`ParticleSystem`](#particlesystem)           | Particles to draw. |
| `options`   | [`ParticleMeshOptions`](#particlemeshoptions) | See below.         |

| Throws                                       | When                                         |
| -------------------------------------------- | -------------------------------------------- |
| `createParticleMesh: invalid particle range` | `range` fails [`assertRange`](#assertrange). |

### ParticleMeshOptions

| Option           | Type                              | Default                | Description                                                                 |
| ---------------- | --------------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| `range`          | [`ParticleRange`](#particlerange) | all particles          | Particles to draw.                                                          |
| `radius`         | `number`                          | `0.9 × particleRadius` | Sphere radius, in m.                                                        |
| `color`          | `number \| string`                | `0x5fb9ff`             | Sphere color. Ignored when `colorNode` is given.                            |
| `colorNode`      | `(position) => node`              | none                   | Returns a TSL `vec3` color from the particle's center (a TSL `vec3`, in m). |
| `widthSegments`  | `number`                          | `8`                    | Sphere segments around.                                                     |
| `heightSegments` | `number`                          | `6`                    | Sphere segments top to bottom.                                              |

## Limitations

- `capacity` and `particleRadius` are fixed after a `ParticleSystem` is constructed; every particle has the same radius.
- `readback()` waits for the GPU and does not return `collisionGroup` or `boundaryVolume`.
- Materials, colliders, contacts, and `iterations` are fixed when the `SimLoop` is constructed; only `substeps` and `gravity` change afterwards.
- Create the `SimLoop` after uploading particles: collision groups it hands out start above the highest group in the CPU copy of `collisionGroup` at construction. Groups written to particles afterwards (or only on the GPU) can clash with them.
- The loop checks a material's `particles` only when the material exposes one; `GasSystem` and `ViscositySolver` are checked through their `FluidSystem`.
- Above 524,288 particles the grid's table stops growing at 1,048,576 buckets, so more particles share each bucket and neighbor queries slow down.
- The grid keeps neighbor lookups local only within `512 × cellSize` of `hashOrigin` per axis. Particles beyond still find their neighbors, more slowly; `readbackOverflow().grid` reports it.
- Contact pairs beyond `maxContacts` in a substep are dropped; `readbackOverflow().contacts` reports it.
- Collider and contact corrections are summed in fixed point: position sums overflow above 10 m per axis per iteration, velocity sums above 50 m/s per axis per substep. `readbackOverflow()` reports it.
- Predict, advect, and accumulator kernels run over every slot up to `capacity`, including unused ones.
- GPU work per step scales with `substeps × iterations × (solve kernels)`.
- `FrameStepper` drops elapsed time beyond `maxStepsPerFrame × fixedDt`, so the simulation runs slower than real time when steps take too long.
- The mesh from `createParticleMesh` applies its transform on top of the simulated positions.
