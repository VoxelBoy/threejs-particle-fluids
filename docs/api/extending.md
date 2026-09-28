[Docs](../README.md) › [API](../README.md#api-reference) › Extending

# Extending

How to add your own physics to a [`SimLoop`](./core.md#simloop). A custom material implements `Material` and returns TSL compute kernels. This page also covers the building blocks the built-in materials use: neighbor search, SPH kernels, XPBD constraints, and atomic sums. [`Simulation`](./simulation.md) doesn't accept custom materials, so build a `ParticleSystem` and `SimLoop` yourself.

```ts
import {
  type Material,
  type MaterialKernels,
  type SolverContext,
  emitForEachNeighbor,
  HashGrid,
  NeighborList,
  MAX_NEIGHBORS,
  createSphKernelUniforms,
  emitPoly6,
  emitPoly6FromRSq,
  emitSpikyGrad,
  buildConstraintGroups,
  colorConstraints,
  constraintKernels,
  createDistanceConstraints,
  xpbdDeltaLambda,
  Accumulator,
  toTriangleMesh,
} from 'threejs-particle-fluids';
```

**Contents**

- Materials: [`Material`](#material), [`MaterialKernels`](#materialkernels), [`SolverContext`](#solvercontext), [Dispatch order](#dispatch-order)
- Neighbor search: [`emitForEachNeighbor`](#emitforeachneighbor), [`HashGrid`](#hashgrid), [`NeighborList`](#neighborlist), [`MAX_NEIGHBORS`](#max_neighbors)
- SPH kernels: [`SphKernelUniforms`](#sphkerneluniforms), [`createSphKernelUniforms`](#createsphkerneluniforms), [`emitPoly6`](#emitpoly6), [`emitPoly6FromRSq`](#emitpoly6fromrsq), [`emitSpikyGrad`](#emitspikygrad)
- Constraints: [`ConstraintType`](#constrainttype), [`ConstraintGroup`](#constraintgroup), [`createDistanceConstraints`](#createdistanceconstraints), [`colorConstraints`](#colorconstraints), [`buildConstraintGroups`](#buildconstraintgroups), [`constraintKernels`](#constraintkernels), [`xpbdDeltaLambda`](#xpbddeltalambda)
- Scatter sums: [`Accumulator`](#accumulator), [`ApplyTarget`](#applytarget)
- Meshes: [`TriangleMesh`](#trianglemesh), [`toTriangleMesh`](#totrianglemesh)

## Material

Physics that runs inside a `SimLoop`. Any object with a `build` method is a material. The loop calls `build` once from its constructor, in the order of `SimLoopOptions.materials`, then dispatches the returned kernels every step.

```ts
interface Material {
  readonly neighborRadius?: number;
  readonly particles?: ParticleSystem;
  build(context: SolverContext): MaterialKernels;
  update?(dt: number): void;
}
```

### Properties

| Property         | Type                                                        | Access    | Description                                                                                                                                                          |
| ---------------- | ----------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `neighborRadius` | `number \| undefined`                                       | read-only | Farthest neighbor distance queried through the grid, in m. Finite and ≥ 0. The loop creates a grid only when some material sets a positive value or contacts are on. |
| `particles`      | [`ParticleSystem`](./core.md#particlesystem) `\| undefined` | read-only | Particles the material was built for. If this is set, `SimLoop` throws unless they're the loop's own particles.                                                      |

### Methods

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

Create the material's buffers and kernels. The `SimLoop` constructor calls it once.

| Parameter | Type                              | Description          |
| --------- | --------------------------------- | -------------------- |
| `context` | [`SolverContext`](#solvercontext) | Shared solver state. |

#### `update(dt)`

```ts
update?(dt: number): void
```

An optional CPU hook. `SimLoop.step` calls it on every material, in order, before colliders update and before any GPU work. `dt` is the full step length in s, not the substep length.

### Example

This material damps the velocity of one range. Its kernel dispatches `range.count` threads and offsets each index by `range.start`.

```ts
import { Fn, instanceIndex, uint, uniform, vec4 } from 'three/tsl';
import { SimLoop, type Material, type ParticleRange } from 'threejs-particle-fluids';

function createDamping(range: ParticleRange, rate: number): Material {
  const k = uniform(rate, 'float'); // 1/s
  return {
    build: ({ particles, dt }) => ({
      postSolve: [
        Fn(() => {
          const v = particles.velocities.element(instanceIndex.add(uint(range.start)));
          v.assign(vec4(v.xyz.mul(k.mul(dt).oneMinus().max(0)), v.w));
        })().compute(range.count),
      ],
    }),
  };
}

const loop = new SimLoop(particles, { materials: [fluid, createDamping(fluid.range, 2)] });
```

## MaterialKernels

The kernels returned by [`Material.build`](#buildcontext). Every field is optional. Within a stage, kernels run in material order, then in array order.

| Field            | Type                                       | When `SimLoop` dispatches it                                                                                                             |
| ---------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `init`           | `readonly ComputeNode[]`                   | Once, on the first `step`, before `beforeStep`. A grid rebuild runs first when there is a grid.                                          |
| `beforeStep`     | `readonly ComputeNode[]`                   | Every step, before the first substep and after the colliders' frame-start kernels.                                                       |
| `preSolve`       | `readonly ComputeNode[]`                   | Every substep, after prediction, the grid rebuild, contact generation, and the colliders' pre-solve kernels.                             |
| `solve`          | `readonly ComputeNode[]`                   | Every solver iteration, before contacts and colliders.                                                                                   |
| `postSolve`      | `readonly ComputeNode[]`                   | Every substep, after velocities are updated from the solved positions and before friction.                                               |
| `noSelfContacts` | [`ParticleRange`](./core.md#particlerange) | Particles in this range don't get contacts with each other, but they still contact every other particle. Only used when contacts are on. |

## SolverContext

The shared solver state passed to [`Material.build`](#buildcontext).

| Property                   | Type                                         | Description                                                                                                                                                                                                                                                                                                                          |
| -------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `particles`                | [`ParticleSystem`](./core.md#particlesystem) | Particle storage the loop advances.                                                                                                                                                                                                                                                                                                  |
| `dt`                       | `UniformNode<'float', number>`               | Substep length in s. Set by `SimLoop.step` to `dt / substeps` before dispatch.                                                                                                                                                                                                                                                       |
| `hashGrid`                 | [`HashGrid`](#hashgrid)                      | The loop's neighbor grid, rebuilt from `predictedPositions` every substep. Its cell size is the largest `neighborRadius` of any material, or `2.2 × particleRadius` if contacts are on and that is larger. If the loop has no grid, this is a placeholder that throws when you read any of its properties. Destructuring it is safe. |
| `allocateCollisionGroup()` | `() => number`                               | Reserve a collision group that no other material has and no particle uploaded before the `SimLoop` was constructed uses. Assign it with `particles.setCollisionGroup`. Groups written to particles after the loop was constructed can clash with it.                                                                                 |

| Throws                                                                          | When                                                                                             |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `SimLoop: a material used the neighbor grid without declaring a neighborRadius` | You read a property of `hashGrid`, but no material sets a `neighborRadius` and contacts are off. |

## Dispatch order

What one `SimLoop.step(dt)` runs, from a material's point of view. [Step order](./core.md#step-order) on the Core page has every detail.

1. On the CPU, `update(dt)` on every material, then every collider. The `dt` uniform is set to `dt / substeps`.
2. First step only: a grid rebuild if there is a grid and any `init` kernels, then all `init` kernels.
3. The colliders' frame-start kernels, then all `beforeStep` kernels.
4. Then, `substeps` times:
   1. **Predict.** Unpinned particles get gravity and move by their velocity into `predictedPositions`.
   2. The grid is rebuilt and, with contacts on, contacts are found and separated. Then the colliders' pre-solve kernels, then all `preSolve` kernels.
   3. `iterations` times: all `solve` kernels, then the contact and collider corrections.
   4. **Advect.** Unpinned particles get a velocity from how far they moved, and move to their predicted positions.
   5. All `postSolve` kernels.
   6. Contact and collider friction, then the colliders' substep-end kernels.

The whole step is sent to the GPU in one `computeAsync` call. A velocity change you make in `postSolve` takes effect at the next prediction.

## emitForEachNeighbor

```ts
emitForEachNeighbor(
  grid: HashGrid,
  position: any,
  onCandidate: (neighborIndex: any, sortedSlot: any) => void,
): void
```

Emit TSL that visits every particle in the 27 grid cells around `position`. Call it inside an `Fn` body. Candidates can include particles beyond your query radius or, through hash collisions, in distant cells, so filter by distance. A particle that queries at its own predicted position finds itself as a candidate.

| Parameter     | Type                                  | Description                                                                                                                                                                                                                                                                |
| ------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grid`        | [`HashGrid`](#hashgrid)               | Grid to walk. Reflects `predictedPositions` at its last rebuild.                                                                                                                                                                                                           |
| `position`    | TSL `vec3` node                       | Query point, in m.                                                                                                                                                                                                                                                         |
| `onCandidate` | `(neighborIndex, sortedSlot) => void` | Runs at shader-build time and must emit TSL. `neighborIndex` is the particle index (`uint`). `sortedSlot` is an index into `grid.sortedIndices` and `grid.sortedPredictedPositions` (`uint`). Call `Continue()` to skip a candidate. `Return()` ends the whole invocation. |

## HashGrid

Finds nearby particles by sorting every particle of a `ParticleSystem` into grid cells by its predicted position, on the GPU. `SimLoop` owns one and rebuilds it every substep, so construct one yourself only for standalone tools.

### Constructor

```ts
new HashGrid(particles: ParticleSystem, options: HashGridOptions)
```

| Parameter   | Type                                         | Description       |
| ----------- | -------------------------------------------- | ----------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particles to bin. |
| `options`   | [`HashGridOptions`](#hashgridoptions)        | See below.        |

#### HashGridOptions

| Option          | Type      | Default                                               | Description                                                                                 |
| --------------- | --------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `cellSize`      | `number`  | required                                              | Cell edge length, in m. Must be at least the largest query radius.                          |
| `hashTableSize` | `number`  | next power of two ≥ `2 × capacity`, at most 1,048,576 | Bucket count. A power of two, at most 1,048,576.                                            |
| `hashOrigin`    | `Vector3` | `(0, 0, 0)`                                           | Where the cells start, in m. Copied. Lookups stay fast within 512 cells of it on each axis. |

| Throws                                                                                       | When                                                  |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `HashGrid: cellSize must be a positive finite number`                                        | `cellSize` is zero, negative, `NaN`, or infinite.     |
| `HashGrid: hashTableSize must be a positive power of two`                                    | `hashTableSize` isn't a power of two (1, 2, 4, 8, …). |
| `HashGrid: hashTableSize=… (capacity …) exceeds the 1048576-bucket limit of the prefix scan` | `hashTableSize` is more than 1,048,576.               |

### Properties

All read-only.

| Property                   | Type                                         | Description                                                                                                                                                                                         |
| -------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer`                 | `WebGPURenderer`                             | From `particles`.                                                                                                                                                                                   |
| `particles`                | [`ParticleSystem`](./core.md#particlesystem) | Binned particles.                                                                                                                                                                                   |
| `cellSize`                 | `number`                                     | Cell edge length at construction, in m.                                                                                                                                                             |
| `hashOrigin`               | `Vector3`                                    | Where the cells start, in m. This is the value of `hashOriginUniform`. To move the grid, change it in place right before a rebuild. Queries read it too.                                            |
| `hashTableSize`            | `number`                                     | Bucket count, a power of two.                                                                                                                                                                       |
| `hashTableSizePadded`      | `number`                                     | `hashTableSize` rounded up to a multiple of 1024.                                                                                                                                                   |
| `cellIndex`                | `StorageBufferNode<'uint'>`                  | Bucket index per particle.                                                                                                                                                                          |
| `counts`                   | `StorageBufferNode<'uint'>`                  | Atomic particle count per bucket. Used during a rebuild and zero outside one; derive a bucket's count as `cellEnd − cellStart`.                                                                     |
| `cellStart`                | `StorageBufferNode<'uint'>`                  | First slot in `sortedIndices` per bucket.                                                                                                                                                           |
| `cellEnd`                  | `StorageBufferNode<'uint'>`                  | One past the last slot per bucket.                                                                                                                                                                  |
| `sortedIndices`            | `StorageBufferNode<'uint'>`                  | Particle indices in bucket order.                                                                                                                                                                   |
| `slotOf`                   | `StorageBufferNode<'uint'>`                  | Each particle's slot in `sortedIndices`, the inverse of `sortedIndices`.                                                                                                                            |
| `sortedPredictedPositions` | `StorageBufferNode<'vec4'>`                  | `predictedPositions` in bucket order, refreshed each rebuild.                                                                                                                                       |
| `overflowFlag`             | `StorageBufferNode<'uint'>`                  | Atomic. Set to 1 when a particle was more than 512 cells from `hashOrigin` on some axis at the last rebuild. Those particles still find their neighbors, through buckets shared with distant cells. |
| `hashOriginUniform`        | `UniformNode<'vec3', Vector3>`               | Origin used by the kernels.                                                                                                                                                                         |
| `cellSizeUniform`          | `UniformNode<'float', number>`               | Cell size used by the kernels, in m.                                                                                                                                                                |
| `rebuildPipeline`          | `readonly ComputeNode[]`                     | The rebuild kernels, for batching into a larger dispatch. Don't modify them. Reading this after `dispose` throws.                                                                                   |

### Methods

#### `rebuild()`

```ts
rebuild(): Promise<void>
```

Rebuild the grid from the current `particles.predictedPositions`.

#### `readback()`

```ts
readback(): Promise<HashGridSnapshot>
```

Copy the grid buffers to the CPU. This waits for the GPU. `HashGridSnapshot` isn't exported by name. Its fields are:

| Field                            | Type          | Description                                                                             |
| -------------------------------- | ------------- | --------------------------------------------------------------------------------------- |
| `capacity`                       | `number`      | `particles.capacity`.                                                                   |
| `hashTableSize`                  | `number`      | Bucket count.                                                                           |
| `hashTableSizePadded`            | `number`      | Padded bucket count.                                                                    |
| `cellIndex`                      | `Uint32Array` | Bucket per particle.                                                                    |
| `counts`, `cellStart`, `cellEnd` | `Uint32Array` | One value per bucket, and zero past `hashTableSize`. `counts` is `cellEnd − cellStart`. |
| `sortedIndices`                  | `Uint32Array` | Particle indices in bucket order. Order within a bucket varies between runs.            |

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<boolean>
```

Return `true` if the last rebuild set `overflowFlag`, which means some particle was more than 512 cells from `hashOrigin` on an axis. This waits for the GPU.

#### `dispose()`

```ts
dispose(): void
```

Free the grid's GPU buffers. Kernels that query the grid can't run afterwards. Calling it again does nothing.

| Throws                       | When                                                                                      |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `HashGrid has been disposed` | You used `rebuild`, `readback`, `readbackOverflow`, or `rebuildPipeline` after `dispose`. |

## NeighborList

Stores the neighbors within a radius of each particle in one range. It's gathered once per substep, so kernels that visit neighbors several times don't each walk the grid.

Lists are stored by grid slot rather than by particle: the particle in slot `s` of the grid's sorted order keeps its list in row `s`, and its neighbor `k` is at index `k × capacity + s`. Kernels that read the list run one thread per slot, starting with `emitThread()`, so each workgroup handles particles that are close together and reads neighbors that are already in cache.

```ts
const sph = createSphKernelUniforms(h);
const material: Material = {
  neighborRadius: h, // the list can't see past the grid's cell size
  build: ({ particles, hashGrid }) => {
    const list = new NeighborList(particles, range);
    return {
      preSolve: list.buildKernels(hashGrid, sph.hSq),
      solve: [
        Fn(() => {
          const { i, row } = list.emitThread();
          list.forEach(row, (j) => {
            /* particle i's neighbor j */
          });
        })().compute(list.threadCount),
      ],
    };
  },
};
```

### Constructor

```ts
new NeighborList(particles: ParticleSystem, range: ParticleRange)
```

| Parameter   | Type                                         | Description                                                          |
| ----------- | -------------------------------------------- | -------------------------------------------------------------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.                                                    |
| `range`     | [`ParticleRange`](./core.md#particlerange)   | Particles whose neighbors are stored. Neighbors can be any particle. |

| Throws                                                              | When                                                          |
| ------------------------------------------------------------------- | ------------------------------------------------------------- |
| `NeighborList: invalid particle range start=… count=… (capacity …)` | `range` is empty, isn't whole slots, or runs past `capacity`. |
| `NeighborList: call buildKernels first`                             | You read `grid` or called `emitThread` before `buildKernels`. |

### Properties

All read-only.

| Property       | Type                                         | Description                                                                                    |
| -------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `particles`    | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.                                                                              |
| `range`        | [`ParticleRange`](./core.md#particlerange)   | Covered particles.                                                                             |
| `indices`      | `StorageBufferNode<'uint'>`                  | `capacity × MAX_NEIGHBORS` neighbor indices, laid out as above.                                |
| `counts`       | `StorageBufferNode<'uint'>`                  | Number of stored neighbors per row. Rows of particles outside the range aren't written.        |
| `overflowFlag` | `StorageBufferNode<'uint'>`                  | Atomic. Set to 1 when some particle had more than `MAX_NEIGHBORS` neighbors in the last build. |
| `threadCount`  | `number`                                     | Threads to dispatch for a kernel that starts with `emitThread()`: `particles.capacity`.        |
| `grid`         | [`HashGrid`](#hashgrid)                      | The grid the list was built from. Reading it before `buildKernels` throws.                     |

### Methods

#### `buildKernels(grid, radiusSq)`

```ts
buildKernels(grid: HashGrid, radiusSq: UniformNode<'float', number>): ComputeNode[]
```

Return two kernels that rebuild the list from `grid`. The first clears `overflowFlag`. The second stores every particle closer than `sqrt(radiusSq)` (m²) to each particle's predicted position, including the particle itself. Dispatch them after the grid rebuild, for example in `preSolve`.

#### `emitThread()`

```ts
emitThread(): { i: any; row: any }
```

Emit the start of a kernel dispatched with `threadCount` threads. `row` is the thread's grid slot, and `i` is the global index of the particle in that slot. Threads whose particle is outside `range` return. Call it inside an `Fn` body, after `buildKernels`.

#### `forEach(row, onNeighbor)`

```ts
forEach(row: any, onNeighbor: (j: any) => void): void
```

Emit TSL that calls `onNeighbor(j)` for each stored neighbor in `row`, the row `emitThread()` returned. `j` is a global particle index (a `uint` node).

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<boolean>
```

Return `true` if some particle's list was cut short in the last build. This waits for the GPU.

## MAX_NEIGHBORS

```ts
const MAX_NEIGHBORS = 64;
```

The most neighbors a [`NeighborList`](#neighborlist) stores per particle. Extra neighbors are dropped, and the list's `overflowFlag` is set.

## SphKernelUniforms

Uniforms for the Poly6 and Spiky smoothing kernels (Müller et al. 2003) with radius `h`. Create them with [`createSphKernelUniforms`](#createsphkerneluniforms). Setting `h.value` recomputes the other three, and throws `createSphKernelUniforms: h must be positive` if the value is zero, negative, `NaN`, or infinite.

| Property    | Type                           | Description             |
| ----------- | ------------------------------ | ----------------------- |
| `h`         | `UniformNode<'float', number>` | Smoothing radius, in m. |
| `hSq`       | `UniformNode<'float', number>` | `h²`, in m².            |
| `poly6Coef` | `UniformNode<'float', number>` | `315 / (64π h⁹)`.       |
| `spikyCoef` | `UniformNode<'float', number>` | `45 / (π h⁶)`.          |

## createSphKernelUniforms

```ts
createSphKernelUniforms(h: number): SphKernelUniforms
```

| Parameter | Type     | Description             |
| --------- | -------- | ----------------------- |
| `h`       | `number` | Smoothing radius, in m. |

| Throws                                        | When                                       |
| --------------------------------------------- | ------------------------------------------ |
| `createSphKernelUniforms: h must be positive` | `h` is zero, negative, `NaN`, or infinite. |

## emitPoly6

```ts
emitPoly6(r_vec: any, u: SphKernelUniforms): any
```

Emit the Poly6 kernel value for the offset `r_vec` (a TSL `vec3`, in m) as a TSL float, in 1/m³. The value is `poly6Coef · (h² − |r|²)³`, or zero when `|r| ≥ h`.

## emitPoly6FromRSq

```ts
emitPoly6FromRSq(rSq: any, u: SphKernelUniforms): any
```

Like [`emitPoly6`](#emitpoly6), but takes a precomputed squared distance (a TSL float, in m²).

## emitSpikyGrad

```ts
emitSpikyGrad(r_vec: any, u: SphKernelUniforms): any
```

Emit the gradient of the Spiky kernel with respect to `r_vec`, as a TSL `vec3` in 1/m⁴. The value is `−spikyCoef · (h − |r|)² · r̂`, or zero when `|r| ≥ h` or `|r| = 0`. With `r_vec = xᵢ − xⱼ` you get the gradient for particle `i`, so negate it for particle `j`.

## ConstraintType

A set of XPBD constraints (Macklin et al. 2016) of one kind, ready to pass to [`constraintKernels`](#constraintkernels). [`createDistanceConstraints`](#createdistanceconstraints) returns one. For other kinds, use [`colorConstraints`](#colorconstraints) and [`buildConstraintGroups`](#buildconstraintgroups).

| Property            | Type                                                 | Description                                                                     |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------- |
| `count`             | `number`                                             | Number of constraints.                                                          |
| `compliance`        | `StorageBufferNode<'float'>`                         | Compliance `α` (inverse stiffness) per constraint. Units depend on the kind.    |
| `lambda`            | `StorageBufferNode<'float'>`                         | Accumulated Lagrange multiplier per constraint.                                 |
| `groups`            | `readonly` [`ConstraintGroup`](#constraintgroup)`[]` | Groups of constraints that share no particles, solved in order every iteration. |
| `resetLambdaKernel` | `ComputeNode`                                        | Zeroes `lambda`. Run it once per substep.                                       |

## ConstraintGroup

A batch of constraints that share no particles, so they can all be solved at once, one thread each.

| Property      | Type                        | Description                                  |
| ------------- | --------------------------- | -------------------------------------------- |
| `constraints` | `StorageBufferNode<'uint'>` | Constraint indices in the group.             |
| `count`       | `number`                    | Number of constraints in the group.          |
| `solveKernel` | `ComputeNode`               | Solves the group, one thread per constraint. |

## createDistanceConstraints

```ts
createDistanceConstraints(args: {
  readonly particles: ParticleSystem;
  readonly pairs: readonly [number, number][];
  readonly compliance: number | ArrayLike<number>;
  readonly restLength: ArrayLike<number>;
  readonly dt: UniformNode<'float', number>;
}): ConstraintType
```

Create distance constraints, which keep each pair of particles at its rest length. They're sorted into groups on the CPU. Each solve writes both particles' `predictedPositions` directly. A constraint is skipped when both particles are pinned (`invMass = 0`) or sit at the same point.

| Argument     | Type                                         | Description                                                                                          |
| ------------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `particles`  | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.                                                                                    |
| `pairs`      | `readonly [number, number][]`                | Particle index pairs.                                                                                |
| `compliance` | `number \| ArrayLike<number>`                | Compliance `α` in s²/kg. A number applies to every pair. An array or typed array gives one per pair. |
| `restLength` | `ArrayLike<number>`                          | Rest length per pair, in m. Array or typed array.                                                    |
| `dt`         | `UniformNode<'float', number>`               | Substep length, usually [`SolverContext.dt`](#solvercontext).                                        |

| Throws                                                             | When                                                       |
| ------------------------------------------------------------------ | ---------------------------------------------------------- |
| `createDistanceConstraints: pairs is empty`                        | `pairs` is empty.                                          |
| `createDistanceConstraints: restLength length … ≠ pairs.length …`  | `restLength` doesn't have one entry per pair.              |
| `createDistanceConstraints: compliance length … ≠ pairs.length …`  | `compliance` is an array without one entry per pair.       |
| `createDistanceConstraints: pair (…) has out-of-range index`       | An index is negative, not a whole number, or ≥ `capacity`. |
| `createDistanceConstraints: pair (…) references the same particle` | A pair uses the same particle twice.                       |

```ts
const rope: Material = {
  build: ({ particles, dt }) =>
    constraintKernels([
      createDistanceConstraints({ particles, pairs, compliance: 1e-6, restLength, dt }),
    ]),
};
```

## colorConstraints

```ts
colorConstraints(args: {
  readonly arity: number;
  readonly nConstraints: number;
  readonly participantsPerConstraint: readonly number[] | Uint32Array;
}): { groupOf: Uint32Array; numGroups: number }
```

Sort constraints into groups on the CPU so that no two in a group share a particle. Each constraint, in index order, gets the lowest group that none of its particles is in yet.

| Argument                    | Type                               | Description                                                                                                    |
| --------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `arity`                     | `number`                           | Particles per constraint.                                                                                      |
| `nConstraints`              | `number`                           | Number of constraints.                                                                                         |
| `participantsPerConstraint` | `readonly number[] \| Uint32Array` | Flat particle indices. Participant `k` of constraint `c` is at `c × arity + k`. Length `arity × nConstraints`. |

| Returns     | Description                                                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `groupOf`   | Group index per constraint.                                                                                                                                  |
| `numGroups` | One past the largest group index, or `0` when `nConstraints` is 0. If one particle is in at most `d` constraints, this is from `d` to `1 + arity × (d − 1)`. |

| Throws                                                                          | When                                                                     |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `colorConstraints: arity must be a positive integer`                            | `arity` isn't a positive whole number.                                   |
| `colorConstraints: nConstraints must be an integer ≥ 0`                         | `nConstraints` is negative or not a whole number.                        |
| `colorConstraints: participantsPerConstraint length … ≠ arity × nConstraints …` | `participantsPerConstraint` doesn't have `arity × nConstraints` entries. |

## buildConstraintGroups

```ts
buildConstraintGroups(
  coloring: { readonly groupOf: Uint32Array; readonly numGroups: number },
  solve: (constraint: any) => void,
  name?: string,
): ConstraintGroup[]
```

Build one [`ConstraintGroup`](#constraintgroup) for each group in a coloring. `solve(constraint)` runs at shader-build time with the constraint index (a `uint` node). It must emit TSL that solves the constraint and writes its particles, and it may call `Return()` early. `name` labels the group kernels in GPU profiles and captures as `<name>.solve`. It defaults to `'constraints'`.

## constraintKernels

```ts
constraintKernels(types: readonly ConstraintType[]): {
  readonly preSolve: ComputeNode[];
  readonly solve: ComputeNode[];
}
```

Turn constraint types into [`MaterialKernels`](#materialkernels). `preSolve` holds each type's `resetLambdaKernel`. `solve` holds every type's group kernels, in order.

## xpbdDeltaLambda

```ts
xpbdDeltaLambda(args: {
  readonly C: any;
  readonly sumGradSqInvMass: any;
  readonly alphaTilde: any;
  readonly lambdaCurrent: any;
}): any
```

Emit the XPBD change `Δλ` in a constraint's Lagrange multiplier, as a TSL float. The caller adds `Δλ` to `λ` and moves each particle `k` by `wₖ ∇ₖC Δλ`, where `wₖ` is its inverse mass. The update is:

```text
Δλ = (−C − α̃λ) / (Σ wₖ |∇ₖC|² + α̃)
```

| Argument           | Type      | Description                                    |
| ------------------ | --------- | ---------------------------------------------- |
| `C`                | TSL float | Constraint value at the predicted positions.   |
| `sumGradSqInvMass` | TSL float | `Σ wₖ ‖∇ₖC‖²` over the constraint's particles. |
| `alphaTilde`       | TSL float | `α / dt²`.                                     |
| `lambdaCurrent`    | TSL float | `λ` accumulated so far this substep.           |

## Accumulator

Lets many GPU threads add to the same particle at once, as kernels that run one thread per pair or contact need to. Each particle gets a `vec3` sum stored as fixed-point `i32` atomics. Integer addition gives the same result in any order, so results repeat from run to run. Call `add` in your kernels, then run an apply kernel.

### Constructor

```ts
new Accumulator(particles: ParticleSystem, maxMagnitude: number, label?: string)
```

| Parameter      | Type                                         | Description                                                                          |
| -------------- | -------------------------------------------- | ------------------------------------------------------------------------------------ |
| `particles`    | [`ParticleSystem`](./core.md#particlesystem) | One sum per particle slot.                                                           |
| `maxMagnitude` | `number`                                     | Largest per-axis sum you expect between applies, in the sum's units. Sets `scale`.   |
| `label`        | `string`                                     | Prefix for the kernels' names in GPU profiles and captures. Default `'accumulator'`. |

| Throws                                       | When                                                  |
| -------------------------------------------- | ----------------------------------------------------- |
| `Accumulator: maxMagnitude must be positive` | `maxMagnitude` is zero, negative, `NaN`, or infinite. |

### Properties

All read-only.

| Property       | Type                                         | Description                                                                                      |
| -------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `particles`    | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.                                                                                |
| `scale`        | `number`                                     | Fixed-point ticks per unit, `floor(2³⁰ / maxMagnitude)`.                                         |
| `delta`        | `StorageBufferNode<'int'>`                   | `3 × capacity` atomic sums, xyz for each particle.                                               |
| `label`        | `string`                                     | Prefix for the kernels' names.                                                                   |
| `overflowFlag` | `StorageBufferNode<'uint'>`                  | Atomic. Set to 1 when an applied per-axis sum reached `2³⁰` ticks since the flag was last reset. |

### Methods

#### `add(index, value)`

```ts
add(index: any, value: any): void
```

Emit TSL that adds `value` (a `vec3` node), rounded to ticks, to the sum for particle `index` (a `uint` node).

#### `buildResetKernel()`

```ts
buildResetKernel(resetOverflow?: boolean): ComputeNode
```

Return a kernel that zeroes every sum, and also `overflowFlag` unless `resetOverflow` is `false`. It dispatches `3 × capacity` threads. Pass `false` to keep the flag across several resets, then clear it with `buildResetOverflowKernel`.

#### `buildResetOverflowKernel()`

```ts
buildResetOverflowKernel(): ComputeNode
```

Return a one-thread kernel that zeroes only `overflowFlag`.

#### `buildApplyKernel(targets, range?)`

```ts
buildApplyKernel(
  targets: readonly (StorageBufferNode<'vec4'> | ApplyTarget)[],
  range?: ParticleRange,
): ComputeNode
```

Return a kernel that adds each particle's sum to the `xyz` of every target and then zeroes the sum. It keeps `w`, and sets `overflowFlag` if a sum saturated.

| Parameter | Type                                                    | Default       | Description                   |
| --------- | ------------------------------------------------------- | ------------- | ----------------------------- |
| `targets` | `readonly (StorageBufferNode<'vec4'> \| ApplyTarget)[]` | required      | Buffers to add to.            |
| `range`   | [`ParticleRange`](./core.md#particlerange)              | all particles | Particles to apply and clear. |

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<boolean>
```

Return `true` if a sum saturated since `overflowFlag` was last reset. This waits for the GPU.

#### `dispose()`

```ts
dispose(): void
```

Free the sums and flag on the GPU. Kernels built from this accumulator can't run afterwards.

## ApplyTarget

A target buffer for [`Accumulator.buildApplyKernel`](#buildapplykerneltargets-range), with an optional scale.

| Property | Type                        | Description                                                                                                   |
| -------- | --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `buffer` | `StorageBufferNode<'vec4'>` | Target buffer.                                                                                                |
| `scale`  | `any` (optional)            | TSL node or number multiplied into the sum first, e.g. `dt` to turn a velocity change into a position change. |

## TriangleMesh

An indexed triangle mesh as flat arrays.

| Property   | Type           | Description                        |
| ---------- | -------------- | ---------------------------------- |
| `vertices` | `Float32Array` | xyz per vertex.                    |
| `indices`  | `Uint32Array`  | Three vertex indices per triangle. |

## toTriangleMesh

```ts
toTriangleMesh(mesh: BufferGeometry | TriangleMesh): TriangleMesh
```

Copy a geometry's `position` attribute and index into a new `TriangleMesh`. Geometry without an index is read as separate triangles, three vertices each. If you pass a `TriangleMesh`, it's checked and copied.

| Throws                                                        | When                                                                                             |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `toTriangleMesh: expected a BufferGeometry or a TriangleMesh` | `mesh` is neither a `BufferGeometry` nor a `TriangleMesh`.                                       |
| `toTriangleMesh: geometry has no position attribute`          | The geometry has no `position` attribute.                                                        |
| `toTriangleMesh: position attribute has itemSize …, need 3`   | The geometry's `position` has fewer than 3 components.                                           |
| `TriangleMesh: vertices length … is not a multiple of 3`      | The `TriangleMesh` you passed has a `vertices` length that isn't a multiple of 3.                |
| `TriangleMesh: indices length … is not a multiple of 3`       | The index count isn't a multiple of 3, or for geometry without an index, the vertex count isn't. |
| `TriangleMesh: index … is out of range`                       | An index points past the last vertex.                                                            |

## Limitations

- You can't add or remove kernels after the `SimLoop` is constructed, because `build` runs only once.
- You can't set the loop's `hashTableSize`, which is always the default. Only the grid's origin can be set, through `SimLoopOptions.hashOrigin`.
- The loop doesn't limit a material's kernels to a range. Each kernel is dispatched with the thread count it was built with.
- `solve` kernels never run when `SimLoopOptions.iterations` is 0.
- Particles more than 512 cells from `hashOrigin` on any axis set `overflowFlag`. Their queries still filter by distance, but they visit extra candidates.
- Changing `cellSizeUniform` doesn't update `HashGrid.cellSize`, which keeps its construction value.
- A grid can't have more than 1,048,576 buckets.
- A `NeighborList` can't store more than 64 neighbors per particle. The rest are dropped.
- You can't change which particles a constraint connects after it's built, because the grouping runs once on the CPU.
- `Accumulator` sums have limited range and precision because they're fixed point. The resolution is `1 / scale`. Saturation is flagged at about `maxMagnitude` per axis, and the sum wraps around at about `2 × maxMagnitude`.
