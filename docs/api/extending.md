[Docs](../README.md) › [API](../README.md#api-reference) › Extending

# Extending

The extension surface: the `Material` interface that adds kernels to a [`SimLoop`](./core.md#simloop), and the building blocks the built-in materials use (neighbor search, SPH kernels, XPBD constraints, atomic accumulation). Kernels are TSL compute nodes. [`Simulation`](./simulation.md) doesn't accept custom materials; build a `ParticleSystem` and `SimLoop` directly.

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

Physics that runs inside a `SimLoop`. Any object with a `build` method. The loop calls `build` once, in its constructor, in the order of `SimLoopOptions.materials`, and dispatches the returned kernels every step.

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
| `particles`      | [`ParticleSystem`](./core.md#particlesystem) `\| undefined` | read-only | Particles the material was built for. When set, `SimLoop` throws unless it is the loop's own.                                                                        |

### Methods

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

Create buffers and kernels. Called once from the `SimLoop` constructor.

| Parameter | Type                              | Description          |
| --------- | --------------------------------- | -------------------- |
| `context` | [`SolverContext`](#solvercontext) | Shared solver state. |

#### `update(dt)`

```ts
update?(dt: number): void
```

Optional CPU hook. `SimLoop.step` calls it for every material, in order, before colliders update and before any GPU work. `dt` is the full step length in s, not the substep.

### Example

Velocity damping on one range. Kernels over a range dispatch `range.count` threads and offset by `range.start`.

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

Kernels returned by [`Material.build`](#buildcontext). Every field is optional. Within a stage, kernels run in material order, then in array order.

| Field            | Type                                       | When `SimLoop` dispatches it                                                                                                                       |
| ---------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `init`           | `readonly ComputeNode[]`                   | Once, on the first `step`, before `beforeStep`. Preceded by a grid rebuild when a grid exists.                                                     |
| `beforeStep`     | `readonly ComputeNode[]`                   | Every step, before the first substep, after colliders' frame-start kernels.                                                                        |
| `preSolve`       | `readonly ComputeNode[]`                   | Every substep, after prediction, grid rebuild, contact generation, and colliders' pre-solve kernels.                                               |
| `solve`          | `readonly ComputeNode[]`                   | Every solver iteration, before contacts and colliders.                                                                                             |
| `postSolve`      | `readonly ComputeNode[]`                   | Every substep, after velocities are derived from solved positions, before friction.                                                                |
| `noSelfContacts` | [`ParticleRange`](./core.md#particlerange) | Particles in this range get no particle–particle contacts with each other; they still contact all other particles. Only used when contacts are on. |

## SolverContext

Passed to [`Material.build`](#buildcontext).

| Property                   | Type                                         | Description                                                                                                                                                                                                                                                                                                            |
| -------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `particles`                | [`ParticleSystem`](./core.md#particlesystem) | Particle storage the loop advances.                                                                                                                                                                                                                                                                                    |
| `dt`                       | `UniformNode<'float', number>`               | Substep length in s. Set by `SimLoop.step` to `dt / substeps` before dispatch.                                                                                                                                                                                                                                         |
| `hashGrid`                 | [`HashGrid`](#hashgrid)                      | The loop's neighbor grid, rebuilt from `predictedPositions` each substep. Cell size is the largest `neighborRadius` of any material, or `2 × particleRadius × 1.1` with contacts on, whichever is larger. Without a grid it is a placeholder that throws when any of its properties is read; destructuring it is safe. |
| `allocateCollisionGroup()` | `() => number`                               | Reserve a collision group unused by other allocations and by particles uploaded before the `SimLoop` was constructed. Assign it with `particles.setCollisionGroup`. Groups written to particles after the loop was constructed can clash with it.                                                                      |

| Throws                                                                          | When                                                                                         |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `SimLoop: a material used the neighbor grid without declaring a neighborRadius` | A property of `hashGrid` is read when no material set `neighborRadius` and contacts are off. |

## Dispatch order

One `SimLoop.step(dt)`:

1. CPU: `update(dt)` on every material, then every collider; `dt` uniform set to `dt / substeps`.
2. First step only: grid rebuild (if a grid exists and any `init` kernels were returned), then all `init` kernels.
3. Colliders' frame-start kernels, then all `beforeStep` kernels.
4. `substeps` times:
   1. Predict: add gravity to velocity, `x* = x + v·dt`, for particles with `invMass > 0`.
   2. Reset shared position sums (if contacts or colliders); grid rebuild; contact generation and stabilization; colliders' pre-solve; all `preSolve` kernels.
   3. `iterations` times: all `solve` kernels, then contact and collider corrections, then apply them to `predictedPositions`.
   4. Advect: `v = (x* − x) / dt`, `x = x*`, for particles with `invMass > 0`.
   5. All `postSolve` kernels.
   6. Contact and collider friction; colliders' substep-end kernels.

The whole step is submitted in one `computeAsync` call. Velocity changes made in `postSolve` take effect at the next prediction.

## emitForEachNeighbor

```ts
emitForEachNeighbor(
  grid: HashGrid,
  position: any,
  onCandidate: (neighborIndex: any, sortedSlot: any) => void,
): void
```

Emit TSL that visits every particle in the 27 grid cells around `position`. Call inside an `Fn` body. Candidates include particles beyond the query radius and, through hash collisions, distant cells; filter by distance. A particle querying at its own predicted position is its own candidate.

| Parameter     | Type                                  | Description                                                                                                                                                                                                                                                    |
| ------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grid`        | [`HashGrid`](#hashgrid)               | Grid to walk. Reflects `predictedPositions` at its last rebuild.                                                                                                                                                                                               |
| `position`    | TSL `vec3` node                       | Query point, in m.                                                                                                                                                                                                                                             |
| `onCandidate` | `(neighborIndex, sortedSlot) => void` | Runs at shader-build time and must emit TSL. `neighborIndex`: particle index (`uint`). `sortedSlot`: index into `grid.sortedIndices` and `grid.sortedPredictedPositions` (`uint`). Use `Continue()` to skip a candidate; `Return()` ends the whole invocation. |

## HashGrid

Spatial hash over every particle of a `ParticleSystem`, rebuilt on the GPU with a counting sort. Bins `predictedPositions`. `SimLoop` owns one and rebuilds it every substep; construct one directly only for standalone tools.

### Constructor

```ts
new HashGrid(particles: ParticleSystem, options: HashGridOptions)
```

| Parameter   | Type                                         | Description       |
| ----------- | -------------------------------------------- | ----------------- |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particles to bin. |
| `options`   | [`HashGridOptions`](#hashgridoptions)        | See below.        |

#### HashGridOptions

| Option          | Type      | Default                                               | Description                                                                                                             |
| --------------- | --------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `cellSize`      | `number`  | required                                              | Cell edge length, in m. Must be ≥ the largest query radius.                                                             |
| `hashTableSize` | `number`  | next power of two ≥ `2 × capacity`, at most 1,048,576 | Bucket count. A power of two, at most 1,048,576.                                                                        |
| `hashOrigin`    | `Vector3` | `(0, 0, 0)`                                           | Subtracted from positions before quantizing to cells, in m. Copied. Lookups stay local within 512 cells of it per axis. |

| Throws                                                                                       | When                                                                |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `HashGrid: cellSize must be a positive finite number`                                        | `cellSize` is ≤ 0, `NaN`, or infinite.                              |
| `HashGrid: hashTableSize must be a positive power of two`                                    | `hashTableSize` is not an integer, is < 1, or isn't a power of two. |
| `HashGrid: hashTableSize=… (capacity …) exceeds the 1048576-bucket limit of the prefix scan` | `hashTableSize` exceeds 1,048,576.                                  |

### Properties

All read-only.

| Property                   | Type                                         | Description                                                                                                                                                                                        |
| -------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer`                 | `WebGPURenderer`                             | From `particles`.                                                                                                                                                                                  |
| `particles`                | [`ParticleSystem`](./core.md#particlesystem) | Binned particles.                                                                                                                                                                                  |
| `cellSize`                 | `number`                                     | Cell edge length at construction, in m.                                                                                                                                                            |
| `hashOrigin`               | `Vector3`                                    | Origin of the cells, in m: the value of `hashOriginUniform`. Mutate it in place to move the grid, right before a rebuild; queries read it too.                                                     |
| `hashTableSize`            | `number`                                     | Bucket count, a power of two.                                                                                                                                                                      |
| `hashTableSizePadded`      | `number`                                     | `hashTableSize` rounded up to a multiple of 1024.                                                                                                                                                  |
| `cellIndex`                | `StorageBufferNode<'uint'>`                  | Bucket index per particle.                                                                                                                                                                         |
| `counts`                   | `StorageBufferNode<'uint'>`                  | Atomic particle count per bucket.                                                                                                                                                                  |
| `cellStart`                | `StorageBufferNode<'uint'>`                  | First slot in `sortedIndices` per bucket.                                                                                                                                                          |
| `cellEnd`                  | `StorageBufferNode<'uint'>`                  | One past the last slot per bucket.                                                                                                                                                                 |
| `sortedIndices`            | `StorageBufferNode<'uint'>`                  | Particle indices in bucket order.                                                                                                                                                                  |
| `sortedPredictedPositions` | `StorageBufferNode<'vec4'>`                  | `predictedPositions` in bucket order, refreshed each rebuild.                                                                                                                                      |
| `overflowFlag`             | `StorageBufferNode<'uint'>`                  | Atomic; 1 when a particle's cell coordinate fell outside ±512 cells of `hashOrigin` during the last rebuild. Such particles still find their neighbors, through buckets shared with distant cells. |
| `hashOriginUniform`        | `UniformNode<'vec3', Vector3>`               | Origin used by the kernels.                                                                                                                                                                        |
| `cellSizeUniform`          | `UniformNode<'float', number>`               | Cell size used by the kernels, in m.                                                                                                                                                               |
| `rebuildPipeline`          | `readonly ComputeNode[]`                     | The rebuild kernels, for batching into a larger dispatch. Don't modify. Throws after `dispose`.                                                                                                    |

### Methods

#### `rebuild()`

```ts
rebuild(): Promise<void>
```

Rebuild from the current `particles.predictedPositions`.

#### `readback()`

```ts
readback(): Promise<HashGridSnapshot>
```

Read the grid buffers to the CPU. Stalls on the GPU. `HashGridSnapshot` isn't exported by name; its fields:

| Field                            | Type          | Description                                                                  |
| -------------------------------- | ------------- | ---------------------------------------------------------------------------- |
| `capacity`                       | `number`      | `particles.capacity`.                                                        |
| `hashTableSize`                  | `number`      | Bucket count.                                                                |
| `hashTableSizePadded`            | `number`      | Padded bucket count.                                                         |
| `cellIndex`                      | `Uint32Array` | Bucket per particle.                                                         |
| `counts`, `cellStart`, `cellEnd` | `Uint32Array` | Per bucket; zero past `hashTableSize`.                                       |
| `sortedIndices`                  | `Uint32Array` | Particle indices in bucket order. Order within a bucket varies between runs. |

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<boolean>
```

`true` if `overflowFlag` was set by the last rebuild: some particle was more than 512 cells from `hashOrigin` on an axis. Stalls on the GPU.

#### `dispose()`

```ts
dispose(): void
```

Free the grid's GPU buffers. Kernels that query the grid can't run afterwards. Calling it again does nothing.

| Throws                       | When                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| `HashGrid has been disposed` | `rebuild`, `readback`, `readbackOverflow`, or `rebuildPipeline` used after `dispose`. |

## NeighborList

Per-particle neighbor indices within a radius, for one range, gathered once per substep so kernels that visit neighbors several times don't each walk the grid. Stored column-major: neighbor `k` of local particle `i` is at `k × range.count + i`.

```ts
const sph = createSphKernelUniforms(h);
const material: Material = {
  neighborRadius: h, // the list can't see past the grid's cell size
  build: ({ particles, hashGrid }) => {
    const list = new NeighborList(particles, range);
    return {
      preSolve: list.buildKernels(hashGrid, sph.hSq),
      solve: [
        /* kernels that call list.forEach */
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

| Throws                                                              | When                                                  |
| ------------------------------------------------------------------- | ----------------------------------------------------- |
| `NeighborList: invalid particle range start=… count=… (capacity …)` | `range` is empty, non-integer, or outside `capacity`. |

### Properties

All read-only.

| Property       | Type                                         | Description                                                                             |
| -------------- | -------------------------------------------- | --------------------------------------------------------------------------------------- |
| `particles`    | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.                                                                       |
| `range`        | [`ParticleRange`](./core.md#particlerange)   | Covered particles.                                                                      |
| `indices`      | `StorageBufferNode<'uint'>`                  | `range.count × MAX_NEIGHBORS` neighbor indices, column-major.                           |
| `counts`       | `StorageBufferNode<'uint'>`                  | Stored neighbor count per local particle.                                               |
| `overflowFlag` | `StorageBufferNode<'uint'>`                  | Atomic; 1 when some particle had more than `MAX_NEIGHBORS` neighbors in the last build. |

### Methods

#### `buildKernels(grid, radiusSq)`

```ts
buildKernels(grid: HashGrid, radiusSq: UniformNode<'float', number>): ComputeNode[]
```

Two kernels: clear `overflowFlag`, then fill the list from `grid` with every particle closer than `sqrt(radiusSq)` (m²) at predicted positions, the particle itself included. Dispatch after the grid rebuild, e.g. in `preSolve`.

#### `forEach(i, onNeighbor)`

```ts
forEach(i: any, onNeighbor: (j: any) => void): void
```

Emit TSL that calls `onNeighbor(j)` for each stored neighbor of particle `i`. `i` is a global particle index (`uint` node) inside `range`; `j` is a global particle index.

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<boolean>
```

`true` if some list was truncated during the last build. Stalls on the GPU.

## MAX_NEIGHBORS

```ts
const MAX_NEIGHBORS = 64;
```

Most neighbors a [`NeighborList`](#neighborlist) stores per particle. Extras are dropped and flagged.

## SphKernelUniforms

Uniforms for the Poly6 and Spiky smoothing kernels (Müller et al. 2003) with radius `h`. Created by [`createSphKernelUniforms`](#createsphkerneluniforms). Setting `h.value` recomputes the other three; it throws `createSphKernelUniforms: h must be positive` for values ≤ 0, `NaN`, or infinite.

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

| Throws                                        | When                            |
| --------------------------------------------- | ------------------------------- |
| `createSphKernelUniforms: h must be positive` | `h` is ≤ 0, `NaN`, or infinite. |

## emitPoly6

```ts
emitPoly6(r_vec: any, u: SphKernelUniforms): any
```

Emit `W = poly6Coef · (h² − |r|²)³`, zero for `|r| ≥ h`, as a TSL float (1/m³). `r_vec` is a TSL `vec3` offset in m.

## emitPoly6FromRSq

```ts
emitPoly6FromRSq(rSq: any, u: SphKernelUniforms): any
```

[`emitPoly6`](#emitpoly6) from a precomputed `|r|²` (TSL float, m²).

## emitSpikyGrad

```ts
emitSpikyGrad(r_vec: any, u: SphKernelUniforms): any
```

Emit `∇W = −spikyCoef · (h − |r|)² · r̂` as a TSL `vec3` (1/m⁴), the gradient with respect to `r_vec`. Zero for `|r| ≥ h` and `|r| = 0`. With `r_vec = xᵢ − xⱼ` this is `∇ᵢW`; negate for `∇ⱼW`.

## ConstraintType

A set of XPBD constraints of one kind, ready for [`constraintKernels`](#constraintkernels). Returned by [`createDistanceConstraints`](#createdistanceconstraints); build others with [`colorConstraints`](#colorconstraints) and [`buildConstraintGroups`](#buildconstraintgroups).

| Property            | Type                                                 | Description                                              |
| ------------------- | ---------------------------------------------------- | -------------------------------------------------------- |
| `count`             | `number`                                             | Number of constraints.                                   |
| `compliance`        | `StorageBufferNode<'float'>`                         | Compliance `α` per constraint. Units depend on the kind. |
| `lambda`            | `StorageBufferNode<'float'>`                         | Accumulated Lagrange multiplier per constraint.          |
| `groups`            | `readonly` [`ConstraintGroup`](#constraintgroup)`[]` | Color groups, solved in order every iteration.           |
| `resetLambdaKernel` | `ComputeNode`                                        | Zeroes `lambda`. Run once per substep.                   |

## ConstraintGroup

One color class: constraints that share no particle, solved one thread each without races.

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

XPBD distance constraints `C = |xᵢ − xⱼ| − L₀`, colored on the CPU. Each solve writes `predictedPositions` of both particles directly. A constraint is skipped when both particles have `invMass = 0` or they coincide.

| Argument     | Type                                         | Description                                                                                          |
| ------------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `particles`  | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.                                                                                    |
| `pairs`      | `readonly [number, number][]`                | Particle index pairs.                                                                                |
| `compliance` | `number \| ArrayLike<number>`                | Compliance `α` in s²/kg; a number applies to every pair, an array or typed array gives one per pair. |
| `restLength` | `ArrayLike<number>`                          | Rest length per pair, in m. Array or typed array.                                                    |
| `dt`         | `UniformNode<'float', number>`               | Substep length, usually [`SolverContext.dt`](#solvercontext).                                        |

| Throws                                                             | When                                                |
| ------------------------------------------------------------------ | --------------------------------------------------- |
| `createDistanceConstraints: pairs is empty`                        | `pairs.length === 0`.                               |
| `createDistanceConstraints: restLength length … ≠ pairs.length …`  | Lengths differ.                                     |
| `createDistanceConstraints: compliance length … ≠ pairs.length …`  | `compliance` is an array of a different length.     |
| `createDistanceConstraints: pair (…) has out-of-range index`       | An index is non-integer, negative, or ≥ `capacity`. |
| `createDistanceConstraints: pair (…) references the same particle` | `i === j`.                                          |

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

Greedy graph coloring on the CPU: assigns each constraint, in index order, the lowest group not used by any of its particles.

| Argument                    | Type                               | Description                                                                                                 |
| --------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `arity`                     | `number`                           | Particles per constraint.                                                                                   |
| `nConstraints`              | `number`                           | Number of constraints.                                                                                      |
| `participantsPerConstraint` | `readonly number[] \| Uint32Array` | Flat particle indices; participant `k` of constraint `c` at `c × arity + k`. Length `arity × nConstraints`. |

| Returns     | Description                                                                                                                                           |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `groupOf`   | Group index per constraint.                                                                                                                           |
| `numGroups` | One past the largest group index; `0` when `nConstraints` is 0. At least the most constraints sharing one particle, at most `1 + arity × (that − 1)`. |

| Throws                                                                          | When                                            |
| ------------------------------------------------------------------------------- | ----------------------------------------------- |
| `colorConstraints: arity must be a positive integer`                            | `arity` is not an integer ≥ 1.                  |
| `colorConstraints: nConstraints must be an integer ≥ 0`                         | `nConstraints` is negative or not an integer.   |
| `colorConstraints: participantsPerConstraint length … ≠ arity × nConstraints …` | The array length is not `arity × nConstraints`. |

## buildConstraintGroups

```ts
buildConstraintGroups(
  coloring: { readonly groupOf: Uint32Array; readonly numGroups: number },
  solve: (constraint: any) => void,
): ConstraintGroup[]
```

One [`ConstraintGroup`](#constraintgroup) per color. `solve(constraint)` runs at shader-build time with the constraint index (`uint` node), must emit TSL that projects that constraint and writes its particles, and may `Return()` early.

## constraintKernels

```ts
constraintKernels(types: readonly ConstraintType[]): {
  readonly preSolve: ComputeNode[];
  readonly solve: ComputeNode[];
}
```

Schedule constraint types as [`MaterialKernels`](#materialkernels): `preSolve` holds each type's `resetLambdaKernel`; `solve` holds every type's group kernels, types in order, groups in order.

## xpbdDeltaLambda

```ts
xpbdDeltaLambda(args: {
  readonly C: any;
  readonly sumGradSqInvMass: any;
  readonly alphaTilde: any;
  readonly lambdaCurrent: any;
}): any
```

Emit the XPBD multiplier update `Δλ = (−C − α̃λ) / (Σ wₖ|∇ₖC|² + α̃)` (Macklin et al. 2016, eq. 18) as a TSL float. The caller adds `Δλ` to `λ` and moves each particle by `wₖ ∇ₖC Δλ`.

| Argument           | Type      | Description                                  |
| ------------------ | --------- | -------------------------------------------- | --- | ----------------------------------- |
| `C`                | TSL float | Constraint value at the predicted positions. |
| `sumGradSqInvMass` | TSL float | `Σ wₖ                                        | ∇ₖC | ²` over the constraint's particles. |
| `alphaTilde`       | TSL float | `α / dt²`.                                   |
| `lambdaCurrent`    | TSL float | `λ` accumulated so far this substep.         |

## Accumulator

Per-particle `vec3` sums built from fixed-point `i32` atomics, for scatter kernels (one thread per pair or contact) that need to add to particles concurrently. Integer sums are order-independent, so results repeat run to run. Usage: `add` in scatter kernels, then an apply kernel.

### Constructor

```ts
new Accumulator(particles: ParticleSystem, maxMagnitude: number)
```

| Parameter      | Type                                         | Description                                                                |
| -------------- | -------------------------------------------- | -------------------------------------------------------------------------- |
| `particles`    | [`ParticleSystem`](./core.md#particlesystem) | One sum per particle slot.                                                 |
| `maxMagnitude` | `number`                                     | Largest per-axis sum expected per apply, in the sum's units. Sets `scale`. |

| Throws                                       | When                                       |
| -------------------------------------------- | ------------------------------------------ |
| `Accumulator: maxMagnitude must be positive` | `maxMagnitude` is ≤ 0, `NaN`, or infinite. |

### Properties

All read-only.

| Property       | Type                                         | Description                                                                               |
| -------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `particles`    | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.                                                                         |
| `scale`        | `number`                                     | Fixed-point ticks per unit: `floor(2³⁰ / maxMagnitude)`.                                  |
| `delta`        | `StorageBufferNode<'int'>`                   | `3 × capacity` atomic sums, xyz per particle.                                             |
| `overflowFlag` | `StorageBufferNode<'uint'>`                  | Atomic; 1 when an applied per-axis sum reached `2³⁰` ticks since the flag was last reset. |

### Methods

#### `add(index, value)`

```ts
add(index: any, value: any): void
```

Emit TSL that adds `value` (`vec3` node), rounded to ticks, to particle `index`'s sum (`uint` node).

#### `buildResetKernel()`

```ts
buildResetKernel(resetOverflow?: boolean): ComputeNode
```

Zero every sum and, unless `resetOverflow` is `false`, `overflowFlag`. Dispatches `3 × capacity` threads. Pass `false` to keep the flag across several resets and clear it with `buildResetOverflowKernel`.

#### `buildResetOverflowKernel()`

```ts
buildResetOverflowKernel(): ComputeNode
```

Zero only `overflowFlag`. One thread.

#### `buildApplyKernel(targets, range?)`

```ts
buildApplyKernel(
  targets: readonly (StorageBufferNode<'vec4'> | ApplyTarget)[],
  range?: ParticleRange,
): ComputeNode
```

Add each particle's sum to the `xyz` of every target (keeping `w`), set `overflowFlag` on saturation, then zero the sum.

| Parameter | Type                                                    | Default       | Description                   |
| --------- | ------------------------------------------------------- | ------------- | ----------------------------- |
| `targets` | `readonly (StorageBufferNode<'vec4'> \| ApplyTarget)[]` | required      | Buffers to add to.            |
| `range`   | [`ParticleRange`](./core.md#particlerange)              | all particles | Particles to apply and clear. |

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<boolean>
```

`true` if a sum saturated since `overflowFlag` was last reset. Stalls on the GPU.

#### `dispose()`

```ts
dispose(): void
```

Free the sums and flag on the GPU. Kernels built from this accumulator can't run afterwards.

## ApplyTarget

A buffer for [`Accumulator.buildApplyKernel`](#buildapplykerneltargets-range) with an optional scale.

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

Read a geometry's `position` attribute and index into a new `TriangleMesh`; non-indexed geometry becomes a triangle soup. A `TriangleMesh` input is validated and copied.

| Throws                                                        | When                                                           |
| ------------------------------------------------------------- | -------------------------------------------------------------- |
| `toTriangleMesh: expected a BufferGeometry or a TriangleMesh` | `mesh` is neither.                                             |
| `toTriangleMesh: geometry has no position attribute`          | Geometry input without `position`.                             |
| `toTriangleMesh: position attribute has itemSize …, need 3`   | Geometry `position` with fewer than 3 components.              |
| `TriangleMesh: vertices length … is not a multiple of 3`      | `TriangleMesh` input only.                                     |
| `TriangleMesh: indices length … is not a multiple of 3`       | Index count (or non-indexed vertex count) not a multiple of 3. |
| `TriangleMesh: index … is out of range`                       | An index ≥ vertex count.                                       |

## Limitations

- `Simulation` builds its own material list; custom materials need a hand-built `SimLoop`.
- `build` runs once in the `SimLoop` constructor; kernels can't be added or removed afterward.
- The loop's grid uses the default `hashTableSize`; only its origin is configurable, through `SimLoopOptions.hashOrigin`.
- Material kernels are dispatched with the thread count they were built with; the loop doesn't restrict them to a range.
- `solve` kernels never run when `SimLoopOptions.iterations` is 0.
- Particles more than 512 cells from `hashOrigin` on any axis set `overflowFlag`; their queries still filter by distance but visit extra candidates.
- `HashGrid.cellSize` is the construction value; changing `cellSizeUniform` doesn't update it.
- `hashTableSize` is capped at 1,048,576 buckets.
- A `NeighborList` keeps at most 64 neighbors per particle; the rest are dropped.
- Constraint coloring runs on the CPU at build time; topology is fixed afterward.
- `Accumulator` sums are fixed point: resolution `1 / scale`, saturation flagged at about `maxMagnitude` per axis, `i32` wraparound at about `2 × maxMagnitude`.
