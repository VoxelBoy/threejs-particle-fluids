[Docs](README.md) › Custom materials

# Custom materials

A material is any object with a `build(context)` method that returns TSL compute kernels. The `SimLoop` calls `build` once, in the order materials are listed, and dispatches the returned kernels at fixed points in every step. This is how you add forces, emitters, or whole new behaviors without changing the library.

## The `Material` interface

```ts
interface Material {
  /** Farthest distance, in metres, this material looks for neighbors. */
  readonly neighborRadius?: number;
  build(context: SolverContext): MaterialKernels;
  /** Called on the CPU before each step's GPU work, with the step length. */
  update?(dt: number): void;
}
```

`build` receives a `SolverContext`:

| Field                      | What it is                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `particles`                | The `ParticleSystem`.                                                                                                          |
| `dt`                       | A uniform holding the **substep** length in seconds, updated every step.                                                       |
| `hashGrid`                 | The neighbor grid, rebuilt at the start of each substep. Only there if some material sets `neighborRadius` or contacts are on. |
| `allocateCollisionGroup()` | Reserves a collision group no other material or uploaded particle uses.                                                        |

`build` returns `MaterialKernels`, every list optional:

| Stage            | When it runs                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `init`           | Once, before the first step, after an initial grid rebuild.                                   |
| `beforeStep`     | At the start of every step, before the first substep.                                         |
| `preSolve`       | Every substep, after positions are predicted and the grid is rebuilt.                         |
| `solve`          | Every solver iteration, before contacts and colliders.                                        |
| `postSolve`      | Every substep, after velocities are updated from the solved positions.                        |
| `noSelfContacts` | A particle range that never gets contacts within itself, because the material keeps it apart. |

Forces and velocity changes usually go in `postSolve`: they take effect when the next substep predicts positions. Position constraints go in `solve`.

## A force: drag toward a target velocity

```ts
import { Fn, instanceIndex, uniform, vec4 } from 'three/tsl';
import type { Material } from 'threejs-particle-fluids';

const wind = uniform(new Vector3(1, 0, 0));
const drag: Material = {
  build: ({ particles, dt }) => ({
    postSolve: [
      Fn(() => {
        const velocity = particles.velocities.element(instanceIndex);
        const v = velocity.xyz;
        const next = v.add(wind.sub(v).mul(dt.mul(2)));
        velocity.assign(vec4(next, velocity.w));
      })().compute(particles.capacity),
    ],
  }),
};

const loop = new SimLoop(particles, { materials: [fluid, drag] });
wind.value.set(0, 0, 2); // uniforms can change at any time
```

The Vortex Plume preset uses the same pattern for its wall vanes, which swirl the air near the walls; see [`demo/presets/vortex.ts`](../demo/presets/vortex.ts).

Kernels run over `particles.capacity` threads unless you choose otherwise. To touch only one material's particles, compute over `range.count` and offset the index by `range.start`.

## Using neighbors

Set `neighborRadius` to have the loop build the neighbor grid, then query it in your kernels with `emitForEachNeighbor`. Candidates can be farther than the radius, so filter by distance; the SPH kernel helpers do that for you by returning 0 outside the smoothing radius.

```ts
import { Fn, float, instanceIndex, instancedArray, vec3 } from 'three/tsl';
import {
  createSphKernelUniforms,
  emitForEachNeighbor,
  emitPoly6FromRSq,
  type Material,
} from 'threejs-particle-fluids';

const h = fluid.smoothingRadius;
const sph = createSphKernelUniforms(h);
const smoothed = instancedArray(particles.capacity, 'vec3');

const average: Material = {
  neighborRadius: h,
  build: ({ particles, hashGrid }) => ({
    postSolve: [
      Fn(() => {
        const xi = particles.positions.element(instanceIndex).xyz.toVar();
        const sum = vec3(0).toVar();
        const weight = float(0).toVar();
        emitForEachNeighbor(hashGrid, xi, (j) => {
          const offset = xi.sub(particles.positions.element(j).xyz);
          const w = emitPoly6FromRSq(offset.dot(offset), sph);
          sum.addAssign(particles.velocities.element(j).xyz.mul(w));
          weight.addAssign(w);
        });
        smoothed.element(instanceIndex).assign(sum.div(weight.max(1e-12)));
      })().compute(particles.capacity),
    ],
  }),
};
```

The callback runs while the shader is built and must emit TSL. Use `Continue()` inside it to skip a candidate; `Return()` would end the whole invocation.

Helpers for neighbor work:

| Export                                           | What it does                                                            |
| ------------------------------------------------ | ----------------------------------------------------------------------- |
| `emitForEachNeighbor(grid, x, fn)`               | Visits every particle in the 27 grid cells around `x`.                  |
| `createSphKernelUniforms(h)`                     | Coefficients for the Poly6 and Spiky kernels with smoothing radius `h`. |
| `emitPoly6(r, sph)`, `emitPoly6FromRSq(r², sph)` | The Poly6 density kernel `W`.                                           |
| `emitSpikyGrad(r, sph)`                          | The Spiky kernel gradient `∇W`, taken with respect to `r = xᵢ − xⱼ`.    |

## CPU-side updates

`update(dt)` runs on the CPU before each step's GPU work. Use it to move emitters, push new data into uniforms, or upload spawn buffers, as `GasSystem` does for its tracers.

## Reading from other materials

Materials expose their buffers for exactly this: `fluid.density`, `smoke.temperature`, `softbody.bodyCenters`, `softbody.bodyRotations`, and every buffer on `ParticleSystem`. Read them in your kernels to build effects on top of the built-in physics.

---

Previous: [Combining materials](combining-materials.md) · Next: [Troubleshooting](troubleshooting.md)
