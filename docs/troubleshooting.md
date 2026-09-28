[Docs](README.md) › Troubleshooting

# Troubleshooting

## Error messages

| Error                                                                                                                   | Fix                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `createParticleRenderer: WebGPU is unavailable, …`                                                                      | Use a browser with WebGPU and hardware acceleration on, served from `localhost` or HTTPS.                                                      |
| ``addSmoke: smoke needs a `container` for the air to fill``                                                             | Pass `container` to `new Simulation`. See [Smoke needs a container](smoke.md#smoke-needs-a-container).                                         |
| `Simulation: addFluid must happen before the first step` (or `addSmoke`, `addSoftbody`, `addCloth`, `adding colliders`) | Everything is added before the first `step()`. See [Add everything before the first step](simulation.md#add-everything-before-the-first-step). |
| `Simulation: smoke can’t share a simulation with liquids, soft bodies, or cloth`                                        | Put the smoke in its own `Simulation`. See [Smoke limits](smoke.md#limits).                                                                    |
| `Simulation: only one smoke source is supported`                                                                        | Call `addSmoke` once per simulation.                                                                                                           |
| `Simulation: add a fluid, smoke, soft body, or cloth before stepping`                                                   | Obstacles alone don't simulate anything. Add something for them to affect.                                                                     |
| ``addFluid: give either `box` or `mesh` ``                                                                              | Pass exactly one of them.                                                                                                                      |
| `addFluid: the fluid has no room; check its box and the container`                                                      | The box is outside the container, or filled by obstacles, soft bodies, or cloth. Move or enlarge it.                                           |
| `addSoftbody: the mesh is too small for the particle size`                                                              | Make the mesh bigger, or raise `particles` so each particle is smaller.                                                                        |
| ``Fluid.thickness: give `thickness` when adding the fluid to change it later``                                          | Pass `thickness` above 0 in `addFluid`. See [Live settings](fluids.md#live-settings).                                                          |
| `… is created when the simulation starts (on the first step)`                                                           | Handle properties such as `fluid.surface`, `softbody.mesh`, and `smoke.emit` exist after the first `step()` or `await sim.start()`.            |
| `Simulation.particles is created on the first step` (or `Simulation.loop`)                                              | The same: read them after the first `step()`.                                                                                                  |
| `bakeMeshToSdf: mesh appears non-watertight — …`                                                                        | A mesh given to `addMesh` has holes or doubled faces. Fix it, or use a simplified closed copy.                                                 |
| `SDFCollider.setTransform: scale must be uniform`                                                                       | A mesh given to `addMesh` is scaled unevenly. Use the same scale on every axis.                                                                |

Errors from the first step reject the promise that `step()` or `start()` returns. After one, that `Simulation` can't be repaired: every later `step()` fails the same way, and every `add` call throws. Call `sim.dispose()`, which shows your meshes again, fix the cause, and build a new one.

## Nothing shows up

- **Not awaiting `step()`.** Call `await sim.step()` before `renderer.render(scene, camera)` every frame.
- **The camera can't see it.** Everything is in metres. A 1 m tank needs the camera a couple of metres away, with a near plane around 0.01.
- **No lights, or the environment map came too late.** Soft bodies and cloth need lights. Liquid needs `scene.environment`, set before the first `step()`. See [Lighting](getting-started.md#lighting).
- **It fell out of view.** Without a container or a floor, everything falls forever. Add a `container` or `sim.addFloor()`.
- **The liquid vanishes as it spreads.** Liquid is drawn only inside the container, or without one, in a box around where it started. Give the simulation a `container` that covers everywhere the liquid can go.

## Seeing what's simulated

```ts
sim.showParticles = true;
```

This hides the rendered water, smoke, soft bodies, and cloth, and draws the raw particles instead. Use it to check that a mesh filled the way you expected, that liquid starts where you think, or that something really is passing through something else. Set it back to `false` to see the normal rendering. It can be toggled at any time, for example from a key press.

## Things pass through each other

- **Fast or thin obstacles.** In one substep, a particle can move far enough to jump right past a thin or fast-moving shape. Make the shape thicker, move it slower, or raise the simulation's [`substeps` option](simulation.md#options). Each substep covers less distance, and each extra one costs about as much as the first.

  ```ts
  new Simulation({ renderer, scene, camera, container, substeps: 12 });
  ```

- **Liquid leaks through cloth.** Raise the cloth's `weight`. See [Catching liquid](cloth.md#catching-liquid).
- **Cloth pokes through a mesh obstacle.** Sharp, thin features like ears slip between the cloth's particles. Raise `particles` for a finer cloth, or smooth the obstacle mesh.

## It's slow

- **Particle count** sets the cost of almost everything. Lower `particles`. Between 5,000 and 50,000 is a sensible range, and 20,000 is the default.
- **Soft bodies and cloth in liquid** are the most expensive combination, because the liquid has to track moving solids every step. Fewer or smaller bodies help.
- **Thick liquids** (`thickness` above 0) do extra work every step.
- **Smoke** costs more with higher `rate` and `lifetime`, and a bigger container means more air.
- **Big meshes** in `addMesh`, `addSoftbody`, or `addFluid({ mesh })` slow down the first step, not the frames after it. Use simplified meshes, or do the setup behind a loading screen; see **Loading** under [Stepping](simulation.md#stepping).
- **Screen size.** Liquid and smoke are drawn per pixel. A capped pixel ratio, such as `renderer.setPixelRatio(Math.min(devicePixelRatio, 2))`, keeps high-density screens from doubling the cost.

`sim.particleCount` tells you how many particles the simulation chose. In this repository, `npm run test:perf` runs GPU benchmarks and writes a local report.

## It looks wrong

- **Blocky soft bodies, missing thin parts, or coarse splashes.** Everything shares one particle size, set by the `particles` budget. Raise it. See [Particles and detail](simulation.md#particles-and-detail).
- **The simulation runs too fast or too slow.** Call `sim.step()` with no argument, or with a fixed value like `1 / 60`. Without an argument it keeps real-time pace at any frame rate. Passing the frame's elapsed time to `step()`, as in `sim.step(delta)`, makes each step a different length and the simulation less stable.
- **Liquid looks springy or bouncy.** Raise `substeps`.
- **Cloth or a soft body is squashed or bunched from the start.** It started partly outside the container, or inside an obstacle. See [The container](simulation.md#the-container).
- **Honey won't coil.** Raise `thickness`, and give the obstacles it lands on high `friction`, around 0.9.

## Smoke problems

- **Smoke is thin or grainy.** Raise `rate`, and set it high from the start, since it can't grow much later. Raise `density` to make the same smoke look thicker.
- **Smoke doesn't rise.** Raise `heat`. At `heat: 0`, heated air doesn't rise.
- **The plume dies out too low.** Lower `cooling`, or raise `lifetime` so tracers live long enough to reach the top.
- **Smoke stops short of the top.** Smoke is removed in the top tenth of the container. Make the container taller.
- **Smoke with water or cloth.** Gas and liquid can't be simulated together, so they can't share one `Simulation`. Use two; see [Smoke and water in one scene](smoke.md#smoke-and-water-in-one-scene).

## Hand-built scenes (low-level API)

This section is for scenes built from the [low-level classes](advanced/low-level-api.md) instead of `Simulation`.

### Low-level error messages

| Error                                                                            | Fix                                                                                              |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| ``FluidSystem: pass `viscosity` in the options to enable it before changing it`` | Optional fluid effects are compiled only when given. Pass the option at construction, even as 0. |
| `FluidSystem.addBoundary: add boundaries before creating the SimLoop`            | Call `addBoundary` first, then create the loop.                                                  |
| `FluidSystem.addBoundary: a boundary cannot overlap the fluid`                   | Give the fluid a `range` that excludes the solid's particles.                                    |
| ``ViscositySolver: list it after its FluidSystem in `materials` ``               | Reorder `materials`: `[fluid, viscosity]`.                                                       |
| ``GasSystem: list it before its FluidSystem in `materials` ``                    | Reorder `materials`: `[gas, air]`.                                                               |
| `SoftbodySystem: body N needs edges for local shape matching`                    | Pass `edges` from `voxelize`, or use `shapeMatching: 'global'`.                                  |
| `SoftbodySystem: particle N belongs to two bodies`                               | Body ranges overlap. Give each body its own block of slots.                                      |
| `SimLoop: every collider must be built for the same ParticleSystem`              | Create colliders with the loop's particle system.                                                |
| `ParticleSystem has been disposed`                                               | Something still uses a disposed system, often a renderer or shader from a previous scene.        |

### Explodes or jitters

- **Particles start overlapping.** Space starting particles `2 × radius` apart.
- **Too few substeps.** Raise `loop.substeps` first. Fast objects, thin colliders, and stiff cloth need the most.
- **The timestep changes every frame.** Use `FrameStepper` instead of passing the frame's elapsed time.
- **Stiffness tuned at one resolution.** Soft-body and bending compliance depend on particle count. Scale them as described in [`SoftbodySystem`](advanced/softbody-system.md#choosing-compliance) and [`ClothSystem`](advanced/cloth-system.md#keeping-the-look-as-resolution-changes).
- **Liquid springs back or compresses.** Add substeps. Lowering the fluid's `compliance` rarely helps without them.
- **Cloth chatters against a collider.** Set the cloth's `damping` to 0.1 to 0.3, or give the collider some friction.
- **Stiff bodies stick together after a fast collision.** A known limitation at low substep counts; add substeps.

### Hand-built scenes are slow

- **Substeps** multiply the simulation cost.
- **Liquid surface:** `voxelBudget` on `FluidSurfaceRenderer` sets the surface grid size and memory.
- **Smoke volume:** `resolution` and `steps` on `GasVolumeRenderer` set its cost. Hidden volumes skip their update.
- **Fluid boundaries** on moving soft bodies and cloth are recomputed every substep and are the largest cost in coupled scenes. Pass `{ dynamic: false }` to `addBoundary` for solids that never move.
- **Readbacks** stall the GPU. Keep `readback()` and `getArrayBufferAsync` out of the frame loop.

Need emitters, custom forces, or your own renderers? See [Advanced](README.md#advanced).

---

Previous: [Smoke](smoke.md) · [Back to the docs home](README.md)
