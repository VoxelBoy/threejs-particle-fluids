# Documentation

`threejs-particle-fluids` adds water, smoke, soft bodies, and cloth to a [three.js](https://threejs.org) scene. You say what you want and where, and it simulates and draws it on the GPU with WebGPU. The [main README](../README.md) has a short quick start.

## Guide

Start here. These pages use the `Simulation` class, which handles the physics setup for you.

1. [Getting started](getting-started.md): install, create a `Simulation`, and put water on screen.
2. [The simulation](simulation.md): the container, the particle budget, stepping, gravity, and cleaning up.
3. [Fluids](fluids.md): water, syrup, and honey, and how the liquid looks.
4. [Obstacles](obstacles.md): floors, spheres, boxes, rods, and any mesh, fixed or moving.
5. [Soft bodies](soft-bodies.md): turn any closed mesh into something squishy that floats or sinks.
6. [Cloth](cloth.md): curtains, flags, and sheets that drape over things.
7. [Smoke](smoke.md): smoke rising from a heated source.
8. [Troubleshooting](troubleshooting.md): error messages, speed, and things that look wrong.

## Advanced

`Simulation` is built from lower-level classes that are also exported. Use them when you need something `Simulation` doesn't do: [emitters](advanced/fluid-system.md#pouring-and-emitting), custom forces, your own renderers, or exact control over every setting. You don't need any of this to get started, and for a single extra setting, [The objects underneath](simulation.md#the-objects-underneath) is usually enough.

These pages use the word _material_ for a kind of physics that runs on a group of particles, such as a liquid or a cloth. It has nothing to do with a three.js `Material`, which only controls how things look.

1. [The low-level API](advanced/low-level-api.md): particles, the solver loop, and building a scene by hand.
2. [Colliders](advanced/colliders.md): `PrimitiveSet`, `SDFCollider`, and turning meshes into solid shapes.
3. [`FluidSystem`](advanced/fluid-system.md): the liquid solver, thick liquids, pouring, and the liquid renderer.
4. [`SoftbodySystem`](advanced/softbody-system.md): filling meshes with particles, shape matching, and drawing the deformed mesh.
5. [`ClothSystem`](advanced/cloth-system.md): cloth from any geometry, pins, wind, and the cloth surface.
6. [`GasSystem`](advanced/gas-system.md): smoke tracers, heat, and the two smoke renderers.
7. [Combining materials](advanced/combining-materials.md): making liquids, soft bodies, and cloth push on each other by hand.
8. [Custom materials](advanced/custom-materials.md): add your own physics to the solver with TSL, three.js's shader language.

## API at a glance

### Simulation API

| Call                                                           | Returns    | Page                                                           |
| -------------------------------------------------------------- | ---------- | -------------------------------------------------------------- |
| `createParticleRenderer`, `new Simulation`                     |            | [Getting started](getting-started.md)                          |
| `sim.step`, `sim.start`, `sim.gravity`, `sim.dispose`          |            | [The simulation](simulation.md)                                |
| `sim.addFluid`                                                 | `Fluid`    | [Fluids](fluids.md)                                            |
| `sim.addFloor`, `addSphere`, `addBox`, `addCapsule`, `addMesh` |            | [Obstacles](obstacles.md)                                      |
| `sim.addSoftbody`                                              | `Softbody` | [Soft bodies](soft-bodies.md)                                  |
| `sim.addCloth`                                                 | `Cloth`    | [Cloth](cloth.md)                                              |
| `sim.addSmoke`                                                 | `Smoke`    | [Smoke](smoke.md)                                              |
| `sim.showParticles`                                            |            | [Troubleshooting](troubleshooting.md#seeing-whats-simulated)   |
| `sim.particleSystem`, `sim.loop`, `fluid.fluidSystem`, …       |            | [The objects underneath](simulation.md#the-objects-underneath) |

### Low-level exports

| Area            | Exports                                                                                                            | Page                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Setup           | `ParticleSystem`, `SimLoop`, `FrameStepper`                                                                        | [The low-level API](advanced/low-level-api.md)                              |
| Fluids          | `FluidSystem`, `ViscositySolver`, `FluidSurfaceRenderer`                                                           | [`FluidSystem`](advanced/fluid-system.md)                                   |
| Soft bodies     | `SoftbodySystem`, `SoftbodyMesh`, `voxelize`                                                                       | [`SoftbodySystem`](advanced/softbody-system.md)                             |
| Cloth           | `ClothSystem`, `createClothGraph`, `createClothSurface`                                                            | [`ClothSystem`](advanced/cloth-system.md)                                   |
| Smoke           | `GasSystem`, `GasVolumeRenderer`, `GasSpriteRenderer`, `SmokeTracers`                                              | [`GasSystem`](advanced/gas-system.md)                                       |
| Colliders       | `PrimitiveSet`, `SDFCollider`, `bakeMeshToSdf`, `encodeSdfBinary`, `decodeSdfBinary`                               | [Colliders](advanced/colliders.md)                                          |
| Reading buffers | `createParticleMesh`, `ParticleSystem.readback`                                                                    | [The low-level API](advanced/low-level-api.md#reading-the-buffers-yourself) |
| Extensions      | `Material`, `SolverContext`, `emitForEachNeighbor`, `createSphKernelUniforms`, `emitPoly6FromRSq`, `emitSpikyGrad` | [Custom materials](advanced/custom-materials.md)                            |

Every option is also documented in the type definitions, so your editor shows it on hover.

## Examples

Each example runs in this repository with `npm run dev`. Open the matching `.html` page, such as `/examples/fluid.html`.

- [`examples/fluid.ts`](../examples/fluid.ts): a block of water collapsing in a tank.
- [`examples/smoke.ts`](../examples/smoke.ts): smoke rising from a heated vent.
- [`examples/floating.ts`](../examples/floating.ts): a light ball floats and a heavy cube sinks in a tank of water.
- [`examples/cloth.ts`](../examples/cloth.ts): a curtain pushed around by a swinging ball.
- [`examples/low-level.ts`](../examples/low-level.ts): the water scene again, built from the low-level classes.
- [`demo/presets/`](../demo/presets): the thirteen demo presets, built with the low-level API.

[`examples/setup.ts`](../examples/setup.ts) holds the renderer, lights, camera, and frame loop the examples share.

The API is young and may change before 1.0. The physics is built for interactive visuals, not engineering analysis.
