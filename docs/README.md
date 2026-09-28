# Documentation

`threejs-particle-fluids` simulates liquids, soft bodies, cloth, and smoke as particles on the GPU, for [Three.js](https://threejs.org) with WebGPU. These pages explain how the pieces fit together and how to use each one. The [main README](../README.md) has a short quick start.

## Guides

1. [Getting started](getting-started.md): install the library, set up a renderer, and run your first simulation.
2. [Core concepts](concepts.md): particles, materials, the solver loop, units, and how to tune stability.
3. [Fluids](fluids.md): liquids with `FluidSystem`, thick liquids with `ViscositySolver`, and drawing them with `FluidSurfaceRenderer`.
4. [Soft bodies](soft-bodies.md): filling meshes with particles, shape matching, and skinning a mesh to the result.
5. [Cloth](cloth.md): building cloth from geometry, pins, wind, and a smooth surface to draw.
6. [Smoke](smoke.md): tracers carried by air, heat and buoyancy, and the two smoke renderers.
7. [Colliders](colliders.md): planes, spheres, boxes, capsules, moving shapes, and arbitrary meshes through distance fields.
8. [Combining materials](combining-materials.md): liquids pushing on soft bodies and cloth, particle contacts, and collision groups.
9. [Custom materials](custom-materials.md): adding your own TSL kernels to the solver.
10. [Troubleshooting](troubleshooting.md): common errors, instability, and performance.

## API at a glance

| Area       | Exports                                                                                                            | Guide                                   |
| ---------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| Setup      | `createParticleRenderer`, `ParticleSystem`, `SimLoop`, `FrameStepper`                                              | [Getting started](getting-started.md)   |
| Fluids     | `FluidSystem`, `ViscositySolver`, `FluidSurfaceRenderer`                                                           | [Fluids](fluids.md)                     |
| Soft body  | `SoftbodySystem`, `SoftbodyMesh`, `voxelize`                                                                       | [Soft bodies](soft-bodies.md)           |
| Cloth      | `ClothSystem`, `createClothGraph`, `createClothSurface`                                                            | [Cloth](cloth.md)                       |
| Smoke      | `GasSystem`, `GasVolumeRenderer`, `GasSpriteRenderer`, `SmokeTracers`                                              | [Smoke](smoke.md)                       |
| Colliders  | `PrimitiveSet`, `SDFCollider`, `bakeMeshToSdf`, `encodeSdfBinary`, `decodeSdfBinary`                               | [Colliders](colliders.md)               |
| Debugging  | `createParticleMesh`, `ParticleSystem.readback`                                                                    | [Troubleshooting](troubleshooting.md)   |
| Extensions | `Material`, `SolverContext`, `emitForEachNeighbor`, `createSphKernelUniforms`, `emitPoly6FromRSq`, `emitSpikyGrad` | [Custom materials](custom-materials.md) |

Every option is also documented in the type definitions, so your editor shows it on hover.

## Examples

- [`examples/fluid.ts`](../examples/fluid.ts): the smallest complete scene, a block of water collapsing in a box.
- [`demo/presets/`](../demo/presets): the thirteen demo presets, which cover every material and most options.

The API is young and may change before 1.0. The solvers are built for interactive visuals, not engineering analysis.
