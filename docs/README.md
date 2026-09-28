# threejs-particle-fluids documentation

GPU particle physics for three.js (WebGPU): liquids, soft bodies, cloth, and smoke.

| Page                          | Contents                                                          |
| ----------------------------- | ----------------------------------------------------------------- |
| [Guide](guide.md)             | Requirements, first scene, particle sizing, frame loop, debugging |
| [Limitations](limitations.md) | Platform, performance, accuracy, and API constraints              |

## API reference

### Simulation API

| Page                            | Exports                                                                                                                             |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [Simulation](api/simulation.md) | `createParticleRenderer`, `Simulation`, `SimulationOptions`, obstacles (`addFloor`, `addSphere`, `addBox`, `addCapsule`, `addMesh`) |
| [Fluid](api/fluid.md)           | `FluidOptions`, `Fluid`                                                                                                             |
| [Softbody](api/softbody.md)     | `SoftbodyOptions`, `Softbody`                                                                                                       |
| [Cloth](api/cloth.md)           | `ClothOptions`, `Cloth`                                                                                                             |
| [Smoke](api/smoke.md)           | `SmokeOptions`, `Smoke`                                                                                                             |

### Low-level API

| Page                                     | Exports                                                                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Core](api/core.md)                      | `ParticleSystem`, `ParticleInit`, `ParticleRange`, `ParticleSnapshot`, `assertRange`, `SimLoop`, `FrameStepper`, `createParticleMesh`            |
| [Colliders](api/colliders.md)            | `Collider`, `PrimitiveSet`, `SDFCollider`, `SDFData`, `bakeMeshToSdf`, `sampleSdf`, `sampleSdfGradient`, `encodeSdfBinary`, `decodeSdfBinary`    |
| [FluidSystem](api/fluid-system.md)       | `FluidSystem`, `ViscositySolver`, `FluidSurfaceRenderer`, `FluidAppearance`                                                                      |
| [SoftbodySystem](api/softbody-system.md) | `SoftbodySystem`, `SoftbodyDef`, `SoftbodyMesh`, `voxelize`                                                                                      |
| [ClothSystem](api/cloth-system.md)       | `ClothSystem`, `createClothGraph`, `createClothSurface`                                                                                          |
| [GasSystem](api/gas-system.md)           | `GasSystem`, `HeatSource`, `GasVolumeRenderer`, `GasSpriteRenderer`, `SmokeTracers`                                                              |
| [Extending](api/extending.md)            | `Material`, `SolverContext`, `HashGrid`, `NeighborList`, `emitForEachNeighbor`, SPH kernels, constraint helpers, `Accumulator`, `toTriangleMesh` |

## Units

| Quantity     | Unit                 |
| ------------ | -------------------- |
| Length       | m                    |
| Time         | s                    |
| Density      | kg/m³ (water = 1000) |
| Acceleration | m/s²                 |
| Velocity     | m/s                  |

## Examples

Run `npm run dev` in this repository and open the matching `.html` page.

| File                                                | Shows                                        |
| --------------------------------------------------- | -------------------------------------------- |
| [`examples/fluid.ts`](../examples/fluid.ts)         | Water block collapsing in a tank             |
| [`examples/floating.ts`](../examples/floating.ts)   | Light soft body floats, heavy one sinks      |
| [`examples/cloth.ts`](../examples/cloth.ts)         | Curtain pushed by a moving sphere            |
| [`examples/smoke.ts`](../examples/smoke.ts)         | Smoke rising from a heated vent              |
| [`examples/low-level.ts`](../examples/low-level.ts) | The water scene built from low-level classes |
| [`demo/presets/`](../demo/presets)                  | The 13 demo presets (low-level API)          |
