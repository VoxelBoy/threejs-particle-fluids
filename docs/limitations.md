[Docs](README.md) › Limitations

# Limitations

Known constraints of the library as of 0.1. Per-class limits are listed at the bottom of each [API page](README.md#api-reference).

## Platform

- WebGPU only. No WebGL fallback. [`createParticleRenderer`](api/simulation.md#createparticlerenderer) throws without it.
- Requires `three` r184 exactly (`>=0.184.0 <0.185.0`).
- Requires the device limits listed under [`createParticleRenderer`](api/simulation.md#createparticlerenderer).
- The API may change before 1.0.

## Performance

- Cost scales with particle count, substeps, and screen resolution (liquid and smoke are ray-marched per pixel).
- Laptop GPUs, including Apple M1 Pro, can struggle at 10,000 particles. Measure on your target hardware.
- Fluid boundaries on moving soft bodies and cloth are recomputed every substep and are the largest cost in coupled scenes.
- `thickness` (implicit viscosity) adds a second solver pass per substep.
- Voxelizing meshes and baking distance fields run on the CPU, on the main thread, during `start()`.
- `readback()` stalls the GPU. Keep it out of the frame loop.

## Accuracy

- Position-based dynamics (PBF, XPBD, shape matching). Built for real-time visuals, not engineering analysis.
- Results depend on substeps and particle radius. Stiffness and bending are scaled with resolution but don't match exactly across resolutions.
- Stiff bodies colliding fast at low substep counts can stick together.
- Fast or thin obstacles can be tunneled through. Raise substeps.
- Each particle tracks at most 64 neighbors. Extra neighbors are dropped silently.

## Simulation API

- Everything is added before the first step. Nothing can be added or removed afterwards.
- One particle radius per simulation.
- Smoke can't share a simulation with liquid, soft bodies, or cloth. One smoke source per simulation.
- The container is fixed.
- Fluids can't be emitted after start.
- Cloth is rectangular only.
- Custom materials can't be added.

## Low-level API

- A [`ParticleSystem`](api/core.md#particlesystem)'s capacity is fixed at construction.
- Materials, colliders, and fluid boundaries are fixed once a [`SimLoop`](api/core.md#simloop) is built.
- Material order in `SimLoop` matters: `ViscositySolver` after its `FluidSystem`, `GasSystem` before its `FluidSystem`.
- [`PrimitiveSet`](api/colliders.md#primitiveset) primitives can't be removed.
- [`SDFCollider`](api/colliders.md#sdfcollider) needs uniform scale and a watertight mesh.
- `dispose()` methods don't free every GPU buffer.
