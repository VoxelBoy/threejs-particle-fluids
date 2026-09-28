[Docs](README.md) › Limitations

# Limitations

What the library can't do yet, and where it's slow. Each [API page](README.md#api-reference) also ends with the limits of its own classes.

## Platform

- The library needs WebGPU. There's no WebGL fallback, so [`createParticleRenderer`](api/simulation.md#createparticlerenderer) throws in browsers without it.
- It works with three.js r184 only.
- The GPU has to support the limits listed under [`createParticleRenderer`](api/simulation.md#createparticlerenderer).
- The API may change before version 1.0.

## Performance

- Frame time grows with the particle count, the number of substeps, and the screen size. Liquid and smoke are drawn per pixel, so high-resolution screens cost more.
- Laptop GPUs can struggle at 10,000 particles, including an Apple M1 Pro. Measure on the hardware you plan to support.
- Liquid touching moving soft bodies or cloth is the most expensive combination, because the solids' boundaries are rebuilt every substep.
- A liquid with `thickness` adds a second solver pass to every substep.
- Filling meshes with particles and preparing `addMesh` obstacles happen on the CPU during `start()`, and they block the page while they run.
- Reading particle data back to the CPU with `readback()` stalls the GPU. Keep it out of the frame loop.

## Accuracy

- The solvers are built for real-time visuals, not engineering. They use position-based methods that trade physical accuracy for speed and stability.
- Results change somewhat with the number of substeps and the particle size. Soft body and cloth stiffness are adjusted for particle count, but not perfectly.
- Stiff bodies that hit each other fast can stick together. Raising substeps helps.
- Fast or thin obstacles can let particles pass through. Raising substeps helps here too.
- Each particle looks at no more than 64 neighbors. In very dense spots, extra neighbors are ignored. [`FluidSystem.readbackOverflow()`](api/fluid-system.md) tells you when that happens.

## Simulation API

- You can't add or remove anything after the simulation starts.
- Every particle in a simulation is the same size.
- Smoke needs its own simulation, and each simulation has one smoke source.
- The container can't move.
- The amount of liquid is fixed once the simulation starts.
- Cloth can only be a rectangle.
- You can't add your own materials.

The [low-level API](api/core.md) removes most of these limits.

## Low-level API

- A [`ParticleSystem`](api/core.md#particlesystem) has a fixed number of slots, set when you create it.
- Once a [`SimLoop`](api/core.md#simloop) is built, you can't add materials, colliders, or liquid boundaries to it.
- The order of materials in a `SimLoop` matters. A `ViscositySolver` must come after its `FluidSystem`, and a `GasSystem` must come before its `FluidSystem`.
- You can't remove a shape from a [`PrimitiveSet`](api/colliders.md#primitiveset).
- An [`SDFCollider`](api/colliders.md#sdfcollider) needs a closed mesh, scaled equally along every axis.
