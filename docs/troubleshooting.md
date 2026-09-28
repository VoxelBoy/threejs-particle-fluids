[Docs](README.md) › Troubleshooting

# Troubleshooting

## Common errors

| Error                                                                            | Fix                                                                                              |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `createParticleRenderer` throws that WebGPU is unavailable                       | Use a browser with WebGPU and hardware acceleration on, served from `localhost` or HTTPS.        |
| ``FluidSystem: pass `viscosity` in the options to enable it before changing it`` | Optional fluid effects are compiled only when given. Pass the option at construction, even as 0. |
| `FluidSystem.addBoundary: add boundaries before creating the SimLoop`            | Call `addBoundary` first, then create the loop.                                                  |
| `FluidSystem.addBoundary: a boundary cannot overlap the fluid`                   | Give the fluid a `range` that excludes the solid's particles.                                    |
| ``ViscositySolver: list it after its FluidSystem in `materials` ``               | Reorder `materials`: `[fluid, viscosity]`.                                                       |
| ``GasSystem: list it before its FluidSystem in `materials` ``                    | Reorder `materials`: `[smoke, air]`.                                                             |
| `SoftbodySystem: body N needs edges for local shape matching`                    | Pass `edges` from `voxelize`, or use `shapeMatching: 'global'`.                                  |
| `SoftbodySystem: particle N belongs to two bodies`                               | Body ranges overlap. Give each body its own block of slots.                                      |
| `SimLoop: every collider must be built for the same ParticleSystem`              | Create colliders with the loop's particle system.                                                |
| `bakeMeshToSdf: mesh appears non-watertight`                                     | The mesh isn't closed. Fix holes and doubled faces, or bake a simplified closed hull.            |
| `ParticleSystem has been disposed`                                               | Something still uses a disposed system, often a renderer or kernel from a previous scene.        |

## The simulation explodes or jitters

- **Particles start overlapping.** Space initial particles `2 × radius` apart. Fluid particles packed closer than the rest spacing burst outward on the first step.
- **Too few substeps.** Raise `loop.substeps` first; it makes every constraint stiffer. Fast objects, thin colliders, and stiff cloth need the most.
- **The timestep changes every frame.** Use a fixed step with `FrameStepper` instead of passing the frame's elapsed time. See [Getting started](getting-started.md#run-at-a-fixed-timestep).
- **Stiffness tuned at one resolution.** Soft-body and bending compliance depend on particle count. Scale them as described in [Soft bodies](soft-bodies.md#choosing-compliance) and [Cloth](cloth.md#keeping-the-look-as-resolution-changes).
- **Liquid springs back or compresses.** Add substeps. Lowering `compliance` below the default rarely helps without them.
- **Cloth chatters against a collider.** Set `damping` to 0.1–0.3, or give the collider some friction so the cloth rides it.
- **Stiff bodies stick together after a fast collision.** A known limitation at low substep counts; add substeps.

## Particles pass through things

- **Colliders**: thin shapes need more substeps, and an `SDFCollider` around cloth needs a `thickness` of about half the particle spacing.
- **Other bodies**: soft bodies and cloth only touch each other with `contact` on. In crowded scenes, raise `maxContacts`; see [Combining materials](combining-materials.md#soft-bodies-and-cloth-together).
- **Liquid through cloth**: make the cloth heavier with a higher `surfaceDensity`.

## Smoke problems

- **Smoke pools at the top.** Pass `bounds` to `GasSystem` so tracers retire before the ceiling.
- **Hot air piles up and the vent empties.** Give the air a little gravity, around 1 m/s². See [Smoke](smoke.md#keep-the-air-settled).
- **Smoke looks sparse or grainy.** Emit more tracers and raise `capacity`, or lower the renderer's `resolution` so each voxel covers more tracers.

## Seeing what's going on

Draw the raw particles under or instead of the rendered surface:

```ts
import { createParticleMesh } from 'threejs-particle-fluids';

const dots = createParticleMesh(particles, {
  range: fluid.range,
  radius: radius * 0.5,
  color: 0x5fb9ff,
});
scene.add(dots);
```

`colorNode(position)` takes a TSL function for per-particle color, for example to color by height or by a buffer such as `smoke.temperature`.

To inspect numbers, `await particles.readback()` copies every particle buffer to the CPU. It stalls until the GPU is idle, so keep it out of the frame loop.

## Performance

- **Particle count** drives the cost of everything. The demo runs its presets at 5,000–50,000 particles.
- **Substeps** multiply the simulation cost. `FrameStepper`'s `maxStepsPerFrame` caps catch-up steps so a slow frame doesn't snowball.
- **Liquid surface**: `voxelBudget` sets the surface grid size and memory. Lower it for speed, raise it for detail.
- **Smoke volume**: `resolution` and `steps` set the cost. Hidden volumes skip their update.
- **Fluid boundaries** on moving soft bodies and cloth are recomputed every substep and are the largest cost in coupled scenes. Pass `{ dynamic: false }` for boundaries that never move.
- **Readbacks** stall the GPU. Avoid `readback()` and `getArrayBufferAsync` in the frame loop.

`npm run test:perf` in this repository runs GPU benchmarks for representative scenes and writes a local report.

---

Previous: [Custom materials](custom-materials.md) · [Back to the docs home](README.md)
