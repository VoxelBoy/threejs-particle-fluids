[Docs](../README.md) › [API](../README.md#api-reference) › Fluid

# Fluid

Liquid added with [`Simulation.addFluid`](./simulation.md#addfluidoptions). The returned `Fluid` handle changes settings live.

```ts
const water = sim.addFluid({
  box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(-0.1, 0.5, 0.3)),
  color: 0x3a9fcf,
});
water.viscosity = 0.05;
```

## FluidOptions

| Option           | Type                                                                | Default                        | Description                                                                                                                                            |
| ---------------- | ------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `box`            | `Box3`                                                              | —                              | Fill this box, clipped one particle radius inside the container. Give `box` or `mesh`, not both.                                                       |
| `mesh`           | `Mesh`                                                              | —                              | Fill this closed mesh at its current world transform. The mesh is hidden until `dispose()`.                                                            |
| `viscosity`      | `number`                                                            | `0.01`                         | XSPH velocity smoothing. About 0.01 for water, up to 0.3 for syrup.                                                                                    |
| `surfaceTension` | `number`                                                            | `0.1`                          | Cohesion. Above about 0.25, streams break into drops.                                                                                                  |
| `vorticity`      | `number`                                                            | `0.02`                         | Vorticity confinement strength.                                                                                                                        |
| `adhesion`       | `number`                                                            | `0.1`                          | Attraction to soft body and cloth particles.                                                                                                           |
| `thickness`      | `number`                                                            | not set                        | Implicit viscosity for honey-like liquids, about 20 for honey. Setting it (even to `0`) adds a [`ViscositySolver`](./fluid-system.md#viscositysolver). |
| `color`          | `number`                                                            | `0x3a9fcf`                     | Shorthand for `appearance.color`.                                                                                                                      |
| `appearance`     | `Partial<`[`FluidAppearance`](./fluid-system.md#fluidappearance)`>` | `{ attenuationDistance: 0.6 }` | Surface look. `color` overrides `appearance.color`.                                                                                                    |

Particles are placed on a `2 × particleRadius` grid, skipping positions inside obstacles, soft bodies, and cloth.

## Fluid

### Properties

| Property         | Type                                                             | Access     | Description                                                    |
| ---------------- | ---------------------------------------------------------------- | ---------- | -------------------------------------------------------------- |
| `viscosity`      | `number`                                                         | read/write | See [FluidOptions](#fluidoptions).                             |
| `surfaceTension` | `number`                                                         | read/write |                                                                |
| `vorticity`      | `number`                                                         | read/write |                                                                |
| `adhesion`       | `number`                                                         | read/write |                                                                |
| `thickness`      | `number`                                                         | read/write | Setting it throws unless `thickness` was passed to `addFluid`. |
| `fluidSystem`    | [`FluidSystem`](./fluid-system.md#fluidsystem-1)                 | read       | Underlying solver. Throws before start.                        |
| `surface`        | [`FluidSurfaceRenderer`](./fluid-system.md#fluidsurfacerenderer) | read       | Surface renderer. Throws before start.                         |
| `mesh`           | `Mesh`                                                           | read       | Surface mesh in the scene. Throws before start.                |

### Methods

#### `setAppearance(appearance)`

```ts
setAppearance(appearance: Partial<FluidAppearance>): void
```

Merges into the current appearance. Works before and after start.

## Limitations

- Fixed volume: fluid can't be emitted or removed after start. For pouring, use [`FluidSystem`](./fluid-system.md#emitting) directly.
- All fluids share the simulation's particle radius.
- `thickness` adds a second solver pass every substep, even when set to `0`.
- Setting a value on `fluidSystem` directly doesn't update the handle's getter.
- The surface is drawn only inside `container` (padded by 3 cm horizontally and 2 cm vertically). Without a container, only inside a box around the starting particles.
