[Docs](../README.md) › [API](../README.md#api-reference) › Fluid

# Fluid

A liquid added with [`Simulation.addFluid`](./simulation.md#addfluidoptions). The `Fluid` it returns lets you change the liquid's settings while it runs.

```ts
const water = sim.addFluid({
  box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(-0.1, 0.5, 0.3)),
  color: 0x3a9fcf,
});
water.viscosity = 0.05;
```

## FluidOptions

| Option           | Type                                                                | Default                        | Description                                                                                                  |
| ---------------- | ------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `box`            | `Box3`                                                              | —                              | A box to fill with liquid. Only the part inside the container is filled. Give `box` or `mesh`, not both.     |
| `mesh`           | `Mesh`                                                              | —                              | A closed mesh to fill with liquid, where it sits in the scene. The mesh is hidden while the simulation runs. |
| `viscosity`      | `number`                                                            | `0.01`                         | How much the liquid resists flowing. About 0.01 for water, up to 0.3 for syrup.                              |
| `surfaceTension` | `number`                                                            | `0.1`                          | Pulls the liquid into round drops and smooth sheets. Above about 0.25, streams break into drops.             |
| `vorticity`      | `number`                                                            | `0.02`                         | Keeps swirls from dying out.                                                                                 |
| `adhesion`       | `number`                                                            | `0.1`                          | How much the liquid clings to soft bodies and cloth.                                                         |
| `thickness`      | `number`                                                            | not set                        | Extra thickness for honey-like liquids, about 20 for honey. See the note below.                              |
| `color`          | `number`                                                            | `0x3a9fcf`                     | Tint of the liquid. Same as `appearance.color`, and wins if you give both.                                   |
| `appearance`     | `Partial<`[`FluidAppearance`](./fluid-system.md#fluidappearance)`>` | `{ attenuationDistance: 0.6 }` | How the liquid looks: color, clarity, roughness, and more.                                                   |

`viscosity` can't make a liquid much thicker than syrup without becoming unstable. For honey and other very thick liquids, use `thickness`, which runs a [second, slower solver](./fluid-system.md#viscositysolver) that stays stable at high values. Passing `thickness`, even as `0`, is also what lets you change it later.

The liquid starts as particles on a grid that fills the box or mesh, skipping space that obstacles, soft bodies, or cloth already take up.

## Fluid

### Properties

| Property         | Type                                                             | Access     | Description                                                         |
| ---------------- | ---------------------------------------------------------------- | ---------- | ------------------------------------------------------------------- |
| `particleCount`  | `number`                                                         | read       | Particles in this liquid. `0` until the simulation starts.          |
| `viscosity`      | `number`                                                         | read/write | See [FluidOptions](#fluidoptions).                                  |
| `surfaceTension` | `number`                                                         | read/write | See [FluidOptions](#fluidoptions).                                  |
| `vorticity`      | `number`                                                         | read/write | See [FluidOptions](#fluidoptions).                                  |
| `adhesion`       | `number`                                                         | read/write | See [FluidOptions](#fluidoptions).                                  |
| `thickness`      | `number`                                                         | read/write | Throws when set unless you passed `thickness` to `addFluid`.        |
| `fluidSystem`    | [`FluidSystem`](./fluid-system.md#fluidsystem-1)                 | read       | The solver underneath. Available after `start()`.                   |
| `surface`        | [`FluidSurfaceRenderer`](./fluid-system.md#fluidsurfacerenderer) | read       | The renderer that draws the liquid. Available after `start()`.      |
| `mesh`           | `Mesh`                                                           | read       | The liquid's surface mesh in your scene. Available after `start()`. |

### Methods

#### `setAppearance(appearance)`

```ts
setAppearance(appearance: Partial<FluidAppearance>): void
```

Changes how the liquid looks. Fields you leave out keep their current values. You can call it before or after the simulation starts.

## Limitations

- The amount of liquid is fixed once the simulation starts. To pour liquid in over time, use [`FluidSystem`](./fluid-system.md#emitting) directly.
- Every liquid in a simulation has the same particle size.
- `thickness` makes each step slower, even when it's set to `0`.
- If you change a setting on `fluidSystem` directly, the matching property on the `Fluid` still shows the old value.
- The liquid is only drawn inside the container, plus a few centimetres of margin. Without a container, it's only drawn in a box around where it started.
