[Docs](README.md) › Fluids

# Fluids

`sim.addFluid` fills a box or a closed mesh with liquid. The liquid is drawn as a smooth, see-through surface that bends and tints the light passing through it.

```ts
const water = sim.addFluid({
  box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(-0.1, 0.5, 0.3)),
  color: 0x3a9fcf,
});
```

The liquid starts at rest wherever you put it, then falls and flows. If the box reaches past the container, it's trimmed to fit. Liquid never starts inside an obstacle, a soft body, or cloth, so you can fill a whole tank around them.

## Options

| Option           | Default    | What it does                                                                                  |
| ---------------- | ---------- | --------------------------------------------------------------------------------------------- |
| `box`            | —          | Fill this `Box3` with liquid. Give `box` or `mesh`, not both.                                 |
| `mesh`           | —          | Fill this closed mesh with liquid, where it sits in the world. The mesh is hidden.            |
| `viscosity`      | 0.01       | How much the liquid resists flowing. About 0.01 for water, up to 0.3 for syrup.               |
| `surfaceTension` | 0.1        | Pulls the liquid into round drops and smooth sheets. Above about 0.25, thin streams break up. |
| `vorticity`      | 0.02       | Keeps swirls and splashes lively. 0.02 to 0.1.                                                |
| `adhesion`       | 0.1        | How much the liquid clings to soft bodies and cloth.                                          |
| `thickness`      | none       | Extra thickness for honey-like liquids that fold and coil. About 20 for honey.                |
| `color`          | `0x3a9fcf` | Tint of the liquid. Shorthand for `appearance.color`.                                         |
| `appearance`     | —          | Everything about the look. See [Appearance](#appearance).                                     |

`viscosity` and `thickness` both make liquid flow more slowly, in different ways. `viscosity` is cheap and works up to about syrup. For anything thicker, such as honey that folds and coils, use `thickness`. Passing it, even as 0, turns on a slower calculation that stays stable at high values, and it costs more every step. Honey keeps a low `viscosity` because `thickness` does the work.

## Live settings

`addFluid` returns a `Fluid`. Change its settings at any time, before or after the simulation starts:

```ts
water.viscosity = 0.2;
water.surfaceTension = 0.05;
water.vorticity = 0.06;
water.adhesion = 0.15;
water.setAppearance({ color: 0xc8102e, roughness: 0.2 });
water.mesh; // the liquid surface in the scene, once the simulation has started
```

`thickness` is the one exception. You can only change it if you passed `thickness` to `addFluid`, even as 0; otherwise setting it throws. For a liquid you can turn from water into honey while it runs, pass `thickness: 0` and raise it later.

For a setting not listed here, reach the object underneath. See [The objects underneath](simulation.md#the-objects-underneath).

## Recipes

| Liquid               | Settings                                                    |
| -------------------- | ----------------------------------------------------------- |
| Water                | the defaults                                                |
| Syrup                | `viscosity: 0.1` to `0.3`                                   |
| Honey                | `thickness: 20`, `viscosity: 0.03`, `surfaceTension: 0.015` |
| Round, beading drops | `surfaceTension: 0.15` to `0.25`                            |
| Sticky, wetting      | `adhesion: 0.15` to `0.2`                                   |

The Honey Bunny demo preset's honey settings:

```ts
sim.addFluid({
  box: new Box3(new Vector3(-0.1, 0.4, -0.1), new Vector3(0.1, 0.6, 0.1)),
  viscosity: 0.03,
  surfaceTension: 0.015,
  thickness: 20,
  appearance: { color: 0xf07a0c, attenuationDistance: 0.06, scattering: 0.4, ior: 1.49 },
});
```

## Appearance

The liquid is drawn like thick colored glass. Light passing through it bends and takes on its color. Pass any of these in `appearance`, or change them later with `setAppearance`:

| Field                 | Default    | What it does                                                                                |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------- |
| `color`               | `0x3a9fcf` | The color white light turns after passing through `attenuationDistance` of liquid.          |
| `attenuationDistance` | 0.6        | Metres of liquid for light to reach `color`. Short distances look dense and dark.           |
| `scattering`          | 0.08       | Light bounced back out of the liquid. 0 is clear water, 1 is milky.                         |
| `ior`                 | 1.333      | Index of refraction: how strongly the surface bends light. Water is 1.333, glass about 1.5. |
| `roughness`           | 0.04       | 0 to 1. Higher values blur reflections and soften highlights.                               |
| `envIntensity`        | 1          | Strength of the environment reflection.                                                     |
| `metalness`           | 0          | Set to 1 for liquid metal.                                                                  |
| `metalColor`          | `0xc8d2da` | Tint of liquid metal.                                                                       |

For where reflections and highlights come from, see [Lighting](getting-started.md#lighting).

## Filling a mesh

Pass a closed mesh to pour liquid in its shape. Position, rotate, and scale the mesh first. The liquid takes its shape where it sits when you call `addFluid`, and the mesh is hidden once the simulation starts.

```ts
const drop = new Mesh(new SphereGeometry(0.15, 32, 24));
drop.position.set(0, 0.6, 0);
sim.addFluid({ mesh: drop });
```

The mesh must be closed, with no holes, so there's a clear inside to fill.

## Clicking the liquid

`water.surface` is the renderer that draws the liquid. Its `pick` method finds where a screen point hits the liquid, or returns `null`. It's available once the simulation has started.

```ts
renderer.domElement.addEventListener('pointerup', async (event) => {
  const rect = renderer.domElement.getBoundingClientRect();
  const uv = new Vector2(
    (event.clientX - rect.left) / rect.width,
    (event.clientY - rect.top) / rect.height,
  );
  const point = await water.surface.pick(uv);
  if (point) console.log('hit the water at', point);
});
```

`uv` runs from 0 to 1 across the canvas, with y pointing down.

`Simulation` has no splash method. The demo's click-to-splash runs a small compute shader that pushes the particles near `point`; see `interact` in [`demo/presets/liquids.ts`](../demo/presets/liquids.ts). You can run the same kind of shader on [`sim.particleSystem`](simulation.md#the-objects-underneath) between steps. Without one, `pick` is still good for placing a mesh where the user clicked.

## More than one liquid

Call `addFluid` more than once for liquids with different settings or colors. They share the space and push on each other.

## Things `addFluid` doesn't do

Pouring from a nozzle, or adding liquid while the simulation runs, needs the low-level API. The Honey Bunny and Tarp Runoff demo presets do it that way. See [Pouring and emitting](advanced/fluid-system.md#pouring-and-emitting).

---

Previous: [The simulation](simulation.md) · Next: [Obstacles](obstacles.md)
