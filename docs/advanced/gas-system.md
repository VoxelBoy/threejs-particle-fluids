[Docs](../README.md) › [Advanced](../README.md#advanced) › `GasSystem`

# `GasSystem`

Smoke is drawn from _tracers_: weightless points that drift with a simulated air flow. The air is simulated with an ordinary [`FluidSystem`](fluid-system.md); the tracers only make its motion visible. `GasSystem` owns the tracers and can also heat the air so it rises. `GasVolumeRenderer` draws the tracers as lit volumetric smoke, and `GasSpriteRenderer` as soft sprites.

`sim.addSmoke` sets up the air, a `GasSystem` with one heat source, and a `GasVolumeRenderer`; the [Smoke guide](../smoke.md) covers it. Build smoke yourself to use several heat sources, change the light direction, draw sprites, or bring your own tracers.

## Air and tracers

The air has to fill the space the smoke moves through, so fill the whole container with fluid particles. The tracers don't take part in the simulation, so you can use far more of them than air particles.

```ts
import { Vector3 } from 'three';
import {
  FluidSystem,
  GasSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  type ParticleInit,
} from 'threejs-particle-fluids';

// `renderer` is as in Getting started.
// Air filling a 1 m × 1.9 m × 1 m tank, on a grid spaced 2 × radius apart.
const radius = 0.02;
const air: ParticleInit[] = [];
for (let x = -0.5 + radius; x < 0.5; x += 2 * radius)
  for (let y = radius; y < 1.9; y += 2 * radius)
    for (let z = -0.5 + radius; z < 0.5; z += 2 * radius) air.push({ position: [x, y, z] });

const particles = new ParticleSystem(renderer, air.length, radius);
particles.uploadParticles(air);

// A closed tank: an inverted box keeps particles inside it.
const tank = new PrimitiveSet(particles);
tank.addBox(new Vector3(0, 0.95, 0), new Vector3(0.5, 0.95, 0.5), { invert: true });

const airFluid = new FluidSystem(particles, { viscosity: 0.02, vorticity: 0.06 });

const gas = new GasSystem(airFluid, { capacity: 30000, lifetime: 6 });

const loop = new SimLoop(particles, {
  substeps: 2,
  gravity: new Vector3(0, -1, 0),
  colliders: [tank],
  // The gas goes before its fluid.
  materials: [gas, airFluid],
});

// Each frame:
for (let i = 0; i < 75; i++) gas.emit([Math.random() * 0.2 - 0.1, 0.05, Math.random() * 0.2 - 0.1]);
await loop.step(1 / 60);
```

List the gas before its fluid in `materials` ([why](combining-materials.md#order-in-materials)). The loop throws otherwise.

### `GasSystem` options

| Option        | Default     | What it does                                                                                                                                   |
| ------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `capacity`    | —           | Most tracers alive at once.                                                                                                                    |
| `lifetime`    | 5           | Seconds each tracer lives. Renderers fade tracers out with age.                                                                                |
| `bounds`      | none        | A `Box3`. Tracers that leave it are retired early, so smoke can vent out of a scene instead of pooling at the top.                             |
| `heatSources` | none        | Regions that heat the air. Passing any turns on air temperature (below).                                                                       |
| `buoyancy`    | 3           | Upward acceleration in m/s² of air at temperature 1, relative to the average. Live: `gas.buoyancy`.                                            |
| `cooling`     | 0.8         | How fast air cools, as an exponential rate per second. Live: `gas.cooling`.                                                                    |
| `phase`       | whole fluid | For a fluid with several [`phases`](fluid-system.md#options), the index of the air phase. Tracers ride only that phase, and only it is heated. |

`gas.emit(position)` releases one tracer at the start of the next step and returns `false` when every tracer is still alive. Tracers are recycled oldest first. To keep a steady stream, size `capacity` for `emission rate × lifetime`. A tracer retired by `bounds` frees its slot only once its lifetime is up.

`gas.aliveCount` counts live tracers on the CPU, from when each was released, so reading it doesn't wait for the GPU. Tracers retired early by `bounds` still count until their lifetime is up.

## Heat and buoyancy

Smoke rises because hot air rises. With `heatSources`, each air particle carries a temperature: air inside a source is set to 1, cools exponentially at `cooling`, and is pushed upward by `buoyancy · (T − T̄)`, where `T̄` is the average air temperature. Because the push is measured against the average, the air as a whole doesn't drift up; only the hot parts rise.

```ts
const gas = new GasSystem(airFluid, {
  capacity: 30000,
  lifetime: 6,
  heatSources: [{ position: new Vector3(0, 0, 0), radius: 0.17 }],
  buoyancy: 3,
  cooling: 0.6,
  bounds: new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1.7, 1)),
});
```

A heat source is a sphere. Mutate `source.position` to move it. `gas.temperature` is a per-air-particle storage buffer, indexed from the start of the fluid's range, which you can read in TSL, for example to color air particles by temperature.

### Keep the air settled

The air particles push apart when they crowd together, but nothing pulls them back when they spread out. Rising air can leave gaps that never refill, and the air piles up against the ceiling. Give the air a little gravity, around 1 m/s², so the column stays settled while the hot air still rises. The Vortex Plume preset also closes the tank with a lid, so hot air can't escape through the top.

## `GasVolumeRenderer`

```ts
const volume = new GasVolumeRenderer(gas, {
  renderer,
  min: new Vector3(-0.5, 0, -0.5),
  max: new Vector3(0.5, 1.7, 0.5),
  resolution: [80, 128, 80],
  steps: 80,
  density: 0.7,
});
scene.add(volume.object);

// every frame, after loop.step():
await volume.update();
```

Each frame the renderer turns the tracers into a 3D grid of smoke density, works out how much light reaches each cell (smoke shadows itself), and draws the box by stepping rays through it. Rays stop at opaque scene geometry, so smoke passes correctly behind and in front of objects. All the detail comes from the tracers, so it moves with the smoke; more tracers make denser, finer smoke.

| Option           | Default             | What it does                                |
| ---------------- | ------------------- | ------------------------------------------- |
| `renderer`       | —                   | The renderer that runs the grid shaders.    |
| `min`, `max`     | —                   | Corners of the box the smoke is drawn in.   |
| `resolution`     | `[32, 56, 32]`      | Density grid size, 4–128 per axis.          |
| `steps`          | 56                  | Steps along each ray, 8–128.                |
| `density`        | 1                   | Opacity multiplier. Live: `volume.density`. |
| `color`          | `0xd8dfe6`          | Color of fully lit smoke.                   |
| `shadowColor`    | `0x3b4758`          | Color of smoke in its own shadow.           |
| `lightDirection` | `(-0.35, 0.8, 0.4)` | Direction toward the light.                 |

Aim for grid voxels about the size of the gap between neighboring tracers. `update()` does nothing while `volume.object` is hidden.

## `GasSpriteRenderer`

```ts
const sprites = new GasSpriteRenderer(gas, { size: 0.04, initialOpacity: 0.5, opacityTau: 3 });
scene.add(sprites.object);
```

Draws each tracer as a soft camera-facing sprite that fades with age. It's cheap and good for seeing individual tracers, but sprites aren't sorted by distance, so overlapping sprites can blend in the wrong order. Options: `color`, `size` (metres, live through `sprites.size`), `initialOpacity`, `opacityTau` (seconds for a sprite to fade to about a third of its starting opacity), and `colorNode(position, velocity)` for a per-sprite TSL color.

## Your own tracers

Both renderers draw anything that implements `SmokeTracers`: a `capacity`, a `lifetime`, and storage buffers `smokePositions`, `smokeAge`, `smokeAlive`, and optionally `smokeVelocities`.

Background reading: smoke tracers carried by simulated air (Macklin et al. 2014, section 7.2.1).

---

Previous: [`ClothSystem`](cloth-system.md) · Next: [Combining materials](combining-materials.md)
