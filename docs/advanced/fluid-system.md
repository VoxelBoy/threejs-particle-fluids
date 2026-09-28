[Docs](../README.md) › [Advanced](../README.md#advanced) › `FluidSystem`

# `FluidSystem`

`FluidSystem` simulates liquids, and the air that carries [smoke](gas-system.md). Each substep it pushes particles apart where they crowd together, so the liquid keeps its volume. `FluidSurfaceRenderer` draws a liquid as a smooth surface that bends, tints, and reflects light. `sim.addFluid` creates one of each; this page is for building them yourself. The [Fluids guide](../fluids.md) covers the settings in plain terms.

## Creating a fluid

Upload the fluid's particles on a grid spaced one diameter apart, then create the fluid over them:

```ts
const particles = new ParticleSystem(renderer, count, radius);
particles.uploadParticles(points); // spaced 2 × radius apart
const water = new FluidSystem(particles, { viscosity: 0.01, surfaceTension: 0.1 });
const loop = new SimLoop(particles, { substeps: 4, materials: [water], colliders: [walls] });
```

The fluid gives its particles the mass of the liquid they stand for, `restDensity · particleSpacing³`, so you don't set `invMass` yourself. Starting particles at the rest spacing means the fluid begins at rest density instead of exploding or collapsing.

## Options

| Option            | Default               | What it does                                                                                                                                                            |
| ----------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `range`           | every particle        | The particles that make up the fluid.                                                                                                                                   |
| `restDensity`     | 1000                  | Density of the liquid when it isn't squeezed, in kg/m³. Water is 1000.                                                                                                  |
| `particleSpacing` | 2 × particle radius   | Distance between particles at rest. Sets each particle's mass.                                                                                                          |
| `smoothingRadius` | 2 × `particleSpacing` | How far away a particle's neighbors can be and still affect it.                                                                                                         |
| `compliance`      | 1e-4                  | How much the fluid may compress. 0 is incompressible, and higher is softer.                                                                                             |
| `viscosity`       | off                   | Blends each particle's velocity with its neighbors'. About 0.01 for water.                                                                                              |
| `vorticity`       | off                   | Puts back swirling motion the solver damps.                                                                                                                             |
| `surfaceTension`  | off                   | Pulls the liquid into drops and smooth sheets.                                                                                                                          |
| `adhesion`        | off                   | Attraction toward [boundary particles](#solids-in-the-liquid), so the liquid wets solids.                                                                               |
| `phases`          | one phase             | Split the fluid into consecutive blocks with their own rest density, such as water with air above it. Lighter phases rise through heavier ones. Replaces `restDensity`. |

The four optional effects are compiled into the solver only when you pass them. Once passed, even as 0, you can change them live through the matching property:

```ts
const water = new FluidSystem(particles, { viscosity: 0.01, surfaceTension: 0 });
water.viscosity = 0.05;
water.surfaceTension = 0.12;
```

Setting a property you didn't pass in the options throws, because its shader was never built.

### Tuning

The [recipes in the Fluids guide](../fluids.md#recipes) apply here too. `Simulation` uses `viscosity: 0.01`, `surfaceTension: 0.1`, `vorticity: 0.02`, and `adhesion: 0.1`. Adhesion only acts on solids added with [`addBoundary`](#solids-in-the-liquid).

If the liquid looks springy or compresses under its own weight, add substeps before raising iterations.

## Thick liquids: `ViscositySolver`

The `viscosity` option works by blending each particle's velocity with its neighbors'. That's cheap, but it becomes unstable well before the liquid is as thick as honey. `ViscositySolver` uses a slower method that stays stable even at honey thickness. It repeats a smoothing pass `iterations` times each substep:

```ts
const honey = new FluidSystem(particles, { viscosity: 0.03, surfaceTension: 0.015 });
const thick = new ViscositySolver(honey, { viscosity: 20, iterations: 16 });
const loop = new SimLoop(particles, { materials: [honey, thick], colliders: [walls] });
```

- `viscosity`: how thick the liquid is, on a different scale from `FluidSystem`'s `viscosity`. Honey is around 20.
- `iterations` (1–64, default 12): smoothing passes per substep. Finer particles need more passes to spread the same distance; the Honey Bunny preset and `Simulation` use `16 · ∛(count / 10000)`, where `count` is the fluid's particle count.
- List it after its fluid in `materials` ([why](combining-materials.md#order-in-materials)).
- `thick.viscosity` can be changed live.

## Solids in the liquid

`addBoundary(range, options?)` makes the fluid treat other particles as a solid: it can't pass through them, it pushes on them, and it wets them when `adhesion` is set.

```ts
for (let i = 0; i < ducks.bodies.length; i++) water.addBoundary(ducks.surfaceRange(i));
water.addBoundary(tarp.range);
```

- Call it before creating the `SimLoop`.
- Pass `{ dynamic: false }` for particles that never move. That skips recomputing their boundary every substep.

[Combining materials](combining-materials.md) covers when to use boundaries, contacts, and collision groups.

## Pouring and emitting

The particle count is fixed, so an emitter doesn't create particles. It keeps them waiting out of sight and moves them to the nozzle a few at a time. The Honey Bunny and Tarp Runoff presets work this way:

1. Upload every liquid particle pinned (`invMass = 0`) in a sparse grid far below the scene. After creating the fluid, call `particles.setInvMass(fluid.range, 0)` again, because the fluid sets its own masses.
2. Each frame, run a small TSL compute shader over the next batch of particles. For each one it:
   - writes the nozzle position to `particles.positions` and `particles.predictedPositions`;
   - sets `particles.velocities` to the stream's velocity;
   - sets `particles.invMass` back to `1 / fluid.mass`;
   - resets `fluid.density` to `fluid.restDensity`. A waiting particle has no neighbors, so its stored density is out of date.
3. Release a new batch only once the stream has moved one particle spacing, so new particles don't overlap the ones before them.

A trimmed-down version of the Honey Bunny emitter, pouring a 4 × 4 square stream straight down:

```ts
import { Vector3 } from 'three';
import { Fn, If, instanceIndex, int, uniform, vec4 } from 'three/tsl';
import { FluidSystem, ParticleSystem, SimLoop } from 'threejs-particle-fluids';

// `renderer` is as in Getting started.
const count = 20000;
const radius = 0.01;
const spacing = 2 * radius;
const particles = new ParticleSystem(renderer, count, radius);

// 1. Every liquid particle waits, pinned, in a sparse grid far below the scene.
particles.uploadParticles(
  Array.from({ length: count }, (_, k) => ({
    position: [(k % 150) * 0.25 - 18.75, -30, Math.floor(k / 150) * 0.25 - 12.5] as const,
  })),
);
const fluid = new FluidSystem(particles, { viscosity: 0.01 });
particles.setInvMass(fluid.range, 0); // the fluid set its own masses, so pin the particles again

// 2. A shader that releases one layer: a 4 × 4 square at the nozzle, one spacing apart.
const side = 4;
const perLayer = side * side;
const nozzle = uniform(new Vector3(0, 0.6, 0));
const first = uniform(0, 'float'); // first slot of the layer
const release = Fn(() => {
  const i = instanceIndex;
  const slot = int(i).sub(first.toInt()).toVar();
  If(slot.greaterThanEqual(0).and(slot.lessThan(perLayer)), () => {
    const x = slot
      .mod(side)
      .toFloat()
      .sub((side - 1) / 2)
      .mul(spacing);
    const z = slot
      .div(side)
      .toFloat()
      .sub((side - 1) / 2)
      .mul(spacing);
    const position = vec4(nozzle.x.add(x), nozzle.y, nozzle.z.add(z), 0);
    particles.positions.element(i).assign(position);
    particles.predictedPositions.element(i).assign(position);
    particles.velocities.element(i).assign(vec4(0, -0.5, 0, 0)); // 0.5 m/s downward
    particles.invMass.element(i).assign(1 / fluid.mass);
    // A waiting particle has no neighbors, so its stored density is out of date.
    fluid.density.element(i).assign(fluid.restDensity);
  });
})().compute(count);

// 3. Release a new layer each time the stream has moved one spacing.
let released = 0;
let travelled = 0;
async function pour(dt: number) {
  travelled += 0.5 * dt;
  if (travelled < spacing || released + perLayer > count) return;
  travelled -= spacing;
  first.value = released;
  released += perLayer;
  await renderer.computeAsync(release);
}

const loop = new SimLoop(particles, { substeps: 4, materials: [fluid] }); // add your colliders

// Each frame, before stepping:
await pour(1 / 60);
await loop.step(1 / 60);
```

This emitter stops when every particle has been poured. [`demo/presets/honey.ts`](../../demo/presets/honey.ts) is the full version, with a round, moving nozzle. [`demo/presets/tarp.ts`](../../demo/presets/tarp.ts) keeps pouring by reusing the oldest liquid once every particle is used.

## Drawing the liquid: `FluidSurfaceRenderer`

```ts
const surface = new FluidSurfaceRenderer(water, {
  renderer,
  scene,
  camera,
  bounds: new Box3(new Vector3(-0.55, -0.02, -0.35), new Vector3(0.55, 0.8, 0.35)),
  colliders: [walls],
  appearance: { color: 0x3a9fcf, attenuationDistance: 0.5, roughness: 0.1 },
});
scene.add(surface.mesh);

// every frame, after loop.step():
await surface.update();
```

Each frame the renderer builds a smooth surface around the particles and draws it. Light bends through the surface, picks up the liquid's color, reflects the environment, and gets a highlight from the scene's main directional light. It writes depth, so ambient occlusion, fog, and later transparent objects see the real surface.

### Options

| Option                        | What it does                                                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `renderer`, `scene`, `camera` | Required. The environment map is read once, when the renderer is created.                                        |
| `bounds`                      | Required. The box the liquid can reach. Leave a few centimetres of margin around walls.                          |
| `colliders`                   | Colliders the liquid wets, drawing the thin edge where it meets them.                                            |
| `solids`                      | Particle range of floating or submerged solids the liquid wets.                                                  |
| `carve`                       | A `PrimitiveSet` of moving shapes cut out of the liquid every frame.                                             |
| `cavities`                    | `{ smokeColor, smokeDensity }`: draw air pockets inside the liquid with a bright rim and an optional smoke fill. |
| `motionStretch`               | Seconds to stretch fast particles along their velocity, which smooths thin streams. Default 0.                   |
| `voxelBudget`                 | Voxels in the surface grid, about 76 bytes each. Default `FluidSurfaceRenderer.defaultVoxelBudget(count)`.       |
| `refraction`                  | Bend light through the surface. Default `true`. Turn it off to avoid dark smears where objects cross it.         |
| `appearance`                  | How the liquid looks (below).                                                                                    |

### Appearance and picking

`appearance` takes the fields listed under [Appearance in the Fluids guide](../fluids.md#appearance). The renderer's own defaults are `color: 0x2a8fb0` and `attenuationDistance: 0.6`; `Simulation` uses `0x3a9fcf` instead. Change it later with `surface.setAppearance({ roughness: 0.3 })`. Screen-space reflections of the scene can be toggled with `surface.reflections`.

`surface.pick(uv)` returns the world-space point where a viewport ray meets the liquid, or `null`. See [Clicking the liquid](../fluids.md#clicking-the-liquid).

Background reading: Position Based Fluids (Macklin & Müller 2013), surface tension (Akinci et al. 2013), and fluid boundaries (Akinci et al. 2012).

---

Previous: [Colliders](colliders.md) · Next: [`SoftbodySystem`](softbody-system.md)
