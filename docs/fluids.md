[Docs](README.md) › Fluids

# Fluids

`FluidSystem` simulates liquids (and the air that carries [smoke](smoke.md)) with Position Based Fluids (Macklin & Müller 2013). `FluidSurfaceRenderer` draws a liquid as a ray-marched surface with refraction, absorption, and reflections.

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

| Option            | Default               | What it does                                                                               |
| ----------------- | --------------------- | ------------------------------------------------------------------------------------------ |
| `range`           | every particle        | The particles that make up the fluid.                                                      |
| `restDensity`     | 1000                  | Rest density in kg/m³.                                                                     |
| `particleSpacing` | 2 × particle radius   | Distance between particles at rest. Sets each particle's mass.                             |
| `smoothingRadius` | 2 × `particleSpacing` | How far particles interact (the SPH smoothing length).                                     |
| `compliance`      | 1e-4                  | How much the fluid may compress. 0 is incompressible.                                      |
| `viscosity`       | off                   | XSPH viscosity: blends each particle's velocity with its neighbors'. About 0.01 for water. |
| `vorticity`       | off                   | Vorticity confinement: puts back swirling motion the solver damps.                         |
| `surfaceTension`  | off                   | Pulls the liquid into drops and smooth sheets (Akinci et al. 2013).                        |
| `adhesion`        | off                   | Attraction toward [boundary particles](#solids-in-the-liquid), so the liquid wets solids.  |

The four optional effects are compiled into the solver only when you pass them. Once passed, even as 0, you can change them live through the matching property:

```ts
const water = new FluidSystem(particles, { viscosity: 0.01, surfaceTension: 0 });
water.viscosity = 0.05;
water.surfaceTension = 0.12;
```

Setting a property you didn't pass in the options throws, because its kernel doesn't exist.

### Tuning

| To get                          | Try                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------ |
| Water                           | `viscosity: 0.01–0.02`, `surfaceTension: 0.1`, `vorticity: 0.02`               |
| Syrup                           | `viscosity: 0.1–0.3`                                                           |
| Honey, tar, anything that coils | a `ViscositySolver` (below)                                                    |
| Round drops and clean sheets    | raise `surfaceTension`; above about 0.25 thin streams start flinging particles |
| Livelier splashes and swirls    | `vorticity: 0.02–0.1`                                                          |
| Liquid that clings to walls     | `adhesion: 0.1–0.2`, with the solid added as a boundary                        |

If the liquid looks springy or compresses under its own weight, add substeps before raising iterations.

## Thick liquids: `ViscositySolver`

XSPH viscosity is explicit, so it becomes unstable long before honey. `ViscositySolver` solves viscosity implicitly with Jacobi sweeps each substep, which stays stable at very high viscosities:

```ts
const honey = new FluidSystem(particles, { viscosity: 0.03, surfaceTension: 0.08 });
const thick = new ViscositySolver(honey, { viscosity: 20, iterations: 16 });
const loop = new SimLoop(particles, { materials: [honey, thick], colliders: [walls] });
```

- `viscosity` is the kinematic viscosity. Honey is around 20.
- `iterations` (1–64, default 12) is the number of sweeps per substep. Finer particles need more sweeps to spread the same distance; the Honey Bunny preset uses `16 · ∛(count / 10000)`.
- List it **after** its fluid in `materials`.
- `thick.viscosity` can be changed live.

## Solids in the liquid

`addBoundary(range)` makes the fluid treat other particles as a solid wall (Akinci et al. 2012): it can't pass through them, it pushes on them, so they float or sink, and it wets them when `adhesion` is set. Use it for the surface particles of soft bodies and for cloth:

```ts
for (let i = 0; i < ducks.bodies.length; i++) water.addBoundary(ducks.surfaceRange(i));
water.addBoundary(tarp.range);
```

Call it before creating the `SimLoop`. Pass `{ dynamic: false }` for particles that never move, which skips recomputing their boundary each substep. See [Combining materials](combining-materials.md).

## Pouring and emitting

The particle count is fixed, so emitters recycle particles instead of creating them. The pattern the Honey Bunny and Tarp Runoff presets use:

1. Upload every liquid particle pinned (`invMass = 0`) in a sparse grid far below the scene, and call `particles.setInvMass(fluid.range, 0)` after creating the fluid, since the fluid sets its own masses.
2. Each frame, run a small TSL kernel over the next batch that writes the nozzle position to both `particles.positions` and `particles.predictedPositions`, sets `particles.velocities` to the stream's velocity, restores `particles.invMass` to `1 / fluid.mass`, and resets `fluid.density` to `fluid.restDensity` (waiting particles have no neighbors, so their density is stale).
3. When all particles are used, reuse the oldest ones.

Place released particles one spacing apart so they don't overlap the stream. [`demo/presets/honey.ts`](../demo/presets/honey.ts) has a complete emitter.

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

Each frame the renderer turns the particles into a smooth signed field, then sphere-traces it, refracts the scene behind it, absorbs light along the refracted path, and adds environment reflections plus a highlight from the scene's main directional light. It writes depth, so ambient occlusion, fog, and later transparent objects see the real surface.

### Options

| Option                        | What it does                                                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `renderer`, `scene`, `camera` | Required. The environment map is read once, when the renderer is created.                                        |
| `bounds`                      | Required. The box the liquid can reach. Leave a few centimetres of margin around walls.                          |
| `colliders`                   | Colliders the liquid wets, drawing a meniscus where it meets them.                                               |
| `solids`                      | Particle range of floating or submerged solids the liquid wets.                                                  |
| `carve`                       | A `PrimitiveSet` of moving shapes cut out of the liquid every frame, such as bubbles.                            |
| `cavities`                    | `{ smokeColor, smokeDensity }`: draw air pockets inside the liquid with a bright rim and an optional smoke fill. |
| `motionStretch`               | Seconds to stretch fast particles along their velocity, which smooths thin streams. Default 0.                   |
| `voxelBudget`                 | Voxels in the surface grid, about 76 bytes each. Default `FluidSurfaceRenderer.defaultVoxelBudget(count)`.       |
| `refraction`                  | Bend light through the surface. Default `true`. Turn it off to avoid dark smears where objects cross it.         |
| `appearance`                  | How the liquid looks (below).                                                                                    |

### Appearance

| Field                 | Meaning                                                                        |
| --------------------- | ------------------------------------------------------------------------------ |
| `color`               | The color white light takes on after crossing `attenuationDistance` of liquid. |
| `attenuationDistance` | Metres of liquid for light to reach `color`. Short distances look dense.       |
| `scattering`          | Light scattered back out of the body: 0 is clear water, 1 is milky or honey.   |
| `ior`                 | Index of refraction. Water is 1.333.                                           |
| `roughness`           | 0–1. Blurs reflections and widens highlights.                                  |
| `envIntensity`        | Strength of the environment reflection.                                        |
| `metalness`           | 0 for a liquid, 1 for liquid metal tinted by `metalColor`.                     |
| `metalColor`          | Tint of liquid metal.                                                          |

Change it later with `surface.setAppearance({ roughness: 0.3 })`. Screen-space reflections of the scene can be toggled with `surface.reflections`.

### Clicking the liquid

`surface.pick(uv)` returns the world-space point where a viewport ray meets the liquid, or `null`. `uv` runs from 0 to 1 with y pointing down, so it matches pointer coordinates divided by the canvas size.

```ts
canvas.addEventListener('pointerup', async (event) => {
  const rect = canvas.getBoundingClientRect();
  const uv = new Vector2(
    (event.clientX - rect.left) / rect.width,
    (event.clientY - rect.top) / rect.height,
  );
  const point = await surface.pick(uv);
  if (point) splashAt(point);
});
```

---

Previous: [Core concepts](concepts.md) · Next: [Soft bodies](soft-bodies.md)
