[Docs](README.md) › The simulation

# The simulation

This page covers the `Simulation` class itself: the container, what you can add, the particle budget, stepping, gravity, and cleaning up. [Getting started](getting-started.md) shows it in a complete scene.

## The container

`container` is a `Box3` that everything stays inside. It has walls on the bottom and all four sides. The top is open unless you pass `closed: true`.

- The container is invisible. Draw your own tank mesh, or an outline with `scene.add(new Box3Helper(container))`.
- Liquid is drawn only inside the container. Without one, it's drawn in a box around where things start, with some room to spread, and the simulation logs a warning to the console.
- Without a container, nothing stops things falling forever. Add a floor with [`addFloor`](obstacles.md#floor) instead.
- [Smoke](smoke.md) always needs a container, because the air fills it.
- Soft bodies and cloth must start fully inside the container, clear of obstacles and of each other. The walls push back anything outside them, which squashes or bunches it. Liquid is the only thing trimmed to fit.
- The container can't move once the simulation starts. To slosh liquid, tilt [gravity](#gravity) (`sim.gravity.set(3, -9.81, 0)`), or push it around with a moving box ([`addBox` with `follow`](obstacles.md#moving-obstacles)). The Wave Chamber demo preset turns its whole tank by building the walls from moving boxes, with the low-level API.

## Adding things

Everything you add is placed in world space, in metres. A 1 m tank is a good size to start with.

| Method        | Adds                                  | Guide                         |
| ------------- | ------------------------------------- | ----------------------------- |
| `addFluid`    | Liquid filling a box or a closed mesh | [Fluids](fluids.md)           |
| `addFloor`    | A floor nothing falls through         | [Obstacles](obstacles.md)     |
| `addSphere`   | A solid ball, fixed or moving         | [Obstacles](obstacles.md)     |
| `addBox`      | A solid box, fixed or moving          | [Obstacles](obstacles.md)     |
| `addCapsule`  | A solid rod with rounded ends         | [Obstacles](obstacles.md)     |
| `addMesh`     | Any closed mesh as a solid            | [Obstacles](obstacles.md)     |
| `addSoftbody` | A soft, squishy copy of a closed mesh | [Soft bodies](soft-bodies.md) |
| `addCloth`    | A rectangle of cloth                  | [Cloth](cloth.md)             |
| `addSmoke`    | Smoke rising from a heated source     | [Smoke](smoke.md)             |

Liquids, soft bodies, and cloth in one simulation affect each other with no extra setup. Water pushes soft bodies around, light ones float and heavy ones sink, cloth catches liquid, and everything collides with the obstacles. Smoke is the exception: it needs a simulation of its own for now. See [Smoke and water in one scene](smoke.md#smoke-and-water-in-one-scene).

Everything is in metres, seconds, and kilograms. Densities are in kg/m³ (water is 1000), gravity in m/s², and wind in m/s. Real-world sizes give real-world motion: a 5 m tank sloshes more slowly than a 50 cm one.

## Add everything before the first step

The first `step()` builds the simulation from everything you added, and nothing can be added after it. Adding later throws, for example `Simulation: addFluid must happen before the first step() or start()`.

You can still change settings such as viscosity, softness, wind, and smoke heat at any time, through the object each `add` method returns. The guide pages list these live settings. Gravity is on the simulation itself, as [`sim.gravity`](#gravity).

To add or remove things in a running scene, throw the simulation away and build a new one. Make the swap between steps, not while `await sim.step()` is still running. A click handler can fire in the middle of a step, so let it build the new simulation and have the frame loop swap it in:

```ts
function build(box: Box3): Simulation {
  const s = new Simulation({ renderer, scene, camera, container });
  s.addFluid({ box });
  // ...add everything else the scene needs
  return s;
}

let sim = build(smallBox);
let next: Simulation | undefined;

button.onclick = () => (next = build(biggerBox));

async function frame() {
  if (next) {
    sim.dispose();
    sim = next;
    next = undefined;
  }
  await sim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

## Particles and detail

The simulation represents everything as many small spheres called particles. Liquid is a heap of them, a soft body is a solid block of them, and cloth is a sheet of them. More particles give finer detail and cost more GPU time.

You choose the total with `particles`, and the simulation picks a particle size that fills everything you added with about that many:

```ts
const sim = new Simulation({ renderer, scene, camera, container, particles: 40000 });
```

The default is 20,000, which runs well on most laptops. Between 5,000 and 50,000 is a sensible range. Every particle has the same size, so a small soft body in a big pool of water gets few particles. If it looks blocky, raise `particles`.

The count is approximate. When liquid fills the space around obstacles or soft bodies, the space they take up is still counted, so you end up with fewer particles than `particles`. In a smoke scene, the budget goes to the air that fills the container.

You can set the size directly instead, as a radius in metres. `particles` is then ignored:

```ts
new Simulation({ renderer, scene, camera, container, particleRadius: 0.01 });
```

Once the simulation has started, `sim.particleCount` and `sim.particleRadius` report what it chose. To see the particles themselves, set `sim.showParticles = true`. It hides the rendered water, smoke, cloth, and soft bodies and draws the raw particles instead, which helps when something looks wrong.

## Options

| Option                        | Default                 | What it does                                                                                                                                                                                                                                                                                                                |
| ----------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer`, `scene`, `camera` | required                | The renderer from `createParticleRenderer`, the scene to draw into, and your camera.                                                                                                                                                                                                                                        |
| `container`                   | none                    | A `Box3` with walls on the bottom and sides. [Smoke](smoke.md) needs one.                                                                                                                                                                                                                                                   |
| `closed`                      | `false`                 | Put a lid on the container. Smoke always gets one.                                                                                                                                                                                                                                                                          |
| `particles`                   | 20,000                  | About how many particles to use in total.                                                                                                                                                                                                                                                                                   |
| `particleRadius`              | fits `particles`        | Radius of every particle in metres. Overrides `particles`.                                                                                                                                                                                                                                                                  |
| `gravity`                     | `(0, -9.81, 0)`         | Gravity in m/s². Smoke scenes default to `(0, -1, 0)`; see [Smoke](smoke.md).                                                                                                                                                                                                                                               |
| `substeps`                    | chosen for you, 2 to 24 | How many smaller pieces each 1/60 s step is cut into. More pieces make collisions and stiff objects (firm soft bodies, taut cloth) more reliable, and cost more. The default depends on the particle size and what's in the scene. See [Things pass through each other](troubleshooting.md#things-pass-through-each-other). |

## Stepping

The simulation moves forward in fixed steps of 1/60 s. Without an argument, `sim.step()` runs as many of those as the time since the last frame calls for, then updates everything it draws. Always `await` it before rendering.

- **Timing.** Without an argument, `step()` keeps real-time speed whatever the display's refresh rate, so it looks the same on a 60 Hz and a 120 Hz screen. On a 120 Hz screen, about every other frame runs no step at all. After a slow frame it catches up by at most four steps and drops the rest, so a stall never snowballs.
- **Exact steps.** `sim.step(1 / 60)` advances exactly 1/60 s, however long the frame took. Use this for recording video or for tests. Don't pass the frame's elapsed time: steps of changing length make the simulation less stable.
- **The frame loop.** Use `requestAnimationFrame` as shown, so each frame waits for `step()` to finish before the next one starts. A loop that doesn't wait, such as `renderer.setAnimationLoop` with an async callback, is also safe: calling `step()` while one is still running returns the same promise instead of starting another.
- **Pausing.** Stop calling `step()`, and keep rendering if you want the scene to stay on screen. If `step()` hasn't been called for more than 0.25 s, the next call restarts the clock instead of catching up, so nothing jumps forward when you resume.
- **Loading.** The first step does the setup: it fills meshes with particles and prepares `addMesh` shapes, which can take a moment for detailed meshes. `await sim.start()` does that part early. It runs on the main thread and blocks the page while it works, so show your loading screen and let the browser draw it before you call `start()`, or the screen never appears. An animated spinner freezes until it finishes.

  ```ts
  showLoadingScreen();
  // Wait two frames, so the loading screen is drawn before the setup blocks the page.
  await new Promise(requestAnimationFrame);
  await new Promise(requestAnimationFrame);
  await sim.start();
  await sim.step(1 / 60); // shaders compile on the first step...
  renderer.render(scene, camera); // ...and the first render
  hideLoadingScreen();
  ```

## Gravity

`sim.gravity` is a live `Vector3`. Change it in place at any time, before or after the simulation starts:

```ts
sim.gravity.set(0, -3, 0); // moon-ish
sim.gravity.set(4, -9.81, 0); // tip the world sideways
```

- It's always the same vector, so you can keep a reference to it. A change takes effect on the next step.
- In a smoke scene, `addSmoke` sets gravity to `(0, -1, 0)` and overwrites any change you made before calling it. To use a different value, pass `gravity` to `new Simulation`, or change `sim.gravity` after `addSmoke`.

## Cleaning up

```ts
sim.dispose();
```

`sim.dispose()` removes everything the simulation added to the scene and frees what it drew: the liquid and smoke renderers, the soft body and cloth meshes and their materials, and the `addMesh` shapes. A `material` you passed to `addCloth` is left alone, since you may use it elsewhere; dispose it yourself when you're done with it. Meshes you passed in, such as the source of a soft body, are shown again. After it, the simulation can't be used again.

It doesn't release the GPU buffers that hold the particles.

## The objects underneath

`Simulation` is built from the [low-level classes](advanced/low-level-api.md). Often you only need one setting it doesn't expose. The simulation and each handle give you the object underneath, once the simulation has started:

| Property                                        | Gives you                                                                                                                   |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `sim.particleSystem`                            | The [`ParticleSystem`](advanced/low-level-api.md#particles) holding every particle.                                         |
| `sim.loop`                                      | The [`SimLoop`](advanced/low-level-api.md#the-solver-loop). Change `loop.substeps` here while running.                      |
| `fluid.fluidSystem`                             | The [`FluidSystem`](advanced/fluid-system.md).                                                                              |
| `fluid.surface`                                 | The [`FluidSurfaceRenderer`](advanced/fluid-system.md#drawing-the-liquid-fluidsurfacerenderer) drawing it.                  |
| `fluid.mesh`                                    | The liquid surface mesh in the scene.                                                                                       |
| `softbody.mesh`                                 | The [`SoftbodyMesh`](advanced/softbody-system.md#softbodymesh) drawing the body.                                            |
| `softbody.softbodySystem`, `softbody.bodyIndex` | The [`SoftbodySystem`](advanced/softbody-system.md) holding every soft body in the simulation, and this body's index in it. |
| `cloth.clothSystem`                             | The [`ClothSystem`](advanced/cloth-system.md).                                                                              |
| `cloth.mesh`                                    | The cloth surface mesh.                                                                                                     |
| `smoke.gasSystem`                               | The [`GasSystem`](advanced/gas-system.md).                                                                                  |

For example, to give a cloth more air drag than `Simulation` sets:

```ts
const curtain = sim.addCloth({ width: 1.2, height: 1.2 });
await sim.start(); // the ClothSystem exists from here on
curtain.clothSystem.drag = 0.6;
```

For settings the handle already has, such as wind, softness, and viscosity, change them on the handle. The handle copies its wind into the `ClothSystem` every step, so a change made on `curtain.clothSystem.wind` is lost. Setting `curtain.softness` rewrites `clothSystem.bendCompliance`. And if you set `fluid.fluidSystem.viscosity` directly, `fluid.viscosity` still reports the old value.

The list of physics the loop runs is fixed once built, so you can't add your own [custom materials](advanced/custom-materials.md) to a `Simulation`. You can still run your own compute shaders on `sim.particleSystem` between steps, for example to push liquid where the user clicks.

---

Previous: [Getting started](getting-started.md) · Next: [Fluids](fluids.md)
