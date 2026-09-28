[Docs](README.md) › Guide

# Guide

Setting up a scene with [`Simulation`](api/simulation.md).

1. [Requirements](#requirements)
2. [First scene](#first-scene)
3. [Choosing particle size and count](#choosing-particle-size-and-count)
4. [Frame loop](#frame-loop)
5. [Lighting](#lighting)
6. [Combining materials](#combining-materials)
7. [Smoke next to liquid](#smoke-next-to-liquid)
8. [Loading](#loading)
9. [Debugging](#debugging)
10. [Low-level API](#low-level-api)

## Requirements

| Requirement    | Version / note                                                        |
| -------------- | --------------------------------------------------------------------- |
| `three`        | r184 (`0.184.x`)                                                      |
| `@types/three` | `0.184.x` (TypeScript only)                                           |
| Browser        | WebGPU enabled. No WebGL fallback. Served from `localhost` or HTTPS.  |
| Build target   | ES2022+ if you use top-level `await` (Vite: `build.target: 'es2022'`) |

```sh
npm install threejs-particle-fluids three
```

## First scene

A block of water collapsing in a 1 × 0.8 × 0.6 m tank.

```ts
import { Box3, Box3Helper, DirectionalLight, PerspectiveCamera, Scene, Vector3 } from 'three';
import { PMREMGenerator } from 'three/webgpu';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Simulation, createParticleRenderer } from 'threejs-particle-fluids';

const renderer = await createParticleRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.append(renderer.domElement);

const scene = new Scene();
scene.environment = new PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const sun = new DirectionalLight(0xffffff, 2);
sun.position.set(1, 3, 2);
scene.add(sun);

const camera = new PerspectiveCamera(40, innerWidth / innerHeight, 0.01, 50);
camera.position.set(1.6, 1.4, 2.2);
camera.lookAt(0, 0.3, 0);

const container = new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(0.5, 0.8, 0.3));
scene.add(new Box3Helper(container));

const sim = new Simulation({
  renderer,
  scene,
  camera,
  container,
  particleRadius: 0.014,
  maxParticles: 5000,
});
sim.addFluid({ box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(-0.1, 0.5, 0.3)) });

async function frame() {
  await sim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

The pattern:

1. `new Simulation({ ... })`
2. `add*` calls: [`addFluid`](api/fluid.md), [`addSoftbody`](api/softbody.md), [`addCloth`](api/cloth.md), [`addSmoke`](api/smoke.md), and [obstacles](api/simulation.md#addflooroptions).
3. `await sim.step()` then `renderer.render()` every frame.

The simulation adds its own meshes to `scene`. Runnable version: [`examples/fluid.ts`](../examples/fluid.ts).

## Choosing particle size and count

Both are required. Nothing is chosen for you.

- **`particleRadius`** sets resolution. Particles sit `2r` apart. Thin features (soft body limbs, splashes, cloth folds) need several particles across.
- **`maxParticles`** is a hard cap. If the scene needs more particles than this at your radius, `start()` throws and the message says how many it needed.

Estimate the count before picking values:

| Content        | Particles                         |
| -------------- | --------------------------------- |
| Box of liquid  | `volume / (2r)³`                  |
| Soft body mesh | `meshVolume / (2r)³`              |
| `w × h` cloth  | `(w / 2.2r + 1) × (h / 2.2r + 1)` |
| Smoke          | `containerVolume / (2r)³`         |

| Scene                         | `particleRadius` | Particles |
| ----------------------------- | ---------------- | --------- |
| 0.4 × 0.5 × 0.6 m water block | 0.014            | ~4,800    |
| 1.2 × 1.2 m cloth             | 0.01             | ~3,100    |
| 1 × 1.9 × 1 m smoke container | 0.035            | ~4,400    |

Halving the radius multiplies volume particle counts by 8.

Frame cost grows with particle count and substeps. Liquid touching moving soft bodies or cloth is the most expensive combination. There is no count that runs well everywhere. Start low (a few thousand), measure on the slowest hardware you target, then raise it. Read the actual count from `sim.particleCount` after `start()`.

## Frame loop

- Call `await sim.step()` once per frame, before `renderer.render()`.
- With no argument, `step()` runs fixed 1/60 s steps to keep up with wall-clock time (at most 4 per call). Behavior is the same on 60 Hz and 120 Hz displays.
- `step(1 / 60)` advances exactly one step. Use it for recording or tests. Don't pass the frame delta.
- To pause, stop calling `step()`. After 250 ms without a call, the clock restarts instead of catching up.
- `renderer.setAnimationLoop` with an async callback is safe: overlapping calls share one step.

## Lighting

- Liquid reflects `scene.environment`. Set it **before** the first `step()`; later changes aren't picked up.
- Liquid takes its specular highlight from one directional light: a shadow-casting one if present, otherwise the first found.
- Soft bodies and cloth use ordinary lit materials and need scene lights.

## Combining materials

Fluids, soft bodies, cloth, and obstacles in one `Simulation` interact with no extra setup.

```ts
const sim = new Simulation({
  renderer,
  scene,
  camera,
  container,
  particleRadius: 0.016,
  maxParticles: 6000,
});
sim.addFluid({ box: water });
sim.addSoftbody({ mesh: ball, density: 400 }); // floats
sim.addSoftbody({ mesh: cube, density: 2000 }); // sinks
sim.addSphere({ radius: 0.1, follow: paddle }); // moving obstacle
```

Soft bodies and cloth must start inside the container and clear of obstacles and each other. Liquid is placed around them. See [`examples/floating.ts`](../examples/floating.ts) and [`examples/cloth.ts`](../examples/cloth.ts).

## Smoke next to liquid

Smoke can't share a `Simulation` with liquid, soft bodies, or cloth. Use two, and step both:

```ts
const smokeSim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(1, 0, -0.5), new Vector3(2, 1.9, 0.5)),
  particleRadius: 0.035,
  maxParticles: 5000,
});
smokeSim.addSmoke();

async function frame() {
  await sim.step();
  await smokeSim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
```

## Loading

The first `step()` (or `start()`) voxelizes meshes and bakes `addMesh` distance fields on the main thread. Shaders compile on the first step and first render. To show a loading screen:

```ts
showLoadingScreen();
await new Promise(requestAnimationFrame); // let the browser draw it
await new Promise(requestAnimationFrame);
await sim.start();
await sim.step(1 / 60);
renderer.render(scene, camera);
hideLoadingScreen();
```

## Debugging

```ts
sim.showParticles = true;
```

Hides the rendered surfaces and draws raw particles. Use it to check mesh filling, starting positions, and tunneling.

| Symptom                                   | Fix                                                                                                                  |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Nothing visible                           | Await `step()` before rendering. Check the camera (units are metres). Add a `container` or `addFloor()`.             |
| Liquid looks flat                         | Set `scene.environment` before the first step.                                                                       |
| Liquid disappears as it spreads           | Give a `container` covering everywhere it flows.                                                                     |
| Objects pass through obstacles            | Thicker or slower obstacles, or raise `substeps`.                                                                    |
| Liquid is springy                         | Raise `substeps`.                                                                                                    |
| Soft bodies are blocky or lose thin parts | Lower `particleRadius` (and raise `maxParticles`).                                                                   |
| Cloth leaks liquid                        | Raise cloth `weight`.                                                                                                |
| Slow                                      | Raise `particleRadius`, lower `substeps`, cap pixel ratio (`renderer.setPixelRatio(Math.min(devicePixelRatio, 2))`). |

All error messages are listed in [Simulation › Errors](api/simulation.md#errors).

## Low-level API

`Simulation` is built from exported classes. Use them for emitters, custom forces, custom renderers, rigid bodies, or settings `Simulation` doesn't expose.

```ts
const particles = new ParticleSystem(renderer, init.length, radius);
particles.uploadParticles(init);
const fluid = new FluidSystem(particles, { viscosity: 0.02 });
const walls = new PrimitiveSet(particles);
walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
const loop = new SimLoop(particles, { substeps: 3, materials: [fluid], colliders: [walls] });
await loop.step(1 / 60);
```

Start at [Core](api/core.md). Full example: [`examples/low-level.ts`](../examples/low-level.ts). The demo presets in [`demo/presets/`](../demo/presets) are larger low-level scenes.

From a running `Simulation`, the underlying objects are available after `start()`: `sim.particleSystem`, `sim.loop`, `fluid.fluidSystem`, `fluid.surface`, `softbody.softbodySystem`, `cloth.clothSystem`, `smoke.gasSystem`.
