# Three.js Particle Fluids

Water, smoke, soft bodies, and cloth for [Three.js](https://threejs.org), simulated and drawn in real time on the GPU with WebGPU. Say what you want and where, then call `sim.step()` each frame. Water pushes soft bodies around, light ones float and heavy ones sink, cloth drapes over meshes and catches liquid, and smoke curls up from a heated vent.

![Wave Chamber](https://raw.githubusercontent.com/dgreenheck/threejs-particle-fluids/main/public/previews/cover.png)

**[Live demo](https://dgreenheck.github.io/threejs-particle-fluids/)** · **[Documentation](docs/README.md)**

## Install

```sh
npm install threejs-particle-fluids three
```

The library supports Three.js r184 and needs a browser with WebGPU. There is no WebGL fallback. TypeScript users also need `@types/three` 0.184. The quick start uses top-level `await`, so build for ES2022 or later (in Vite, `build.target: 'es2022'`).

## Quick start

A block of water falling in a tank:

```ts
import { Box3, Vector3 } from 'three';
import { Simulation, createParticleRenderer } from 'threejs-particle-fluids';

// `scene` and `camera`: an ordinary Scene and PerspectiveCamera. Put the camera about
// 2 m back, and set scene.environment first, or the water looks flat.
const renderer = await createParticleRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.append(renderer.domElement);

const sim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(0.5, 0.8, 0.3)),
});
sim.addFluid({ box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(-0.1, 0.5, 0.3)) });

async function frame() {
  await sim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

For a complete file you can paste, with the scene, lighting, and camera, see [Getting started](docs/getting-started.md#water-in-a-tank). In this repository, [`examples/fluid.ts`](examples/fluid.ts) runs the same scene: run `npm run dev` and open `/examples/fluid.html`.

Add everything before the first `sim.step()`. To put more in the same tank, add these lines after `addFluid` and before the frame loop starts:

```ts
sim.addSoftbody({ mesh: duck, density: 400 }); // `duck` is any closed mesh placed in the tank; it floats
sim.addSphere({ radius: 0.1, follow: ball }); // a solid ball that moves with your `ball` mesh
```

Liquids, soft bodies, and cloth in one simulation push on each other with no extra setup.

Gas and liquid can't be simulated together, so smoke gets a simulation of its own, with a container beside the water tank for the air to fill. Replace the frame loop above with one that steps both:

```ts
const smokeSim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(1, 0, -0.5), new Vector3(2, 1.9, 0.5)), // beside the water tank
});
smokeSim.addSmoke(); // rises from the middle of its floor

async function frame() {
  await sim.step();
  await smokeSim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
```

See [Smoke and water in one scene](docs/smoke.md#smoke-and-water-in-one-scene).

## Documentation

- [Getting started](docs/getting-started.md): the full first scene.
- [The simulation](docs/simulation.md): the container, the particle budget, stepping, and cleaning up.
- One page per thing you can add: [fluids](docs/fluids.md), [obstacles](docs/obstacles.md), [soft bodies](docs/soft-bodies.md), [cloth](docs/cloth.md), and [smoke](docs/smoke.md).
- [Troubleshooting](docs/troubleshooting.md): error messages, speed, and things that look wrong.
- [Advanced](docs/README.md#advanced): the low-level classes `Simulation` is built on, for emitters, custom forces, and full control.

The API is young and may change before 1.0. The physics is built for interactive visuals, not engineering analysis.

## Demo

The [live demo](https://dgreenheck.github.io/threejs-particle-fluids/) runs thirteen presets in any browser with WebGPU. To run it locally, install Node.js 22.12 or newer, then:

```sh
npm ci
npm run dev
```

| Preset                | Shows                                                                            |
| --------------------- | -------------------------------------------------------------------------------- |
| **Wave Chamber**      | A sealed tank turning end over end, driving water through its walls              |
| **Water Drop**        | A falling drop splashing into a shallow pool                                     |
| **Dam Break**         | A column of water collapses and floods into a large Stanford bunny               |
| **Liquid Marble**     | Inward gravity pulling a drop back together after a click bursts it              |
| **Honey Bunny**       | A circling nozzle drizzles viscous honey over the Stanford bunny                 |
| **Buoyancy**          | Textured rubber ducks floating or sinking as their density changes               |
| **Soft Body Squeeze** | 20 textured CC0 forms squeezed between closing plates                            |
| **Bunny Lineup**      | Five jelly bunnies dropped side by side, from firm to very soft                  |
| **Banana Blender**    | Soft bananas as dense as the liquid, swirled by a tall paddle                    |
| **Velvet Curtain**    | Soft red velvet displaced by a moving chrome sphere                              |
| **Velvet Drape**      | A square of red velvet dropped onto the Stanford bunny                           |
| **Tarp Runoff**       | Red liquid pouring onto a sloped canvas tarp and spilling off its edge           |
| **Vortex Plume**      | A heated vent drives a buoyant, swirling plume that carries lit volumetric smoke |

Each preset has live controls; controls marked **↻** restart it. **Space** pauses, **R** restarts, dragging orbits, and clicking the liquid splashes it. The **Particles** menu sets the particle budget from 5,000 to 50,000, and **Surface / Particles** shows the particles under the rendering. The overlay reports frame rate, frame time, particle count, and solver settings.

The preset sources in [`demo/presets/`](demo/presets) are larger examples built with the [low-level API](docs/advanced/low-level-api.md).

## Development

| Command                  | Purpose                                                       |
| ------------------------ | ------------------------------------------------------------- |
| `npm run dev`            | Start the demo and examples                                   |
| `npm run build`          | Build the library into `dist/` and the demo into `dist-demo/` |
| `npm run typecheck`      | Type-check the library, demo, examples, tests, and scripts    |
| `npm run lint`           | Lint                                                          |
| `npm run format:check`   | Check formatting                                              |
| `npm test`               | Run CPU tests                                                 |
| `npm run test:gpu`       | Run WebGPU tests in installed Google Chrome                   |
| `npm run test:demo`      | Run browser checks of the demo                                |
| `npm run test:perf`      | Run GPU benchmarks and write a local report                   |
| `npm run assets:elastic` | Rebuild the soft-body meshes and particle templates           |
| `npm run assets:honey`   | Rebuild the Stanford bunny mesh and distance field            |

GPU tests and benchmarks need a local GPU; the runner passes Chrome's `--enable-unsafe-webgpu` flag. Benchmark results depend on the browser, adapter, and driver.

```text
src/
  simulation/ The Simulation class and its handles
  core/       Particles, the solver loop, contacts, colliders, neighbor grid
  fluids/     Fluid solver and liquid surface renderer
  softbody/   Shape matching, voxelization, mesh skinning
  cloth/      Cloth constraints, wind, and surface rendering
  gas/        Smoke tracers and renderers
  sdf/        Mesh-to-distance-field baking
  render/     Particle debug rendering
demo/         The preset gallery
examples/     Small examples of the API
tests/        Numerical GPU tests and benchmarks
```

## License

[MIT](LICENSE). The bundled Kenney and Poly Haven models are CC0; the Stanford bunny is courtesy of the Stanford 3D Scanning Repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for sources and modifications.
