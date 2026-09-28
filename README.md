# Three.js Particle Fluids

GPU particle physics for [three.js](https://threejs.org): liquids, soft bodies, cloth, and smoke, simulated and rendered with WebGPU.

![Wave Chamber](https://raw.githubusercontent.com/dgreenheck/threejs-particle-fluids/main/public/previews/cover.png)

**[Live demo](https://dgreenheck.github.io/threejs-particle-fluids/)** · **[Documentation](docs/README.md)** · **[API reference](docs/README.md#api-reference)**

## Install

```sh
npm install threejs-particle-fluids three
```

Requires three.js r184 and a browser with WebGPU (no WebGL fallback). See [Requirements](docs/guide.md#requirements).

## Quick start

```ts
import { Box3, Vector3 } from 'three';
import { Simulation, createParticleRenderer } from 'threejs-particle-fluids';

const renderer = await createParticleRenderer({ antialias: true });

const sim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(0.5, 0.8, 0.3)),
  particles: 5000, // total budget; sets the cost of each step
});
sim.addFluid({ box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(-0.1, 0.5, 0.3)) });

async function frame() {
  await sim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

Full scene with lighting and camera: [Guide › First scene](docs/guide.md#first-scene).

## Documentation

| Page                                          | Contents                                                |
| --------------------------------------------- | ------------------------------------------------------- |
| [Guide](docs/guide.md)                        | Setup, particle sizing, frame loop, combining materials |
| [API reference](docs/README.md#api-reference) | Every exported class, option, property, and method      |
| [Limitations](docs/limitations.md)            | Platform, performance, accuracy, and API constraints    |

## Demo

[Live demo](https://dgreenheck.github.io/threejs-particle-fluids/). To run locally (Node.js 22.12+):

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

| Control        | Action                           |
| -------------- | -------------------------------- |
| Space          | Pause                            |
| R              | Restart                          |
| Drag           | Orbit                            |
| Click liquid   | Splash                           |
| Particles menu | Particle count (5,000 to 50,000) |
| ↻ controls     | Restart the preset when changed  |

Preset sources: [`demo/presets/`](demo/presets), built with the [low-level API](docs/api/core.md).

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
