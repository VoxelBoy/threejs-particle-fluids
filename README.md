# Three.js Particle Fluids

GPU particle physics for [Three.js](https://threejs.org): liquids, soft bodies, cloth, and smoke, simulated with extended position-based dynamics and rendered in real time with WebGPU and TSL. Everything runs on the GPU, and every material shares one particle system, so water pushes on soft bodies, cloth drapes over meshes, and smoke rides the air.

![Wave Chamber](https://raw.githubusercontent.com/dgreenheck/threejs-particle-fluids/main/public/previews/cover.png)

## Install

```sh
npm install threejs-particle-fluids three
```

The library supports Three.js r184 and needs a browser with WebGPU. There is no WebGL fallback. TypeScript users also need `@types/three` 0.184. The quick start below uses top-level `await`, so build for ES2022 or later (in Vite, `build.target: 'es2022'`).

## Quick start

A block of water collapsing in a box:

```ts
import { Box3, Vector3 } from 'three';
import {
  FluidSurfaceRenderer,
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from 'threejs-particle-fluids';

// A WebGPURenderer with the device limits the solver needs. `scene` and
// `camera` below are an ordinary three.js Scene and PerspectiveCamera.
const renderer = await createParticleRenderer({ antialias: true });

// 1. Particles. Every particle in a simulation shares one radius.
const radius = 0.012;
const water: ParticleInit[] = [];
for (let x = -0.48; x < -0.1; x += radius * 2)
  for (let y = radius; y < 0.5; y += radius * 2)
    for (let z = -0.28; z < 0.28; z += radius * 2) water.push({ position: [x, y, z] });
const particles = new ParticleSystem(renderer, water.length, radius);
particles.uploadParticles(water);

// 2. Physics: a fluid, the walls it collides with, and the solver loop.
const fluid = new FluidSystem(particles, { viscosity: 0.02, surfaceTension: 0.1 });
const walls = new PrimitiveSet(particles);
walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
walls.addPlane(new Vector3(1, 0, 0), new Vector3(-0.5, 0, 0));
walls.addPlane(new Vector3(-1, 0, 0), new Vector3(0.5, 0, 0));
walls.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -0.3));
walls.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, 0.3));
const loop = new SimLoop(particles, { substeps: 3, materials: [fluid], colliders: [walls] });

// 3. Rendering: a ray-marched liquid surface.
const surface = new FluidSurfaceRenderer(fluid, {
  renderer,
  scene,
  camera,
  bounds: new Box3(new Vector3(-0.53, -0.02, -0.33), new Vector3(0.53, 0.8, 0.33)),
  colliders: [walls],
});
scene.add(surface.mesh);

async function frame() {
  await loop.step(1 / 60);
  await surface.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

The complete version, with a camera and lights, is [`examples/fluid.ts`](examples/fluid.ts). Run `npm run dev` and open `/examples/fluid.html`.

## How it fits together

- **`ParticleSystem`** holds every particle's position, velocity, and inverse mass in GPU buffers. Materials own ranges of it.
- **Materials** add physics to a range of particles:
  - `FluidSystem`: liquids and gases (Position Based Fluids), with optional viscosity, vorticity confinement, surface tension, and adhesion. `ViscositySolver` adds implicit viscosity for very thick liquids such as honey.
  - `SoftbodySystem`: soft and near-rigid bodies by shape matching, either per body (`'global'`) or per particle neighborhood (`'local'`, which bends and folds). Global matching respects per-particle mass, so a body with a heavy base floats upright. `voxelize` fills a mesh or distance field with particles.
  - `ClothSystem`: cloth with stretch, bending, and long-range attachment constraints, plus wind. `createClothGraph` builds it from any indexed geometry.
  - `GasSystem`: smoke tracers carried by a fluid's velocity field.
  - Anything with a `build(context)` method, for your own TSL kernels.
- **Colliders** are shapes particles can't enter: `PrimitiveSet` (planes, spheres, boxes, capsules that can follow `Object3D`s) and `SDFCollider` (any mesh, through a baked distance field from `bakeMeshToSdf`).
- **`SimLoop`** advances everything with substepped XPBD. Pass `contact: true` to make particles collide with each other, which soft bodies and cloth need to touch one another.
- **Renderers** read the particle buffers directly on the GPU: `FluidSurfaceRenderer` (ray-marched liquid with refraction, absorption, and reflections), `SoftbodyMesh` (skins any mesh to a soft body), `createClothSurface` (smooth bicubic cloth), `GasVolumeRenderer` and `GasSpriteRenderer` (smoke), and `createParticleMesh` (raw particles).

Materials interact through the shared particle buffer. For example, a soft body floats because the fluid treats its surface particles as a boundary:

```ts
const ducks = new SoftbodySystem(particles, { bodies });
const water = new FluidSystem(particles, { range: { start: 0, count: waterCount } });
for (let i = 0; i < bodies.length; i++) water.addBoundary(ducks.surfaceRange(i));
const loop = new SimLoop(particles, { materials: [water, ducks], contact: true });
```

The API is young and may change before 1.0. The solvers are built for interactive visuals, not engineering analysis.

## Demo

The repository includes a demo of thirteen presets. Install Node.js 22.12 or newer, then:

```sh
npm ci
npm run dev
```

| Preset                | Shows                                                                  |
| --------------------- | ---------------------------------------------------------------------- |
| **Wave Chamber**      | A sealed tank turning end over end, driving water through its walls    |
| **Water Drop**        | A falling drop splashing into a shallow pool                           |
| **Liquid Marble**     | Inward gravity pulling a drop back together after a click bursts it    |
| **Honey Bunny**       | A circling nozzle drizzles viscous honey over the Stanford bunny       |
| **Buoyancy**          | Textured rubber ducks floating or sinking as their density changes     |
| **Soft Body Squeeze** | 20 textured CC0 forms squeezed between closing plates                  |
| **Bunny Lineup**      | Five jelly bunnies dropped side by side, from firm to very soft        |
| **Banana Blender**    | Soft bananas as dense as the liquid, swirled by a tall paddle          |
| **Velvet Curtain**    | Soft red velvet displaced by a moving chrome sphere                    |
| **Velvet Drape**      | A square of red velvet dropped onto the Stanford bunny                 |
| **Tarp Runoff**       | Red liquid pouring onto a sloped canvas tarp and spilling off its edge |
| **Vortex Plume**      | Lit volumetric smoke with filtered density and correct scene occlusion |
| **Smoke Bubbles**     | Smoke-filled bubbles rise through water and burst into drifting puffs  |

Each preset has live controls; controls marked **↻** restart it. **Space** pauses, **R** restarts, dragging orbits, and clicking the liquid splashes it. The **Particles** menu sets the particle budget from 5,000 to 50,000, and **Surface / Particles** shows the particles under the rendering. The overlay reports frame rate, frame time, particle count, and solver settings.

The preset sources in [`demo/presets/`](demo/presets) are larger examples of the API.

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
  core/       Particles, the solver loop, contacts, colliders, neighbor grid
  fluids/     Fluid solver and liquid surface renderer
  softbody/   Shape matching, voxelization, mesh skinning
  cloth/      Cloth constraints, wind, and surface rendering
  gas/        Smoke tracers and renderers
  sdf/        Mesh-to-distance-field baking
  render/     Particle debug rendering
demo/         The preset gallery
examples/     Minimal examples
tests/        Numerical GPU tests and benchmarks
```

## License

[MIT](LICENSE). The bundled Kenney and Poly Haven models are CC0; the Stanford bunny is courtesy of the Stanford 3D Scanning Repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for sources and modifications.
