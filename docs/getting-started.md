[Docs](README.md) › Getting started

# Getting started

## Requirements

- Three.js r184 (`three@0.184`). TypeScript users also need `@types/three@0.184`.
- A browser with WebGPU. There is no WebGL fallback. Pages must be served from `localhost` or HTTPS.
- A build target of ES2022 or later if you use top-level `await`, as these examples do. In Vite, set `build.target: 'es2022'`.

```sh
npm install threejs-particle-fluids three
```

## Create the renderer

The solver needs a few WebGPU device limits raised above the defaults. `createParticleRenderer` creates a `WebGPURenderer` with those limits and waits for it to initialize. It takes the same options as `WebGPURenderer` and throws if WebGPU isn't available.

```ts
import { createParticleRenderer } from 'threejs-particle-fluids';

const renderer = await createParticleRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.append(renderer.domElement);
```

Use it like any other three.js renderer: build a `Scene` and a `PerspectiveCamera`, add lights, and call `renderer.render(scene, camera)`. The liquid renderer reads the scene's environment map and its main directional light, so set `scene.environment` if you want reflections.

## A first simulation

Every simulation has the same three parts:

1. A **`ParticleSystem`**: GPU buffers for every particle. All particles share one radius.
2. **Materials** that give the particles behavior, and **colliders** they can't pass through, run by a **`SimLoop`**.
3. **Renderers** that draw the particles as a liquid surface, a cloth, a skinned mesh, or smoke.

This block of water collapses in a box:

```ts
import { Box3, Vector3 } from 'three';
import {
  FluidSurfaceRenderer,
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  type ParticleInit,
} from 'threejs-particle-fluids';

// 1. Particles on a grid, spaced one diameter apart.
const radius = 0.012;
const water: ParticleInit[] = [];
for (let x = -0.48; x < -0.1; x += radius * 2)
  for (let y = radius; y < 0.5; y += radius * 2)
    for (let z = -0.28; z < 0.28; z += radius * 2) water.push({ position: [x, y, z] });
const particles = new ParticleSystem(renderer, water.length, radius);
particles.uploadParticles(water);

// 2. A fluid, the walls of a 1 m × 0.6 m box, and the solver.
const fluid = new FluidSystem(particles, { viscosity: 0.02, surfaceTension: 0.1 });
const walls = new PrimitiveSet(particles);
walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0)); // floor
walls.addPlane(new Vector3(1, 0, 0), new Vector3(-0.5, 0, 0));
walls.addPlane(new Vector3(-1, 0, 0), new Vector3(0.5, 0, 0));
walls.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -0.3));
walls.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, 0.3));
const loop = new SimLoop(particles, { substeps: 3, materials: [fluid], colliders: [walls] });

// 3. A ray-marched liquid surface.
const surface = new FluidSurfaceRenderer(fluid, {
  renderer,
  scene,
  camera,
  bounds: new Box3(new Vector3(-0.53, -0.02, -0.33), new Vector3(0.53, 0.8, 0.33)),
  colliders: [walls],
});
scene.add(surface.mesh);
```

[`examples/fluid.ts`](../examples/fluid.ts) is the complete version with a camera, lights, and orbit controls. In this repository, run `npm run dev` and open `/examples/fluid.html`.

## The frame loop

Each frame, step the simulation, let the renderers rebuild what they draw from the particles, then render:

```ts
async function frame() {
  await loop.step(1 / 60);
  await surface.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

`loop.step(dt)` submits the whole step to the GPU at once and resolves when it has been queued. Always `await` it before rendering.

### Run at a fixed timestep

The solver is tuned for a fixed step length. On a 120 Hz display, calling `step(1 / 60)` every frame runs the simulation at double speed. `FrameStepper` takes as many fixed steps as the elapsed time allows, and drops time when the GPU falls behind instead of spiraling:

```ts
import { FrameStepper } from 'threejs-particle-fluids';

const stepper = new FrameStepper({ fixedDt: 1 / 60, maxStepsPerFrame: 4 });

async function frame(now: number) {
  await stepper.pump(now, (dt) => loop.step(dt));
  await surface.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

Call `stepper.reset()` after pausing, so the next frame doesn't try to catch up.

## Cleaning up

`SimLoop.dispose()` releases only the loop's own GPU helpers. Dispose the renderers, the colliders, the loop, and the particle system separately when you tear a scene down:

```ts
surface.dispose();
walls.dispose();
loop.dispose();
particles.dispose();
```

---

Next: [Core concepts](concepts.md)
