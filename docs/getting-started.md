[Docs](README.md) › Getting started

# Getting started

This page puts a block of water in a tank and lets it fall. The next page, [The simulation](simulation.md), explains the rules behind it.

## Requirements

- Three.js r184 (`three@0.184`). TypeScript users also need `@types/three@0.184`.
- A browser with WebGPU. There is no WebGL fallback. Pages must be served from `localhost` or HTTPS.
- A build target of ES2022 or later if you use top-level `await`, as these examples do. In Vite, set `build.target: 'es2022'`.

```sh
npm install threejs-particle-fluids three
```

## Water in a tank

```ts
import { Box3, Box3Helper, DirectionalLight, PerspectiveCamera, Scene, Vector3 } from 'three';
import { PMREMGenerator } from 'three/webgpu';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Simulation, createParticleRenderer } from 'threejs-particle-fluids';

// A WebGPURenderer set up for the simulation.
const renderer = await createParticleRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.append(renderer.domElement);

// An ordinary three.js scene, light, and camera.
const scene = new Scene();
scene.environment = new PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const sun = new DirectionalLight(0xffffff, 2);
sun.position.set(1, 3, 2);
scene.add(sun);
const camera = new PerspectiveCamera(40, innerWidth / innerHeight, 0.01, 50);
camera.position.set(1.6, 1.4, 2.2);
camera.lookAt(0, 0.3, 0);

// A tank 1 m wide, 0.8 m tall, and 0.6 m deep.
const container = new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(0.5, 0.8, 0.3));
scene.add(new Box3Helper(container)); // draw the tank's outline
const sim = new Simulation({ renderer, scene, camera, container });

// Fill the left end of the tank with water.
sim.addFluid({
  box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(-0.1, 0.5, 0.3)),
});

async function frame() {
  await sim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

That's the whole pattern:

1. Create a `Simulation` with your renderer, scene, and camera.
2. Add what you want: `addFluid`, `addSoftbody`, `addCloth`, `addSmoke`, and obstacles such as `addFloor` or `addSphere`.
3. Call `await sim.step()` once per frame, before `renderer.render`.

The simulation adds what it draws to your scene. The water above appears as a mesh in `scene`, and you never create it yourself.

[`examples/fluid.ts`](../examples/fluid.ts) is this scene with orbit controls and softer lighting. In this repository, run `npm run dev` and open `/examples/fluid.html`.

## The renderer

`createParticleRenderer` creates a `WebGPURenderer` that asks the GPU for the larger limits the simulation's shaders need, such as more storage buffers per shader, and waits for it to start. It takes the same options as `WebGPURenderer` and throws if the browser has no WebGPU. Use it like any other three.js renderer: size it and add its canvas to the page yourself.

## Lighting

Liquids reflect the scene's environment map (`scene.environment`). Without one they look flat, which is why the example sets one.

Set `scene.environment` **before the first `sim.step()`**. The liquid reads it once, when the simulation starts, so an environment map you assign later (for example, once an HDR file finishes loading) won't show up in the water.

Liquids also take their highlight from one directional light: a shadow-casting one if the scene has one, otherwise the first one found. If the scene has no directional light when the simulation starts, the liquid picks one up when you add it. Otherwise it keeps the light it found first.

Soft bodies and cloth are ordinary lit meshes, so they need lights like anything else.

## Next

- [The simulation](simulation.md): the container, the particle budget, stepping, gravity, and cleaning up.
- [Fluids](fluids.md): thicker liquids, colors, and filling a mesh.

---

Previous: [Docs home](README.md) · Next: [The simulation](simulation.md)
