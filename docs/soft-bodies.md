[Docs](README.md) › Soft bodies

# Soft bodies

`sim.addSoftbody` turns a closed mesh into a soft body: something that squashes, wobbles, and bends, then springs back to its shape. It keeps its look, because your mesh is drawn bending along with it.

```ts
const ball = new Mesh(
  new SphereGeometry(0.12, 32, 24),
  new MeshStandardMaterial({ color: 0xf2c14e }),
);
ball.position.set(0, 0.65, 0);
scene.add(ball);

const jelly = sim.addSoftbody({ mesh: ball, softness: 0.4 });
```

Place the mesh where the body should start before you call `addSoftbody`. The body takes the mesh's shape, position, rotation, and scale at that moment. When the simulation starts, your mesh is hidden and a deforming copy takes its place in the scene. `sim.dispose()` shows your mesh again.

## Options

| Option     | Default  | What it does                                                       |
| ---------- | -------- | ------------------------------------------------------------------ |
| `mesh`     | required | A closed mesh, where it sits in the world.                         |
| `softness` | 0.3      | 0 is firm rubber, 1 is loose jelly.                                |
| `density`  | 500      | Density in kg/m³. Water is 1000, so lower floats and higher sinks. |

## Live settings

`addSoftbody` returns a `Softbody`:

```ts
jelly.softness = 0.8; // go floppy
jelly.mesh; // the deforming mesh in the scene, once the simulation has started
jelly.source; // the mesh you passed in
```

- `density` is fixed once added.
- `jelly.mesh` stays at the origin. Its vertices move, not its transform, so `jelly.mesh.position` won't tell you where the body is. To find the body, read its particles back from the GPU and average them:

  ```ts
  const { positions } = await sim.particles.readback(); // x, y, z, w for every particle
  const { start, count } = jelly.mesh.softbody.particleRange(jelly.mesh.bodyIndex);
  const center = new Vector3();
  const point = new Vector3();
  for (let i = start; i < start + count; i++) center.add(point.fromArray(positions, 4 * i));
  center.divideScalar(count);
  ```

  A readback makes the page wait for the GPU, so do it now and then, not every frame. See [Reading the buffers yourself](advanced/low-level-api.md#reading-the-buffers-yourself).

- The source mesh's material is copied once, when the simulation starts. To recolor the body while it runs, change `jelly.mesh.material`, for example `jelly.mesh.material.color.set(0x44aa88)`.
- For a setting not listed here, reach the object underneath. See [The objects underneath](simulation.md#the-objects-underneath).

The softness scale is the same for every body and every particle count, so a body with `softness: 0.5` feels about the same whether the simulation uses 10,000 or 50,000 particles.

## Floating and sinking

Put a soft body in a liquid and it floats or sinks by its density, like a real object. Water has a density of 1000.

```ts
const ball = new Mesh(
  new SphereGeometry(0.12, 32, 24),
  new MeshStandardMaterial({ color: 0xf2c14e }),
);
ball.position.set(-0.2, 0.65, 0); // above the water, so it drops in
const cube = new Mesh(
  new BoxGeometry(0.16, 0.16, 0.16),
  new MeshStandardMaterial({ color: 0xd9534f }),
);
cube.position.set(0.2, 0.7, 0);
scene.add(ball, cube);

const sim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.6, 0, -0.4), new Vector3(0.6, 1, 0.4)),
  particles: 30000,
});
sim.addFluid({ box: new Box3(new Vector3(-0.6, 0, -0.4), new Vector3(0.6, 0.35, 0.4)) });
sim.addSoftbody({ mesh: ball, density: 400, softness: 0.4 }); // floats
sim.addSoftbody({ mesh: cube, density: 2000, softness: 0.1 }); // sinks
```

[`examples/floating.ts`](../examples/floating.ts) is this scene. Liquid pushes on bodies, bodies push on liquid and make waves, and the liquid wets their surface. If a body starts inside the water box, the water is cleared around it.

## Collisions

Soft bodies collide with each other, with [cloth](cloth.md), with [obstacles](obstacles.md), and with the container walls. You don't need to set anything up.

## The mesh

- **It must be closed**, with no holes, so there's a clear inside to fill.
- **Its material carries over.** If the mesh's material is a `MeshStandardMaterial` or `MeshPhysicalMaterial` from `three`, the deforming copy uses its color, roughness, metalness, and texture maps. The maps need the geometry to have UVs. Anything else gets a plain orange-brown. That includes the node materials from `three/webgpu`, such as `MeshStandardNodeMaterial`, and meshes with an array of materials.
- **Loaded models.** A glTF file loads as a group. Pass the mesh inside it. The position, rotation, and scale of its parents count.

  ```ts
  import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

  const gltf = await new GLTFLoader().loadAsync('duck.glb');
  const duck = gltf.scene.getObjectByProperty('isMesh', true) as Mesh;
  sim.addSoftbody({ mesh: duck });
  ```

- **Detail follows the particle size.** Small parts, such as thin ears or fingers, need several particles across to bend nicely. A thin part that ends up disconnected from the rest is dropped. If a body looks blocky or loses a part, raise the simulation's `particles` budget. See [Particles and detail](simulation.md#particles-and-detail).
- **Too small to fill** throws `addSoftbody: the mesh is too small for the particle size` on the first step. Make the mesh bigger or the particles smaller.
- Each call makes one body. Call `addSoftbody` again for every body, even from the same mesh; move the mesh between calls.

---

Previous: [Obstacles](obstacles.md) · Next: [Cloth](cloth.md)
