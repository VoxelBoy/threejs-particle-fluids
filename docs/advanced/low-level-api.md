[Docs](../README.md) › [Advanced](../README.md#advanced) › The low-level API

# The low-level API

`Simulation` is built from classes the library also exports. Use them directly when you need something `Simulation` doesn't offer: liquid poured from a nozzle, your own forces, your own renderers, or exact control over every solver setting.

On these pages, a _material_ is a kind of physics that runs on a group of particles, such as a liquid or a cloth. It has nothing to do with a three.js `Material`. The _solver_ is the code that moves the particles forward each step.

The cost is that you do the setup `Simulation` does for you: size the particles, lay them out, connect the materials to each other, pick solver settings, and create the renderers.

Only need one setting? See [The objects underneath](../simulation.md#the-objects-underneath).

## The same scene by hand

This is the water tank from [Getting started](../getting-started.md), built without `Simulation`. [`examples/low-level.ts`](../../examples/low-level.ts) is the complete file.

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

const renderer = await createParticleRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.append(renderer.domElement);
// `scene` and `camera`: as in Getting started.

// 1. Particles on a grid spaced one diameter apart.
const radius = 0.012;
const water: ParticleInit[] = [];
for (let x = -0.48; x < -0.1; x += radius * 2)
  for (let y = radius; y < 0.5; y += radius * 2)
    for (let z = -0.28; z < 0.28; z += radius * 2) water.push({ position: [x, y, z] });
const particles = new ParticleSystem(renderer, water.length, radius);
particles.uploadParticles(water);

// 2. Physics: a fluid, the walls of a 1 m × 0.6 m tank, and the solver loop.
const fluid = new FluidSystem(particles, { viscosity: 0.02, surfaceTension: 0.1, vorticity: 0.02 });
const walls = new PrimitiveSet(particles);
walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0)); // floor
walls.addPlane(new Vector3(1, 0, 0), new Vector3(-0.5, 0, 0));
walls.addPlane(new Vector3(-1, 0, 0), new Vector3(0.5, 0, 0));
walls.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -0.3));
walls.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, 0.3));
const loop = new SimLoop(particles, { substeps: 3, materials: [fluid], colliders: [walls] });

// 3. Rendering: the liquid surface.
const surface = new FluidSurfaceRenderer(fluid, {
  renderer,
  scene,
  camera,
  bounds: new Box3(new Vector3(-0.53, -0.02, -0.33), new Vector3(0.53, 0.8, 0.33)),
  colliders: [walls],
  appearance: { color: 0x3a9fcf },
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

Every hand-built scene has these three parts: a `ParticleSystem`, materials and colliders run by a `SimLoop`, and renderers.

## Particles

A `ParticleSystem` holds every particle in GPU buffers: positions, velocities, inverse masses, collision groups, and a few fields some materials use. Each particle lives in a numbered _slot_, from 0 up to `capacity − 1`. Liquids, soft bodies, and cloth are all particles in the same buffers, which is what lets them push on each other.

```ts
const particles = new ParticleSystem(renderer, capacity, particleRadius);
```

- **`capacity`** is fixed when the system is created. Size it for every material you'll add.
- **`particleRadius`** is shared by every particle. One radius lets a single grid find every particle's neighbors. Pick it for the finest detail you need, then size your materials around it.

### Uploading particles

`uploadParticles(data, start = 0)` writes starting state into consecutive slots:

```ts
particles.uploadParticles([
  { position: [0, 1, 0] },
  { position: [0.02, 1, 0], velocity: [0, -1, 0] },
  { position: [0.04, 1, 0], invMass: 0 }, // pinned in place
]);
```

Each `ParticleInit` has a `position` and optional `velocity`, `invMass`, and `collisionGroup`. `invMass` is 1 divided by the particle's mass, and defaults to 1. 0 means infinitely heavy, so the particle never moves. Slots you never write keep `invMass = 0`, so unused capacity sits still.

Space particles `2 × radius` apart. Liquid particles packed closer than that burst apart on the first step.

Some materials set masses themselves. `FluidSystem` gives its particles the mass of the liquid they stand for, and `ClothSystem` writes its whole graph. See each material's page.

### Ranges

Each material owns a **range** of the buffer, `{ start, count }`. A fluid might own slots 0 to 9,999 and a soft body slots 10,000 to 12,499:

```ts
const water = new FluidSystem(particles, { range: { start: 0, count: 10000 } });
```

Ranges of different materials must not overlap. `FluidSystem` defaults to every particle in the system, so pass a `range` whenever you add another material. [Combining materials](combining-materials.md) walks through a shared layout.

### Reading the buffers yourself

The buffers, such as `particles.positions`, `particles.velocities`, and `particles.invMass`, are public TSL storage nodes. TSL is three.js's shader language, so your own shaders can read the buffers directly. `createParticleMesh` does this to draw particles with no CPU work:

```ts
import { createParticleMesh } from 'threejs-particle-fluids';

const dots = createParticleMesh(particles, { range: fluid.range, radius: radius * 0.5 });
scene.add(dots);
```

Its `colorNode(position)` option takes a TSL function for per-particle color, for example to color by height.

If you write to a buffer's `.value.array` on the CPU, set `.value.needsUpdate = true` afterward. `await particles.readback()` copies everything back to the CPU. It stalls until the GPU is idle, so use it for debugging and tests, not every frame.

## Materials

Each material works on a range of particles:

| Material          | What it simulates                                                 | Page                                                           |
| ----------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| `FluidSystem`     | Liquids, and the air that carries smoke                           | [`FluidSystem`](fluid-system.md)                               |
| `ViscositySolver` | Very thick liquids, on top of a `FluidSystem`                     | [`FluidSystem`](fluid-system.md#thick-liquids-viscositysolver) |
| `SoftbodySystem`  | Soft and near-rigid bodies by shape matching                      | [`SoftbodySystem`](softbody-system.md)                         |
| `ClothSystem`     | Cloth with stretch, bending, pins, and wind                       | [`ClothSystem`](cloth-system.md)                               |
| `GasSystem`       | Smoke tracers carried by a fluid, with optional heat and buoyancy | [`GasSystem`](gas-system.md)                                   |

Anything with a `build(context)` method is a material too. See [Custom materials](custom-materials.md).

## The solver loop

`SimLoop` advances the particles and everything attached to them:

```ts
const loop = new SimLoop(particles, {
  substeps: 4,
  iterations: 2,
  gravity: new Vector3(0, -9.81, 0),
  materials: [water, ducks],
  colliders: [walls],
  contact: true,
});
await loop.step(1 / 60);
```

It uses extended position-based dynamics (XPBD): each step moves particles freely, then corrects their positions to satisfy rules, called _constraints_, such as "keep this liquid at its rest density" or "keep this cloth edge its rest length".

Stiffness settings are given as _compliance_, which is the opposite of stiffness: 0 is completely rigid, and bigger numbers are softer. Useful values are tiny, like 1e-6.

Each call to `step(dt)` splits `dt` into **substeps**, and each substep:

1. predicts new positions from velocity and gravity;
2. rebuilds the neighbor grid, finds particle contacts, and runs the materials' pre-solve shaders;
3. runs **iterations** of the material constraints, then contacts and colliders;
4. derives velocities from how far the particles moved;
5. runs the materials' post-solve shaders (viscosity, vorticity, moving the smoke tracers, …), then friction.

Materials run in the order they're listed. Two depend on it: `ViscositySolver` goes after its fluid and `GasSystem` before it ([why](combining-materials.md#order-in-materials)).

### Substeps and iterations

- **Substeps** matter most. More substeps make every constraint stiffer and keep fast particles from skipping through thin things. `loop.substeps` can change at any time.
- **Iterations** solve constraints more precisely within a substep. Two is usually enough; extra substeps are a better use of the same work.

Start with 3 to 6 substeps at 60 steps per second. When you shrink the particle radius and keep the scene the same size, scale substeps up roughly in proportion to `1 / radius`. `Simulation` picks its default this way.

### Gravity

`gravity` defaults to Earth gravity. `loop.gravity` is the live vector:

```ts
loop.gravity.set(0, -3, 0);
```

### Colliders and contacts

Colliders are shapes particles can't enter, passed in `colliders`; see [Colliders](colliders.md). Soft bodies and cloth also need **particle contacts** (`contact: true`) to touch each other, and liquids need **boundaries** to push on them; [Combining materials](combining-materials.md) covers both.

## The frame loop

The solver is tuned for a fixed step length. On a 120 Hz display, calling `step(1 / 60)` every frame runs the simulation at double speed. `FrameStepper` takes as many fixed steps as the elapsed time allows, and drops time when the GPU falls behind, instead of falling further and further behind. `Simulation.step()` uses one internally.

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

`loop.step(dt)` submits the whole step to the GPU at once and resolves when it's queued. Always `await` it before rendering. Call `stepper.reset()` after pausing, so the next frame doesn't try to catch up.

## Cleaning up

Dispose the renderers, the colliders, the loop, and the particle system separately:

```ts
surface.dispose();
walls.dispose();
loop.dispose();
particles.dispose();
```

The renderers and `SDFCollider` free their GPU resources. `loop.dispose()` and `particles.dispose()` don't release the particle buffers on the GPU. `particles.dispose()` marks the system as unusable, so later uploads and readbacks throw.

## When something goes wrong

Error messages from the low-level classes, and fixes for scenes that explode, jitter, or run slowly, are in [Hand-built scenes](../troubleshooting.md#hand-built-scenes-low-level-api) in Troubleshooting.

---

Previous: [Docs home](../README.md) · Next: [Colliders](colliders.md)
