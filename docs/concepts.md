[Docs](README.md) › Core concepts

# Core concepts

## One particle system for everything

A `ParticleSystem` holds every particle in the simulation in GPU storage buffers: positions, velocities, inverse masses, collision groups, and a few per-particle fields some materials use. Liquids, soft bodies, and cloth are all particles in the same buffers, which is what lets them push on each other without any special coupling code.

```ts
const particles = new ParticleSystem(renderer, capacity, particleRadius);
```

- **`capacity`** is fixed when the system is created. Size it for every material you will add.
- **`particleRadius`** is shared by every particle. A single radius lets one uniform grid find every particle's neighbors. Pick it for the finest detail you need, then size your materials around it.

### Uploading particles

`uploadParticles(data, start = 0)` writes initial state into consecutive slots:

```ts
particles.uploadParticles([
  { position: [0, 1, 0] },
  { position: [0.02, 1, 0], velocity: [0, -1, 0] },
  { position: [0.04, 1, 0], invMass: 0 }, // pinned in place
]);
```

Each `ParticleInit` has a `position` and optional `velocity`, `invMass` (default 1, and 0 pins the particle), and `collisionGroup`. Slots you never write keep `invMass = 0`, so unused capacity sits still.

Some materials set masses themselves: `FluidSystem` gives its particles the mass of the fluid they represent, and `ClothSystem` writes its whole graph. See each material's guide.

### Ranges

Materials own **ranges** of the buffer, `{ start, count }`. A fluid might own slots 0–9,999 and a soft body slots 10,000–12,499:

```ts
const water = new FluidSystem(particles, { range: { start: 0, count: 10000 } });
```

Ranges of different materials must not overlap. `FluidSystem` defaults to every particle in the system, so pass a `range` whenever you add another material.

### Reading the buffers yourself

The buffers are public TSL storage nodes (`particles.positions`, `particles.velocities`, `particles.invMass`, …), so your own shaders can read them directly. `createParticleMesh` does this to draw particles with no CPU work. If you write to a buffer's `.value.array` on the CPU, set `.value.needsUpdate = true` afterward.

`await particles.readback()` copies everything back to the CPU. It stalls until the GPU is idle, so use it for debugging and tests, not every frame.

## Materials

A material adds physics to a range of particles. The library includes:

| Material          | What it simulates                                                 | Guide                         |
| ----------------- | ----------------------------------------------------------------- | ----------------------------- |
| `FluidSystem`     | Liquids and gases with Position Based Fluids                      | [Fluids](fluids.md)           |
| `ViscositySolver` | Very thick liquids, on top of a `FluidSystem`                     | [Fluids](fluids.md)           |
| `SoftbodySystem`  | Soft and near-rigid bodies by shape matching                      | [Soft bodies](soft-bodies.md) |
| `ClothSystem`     | Cloth with stretch, bending, pins, and wind                       | [Cloth](cloth.md)             |
| `GasSystem`       | Smoke tracers carried by a fluid, with optional heat and buoyancy | [Smoke](smoke.md)             |

Anything with a `build(context)` method is a material too, so you can add your own kernels. See [Custom materials](custom-materials.md).

## The solver loop

`SimLoop` advances the particle system and everything attached to it:

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

It uses extended position-based dynamics (XPBD). Each call to `step(dt)` splits `dt` into **substeps**, and each substep:

1. predicts new positions from velocity and gravity;
2. rebuilds the neighbor grid, finds particle contacts, and runs the materials' pre-solve kernels;
3. runs **iterations** of the material constraints, then contacts and colliders;
4. derives velocities from how far the particles moved;
5. runs the materials' post-solve kernels (viscosity, vorticity, smoke advection, …), then friction.

Materials run in the order they're listed. A few materials depend on that order: `ViscositySolver` goes after its fluid, and `GasSystem` goes before its fluid. Both throw a clear error if listed the wrong way round.

### Substeps and iterations

- **Substeps** matter most. More substeps make every constraint stiffer and keep fast particles from tunneling. `loop.substeps` can be changed at any time without rebuilding.
- **Iterations** solve constraints more precisely within a substep. Two is usually enough; extra substeps are a better use of the same work.

A good starting point is 3–6 substeps at 60 steps per second. When you shrink the particle radius and keep the scene the same size, particles cover less distance per substep before they overlap, so scale substeps up roughly in proportion to `1 / radius`.

### Gravity

`gravity` defaults to Earth gravity. `loop.gravity` returns the live vector, so mutate it to change gravity while running:

```ts
loop.gravity.set(0, -3, 0);
```

## Colliders

Colliders are shapes particles can't enter. `PrimitiveSet` holds planes, spheres, boxes, and capsules that can follow `Object3D`s; `SDFCollider` handles any mesh through a baked distance field. Pass them in `colliders`. See [Colliders](colliders.md).

## Particle contacts and collision groups

Fluids keep their own particles apart with their pressure solve. Soft bodies and cloth don't, so they need **particle contacts** to touch each other or themselves: pass `contact: true`, or `{ muS, muK, maxContacts }` to set friction and the contact budget.

A particle's **collision group** switches contacts off within the group: particles that share a non-zero group never collide with each other, and group 0 collides with everything. Soft bodies assign groups to their own bodies automatically. Use groups yourself when neighboring particles overlap on purpose, as in a tightly sampled cloth. See [Combining materials](combining-materials.md).

## Units

Everything is SI: metres, seconds, kilograms, kg/m³ for density, and m/s² for gravity. Stiffness is given as XPBD **compliance**, the inverse of stiffness: 0 is rigid, and larger values are softer.

---

Previous: [Getting started](getting-started.md) · Next: [Fluids](fluids.md)
