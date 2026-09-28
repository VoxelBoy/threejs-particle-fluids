[Docs](../README.md) › [Advanced](../README.md#advanced) › Combining materials

# Combining materials

`Simulation` makes everything you add push on each other without any setup. This page shows how it's done by hand, for scenes built from the low-level classes. It's the one place that covers fluid boundaries, particle contacts, and collision groups.

A _material_ here is a kind of physics, such as a liquid, a soft body, or a cloth (see [Materials](low-level-api.md#materials)). All materials store their particles in one `ParticleSystem`. That's what lets them touch. Three things make them interact:

- **Fluid boundaries**: a fluid treats other particles as a solid wall, pushes on them, and can wet them.
- **Particle contacts**: soft bodies and cloth collide with each other, and with themselves when allowed.
- **Collision groups**: switch contacts off between particles that shouldn't collide.

## Laying out the buffer

Give each material its own range, and size the particle system for all of them:

```ts
// `waterPoints` is a ParticleInit[] of liquid on a grid spaced 2 × radius apart,
// as in The low-level API. `duckGeometry` is a closed geometry in world space.
const duck = voxelize(duckGeometry, { particleRadius: radius });
const invMass = 1 / (400 * (2 * radius) ** 3); // a density of 400 kg/m³, so it floats
const duckParticles: ParticleInit[] = [];
for (let i = 0; i < duck.count; i++) {
  const p = duck.positions;
  duckParticles.push({ position: [p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!], invMass });
}

const waterCount = waterPoints.length;
const duckCount = duck.count;
const particles = new ParticleSystem(renderer, waterCount + duckCount, radius);
particles.uploadParticles(waterPoints, 0);
particles.uploadParticles(duckParticles, waterCount);

const fluid = new FluidSystem(particles, { range: { start: 0, count: waterCount } });
const ducks = new SoftbodySystem(particles, {
  bodies: [
    {
      range: { start: waterCount, count: duckCount },
      surfaceCount: duck.surfaceCount,
      compliance: 1e-6,
    },
  ],
});
```

Every particle shares one radius, so voxelize soft bodies and size cloth grids at the fluid's radius.

## Floating soft bodies

A soft body floats when the fluid sees its surface as a boundary:

```ts
for (let i = 0; i < ducks.bodies.length; i++) fluid.addBoundary(ducks.surfaceRange(i));

const loop = new SimLoop(particles, {
  materials: [fluid, ducks],
  colliders: [walls],
  contact: { muS: 0.35, muK: 0.2 },
});
```

- Only the surface particles need to be boundaries; the fluid never reaches the inside.
- Whether a body floats depends on its mass. See [Mass and balance](softbody-system.md#mass-and-balance) for how to set it from a density.
- For solids that never move, pass `{ dynamic: false }` to `addBoundary` to save work every substep.
- `contact` lets the bodies bump into each other. Without it they pass through one another, though the fluid still pushes on each.
- Pass the body particles to the liquid renderer so it wets them: `solids: { start: waterCount, count: duckCount }`.

## Liquid on cloth

The same boundary makes a cloth hold liquid:

```ts
const tarp = new ClothSystem(particles, { graph, stretchCompliance: 1e-8 });
particles.setCollisionGroup(tarp.range, 1); // its overlapping neighbors aren't collisions
const liquid = new FluidSystem(particles, { range: liquidRange, adhesion: 0.15 });
liquid.addBoundary(tarp.range);

const loop = new SimLoop(particles, {
  materials: [liquid, tarp],
  colliders: [walls],
  contact: { muS: 0.05, muK: 0.03 },
});
```

Make the cloth heavy (a high `surfaceDensity` in `createClothGraph`) so pooled liquid can't push its particles apart and leak through. [`demo/presets/tarp.ts`](../../demo/presets/tarp.ts) is the full scene.

## Soft bodies and cloth together

Particle contacts are what let soft bodies land on cloth, cloth drape over bodies, and bodies stack. Turn them on with `contact`, and raise the contact budget for dense scenes:

```ts
contact: { muS: 0.12, muK: 0.06, maxContacts: particles.capacity * 20 }
```

`maxContacts` defaults to `8 × capacity`. Extra pairs are dropped, which shows up as particles passing through each other in crowded piles. To check, `await loop.contacts!.readbackCount()` returns the pairs found in the last substep, including dropped ones.

## Collision groups

Particles that share a non-zero collision group never collide with each other; group 0 collides with everything.

- `SoftbodySystem` puts each body in its own group unless you pass `selfCollision: true`, so a body's tightly packed particles don't collide with themselves.
- For a cloth sampled so densely that neighbors overlap, call `particles.setCollisionGroup(cloth.range, group)`.
- Fluids already keep their own particles apart, so they skip contacts with themselves.
- A cloth can only collide with itself when its particles are smaller than half the grid spacing; see [Particle radius](cloth-system.md#particle-radius).

Set your own collision groups before you create the `SimLoop`. When the loop is created, it gives materials new group numbers above the highest one already in use.

## Order in `materials`

Most materials don't care about order. Two do:

| Material          | Must come        | Why                                                                              |
| ----------------- | ---------------- | -------------------------------------------------------------------------------- |
| `ViscositySolver` | after its fluid  | It smooths the fluid's velocities after the fluid computes them.                 |
| `GasSystem`       | before its fluid | Tracers follow the solved velocities before vorticity and viscosity adjust them. |

Both throw an error that says so if they're listed the wrong way round.

---

Previous: [`GasSystem`](gas-system.md) · Next: [Custom materials](custom-materials.md)
