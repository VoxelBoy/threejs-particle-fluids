[Docs](README.md) › Combining materials

# Combining materials

All materials share one `ParticleSystem`, so they meet in the same buffers. Three mechanisms make them interact:

- **Fluid boundaries**: a fluid treats other particles as a solid wall, pushes on them, and can wet them.
- **Particle contacts**: soft bodies and cloth collide with each other, and with themselves when allowed.
- **Collision groups**: switch contacts off between particles that shouldn't collide.

## Laying out the buffer

Give each material its own range, and size the particle system for all of them:

```ts
const waterCount = water.length;
const duckCount = duck.count;
const particles = new ParticleSystem(renderer, waterCount + duckCount, radius);
particles.uploadParticles(water, 0);
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
- Whether a body floats depends on its mass. With particles spaced `2 · radius` apart, a body of density `d` kg/m³ has `invMass = 1 / (d · spacing³)` per particle. Water is 1000.
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

Make the cloth heavy (a high `surfaceDensity` in `createClothGraph`) so pooled liquid can't push its particles apart and leak through. [`demo/presets/tarp.ts`](../demo/presets/tarp.ts) is the full scene.

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
- Fluids skip contacts within themselves on their own, since their pressure solve keeps them apart.

The loop hands out new groups to materials above the highest group already set when it's created, so set your own groups before creating the `SimLoop`.

## Order in `materials`

Most materials don't care about order. Two do:

| Material          | Must come        | Why                                                                              |
| ----------------- | ---------------- | -------------------------------------------------------------------------------- |
| `ViscositySolver` | after its fluid  | It smooths the fluid's velocities after the fluid computes them.                 |
| `GasSystem`       | before its fluid | Tracers follow the solved velocities before vorticity and viscosity adjust them. |

Both throw an error that says so if they're listed the wrong way round.

---

Previous: [Colliders](colliders.md) · Next: [Custom materials](custom-materials.md)
