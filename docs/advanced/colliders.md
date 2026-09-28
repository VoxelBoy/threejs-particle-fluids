[Docs](../README.md) › [Advanced](../README.md#advanced) › Colliders

# Colliders

Colliders are shapes particles can't enter. Particles slide along them, slowed by friction. `Simulation`'s [obstacles](../obstacles.md) are built from these two classes. Pass colliders to the `SimLoop`:

```ts
const loop = new SimLoop(particles, { materials: [water], colliders: [walls, bunny] });
```

Every collider is built for one `ParticleSystem` and must use the same one as the loop.

## `PrimitiveSet`

A set of planes, spheres, boxes, and capsules.

```ts
import { Quaternion, Vector3 } from 'three';
import { PrimitiveSet } from 'threejs-particle-fluids';

const walls = new PrimitiveSet(particles);
walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0)); // floor
walls.addSphere(new Vector3(0, 0.3, 0), 0.1);
walls.addBox(new Vector3(0.3, 0.1, 0), new Vector3(0.05, 0.1, 0.2), {
  rotation: new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), 0.4),
});
walls.addCapsule(new Vector3(-0.3, 0, 0), new Vector3(-0.3, 0.5, 0), 0.03);
```

| Method                                  | Shape                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `addPlane(normal, point, options?)`     | A plane through `point`. Particles stay on the side `normal` points to. |
| `addSphere(center, radius, options?)`   | A sphere.                                                               |
| `addBox(center, halfExtents, options?)` | A box, optionally rotated with `options.rotation`.                      |
| `addCapsule(a, b, radius, options?)`    | A capsule between two points. Handy for rods, shafts, and limbs.        |

Each returns a slot number for moving the shape later.

### Options

- `muS`, `muK`: static friction (how hard it is to start sliding) and kinetic friction (how much sliding slows down). Defaults 0.5 and 0.4. Use about 1 for sticky honey, near 0 for slick walls.
- `velocity`: surface velocity in m/s, used for friction, such as a conveyor belt.
- `invert` (spheres and boxes): keep particles **inside** the shape. An inverted box is a quick container.

### Moving shapes

`attach(slot, object)` makes a primitive follow an `Object3D`'s world position and orientation. Move the object as usual; the set follows it every step and derives its velocity, so friction drags particles along:

```ts
const paddle = walls.addBox(new Vector3(), new Vector3(0.2, 0.05, 0.01));
walls.attach(paddle, paddleMesh);

// later
paddleMesh.rotation.y += 0.05;
```

The solver sweeps a moving primitive smoothly across the frame's substeps, so a shape that jumps once per frame still pushes particles cleanly. `setSphere(slot, center, radius, velocity?)` moves and resizes a sphere directly.

Shapes added after the simulation has started need room reserved up front:

```ts
const set = new PrimitiveSet(particles, { capacity: 32 });
```

## `SDFCollider`: any mesh

For any other shape, bake the mesh once into a _signed distance field_: a 3D grid that stores how far each point is from the surface, negative inside. Then collide against it:

```ts
import { SDFCollider, bakeMeshToSdf } from 'threejs-particle-fluids';

const sdf = bakeMeshToSdf(bunnyGeometry, { resolution: 64, padding: 0.02 });
const bunny = new SDFCollider(particles, sdf, {
  position: new Vector3(0, 0, 0),
  rotation: new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), 2.16),
  scale: 1,
  muS: 0.6,
  muK: 0.5,
});
```

The field is stored as a 3D texture on the GPU, so a detailed mesh costs no more per frame than a simple one. Move the collider with `setPosition`, `setRotation`, `setScale`, or `setTransform(matrix)` (scale must be uniform). Friction treats the collider as standing still, so moving it pushes particles but doesn't drag them along. `bunny.muS` and `bunny.muK` can be changed live.

`thickness` adds contact distance beyond the particle radius. Sparse surfaces, such as cloth, leave gaps that thin features (ears, fins) can slip through; about half the particle spacing closes them.

### Baking

`bakeMeshToSdf(geometry, { resolution, padding })` takes a `BufferGeometry`, not a `Mesh`. It fits a cubic grid around the geometry, in the geometry's own local space, and computes the distance at each grid point. Place the field in the world with the collider's `position`, `rotation`, and `scale`, or with `setTransform(mesh.matrixWorld)`, as `Simulation` does. It's a slow CPU bake, and the time grows with the grid size times the triangle count, so:

- Bake offline and ship the result. `encodeSdfBinary(sdf)` writes a compact `.sdf.bin` buffer, and `decodeSdfBinary(buffer)` reads it back:

  ```ts
  const response = await fetch('/models/bunny.sdf.bin');
  const sdf = decodeSdfBinary(await response.arrayBuffer());
  ```

  [`scripts/prepare-honey-assets.ts`](../../scripts/prepare-honey-assets.ts) bakes the Stanford bunny this way.

- Keep collision meshes to a few thousand triangles.
- The mesh must be closed, with every edge shared by exactly two triangles. Holes and doubled faces make the inside/outside test disagree, and the bake throws with counts so you can fix the mesh instead of getting a wrong field.

`sampleSdf(sdf, x, y, z)` and `sampleSdfGradient(sdf, x, y, z)` sample the field on the CPU the same way the GPU does, which helps when placing objects or writing tests. A baked field also works as input to [`voxelize`](softbody-system.md#voxelize).

## Colliders and the liquid surface

Pass the same colliders to `FluidSurfaceRenderer`'s `colliders` option to draw the thin edge where the liquid meets them. A `PrimitiveSet` given as `carve` is cut out of the liquid every frame. See [`FluidSystem`](fluid-system.md#drawing-the-liquid-fluidsurfacerenderer).

---

Previous: [The low-level API](low-level-api.md) · Next: [`FluidSystem`](fluid-system.md)
