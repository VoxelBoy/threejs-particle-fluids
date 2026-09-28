[Docs](../README.md) › [Advanced](../README.md#advanced) › `SoftbodySystem`

# `SoftbodySystem`

`SoftbodySystem` simulates soft and near-rigid bodies made of particles. Each body remembers its starting shape. Every substep it finds the closest moved-and-rotated copy of that shape and pulls its particles toward it. This is called _shape matching_. `voxelize` fills a mesh with particles, and `SoftbodyMesh` draws the original mesh bending along with them.

`sim.addSoftbody` builds bodies this way for you; the [Soft bodies guide](../soft-bodies.md) covers it. Build them yourself for global shape matching, uneven mass, self-collision, or bodies read from a baked distance field. (`addSoftbody` uses local shape matching and the formula under [Choosing compliance](#choosing-compliance).)

## From a mesh to a body

```ts
import { MeshStandardMaterial, Vector3 } from 'three';
import {
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  SoftbodyMesh,
  SoftbodySystem,
  voxelize,
  type ParticleInit,
} from 'threejs-particle-fluids';

// `source` is a closed Mesh with a MeshStandardMaterial.
// `renderer`, `scene`, and `camera` are as in Getting started.
const radius = 0.015;

// Place the geometry in world space where the body starts.
source.updateWorldMatrix(true, false);
const geometry = source.geometry.clone().applyMatrix4(source.matrixWorld).translate(0, 0.5, 0);

// 1. Fill it with particles on a grid spaced 2 × radius apart.
const shape = voxelize(geometry, { particleRadius: radius });

// 2. Upload them. Rest shape defaults to these positions, so upload first.
const initial: ParticleInit[] = [];
for (let i = 0; i < shape.count; i++) {
  const [x, y, z] = shape.positions.subarray(i * 3, i * 3 + 3);
  initial.push({ position: [x!, y!, z!] });
}
const particles = new ParticleSystem(renderer, initial.length, radius);
particles.uploadParticles(initial);

// 3. Make it a body.
const bodies = new SoftbodySystem(particles, {
  bodies: [
    { range: { start: 0, count: shape.count }, surfaceCount: shape.surfaceCount, compliance: 1e-6 },
  ],
});

// 4. Draw the original mesh, deformed by the particles.
const mesh = new SoftbodyMesh(bodies, 0, geometry, source.material as MeshStandardMaterial);
scene.add(mesh);
source.visible = false;

// 5. A floor to land on, and the solver loop.
const floor = new PrimitiveSet(particles);
floor.addPlane(new Vector3(0, 1, 0), new Vector3());
const loop = new SimLoop(particles, {
  substeps: 6,
  materials: [bodies],
  colliders: [floor],
  contact: true,
});

// Each frame. The mesh follows the particles on the GPU.
await loop.step(1 / 60);
renderer.render(scene, camera);
```

## `voxelize`

`voxelize(shape, options)` places particles on a cubic grid inside a closed shape. The shape can be a `BufferGeometry` or `TriangleMesh` (it tests whether each point is inside, so the mesh must be closed) or a baked distance field (`SDFData`, see [Colliders](colliders.md#baking)).

| Option           | Default | What it does                                                                                                     |
| ---------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `particleRadius` | —       | Particle radius. Particles are spaced `2 · particleRadius` apart. Use the particle system's radius.              |
| `largestPiece`   | `false` | Keep only the largest connected piece. Thin features can voxelize into loose islands.                            |
| `dilation`       | 0       | Distance field input only: also fill points up to this far outside the surface, which keeps thin parts attached. |

It returns `positions` (xyz per particle), `count`, `surfaceCount`, and `edges` (pairs of face-adjacent particles, needed for local shape matching). Particles come **surface first**: the first `surfaceCount` particles are the ones missing a grid neighbor.

Every particle in a simulation shares one radius, so a body's resolution follows from the radius. To hit a target particle count, search on the radius; see `sampleBody` in [`demo/presets/softbodies.ts`](../../demo/presets/softbodies.ts).

## Bodies

Each entry in `bodies` is a `SoftbodyDef`:

| Field           | Default                | What it does                                                                                                                                                                                                                                         |
| --------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `range`         | —                      | The body's particles. Surface particles must come first.                                                                                                                                                                                             |
| `surfaceCount`  | all of them            | How many leading particles lie on the surface. Fluids push on these.                                                                                                                                                                                 |
| `compliance`    | 0                      | How far the body may bend away from its shape (0 is rigid, higher is softer). Around 1e-6 is a firm toy for a body of a few hundred particles; bigger bodies need bigger values for the same feel (see [Choosing compliance](#choosing-compliance)). |
| `restPositions` | the uploaded positions | Rest shape, xyz per particle.                                                                                                                                                                                                                        |
| `edges`         | none                   | Neighbor pairs local to the body. Required for local shape matching; `voxelize` produces them.                                                                                                                                                       |

Change a body's stiffness live with `bodies.setCompliance(index, compliance)`.

## Global or local shape matching

```ts
new SoftbodySystem(particles, { bodies, shapeMatching: 'local' });
```

- **`'global'`** (default): each body matches its rest shape as a whole. Cheap and stiff. Bodies wobble and squash but don't bend much. Use it for near-rigid objects, such as the floating ducks.
- **`'local'`**: every particle matches the shape of its own neighborhood, so bodies bend, fold, and droop. Needs `edges`. Use it for jelly and anything with thin parts, like the Bunny Lineup's ears.

### Choosing compliance

Compliance depends on the particle count: with more particles per body, the same compliance feels stiffer, so scale it with the count to keep the same look at every resolution. The Soft Body Squeeze preset and `Simulation` use, for local matching,

```ts
// softness 0 is firm rubber, 1 is loose jelly
const compliance = 10 ** (-6 + softness * 3) * (count / 200);
```

where `count` is the body's particle count. For global matching, the Buoyancy preset's ducks use `1e-6` for a stiff toy. 0 makes a body rigid in either mode.

### Mass and balance

Particle masses come from `invMass` at upload. Global shape matching weights each particle by the mass it has when the system is created, so a body with a heavier base settles base-down. The Buoyancy preset makes each duck's lowest third four times heavier than the rest, which keeps it floating upright without any extra constraint.

To give a body a density of `density` kg/m³ (water is 1000), set each particle's `invMass` (1 divided by its mass) to `1 / (density · spacing³)`, where `spacing = 2 · radius`. A body denser than the liquid sinks; a lighter one floats.

## Collisions

Bodies need particle contacts (`contact: true` on the `SimLoop`) to touch each other, and a fluid boundary to float in a liquid. [Combining materials](combining-materials.md) covers both.

One thing is specific to soft bodies: a body doesn't collide with itself, because each body gets its own collision group. Pass `selfCollision: true` to `SoftbodySystem` so a ring or a limb can fold onto itself without passing through.

## `SoftbodyMesh`

```ts
new SoftbodyMesh(softbody, bodyIndex, geometry, material?);
```

Each vertex of your mesh follows its nearest particles, on the GPU. Two rules:

- `geometry` must sit exactly where the body's rest shape is, in world space. Pass the same transformed geometry to `voxelize` and to `SoftbodyMesh`.
- Leave the mesh's own transform at identity. The mesh writes world positions itself.

`material` is a `MeshStandardMaterial` (or `MeshPhysicalMaterial`) from `three` whose color, roughness, metalness, and maps are copied onto the deforming mesh. Without one you get a plain orange-brown.

## Reading body transforms

With global shape matching, `bodies.bodyCenters` holds each body's current center and `bodies.bodyRotations` its rotation (three row vectors per body, at rows `3b`, `3b + 1`, `3b + 2`). Read them in TSL to attach effects or objects to a body.

Background reading: shape matching (Müller et al. 2005) and its local variant (Müller & Chentanez 2011).

---

Previous: [`FluidSystem`](fluid-system.md) · Next: [`ClothSystem`](cloth-system.md)
