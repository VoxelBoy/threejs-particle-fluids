[Docs](README.md) › Soft bodies

# Soft bodies

`SoftbodySystem` simulates soft and near-rigid bodies made of particles. Each body pulls its particles back toward its rest shape by shape matching (Müller et al. 2005; Müller & Chentanez 2011), solved as XPBD constraints. `voxelize` fills a mesh with particles, and `SoftbodyMesh` skins the original mesh to the result.

## From a mesh to a body

```ts
import { SoftbodyMesh, SoftbodySystem, voxelize, type ParticleInit } from 'threejs-particle-fluids';

// Place the geometry in world space where the body starts.
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
const mesh = new SoftbodyMesh(bodies, 0, geometry, source.material);
scene.add(mesh);

const loop = new SimLoop(particles, {
  substeps: 6,
  materials: [bodies],
  colliders: [floor],
  contact: true,
});
```

## `voxelize`

`voxelize(shape, options)` places particles on a cubic grid inside a closed shape. The shape can be a `BufferGeometry` or `TriangleMesh` (inside is decided by ray-cast parity, so the mesh must be closed) or a baked distance field (`SDFData`, see [Colliders](colliders.md)).

| Option           | Default | What it does                                                                                                     |
| ---------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `particleRadius` | —       | Particle radius. Particles are spaced `2 · particleRadius` apart. Use the particle system's radius.              |
| `largestPiece`   | `false` | Keep only the largest connected piece. Thin features can voxelize into loose islands.                            |
| `dilation`       | 0       | Distance field input only: also fill points up to this far outside the surface, which keeps thin parts attached. |

It returns `positions` (xyz per particle), `count`, `surfaceCount`, and `edges` (pairs of face-adjacent particles, needed for local shape matching). Particles come **surface first**: the first `surfaceCount` particles are the ones missing a grid neighbor.

Every particle in a simulation shares one radius, so a body's resolution follows from the radius. To hit a target particle count, search on the radius; see `sampleBody` in [`demo/presets/softbodies.ts`](../demo/presets/softbodies.ts).

## Bodies

Each entry in `bodies` is a `SoftbodyDef`:

| Field           | Default                | What it does                                                                                   |
| --------------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| `range`         | —                      | The body's particles. Surface particles must come first.                                       |
| `surfaceCount`  | all of them            | How many leading particles lie on the surface. Fluids push on these.                           |
| `compliance`    | 0                      | Shape-matching compliance in s²/kg. 0 is rigid; around 1e-6 is soft.                           |
| `restPositions` | the uploaded positions | Rest shape, xyz per particle.                                                                  |
| `edges`         | none                   | Neighbor pairs local to the body. Required for local shape matching; `voxelize` produces them. |

Change a body's stiffness live with `bodies.setCompliance(index, compliance)`.

## Global or local shape matching

```ts
new SoftbodySystem(particles, { bodies, shapeMatching: 'local' });
```

- **`'global'`** (default): each body matches its rest shape as a whole. Cheap and stiff. Bodies wobble and squash but don't bend much. Use it for near-rigid objects, such as the floating ducks.
- **`'local'`**: every particle matches the shape of its own neighborhood, so bodies bend, fold, and droop. Needs `edges`. Use it for jelly and anything with thin parts, like the Bunny Lineup's ears.

### Choosing compliance

Compliance depends on the particle count: with more particles per body, the same compliance feels stiffer, so scale it with the count to keep the same look at every resolution. The Soft Body Squeeze preset uses, for local matching,

```ts
// softness 0 is firm rubber, 1 is loose jelly
const compliance = 10 ** (-6 + softness * 3) * (count / 200);
```

where `count` is the body's particle count. For global matching, the Buoyancy preset's ducks use `1e-6` for a stiff toy. 0 makes a body rigid in either mode.

### Mass and balance

Particle masses come from `invMass` at upload. Global shape matching weights each particle by the mass it has when the system is created, so a body with a heavier base settles base-down. The Buoyancy preset makes each duck's lowest third four times heavier than the rest, which keeps it floating upright without any extra constraint.

To give a body a density of `density` kg/m³ (water is 1000), set each particle's inverse mass to `1 / (density · spacing³)`, where `spacing = 2 · radius`. A body denser than the liquid sinks; a lighter one floats.

## Collisions

- **Bodies touching each other** need particle contacts: pass `contact: true` (or `{ muS, muK }`) to the `SimLoop`.
- **A body touching itself** is off by default, because each body gets its own collision group. Pass `selfCollision: true` so a ring or a limb can fold onto itself without passing through.
- **Floating and sinking** in a liquid: add each body's `surfaceRange(i)` as a fluid boundary. See [Combining materials](combining-materials.md).

## `SoftbodyMesh`

```ts
new SoftbodyMesh(softbody, bodyIndex, geometry, material?);
```

Each vertex follows its nearest particles by dual-quaternion blending on the GPU. Two rules:

- `geometry` must sit exactly where the body's rest shape is, in world space. Voxelize and skin the same transformed geometry.
- Leave the mesh's own transform at identity. The skinning writes world positions.

`material` is a `MeshStandardMaterial` whose colors and maps are copied onto the skinned material. Without one you get a plain orange-brown.

## Reading body transforms

With global shape matching, `bodies.bodyCenters` holds each body's current center and `bodies.bodyRotations` its rotation (three row vectors per body, at rows `3b`, `3b + 1`, `3b + 2`). Read them in TSL to attach effects or objects to a body.

---

Previous: [Fluids](fluids.md) · Next: [Cloth](cloth.md)
