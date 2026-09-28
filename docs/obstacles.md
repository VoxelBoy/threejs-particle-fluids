[Docs](README.md) › Obstacles

# Obstacles

Obstacles are solid shapes that liquid, smoke, soft bodies, and cloth can't enter. Liquid flows around them, cloth drapes over them, and soft bodies bounce off them.

```ts
sim.addFloor();
sim.addSphere({ center: new Vector3(0, 0.3, 0), radius: 0.15 });
sim.addBox({ center: new Vector3(0.4, 0.1, 0), size: new Vector3(0.1, 0.2, 0.4) });
sim.addCapsule({ start: new Vector3(-0.4, 0, 0), end: new Vector3(-0.4, 0.5, 0), radius: 0.03 });
sim.addMesh({ mesh: bunny });
```

**Obstacles are invisible.** They only affect the simulation. To see one, add a matching mesh to your scene yourself. The exception is `addMesh`, which uses a mesh you already have.

Like everything else, obstacles must be added before the first `step()`.

## Floor

```ts
sim.addFloor(); // at y = 0
sim.addFloor({ height: -0.5 });
```

An endless flat floor at `height` (default 0). A [container](simulation.md#the-container) already has a floor at its bottom, so you only need this without one.

## Sphere

```ts
sim.addSphere({ center: new Vector3(0, 0.3, 0), radius: 0.15 });
```

| Option   | Default     | What it does                                                        |
| -------- | ----------- | ------------------------------------------------------------------- |
| `radius` | required    | Radius in metres.                                                   |
| `center` | `(0, 0, 0)` | Where the sphere sits.                                              |
| `follow` | none        | An `Object3D` to follow. See [Moving obstacles](#moving-obstacles). |

## Box

```ts
sim.addBox({
  center: new Vector3(0, 0.1, 0),
  size: new Vector3(0.4, 0.2, 0.1),
  rotation: new Euler(0, Math.PI / 4, 0),
});
```

| Option     | Default     | What it does                                    |
| ---------- | ----------- | ----------------------------------------------- |
| `size`     | required    | Full width, height, and depth in metres.        |
| `center`   | `(0, 0, 0)` | Where the box sits.                             |
| `rotation` | none        | An `Euler`.                                     |
| `follow`   | none        | An `Object3D` to follow, position and rotation. |

`size` is the full size, the same numbers you'd pass to `BoxGeometry`.

## Capsule

```ts
sim.addCapsule({ start: new Vector3(0, 0, 0), end: new Vector3(0, 0.6, 0), radius: 0.03 });
```

A rod with rounded ends, from `start` to `end`. Good for posts, handles, and limbs. Capsules can't move.

## Any mesh

```ts
sim.addMesh({ mesh: bunny });
```

Makes any closed mesh solid. The obstacle follows the mesh as you move, rotate, or scale it, so there's no `follow` option.

- **Closed meshes only.** Holes and doubled faces confuse which side is inside. The first `step()` throws `bakeMeshToSdf: mesh appears non-watertight` if the mesh isn't closed.
- **Keep it simple.** The shape is worked out on the CPU during the first step, and that takes longer for detailed meshes. A few thousand triangles is plenty. Use a simplified copy for the obstacle and draw the detailed one (see below).
- **Scale evenly.** Use the same scale on all three axes. Uneven scale throws `SDFCollider.setTransform: scale must be uniform`.
- `resolution` (default 64) sets how finely the shape is captured. Raise it for small details at the cost of a slower start.

To use a simplified copy, make it a hidden child of the detailed mesh. A hidden mesh is still solid, and as a child it moves with the mesh you draw:

```ts
const proxy = new Mesh(simplifiedGeometry);
proxy.visible = false; // hidden, but still solid
bunny.add(proxy); // moves with the detailed bunny
sim.addMesh({ mesh: proxy });
```

The obstacle follows `proxy`'s world position, rotation, and scale, including those of its parents.

## Moving obstacles

`addSphere` and `addBox` take a `follow` object. Move the object as you normally would, and the obstacle moves with it and pushes things out of the way:

```ts
const ball = new Mesh(new SphereGeometry(0.2), new MeshStandardMaterial());
scene.add(ball);
sim.addSphere({ radius: 0.2, follow: ball });

async function frame(time: number) {
  ball.position.set(0, 0.7, Math.cos(time / 1000) * 0.8);
  await sim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

With `follow`, the obstacle's position comes from the object, so `center` is ignored. A box also takes the object's rotation. The object's scale is ignored, so keep `radius` and `size` matched to what you draw.

A moving sphere or box also drags what it touches along with it by friction, like a spoon pulling honey. That drag follows the object's movement from place to place, not its spin. A moving mesh from `addMesh` pushes things but doesn't drag them.

## Friction

Every obstacle takes `friction`, from 0 for slick to about 1 for sticky. The default is 0.5.

```ts
sim.addMesh({ mesh: bunny, friction: 0.9 }); // honey clings to it
sim.addFloor({ friction: 0.05 }); // ice
```

---

Previous: [Fluids](fluids.md) · Next: [Soft bodies](soft-bodies.md)
