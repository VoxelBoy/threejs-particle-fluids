[Docs](README.md) › Cloth

# Cloth

`sim.addCloth` adds a rectangle of cloth. It hangs, drapes, folds over itself, blows in the wind, and catches liquid. It's drawn as a smooth, double-sided fabric surface.

```ts
const sim = new Simulation({ renderer, scene, camera }); // no container: the cloth hangs in open air
const curtain = sim.addCloth({
  width: 1.2,
  height: 1.2,
  position: new Vector3(0, 1, 0),
  pin: 'top',
});
sim.addFloor();
```

Cloth must start fully inside the container, if there is one. This curtain is wider than the tank from [Getting started](getting-started.md) and hangs above it, so it gets a simulation with no container, and a floor instead.

[`examples/cloth.ts`](../examples/cloth.ts) hangs this curtain and swings a ball through it.

## Options

| Option     | Default          | What it does                                                                                                                                                                          |
| ---------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `width`    | required         | Width in metres.                                                                                                                                                                      |
| `height`   | required         | Height in metres.                                                                                                                                                                     |
| `position` | `(0, 1, 0)`      | Center of the cloth.                                                                                                                                                                  |
| `rotation` | none             | An `Euler`. The cloth starts upright in the XY plane, facing +z.                                                                                                                      |
| `pin`      | `'top'`          | What holds it up: `'top'` (the whole top edge), `'top-corners'`, `'corners'` (all four), or `'none'`.                                                                                 |
| `softness` | 0.75             | 0 is stiff canvas, 1 drapes like silk.                                                                                                                                                |
| `weight`   | 0.1              | Mass per area in kg/m². 0.1 is a light fabric. Heavier cloth holds liquid without leaking.                                                                                            |
| `wind`     | none             | Wind velocity in m/s, as a `Vector3`.                                                                                                                                                 |
| `color`    | `0x870b21`       | Fabric color.                                                                                                                                                                         |
| `material` | a sheen material | Your own `MeshPhysicalNodeMaterial` from `'three/webgpu'` (not `MeshPhysicalMaterial`). The cloth replaces its `positionNode` and `normalNode`, so anything you put there is ignored. |

Pinned points stay exactly where they started. With `pin: 'none'`, the cloth falls freely.

## Laying it flat

The cloth starts upright. Rotate it to lay it flat, for example to drop a sheet onto something:

```ts
sim.addCloth({
  width: 1,
  height: 1,
  position: new Vector3(0, 1.2, 0),
  rotation: new Euler(-Math.PI / 2, 0, 0),
  pin: 'none',
});
sim.addSphere({ center: new Vector3(0, 0.4, 0), radius: 0.3 });
sim.addFloor();
```

`pin` names edges and corners of the upright cloth, before rotation. After the rotation above, `'top'` would be the back edge (toward -z).

## Live settings

`addCloth` returns a `Cloth`:

```ts
curtain.wind.set(0, 0, 2); // wind is a Vector3 you change in place
curtain.softness = 0.3; // stiffer
curtain.mesh; // the cloth surface in the scene, once the simulation has started
```

A gusty breeze, updated every frame:

```ts
async function frame(time: number) {
  curtain.wind.set(0, 0, 1 + Math.sin(time / 1000));
  await sim.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

`weight`, `pin`, `color`, and `material` are fixed once added. To restyle the cloth while it runs, change the properties of the material you passed in, or of `curtain.mesh.material`.

For a setting not listed here, such as air drag, reach the object underneath. See [The objects underneath](simulation.md#the-objects-underneath).

## Collisions

Cloth collides with itself, with soft bodies, with [obstacles](obstacles.md), and with the container walls. Folds pile up on each other instead of passing through.

Cloth is thin, so fast or sharp obstacles can poke through it. If that happens, see [Things pass through each other](troubleshooting.md#things-pass-through-each-other).

## Catching liquid

Pour liquid onto cloth and it pools, runs off, and drips from the edges. The liquid wets the fabric using the fluid's `adhesion`.

```ts
const sim = new Simulation({
  renderer,
  scene,
  camera,
  // Reaches down to the floor, so liquid that drips off the cloth is still drawn.
  container: new Box3(new Vector3(-0.8, 0, -0.7), new Vector3(0.8, 1.3, 0.7)),
});
sim.addCloth({
  width: 1.2,
  height: 1,
  position: new Vector3(0, 0.6, 0),
  rotation: new Euler(-Math.PI / 2, 0, 0),
  pin: 'corners',
  weight: 5,
});
sim.addFluid({ box: new Box3(new Vector3(-0.2, 0.8, -0.2), new Vector3(0.2, 1.1, 0.2)) });
```

Give the scene a `container` that covers everywhere the liquid can fall. Without one, liquid is drawn only in a box around where things started, so drips vanish on their way to the floor.

Light cloth lets pooled liquid push its threads apart and leak through. Raise `weight` until it holds. The Tarp Runoff demo preset uses a heavy canvas, about 10 kg/m².

---

Previous: [Soft bodies](soft-bodies.md) · Next: [Smoke](smoke.md)
