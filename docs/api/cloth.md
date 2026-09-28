[Docs](../README.md) › [API](../README.md#api-reference) › Cloth

# Cloth

A rectangle of cloth added with [`Simulation.addCloth`](./simulation.md#addclothoptions). The `Cloth` it returns lets you change the wind and softness while it runs.

```ts
const curtain = sim.addCloth({
  width: 1.2,
  height: 1.2,
  position: new Vector3(0, 1, 0),
  pin: 'top',
});
curtain.wind.set(0, 0, 2);
```

## ClothOptions

| Option     | Type                                            | Default      | Description                                                                                                             |
| ---------- | ----------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `width`    | `number`                                        | required     | Width, m.                                                                                                               |
| `height`   | `number`                                        | required     | Height, m.                                                                                                              |
| `position` | `Vector3`                                       | `(0, 1, 0)`  | Center of the cloth.                                                                                                    |
| `rotation` | `Euler`                                         | none         | The cloth starts upright, facing +z. `new Euler(-Math.PI / 2, 0, 0)` lays it flat.                                      |
| `pin`      | `'top' \| 'top-corners' \| 'corners' \| 'none'` | `'top'`      | Which part of the cloth is held in place.                                                                               |
| `softness` | `number`                                        | `0.75`       | From 0 for stiff canvas to 1 for silk that drapes.                                                                      |
| `weight`   | `number`                                        | `0.1`        | Weight per area, kg/m². Heavier cloth holds liquid with less leaking.                                                   |
| `wind`     | `Vector3`                                       | `(0, 0, 0)`  | Wind speed and direction, m/s.                                                                                          |
| `color`    | `number`                                        | `0x870b21`   | Color of the default fabric material.                                                                                   |
| `material` | `MeshPhysicalNodeMaterial`                      | sheen fabric | Your own material in place of the default. Its position and normal nodes are replaced. You dispose it when you're done. |

Cloth particles sit a little more than one particle width apart, so the cloth can fold onto itself without its neighbors colliding. A 1.2 m square cloth with a particle radius of 0.01 m uses about 3,100 particles.

Like `softness` on soft bodies, `softness` here accounts for the particle count, so a cloth drapes about the same way at any particle budget. The [formula](./cloth-system.md#bend-compliance-and-resolution) is on the `ClothSystem` page.

The simulation also gives the cloth some air drag, lift, and damping. You can change these on `clothSystem` after `start()`.

## Cloth

### Properties

| Property        | Type                               | Access     | Description                                                             |
| --------------- | ---------------------------------- | ---------- | ----------------------------------------------------------------------- |
| `particleCount` | `number`                           | read       | Particles in this cloth. `0` until the simulation starts.               |
| `wind`          | `Vector3`                          | read       | Wind, m/s. Change it in place, for example `curtain.wind.set(0, 0, 2)`. |
| `softness`      | `number`                           | read/write | Can be changed while the simulation runs.                               |
| `mesh`          | `Mesh`                             | read       | The cloth surface in your scene. Available after `start()`.             |
| `clothSystem`   | [`ClothSystem`](./cloth-system.md) | read       | The solver underneath. Available after `start()`.                       |

## Limitations

- Only rectangles are supported here. For other shapes, build the cloth from any geometry with [`createClothGraph`](./cloth-system.md).
- Pinned points stay where they started. To move them, use [`ClothSystem`](./cloth-system.md) directly.
- You can't change `weight` after adding the cloth.
- Set wind on the `Cloth`, not on `clothSystem`. The `Cloth` copies its wind into `clothSystem` every step, which overwrites any change made there.
- Liquid can leak through light cloth. Raise `weight`, or raise `particles` so the cloth has more of them.
- Thin, sharp parts of an obstacle can poke through the gaps between cloth particles.
