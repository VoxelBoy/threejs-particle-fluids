[Docs](../README.md) › [API](../README.md#api-reference) › Cloth

# Cloth

Rectangular cloth added with [`Simulation.addCloth`](./simulation.md#addclothoptions).

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

| Option     | Type                                            | Default      | Description                                                                                          |
| ---------- | ----------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------- |
| `width`    | `number`                                        | required     | m.                                                                                                   |
| `height`   | `number`                                        | required     | m.                                                                                                   |
| `position` | `Vector3`                                       | `(0, 1, 0)`  | Center.                                                                                              |
| `rotation` | `Euler`                                         | none         | Starts upright in the XY plane facing +z. `new Euler(-Math.PI / 2, 0, 0)` lays it flat.              |
| `pin`      | `'top' \| 'top-corners' \| 'corners' \| 'none'` | `'top'`      | Fixed particles.                                                                                     |
| `softness` | `number`                                        | `0.75`       | 0 is stiff canvas, 1 drapes like silk.                                                               |
| `weight`   | `number`                                        | `0.1`        | Mass per area, kg/m². Heavier cloth holds liquid with less leaking.                                  |
| `wind`     | `Vector3`                                       | `(0, 0, 0)`  | m/s.                                                                                                 |
| `color`    | `number`                                        | `0x870b21`   | Color of the default sheen material.                                                                 |
| `material` | `MeshPhysicalNodeMaterial`                      | sheen fabric | Replaces the default. Its position and normal nodes are overwritten. Not disposed by the simulation. |

Particles are spaced `2.2 × particleRadius` apart, so a `w × h` cloth uses `(round(w / 2.2r) + 1) × (round(h / 2.2r) + 1)` particles.

Softness maps to bending compliance as:

```
bendCompliance = 10^(-1 + 5 × softness) × (columns / 30)^4 × (0.35 / weight)
```

Set at construction: stretch tolerance `0.06` (fixed), drag `0.18`, lift `0.02`, damping `0.1`. Change drag, lift, and damping on `clothSystem` after start.

## Cloth

### Properties

| Property      | Type                               | Access     | Description                                                               |
| ------------- | ---------------------------------- | ---------- | ------------------------------------------------------------------------- |
| `wind`        | `Vector3`                          | read       | Live wind, m/s. Mutate in place. Copied to `clothSystem.wind` every step. |
| `softness`    | `number`                           | read/write | Live. Setting it overwrites `clothSystem.bendCompliance`.                 |
| `mesh`        | `Mesh`                             | read       | Cloth surface in the scene. Throws before start.                          |
| `clothSystem` | [`ClothSystem`](./cloth-system.md) | read       | Underlying solver. Throws before start.                                   |

## Limitations

- Rectangles only. For other shapes, use [`createClothGraph`](./cloth-system.md) with any geometry.
- Pins are fixed in place; pinned particles can't be moved through this API.
- `weight` can't change after creation.
- Writing `clothSystem.wind` directly has no effect: the handle overwrites it every step.
- Liquid can leak through light cloth. Raise `weight` or lower `particleRadius`.
- Thin, sharp obstacle features can pass between cloth particles.
