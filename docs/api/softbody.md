[Docs](../README.md) › [API](../README.md#api-reference) › Softbody

# Softbody

Deformable body added with [`Simulation.addSoftbody`](./simulation.md#addsoftbodyoptions). The source mesh is voxelized into particles, hidden, and replaced by a skinned copy that follows them.

```ts
const duck = sim.addSoftbody({ mesh: duckMesh, density: 400, softness: 0.3 });
```

## SoftbodyOptions

| Option     | Type     | Default  | Description                                                                           |
| ---------- | -------- | -------- | ------------------------------------------------------------------------------------- |
| `mesh`     | `Mesh`   | required | Closed mesh at its current world transform. Only its largest connected piece is kept. |
| `softness` | `number` | `0.3`    | 0 is firm rubber, 1 is loose jelly.                                                   |
| `density`  | `number` | `500`    | kg/m³. Water is 1000: lower floats, higher sinks.                                     |

The copy keeps the source material's `color`, `map`, `roughness`, and other standard material fields when the source uses `MeshStandardMaterial` or a subclass. It casts and receives shadows.

Softness maps to shape-matching compliance as:

```
compliance = 10^(-6 + 3 × softness) × (particleCount / 200)
```

## Softbody

### Properties

| Property         | Type                                                | Access     | Description                                                       |
| ---------------- | --------------------------------------------------- | ---------- | ----------------------------------------------------------------- |
| `particleCount`  | `number`                                            | read       | Particles in this body. `0` until the simulation starts.          |
| `softness`       | `number`                                            | read/write | Live.                                                             |
| `density`        | `number`                                            | read       | Fixed at creation.                                                |
| `source`         | `Mesh`                                              | read       | The mesh passed in. Hidden while the simulation runs.             |
| `mesh`           | [`SoftbodyMesh`](./softbody-system.md#softbodymesh) | read       | Deforming mesh in the scene. Throws before start.                 |
| `softbodySystem` | [`SoftbodySystem`](./softbody-system.md)            | read       | Shared by every soft body in the simulation. Throws before start. |
| `bodyIndex`      | `number`                                            | read       | This body's index in `softbodySystem`. `-1` before start.         |

## Limitations

- Mesh must be closed. Thin parts narrower than about `2 × particleRadius` are lost.
- Detail is limited by the simulation's single particle radius. Small bodies next to a large fluid volume get few particles.
- `density` can't change after creation.
- Uses local shape matching. For rigid objects, use [`SoftbodySystem`](./softbody-system.md) directly with global shape matching and compliance `0`.
- Stiff bodies colliding fast at low substep counts can stick together. Raise `substeps`.
- Voxelizing large meshes blocks the main thread during `start()`.
