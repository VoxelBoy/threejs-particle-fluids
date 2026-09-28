[Docs](../README.md) › [API](../README.md#api-reference) › Softbody

# Softbody

A soft body added with [`Simulation.addSoftbody`](./simulation.md#addsoftbodyoptions). The simulation fills your mesh with particles, hides it, and draws a copy that bends as the particles move. See [Using your own meshes](../guide.md#using-your-own-meshes) for how the filling works.

```ts
const duck = sim.addSoftbody({ mesh: duckMesh, density: 400, softness: 0.3 });
```

## SoftbodyOptions

| Option     | Type     | Default  | Description                                                                          |
| ---------- | -------- | -------- | ------------------------------------------------------------------------------------ |
| `mesh`     | `Mesh`   | required | A closed mesh, where it sits in the scene. Only its largest connected piece is used. |
| `softness` | `number` | `0.3`    | From 0 for firm rubber to 1 for loose jelly.                                         |
| `density`  | `number` | `500`    | kg/m³. Water is 1000, so lower values float and higher values sink.                  |

The copy uses your mesh's material settings, such as `color`, `map`, and `roughness`, when the material is a `MeshStandardMaterial` or a subclass of it. It casts and receives shadows.

A body with more particles needs different stiffness settings to feel equally soft. `softness` takes care of that, so a body feels about the same at any particle count. The [formula](./softbody-system.md#compliance) is on the `SoftbodySystem` page.

## Softbody

### Properties

| Property         | Type                                                | Access     | Description                                                                        |
| ---------------- | --------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------- |
| `particleCount`  | `number`                                            | read       | Particles in this body. `0` until the simulation starts.                           |
| `softness`       | `number`                                            | read/write | Can be changed while the simulation runs.                                          |
| `density`        | `number`                                            | read       | Can't be changed after `addSoftbody`.                                              |
| `source`         | `Mesh`                                              | read       | The mesh you passed in. It's hidden while the simulation runs.                     |
| `mesh`           | [`SoftbodyMesh`](./softbody-system.md#softbodymesh) | read       | The bending copy drawn in your scene. Available after `start()`.                   |
| `softbodySystem` | [`SoftbodySystem`](./softbody-system.md)            | read       | The solver shared by every soft body in the simulation. Available after `start()`. |
| `bodyIndex`      | `number`                                            | read       | This body's index in `softbodySystem`. `-1` until the simulation starts.           |

## Limitations

- The mesh must be closed. Parts thinner than about one particle width are lost.
- All particles in a simulation share one size, so a small body in a lot of liquid gets few particles. The console warns you when a body has fewer than 100.
- You can't change `density` after adding the body.
- Soft bodies made this way always bend a little. For objects that shouldn't bend at all, use [`SoftbodySystem`](./softbody-system.md) directly with global shape matching and a compliance of `0`.
- Stiff bodies that hit each other fast can stick together. Raising `substeps` helps.
- Filling a detailed mesh with particles takes time during `start()` and blocks the page while it runs.
