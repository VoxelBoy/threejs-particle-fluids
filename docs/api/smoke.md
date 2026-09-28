[Docs](../README.md) › [API](../README.md#api-reference) › Smoke

# Smoke

Heated smoke source added with [`Simulation.addSmoke`](./simulation.md#addsmokeoptions). Air particles fill the whole container; the source heats them and releases passive tracers that are rendered as a lit volume.

```ts
const smokeSim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.5, 0, -0.5), new Vector3(0.5, 1.9, 0.5)),
  particleRadius: 0.035,
  maxParticles: 5000,
});
const smoke = smokeSim.addSmoke({ radius: 0.17 });
```

## SmokeOptions

| Option        | Type      | Default                       | Description                                           |
| ------------- | --------- | ----------------------------- | ----------------------------------------------------- |
| `source`      | `Vector3` | center of the container floor | Where tracers are released and air is heated. Copied. |
| `radius`      | `number`  | `0.15`                        | Source radius, m.                                     |
| `rate`        | `number`  | `4000`                        | Tracers released per second.                          |
| `maxRate`     | `number`  | `2 × rate`                    | Highest `rate` allowed later. Sizes the tracer pool.  |
| `lifetime`    | `number`  | `6`                           | Seconds each tracer lives.                            |
| `heat`        | `number`  | `3`                           | Upward acceleration of heated air, m/s².              |
| `cooling`     | `number`  | `0.6`                         | Heat decay rate, 1/s.                                 |
| `opacity`     | `number`  | `0.7`                         | Volume density scale.                                 |
| `color`       | `number`  | `0xd8dfe6`                    | Lit smoke color.                                      |
| `shadowColor` | `number`  | `0x3b4758`                    | Self-shadowed smoke color.                            |

Tracer pool size: `ceil(maxRate × lifetime × 1.1) + 1000`.

Adding smoke sets `gravity` to `(0, -1, 0)` unless `gravity` was passed to the constructor.

## Smoke

### Properties

| Property    | Type                           | Access     | Description                                                                      |
| ----------- | ------------------------------ | ---------- | -------------------------------------------------------------------------------- |
| `rate`      | `number`                       | read/write | Tracers per second. Values above `maxRate` are capped, with one console warning. |
| `heat`      | `number`                       | read/write | m/s². `0` stops the air rising. Throws `Smoke.heat: must be finite` otherwise.   |
| `cooling`   | `number`                       | read/write | 1/s. Throws `Smoke.cooling: must be finite and ≥ 0` otherwise.                   |
| `opacity`   | `number`                       | read/write |                                                                                  |
| `source`    | `Vector3`                      | read       | Live source position. Mutate in place to move it.                                |
| `gasSystem` | [`GasSystem`](./gas-system.md) | read       | Underlying system. Throws before start.                                          |

### Methods

#### `emit(position)`

```ts
emit(position: Vector3 | readonly [number, number, number]): boolean
```

Releases one extra tracer. Returns `false` when the pool is full. Throws before start.

## Limitations

- Requires `container`. The container is always closed.
- Can't share a `Simulation` with fluid, soft bodies, or cloth, and only one source per simulation. Use a separate `Simulation` for smoke next to water.
- Air fills the whole container, so the particle count is `containerVolume / (2r)³` regardless of how much smoke is visible.
- Tracers are removed in the top 10% of the container, and the volume is only rendered below it. The rendered density fades out near the side walls, the floor, and the top of that volume (the default [`edgeFade`](./gas-system.md#gasvolumerendereroptions)).
- `radius`, `lifetime`, `maxRate`, `color`, and `shadowColor` are fixed at creation.
- Volume resolution is up to 128 cells on the longest axis with 80 ray-march steps; not configurable through this API. Use [`GasVolumeRenderer`](./gas-system.md#gasvolumerenderer) directly.
