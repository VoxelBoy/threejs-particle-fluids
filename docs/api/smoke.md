[Docs](../README.md) › [API](../README.md#api-reference) › Smoke

# Smoke

Smoke rising from a heated source, added with [`Simulation.addSmoke`](./simulation.md#addsmokeoptions). Particles of air fill the whole container, and the source heats the air above it so it rises. The smoke you see is made of tiny markers, called tracers, that the moving air carries along. Tracers don't push on anything.

```ts
const smokeSim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.5, 0, -0.5), new Vector3(0.5, 1.9, 0.5)),
  particles: 5000,
});
const smoke = smokeSim.addSmoke({ radius: 0.17 });
```

## SmokeOptions

| Option        | Type      | Default                       | Description                                                               |
| ------------- | --------- | ----------------------------- | ------------------------------------------------------------------------- |
| `source`      | `Vector3` | center of the container floor | Where smoke is released and the air is heated.                            |
| `radius`      | `number`  | `0.15`                        | Radius of the source, m.                                                  |
| `rate`        | `number`  | `4000`                        | Tracers released per second. More gives thicker, finer smoke.             |
| `maxRate`     | `number`  | `2 × rate`                    | The highest `rate` you plan to set later.                                 |
| `lifetime`    | `number`  | `6`                           | Seconds each tracer lasts.                                                |
| `heat`        | `number`  | `3`                           | How hard heated air rises, as an upward acceleration in m/s².             |
| `cooling`     | `number`  | `0.6`                         | How fast the air cools, per second. Faster cooling stops the rise sooner. |
| `opacity`     | `number`  | `0.7`                         | How thick the smoke looks.                                                |
| `color`       | `number`  | `0xd8dfe6`                    | Color of smoke facing the light.                                          |
| `shadowColor` | `number`  | `0x3b4758`                    | Color of smoke in its own shadow.                                         |

Adding smoke changes the simulation's default `gravity` to `(0, -1, 0)`, which suits drifting smoke. Pass `gravity` to the `Simulation` to use a different value.

## Smoke

### Properties

| Property        | Type                           | Access     | Description                                                                        |
| --------------- | ------------------------------ | ---------- | ---------------------------------------------------------------------------------- |
| `particleCount` | `number`                       | read       | Air particles filling the container. `0` until the simulation starts.              |
| `rate`          | `number`                       | read/write | Tracers per second. Values above `maxRate` are capped, and the console warns once. |
| `heat`          | `number`                       | read/write | `0` stops the air rising. Throws if it isn't a finite number.                      |
| `cooling`       | `number`                       | read/write | Throws if it's negative or not a finite number.                                    |
| `opacity`       | `number`                       | read/write | How thick the smoke looks.                                                         |
| `source`        | `Vector3`                      | read       | The source position. Change it in place to move the source.                        |
| `gasSystem`     | [`GasSystem`](./gas-system.md) | read       | The system underneath. Available after `start()`.                                  |

### Methods

#### `emit(position)`

```ts
emit(position: Vector3 | readonly [number, number, number]): boolean
```

Releases one extra tracer at `position`, on top of the steady `rate`. It returns `false` if every tracer is still in use. You can only call it after `start()`.

## Limitations

- Smoke needs a `container`, and the container always gets a lid.
- Smoke can't share a simulation with liquid, soft bodies, or cloth, and each simulation has one source. For smoke next to water, use a second `Simulation`.
- Air fills the whole container, so a bigger container uses more particles, however little smoke is in it.
- Tracers disappear in the top tenth of the container. Make the container taller if the smoke stops short.
- Smoke fades out near the walls and floor of the container.
- You can't change `radius`, `lifetime`, `maxRate`, `color`, or `shadowColor` after adding the smoke.
- The smoke's render quality is fixed. For control over it, use [`GasVolumeRenderer`](./gas-system.md#gasvolumerenderer) directly.
