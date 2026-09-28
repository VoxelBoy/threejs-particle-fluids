[Docs](README.md) › Smoke

# Smoke

`sim.addSmoke` releases smoke from a heated spot. The heat makes the air rise, and the smoke rides along with it, curling and spreading as it goes. It's drawn as lit, shadowed volumetric smoke.

```ts
const sim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.5, 0, -0.5), new Vector3(0.5, 1.9, 0.5)),
});
const smoke = sim.addSmoke({ source: new Vector3(0, 0, 0), radius: 0.17 });
```

[`examples/smoke.ts`](../examples/smoke.ts) is this scene. Run `npm run dev` and open `/examples/smoke.html`.

## How it works

The simulation fills the whole container with air. The source heats the air around it, and warm air rises, cools, and sinks again, stirring up swirls. The smoke you see is a cloud of tiny points called _tracers_ that drift with the air. Tracers don't push anything, so there can be many more of them than air particles, and they don't count toward the `particles` budget.

## Smoke needs a container

The air has to fill something, so smoke needs a [container](simulation.md#the-container). `addSmoke` throws without one. Smoke scenes also get two defaults you'd otherwise set yourself:

- **A lid.** The container is always closed, so warm air can't escape out the top.
- **Light gravity**, `(0, -1, 0)`, unless you pass `gravity`. Full gravity squashes the air to the bottom of the tank. A little gravity keeps the air evenly spread while warm air still rises.

## Options

| Option        | Default             | What it does                                                            |
| ------------- | ------------------- | ----------------------------------------------------------------------- |
| `source`      | center of the floor | Where smoke is released and the air is heated.                          |
| `radius`      | 0.15                | Radius of the source in metres.                                         |
| `rate`        | 4000                | Tracers released per second. More makes denser, finer smoke.            |
| `lifetime`    | 6                   | Seconds each tracer lives before it fades out.                          |
| `heat`        | 3                   | How hard heated air rises, as upward acceleration in m/s². 0 stops it.  |
| `cooling`     | 0.6                 | How fast the air cools, per second. Higher values make a shorter plume. |
| `density`     | 0.7                 | How opaque the smoke looks.                                             |
| `color`       | `0xd8dfe6`          | Color of lit smoke.                                                     |
| `shadowColor` | `0x3b4758`          | Color of smoke in its own shadow.                                       |

Smoke is lit from a fixed direction, from above and to the left, not by your scene's lights. `color` and `shadowColor` set how its lit and shaded sides look. To change the light direction, build the smoke renderer yourself; see [`GasVolumeRenderer`](advanced/gas-system.md#gasvolumerenderer).

## Live settings

`addSmoke` returns a `Smoke`:

```ts
smoke.heat = 5; // a stronger plume
smoke.density = 1.2; // thicker-looking smoke
smoke.rate = 2000; // less smoke
smoke.source.x += 0.1; // move the source; it's a Vector3 you change in place
smoke.emit(new Vector3(0.2, 0.5, 0)); // one extra puff anywhere
```

- `rate` can go down freely, but it can't go much above its starting value. The pool of tracers is sized from the starting `rate` and `lifetime`, and extra tracers are dropped once it's full. Start with the highest rate you plan to use.
- `emit` works once the simulation has started. For a burst, call it many times in one frame.
- `lifetime`, `color`, and `shadowColor` are fixed once added. Cooling can change after the first step through `smoke.gasSystem.cooling`. For other settings, see [The objects underneath](simulation.md#the-objects-underneath).

## Where smoke is drawn

Smoke is drawn in the bottom 90% of the container. Tracers that drift into the top tenth are removed, so smoke doesn't pile up against the lid. Make the container a little taller than the plume you want.

The smoke is hidden behind solid objects in your scene and shows in front of them, so you can put a vent mesh or other props inside the container.

## Limits

- **Gas and liquid can't be simulated together.** Air is about 800 times lighter than water, and particle solvers like this one can't handle that at the interface: in our tests, air either shot through the water as loose particles or got stuck at the bottom, never forming believable bubbles. So smoke can't share a simulation with liquids, soft bodies, or cloth, and the first `step()` throws if you add both. For smoke and water in the same scene, use two simulations, as below.
- **One smoke source per simulation.** A second `addSmoke` throws on the first step.
- **Obstacles work, with a jolt at the start.** The air flows around them once it's running. But the air first fills the whole container, including the inside of obstacles, and the first step pushes that air out, so there's a short burst of motion. The air inside an obstacle still counts toward the `particles` budget. Keep obstacles in a smoke scene small.

## Smoke and water in one scene

Give each its own `Simulation`, with its own container, and step both every frame:

```ts
const water = new Simulation({ renderer, scene, camera, container: tank });
water.addFluid({ box: waterBox });

const air = new Simulation({ renderer, scene, camera, container: smokeBox });
air.addSmoke();

async function frame() {
  await water.step();
  await air.step();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

- Both share the renderer, the scene, and the camera.
- Each has its own `particles` budget, 20,000 by default, so the cost adds up. Lower `particles` on one or both if it's slow.
- Each has its own gravity. The smoke simulation's light gravity doesn't affect the water.
- The two don't affect each other. Smoke passes straight through the water, and the water doesn't feel the air. Keep the containers apart, or put the smoke somewhere the water can't reach.

---

Previous: [Cloth](cloth.md) · Next: [Troubleshooting](troubleshooting.md)
