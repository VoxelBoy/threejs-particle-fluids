# Three.js Particle Fluids

Interactive particle simulations built with **Three.js, WebGPU, and the Three.js Shading Language (TSL)**. Eight curated presets explore fluid, form, and motion, with live controls and GPU performance diagnostics.

![Tidal chamber](public/previews/cover.png)

## Run locally

Install **Node.js 22.12 or newer**, then run:

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. No API keys, accounts, downloaded assets, or additional services are needed. The demo makes no external requests.

A desktop browser and GPU with WebGPU support are required. Enable hardware acceleration and serve the app on localhost or HTTPS. The solver requires 1,024 compute invocations per workgroup, a workgroup X size of 1,024, and 10 storage buffers per shader stage. Adapters with lower limits cannot run it. There is no WebGL simulation fallback.

## The collection

| Preset              | Explore                                                             |
| ------------------- | ------------------------------------------------------------------- |
| **Tidal chamber**   | A collapsing column of water, an obstacle, and changing currents    |
| **Crown impact**    | A falling drop, a pool, and the resulting splash                    |
| **Liquid marble**   | Surface tension and angular motion in zero gravity                  |
| **Amber cascade**   | Viscous liquid flowing over ceramic steps                           |
| **Floating forms**  | Two-way interaction between fluid and deformable objects            |
| **Elastic studies** | Shape matching, compliance, and collisions                          |
| **Silk in motion**  | Cloth stretch, bending, aerodynamic drag, and lift                  |
| **Vortex plume**    | Passive tracer particles carried by a rotating fluid velocity field |

Each preset has a focused parameter panel. Sliders apply immediately; controls marked **↻** restart the experiment when released. Settings are remembered per preset during the current session. **Reset all** restores that preset's defaults.

- **Space** pauses or plays; **R** restarts. Shortcuts do not intercept focused controls.
- **Drag** to orbit, **right-drag** to pan, and **scroll or pinch** to zoom.
- **Surface / Particles** reveals the simulation beneath the rendering. In Vortex plume, Particles also reveals the carrier fluid.
- **Balanced / High fidelity** changes particle density and render resolution. Changing quality restarts the scene.
- **Loop experiment** automatically replays a study after its duration. Disable it to continue experimenting with the settled state.
- **Disturb** applies a small impulse in fluid presets.
- **Save image** downloads a PNG of the current viewport.
- On smaller screens, open **Parameters** to access the controls.

The selected preset is reflected in the URL, for example `?preset=liquid-marble`. Reduced-motion preferences start the simulation paused. Background tabs suspend simulation time.

The upper-left overlay reports rendered FPS, frame interval, active particle count, simulation time, and solver settings. Gas counts include both carrier particles and live tracers. Frame time measures the interval between rendered frames, including GPU completion; it is not an isolated GPU kernel benchmark. Under sustained load, the simulation limits catch-up work and may advance more slowly than real time.

## Project layout

One npm project, with engine code and demo code kept separate:

```text
src/
  core/       Particle buffers, integration, constraints, contacts, colliders
  fluids/     Fluid solver and screen-space surface rendering
  softbody/   Soft and rigid body solvers, voxelization, mesh skinning
  cloth/      Stretch, bending, tether, and aerodynamic constraints
  gas/        Passive tracer advection and billboard rendering
  render/     GPU-driven particle rendering
  sdf/        CPU mesh-to-SDF baking and binary utilities
  index.ts    Engine exports
demo/
  presets/    Eight presets and their controls
  runtime/    Scene lifecycle, lighting, camera, and frame pacing
  main.ts     Gallery and parameter interface
  style.css   Responsive interface styles
public/       Local icons and rendered preset previews
tests/        Numerical tests and GPU benchmark harness
```

The engine is source code in this repository, not a published npm package. Import from `src/index.ts` or an individual module when integrating it. Start with `demo/presets/liquids.ts` for a complete example of particle allocation, a fluid material, colliders, a simulation loop, and surface rendering. The demo runtime handles device initialization and resource disposal.

The public API is experimental and may change. The soft-body solver and screen-space fluid renderer have numerical and rendering limitations; this project is intended for interactive visualization, not engineering analysis.

## Development

| Command                  | Purpose                                                  |
| ------------------------ | -------------------------------------------------------- |
| `npm run dev`            | Start the local demo                                     |
| `npm run build`          | Type-check and build the demo into `dist/`               |
| `npm run preview`        | Serve the production build locally                       |
| `npm run typecheck`      | Check engine, demo, and test types                       |
| `npm run lint`           | Lint engine, demo, and scripts                           |
| `npm test`               | Run CPU tests                                            |
| `npm run test:gpu`       | Run numerical WebGPU tests using installed Google Chrome |
| `npm run test:demo`      | Run browser checks for presets and application controls  |
| `npm run test:perf`      | Run local GPU benchmarks and create a report             |
| `npm run bench:baseline` | Record a local benchmark baseline                        |

GPU checks require a compatible local GPU. The test runner passes Chrome's `--enable-unsafe-webgpu` flag for local testing. Performance results are specific to the browser, adapter, driver, and settings; they are not cross-device guarantees. Benchmark output remains local and is ignored by Git.

A small number of numerical checks are explicitly skipped with explanations in their test files. Retained cloth-compliance and buoyancy checks have shown intermittent failures on the tested adapter; their assertions have not been relaxed.

## License

[MIT](LICENSE). Dependency notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
