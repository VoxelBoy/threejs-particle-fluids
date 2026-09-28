// Smoke rising from a heated vent.
// Run `npm run dev` and open /examples/smoke.html.
import { Box3, Vector3 } from 'three';
import { Simulation } from '../src/index.js';
import { setup } from './setup.js';

const { renderer, scene, camera, run } = await setup({
  camera: new Vector3(1.9, 1.4, 2.6),
  target: new Vector3(0, 0.8, 0),
});

// The air fills the container; the source heats it and releases smoke.
const sim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.5, 0, -0.5), new Vector3(0.5, 1.9, 0.5)),
});
sim.addSmoke({ source: new Vector3(0, 0, 0), radius: 0.17 });

run(() => sim.step());
