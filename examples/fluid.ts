// A block of water collapsing in a box.
// Run `npm run dev` and open /examples/fluid.html.
import { Box3, Vector3 } from 'three';
import { Simulation } from '../src/index.js';
import { setup } from './setup.js';

const { renderer, scene, camera, run } = await setup({
  camera: new Vector3(1.6, 1.4, 2.2),
  target: new Vector3(0, 0.3, 0),
});

// A 1 m × 0.6 m tank, with water filling one corner.
const sim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(0.5, 0.8, 0.3)),
  particles: 5000,
});
sim.addFluid({
  box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(-0.1, 0.5, 0.3)),
  color: 0x3a9fcf,
});

run(() => sim.step());
