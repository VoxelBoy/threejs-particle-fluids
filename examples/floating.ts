// Soft bodies dropped into a tank of water: a light ball floats, a heavy
// cube sinks.
// Run `npm run dev` and open /examples/floating.html.
import { Box3, BoxGeometry, Mesh, MeshStandardMaterial, SphereGeometry, Vector3 } from 'three';
import { Simulation } from '../src/index.js';
import { setup } from './setup.js';

const { renderer, scene, camera, run } = await setup({
  camera: new Vector3(1.6, 1.3, 2.1),
  target: new Vector3(0, 0.3, 0),
});

const ball = new Mesh(
  new SphereGeometry(0.12, 32, 24),
  new MeshStandardMaterial({ color: 0xf2c14e }),
);
ball.position.set(-0.2, 0.65, 0);
const cube = new Mesh(
  new BoxGeometry(0.16, 0.16, 0.16),
  new MeshStandardMaterial({ color: 0xd9534f }),
);
cube.position.set(0.2, 0.7, 0);
scene.add(ball, cube);

const sim = new Simulation({
  renderer,
  scene,
  camera,
  container: new Box3(new Vector3(-0.4, 0, -0.3), new Vector3(0.4, 0.9, 0.3)),
  particles: 5000, // shared by the water and both bodies
});
sim.addFluid({ box: new Box3(new Vector3(-0.4, 0, -0.3), new Vector3(0.4, 0.3, 0.3)) });
sim.addSoftbody({ mesh: ball, density: 400, softness: 0.4 });
sim.addSoftbody({ mesh: cube, density: 2000, softness: 0.1 });

run(() => sim.step());
