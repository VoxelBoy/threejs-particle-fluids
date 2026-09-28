// A velvet curtain pushed around by a swinging ball.
// Run `npm run dev` and open /examples/cloth.html.
import { Mesh, MeshStandardMaterial, SphereGeometry, Vector3 } from 'three';
import { Simulation } from '../src/index.js';
import { setup } from './setup.js';

const { renderer, scene, camera, run } = await setup({
  camera: new Vector3(1.8, 1.2, 2.4),
  target: new Vector3(0, 0.8, 0),
});

const ball = new Mesh(
  new SphereGeometry(0.2, 48, 32),
  new MeshStandardMaterial({ color: 0xe3e7ec, metalness: 1, roughness: 0.1 }),
);
scene.add(ball);

// A 1.2 m square of cloth at this radius takes about 3,100 particles.
const sim = new Simulation({ renderer, scene, camera, particleRadius: 0.01, maxParticles: 4000 });
const curtain = sim.addCloth({
  width: 1.2,
  height: 1.2,
  position: new Vector3(0, 1, 0),
  pin: 'top',
});
sim.addSphere({ radius: 0.2, follow: ball });
sim.addFloor();

run((time) => {
  // Swing the ball through the curtain; the sphere collider follows it.
  ball.position.set(0, 0.7, Math.cos(time * 0.8) * 0.8);
  curtain.wind.set(0, 0, 1 + Math.sin(time));
  return sim.step();
});
