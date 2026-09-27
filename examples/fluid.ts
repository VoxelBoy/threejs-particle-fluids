// A minimal scene: a block of water collapsing in a box.
// Run `npm run dev` and open /examples/fluid.html.
import {
  ACESFilmicToneMapping,
  Box3,
  BoxGeometry,
  DirectionalLight,
  EdgesGeometry,
  LineBasicMaterial,
  LineSegments,
  PerspectiveCamera,
  Scene,
  Vector3,
} from 'three';
import { PMREMGenerator } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  FluidSurfaceRenderer,
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../src/index.js';

const renderer = await createParticleRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = ACESFilmicToneMapping;
document.body.append(renderer.domElement);

const scene = new Scene();
scene.environment = new PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const sun = new DirectionalLight(0xffffff, 2);
sun.position.set(1, 3, 2);
scene.add(sun);
const camera = new PerspectiveCamera(40, innerWidth / innerHeight, 0.01, 50);
camera.position.set(1.6, 1.4, 2.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.3, 0);

// 1. Particles: a block of water in one corner of a 1 m × 0.6 m tank.
const radius = 0.012;
const water: ParticleInit[] = [];
for (let x = -0.48; x < -0.1; x += radius * 2)
  for (let y = radius; y < 0.5; y += radius * 2)
    for (let z = -0.28; z < 0.28; z += radius * 2) water.push({ position: [x, y, z] });
const particles = new ParticleSystem(renderer, water.length, radius);
particles.uploadParticles(water);

// 2. Physics: the fluid, the walls it collides with, and the solver loop.
const fluid = new FluidSystem(particles, { viscosity: 0.02, surfaceTension: 0.1, vorticity: 0.02 });
const walls = new PrimitiveSet(particles);
const outline = new LineSegments(
  new EdgesGeometry(new BoxGeometry(1, 0.6, 0.6).translate(0, 0.3, 0)),
  new LineBasicMaterial({ color: 0x4d6272 }),
);
scene.add(outline);
walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
walls.addPlane(new Vector3(1, 0, 0), new Vector3(-0.5, 0, 0));
walls.addPlane(new Vector3(-1, 0, 0), new Vector3(0.5, 0, 0));
walls.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -0.3));
walls.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, 0.3));
const loop = new SimLoop(particles, { substeps: 3, materials: [fluid], colliders: [walls] });

// 3. Rendering: a ray-marched liquid surface.
const surface = new FluidSurfaceRenderer(fluid, {
  renderer,
  scene,
  camera,
  bounds: new Box3(new Vector3(-0.53, -0.02, -0.33), new Vector3(0.53, 0.8, 0.33)),
  colliders: [walls],
  appearance: { color: 0x3a9fcf },
});
scene.add(surface.mesh);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

async function frame(): Promise<void> {
  await loop.step(1 / 60);
  await surface.update();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
