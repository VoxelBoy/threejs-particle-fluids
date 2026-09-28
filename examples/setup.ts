// Shared scaffolding for the examples: a renderer, a lit scene, a camera
// with orbit controls, and a frame loop. The simulation code lives in each
// example file.
import {
  ACESFilmicToneMapping,
  Color,
  DirectionalLight,
  HemisphereLight,
  PerspectiveCamera,
  Scene,
  Vector3,
} from 'three';
import { PMREMGenerator } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { createParticleRenderer } from '../src/index.js';

export async function setup(options: { camera: Vector3; target: Vector3 }) {
  const renderer = await createParticleRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = ACESFilmicToneMapping;
  document.body.append(renderer.domElement);

  const scene = new Scene();
  scene.background = new Color(0x10161e);
  scene.environment = new PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.6;
  const sun = new DirectionalLight(0xffffff, 2);
  sun.position.set(1, 3, 2);
  scene.add(sun, new HemisphereLight(0xdde8ff, 0x202830, 0.6));

  const camera = new PerspectiveCamera(40, innerWidth / innerHeight, 0.01, 50);
  camera.position.copy(options.camera);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(options.target);

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  /** Call `frame` every animation frame, then render. */
  const run = (frame: (time: number) => Promise<void> | void) => {
    const tick = async (time: number) => {
      await frame(time / 1000);
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  return { renderer, scene, camera, run };
}
