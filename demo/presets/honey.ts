import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  CylinderGeometry,
  Float32BufferAttribute,
  Mesh,
  MeshStandardMaterial,
  Quaternion,
  TorusGeometry,
  Vector3,
} from 'three';
import { Fn, If, instanceIndex, instancedArray, int, uniform, vec3, vec4 } from 'three/tsl';
import {
  FluidSystem,
  ParticleSystem,
  SDFCollider,
  SimLoop,
  ViscositySolver,
  decodeSdfBinary,
  type SDFData,
} from '../../src/index.js';
import { basin, material } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { liquidVisual } from './liquids.js';
import { scaledSubsteps, tank } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export const BUNNY_YAW = 2.16;
const MESH_MAGIC = 0x4e554242;
/** Height of the circling nozzle above the floor, and the speed of the stream leaving it. */
const NOZZLE_HEIGHT = 0.75,
  POUR_SPEED = 1.6;

/** Decode the quantized Stanford bunny written by `npm run assets:honey`. */
export async function loadBunny(): Promise<{ mesh: Mesh; sdf: SDFData; aim: Vector3 }> {
  const base = `${import.meta.env.BASE_URL}models/honey/`;
  const [meshBytes, sdfBytes] = await Promise.all(
    ['bunny.mesh.bin', 'bunny.sdf.bin'].map(async (file) => {
      const response = await fetch(base + file);
      if (!response.ok) throw new Error(`Could not load ${file}.`);
      return response.arrayBuffer();
    }),
  );
  const view = new DataView(meshBytes!);
  if (view.getUint32(0, true) !== MESH_MAGIC) throw new Error('The bunny mesh is invalid.');
  const vertices = view.getUint32(4, true),
    indexCount = view.getUint32(8, true);
  const min = [0, 1, 2].map((k) => view.getFloat32(12 + k * 4, true));
  const max = [0, 1, 2].map((k) => view.getFloat32(24 + k * 4, true));
  const quantized = new Uint16Array(meshBytes!, 36, vertices * 3);
  const positions = new Float32Array(vertices * 3);
  for (let i = 0; i < positions.length; i++)
    positions[i] = min[i % 3]! + (quantized[i]! / 65535) * (max[i % 3]! - min[i % 3]!);
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setIndex(
    new BufferAttribute(new Uint16Array(meshBytes!, 36 + vertices * 6, indexCount), 1),
  );
  geometry.computeVertexNormals();
  const mesh = new Mesh(
    geometry,
    new MeshStandardMaterial({ color: 0xf4f0ea, roughness: 0.32, metalness: 0 }),
  );
  mesh.rotation.y = BUNNY_YAW;
  mesh.castShadow = mesh.receiveShadow = true;
  // Aim the nozzle's circle at the head and upper back, below the ears.
  const top = max[1]!;
  const aim = new Vector3();
  let n = 0;
  for (let i = 0; i < positions.length; i += 3)
    if (positions[i + 1]! > top * 0.5 && positions[i + 1]! < top * 0.78) {
      aim.x += positions[i]!;
      aim.z += positions[i + 2]!;
      n++;
    }
  aim.divideScalar(Math.max(1, n)).applyAxisAngle(new Vector3(0, 1, 0), BUNNY_YAW);
  return { mesh, sdf: decodeSdfBinary(sdfBytes!), aim };
}

export async function buildHoney(ctx: BuildContext, values: Values): Promise<Experiment> {
  // A fixed ~34 L of honey, split into the requested number of particles.
  const count = ctx.particles;
  const spacing = Math.cbrt(0.0343 / count);
  const radius = spacing / 2;
  // Every particle waits pinned in a sparse grid far below the floor until the
  // nozzle releases it.
  const particles = new ParticleSystem(ctx.renderer, count, radius);
  particles.uploadParticles(
    Array.from({ length: count }, (_, k) => ({
      position: [(k % 150) * 0.25 - 18.75, -30, Math.floor(k / 150) * 0.25 - 12.5] as const,
    })),
  );
  const fluid = new FluidSystem(particles, {
    viscosity: 0.03,
    surfaceTension: values['tension']!,
  });
  particles.setInvMass(fluid.range, 0);
  const viscosity = new ViscositySolver(fluid, {
    viscosity: values['viscosity']!,
    // Finer particles need more sweeps to diffuse across the same distance.
    iterations: Math.round(16 * Math.cbrt(count / 10000)),
  });

  // Honey sticks to what it touches: high friction everywhere.
  const walls = tank(particles, 0.8, 0.55, { muS: 1.2, muK: 1 });
  const bunny = await loadBunny();
  const bunnyCollider = new SDFCollider(particles, bunny.sdf, {
    rotation: new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), BUNNY_YAW),
    muS: values['friction']!,
    muK: values['friction']! * 0.85,
  });
  const substeps = scaledSubsteps(3, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    colliders: [walls, bunnyCollider],
    materials: [fluid, viscosity],
  });

  const nozzle = new Mesh(
    new CylinderGeometry(0.03, 0.03, 0.14, 48, 1, true),
    material(0xb2a896, 0.23, 0.8),
  );
  const lip = new Mesh(new TorusGeometry(0.03, 0.006, 12, 64), nozzle.material);
  lip.rotation.x = Math.PI / 2;
  lip.position.y = -0.07;
  nozzle.add(lip);

  const visual = liquidVisual(ctx, fluid, {
    bounds: new Box3(new Vector3(-0.83, -0.02, -0.58), new Vector3(0.83, 1.15, 0.58)),
    colliders: [walls, bunnyCollider],
    motionStretch: 0.025,
    appearance: {
      color: 0xf07a0c,
      attenuationDistance: 0.06,
      scattering: 0.4,
      ior: 1.49,
      roughness: 0.12,
    },
  });

  // Release whole layers of a square lattice clipped to the nozzle disc, one
  // particle spacing apart, so new particles never overlap the stream.
  const disc: number[] = [];
  const reach = Math.floor(0.02 / spacing);
  for (let i = -reach; i <= reach; i++)
    for (let j = -reach; j <= reach; j++)
      if ((i * i + j * j) * spacing * spacing <= 0.02 ** 2)
        disc.push(i * spacing, 0, j * spacing, 0);
  const discOffsets = instancedArray(new Float32Array(disc), 'vec4');
  const perLayer = disc.length / 4;
  const source = uniform(new Vector3());
  const sweep = uniform(new Vector3());
  const emitStart = uniform(0, 'float');
  const emitLayers = uniform(0, 'float');
  const newestDrop = uniform(0, 'float');
  const speed = uniform(0.5, 'float');
  const emit = Fn(() => {
    const i: Any = instanceIndex;
    const slot: Any = int(i).sub(emitStart.toInt()).toVar();
    const layers: Any = emitLayers.toInt();
    If(slot.greaterThanEqual(0).and(slot.lessThan(layers.mul(perLayer))), () => {
      const layer: Any = slot.div(perLayer);
      const offset: Any = discOffsets.element(slot.mod(perLayer)).xyz;
      const drop: Any = newestDrop.add(layers.sub(1).sub(layer).toFloat().mul(spacing));
      const position: Any = vec4(source.add(offset).sub(vec3(0, drop, 0)), 0);
      particles.positions.element(i).assign(position);
      particles.predictedPositions.element(i).assign(position);
      particles.velocities.element(i).assign(vec4(sweep.x, speed.negate(), sweep.z, 0));
      particles.invMass.element(i).assign(1 / fluid.mass);
      // Waiting particles have no neighbors, so their density is stale.
      fluid.density.element(i).assign(fluid.restDensity);
    });
  })().compute(count);
  let released = 0,
    travelled = 0;

  return {
    particles,
    loop,
    objects: [basin(1.6, 1.1), bunny.mesh, nozzle, visual.surface.mesh, visual.dots],
    particleCount: count,
    substeps,
    iterations,
    async update(dt, time) {
      // The nozzle circles over the bunny's back and head; the stream leaves
      // with the nozzle's sideways velocity.
      const angle = time * 0.9;
      const x = bunny.aim.x + Math.cos(angle) * 0.07,
        z = bunny.aim.z + Math.sin(angle) * 0.07;
      // Release 11 cm up inside the 14 cm nozzle so the start of the flow is hidden.
      source.value.set(x, NOZZLE_HEIGHT + 0.11, z);
      sweep.value.set(-Math.sin(angle) * 0.063, 0, Math.cos(angle) * 0.063);
      nozzle.position.set(x, NOZZLE_HEIGHT + 0.07, z);
      speed.value = POUR_SPEED;
      travelled += speed.value * dt;
      const layers = Math.min(
        Math.floor(travelled / spacing),
        Math.floor((count - released) / perLayer),
      );
      travelled -= Math.floor(travelled / spacing) * spacing;
      if (layers <= 0) return;
      emitStart.value = released;
      emitLayers.value = layers;
      newestDrop.value = travelled;
      released += layers * perLayer;
      await ctx.renderer.computeAsync(emit);
    },
    prepareRender: () => visual.update(),
    setReflections: (enabled) => (visual.surface.reflections = enabled),
    setParticleView: (enabled) => visual.setParticleView(enabled),
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'viscosity') viscosity.viscosity = value;
      if (key === 'tension') fluid.surfaceTension = value;
      if (key === 'friction') {
        bunnyCollider.muS = value;
        bunnyCollider.muK = value * 0.85;
      }
    },
    dispose() {
      visual.surface.dispose();
      bunnyCollider.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
