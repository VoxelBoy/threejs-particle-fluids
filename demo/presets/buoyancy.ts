import {
  CanvasTexture,
  Mesh,
  RepeatWrapping,
  SRGBColorSpace,
  type BufferGeometry,
  type MeshStandardMaterial,
  type Texture,
} from 'three';
import type { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import {
  Fn,
  Loop,
  cross,
  float,
  instanceIndex,
  instancedArray,
  uint,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { HashGrid, ParticleSystem, SimLoop, createXpbdUniforms } from '../../src/core/index.js';
import { FluidSystem } from '../../src/fluids/index.js';
import {
  SoftbodyMesh,
  SoftbodySystem,
  voxelize,
  type SoftbodyDef,
} from '../../src/softbody/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { basin } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { liquidVisual } from './liquids.js';
import { lattice, tank, triangleMesh } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export async function buildBuoyancy(ctx: BuildContext, values: Values): Promise<Experiment> {
  const radius = ctx.quality === 'high' ? 0.016 : 0.021;
  const spacing = radius * 2;
  const initial = lattice([-0.76, radius, -0.5], [0.76, 0.42, 0.5], spacing);
  const waterCount = initial.length;
  const model = await new GLTFLoader().loadAsync(
    `${import.meta.env.BASE_URL}models/buoyancy/rubber-duck.glb`,
  );
  model.scene.updateMatrixWorld(true);
  const source = model.scene.getObjectByProperty('isMesh', true) as Mesh<
    BufferGeometry,
    MeshStandardMaterial
  >;
  if (!source) throw new Error('The rubber duck asset has no mesh.');
  const geometries = [0.92, 1, 0.86].map((size, i) =>
    source.geometry
      .clone()
      .applyMatrix4(source.matrixWorld)
      .scale(size, size, size)
      .rotateY([-0.5, 0.65, -0.8][i]!)
      .translate((i - 1) * 0.45, 0.47, [0.08, -0.12, 0.06][i]!),
  );
  const bodies: SoftbodyDef[] = geometries.map((geometry, index) => {
    const voxels = voxelize(triangleMesh(geometry), { particleRadius: radius });
    const start = initial.length,
      phaseId = (index + 1) << 16;
    for (let i = 0; i < voxels.count; i++)
      initial.push({
        position: [
          voxels.positions[i * 3]!,
          voxels.positions[i * 3 + 1]!,
          voxels.positions[i * 3 + 2]!,
        ],
        velocity: [0, 0, 0],
        invMass: 1 / (1000 * spacing ** 3 * values['density']!),
        phase: phaseId,
      });
    return {
      particleRange: { start, count: voxels.count },
      restPositions: voxels.positions,
      surfaceFlag: voxels.surfaceFlag,
      phaseId,
      matchCompliance: 1e-6,
      edges: voxels.edges,
    };
  });
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const xpbd = createXpbdUniforms(1 / 60);
  const hashGrid = new HashGrid(particles, { cellSize: radius * 4 });
  const softbody = new SoftbodySystem({ particles, xpbd, bodies, shapeMatchMode: 'explicit' });
  const fluid = new FluidSystem({
    particles,
    hashGrid,
    xpbd,
    restDensity: 1000,
    particleSpacing: spacing,
    h: radius * 4,
    compliance: 1e-4,
    fluidParticles: { start: 0, count: waterCount },
    xsph: { c: values['viscosity']! },
    surfaceTension: values['tension']!,
  });
  // Registration installs the pressure reaction kernels. Complete it before SimLoop snapshots them.
  for (let i = 0; i < bodies.length; i++)
    await fluid.registerBoundaryParticles(softbody.surfaceRange(i));
  const colliders = tank(particles, 0.8, 0.55);
  colliders.upload();
  const substeps = 4,
    iterations = 3;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    hashGrid,
    materials: [fluid, softbody],
    colliders: { colliders },
    contact: {
      hashGrid,
      maxContacts: initial.length * 8,
      friction: { muS: 0.35, muK: 0.2 },
      emittingRanges: bodies.map((body) => body.particleRange),
    },
  });
  loop.kernels.floorY.value = -1e9;
  loop.gravity.set(0, -values['gravity']!, 0);
  // Model the righting moment of a weighted toy base. This supplies only a
  // pitch/roll torque, with zero net force: all vertical support still comes
  // from the coupled water pressure, so dense ducks remain free to sink.
  const balanceDt = uniform(1 / 60);
  const balanceGravity = uniform(values['gravity']!);
  const angularAcceleration = instancedArray(bodies.length, 'vec4');
  const balance = Fn(() => {
    const body: Any = instanceIndex;
    const row: Any = body.mul(uint(3));
    const r0: Any = softbody.bodyRotations.element(row).xyz;
    const r1: Any = softbody.bodyRotations.element(row.add(uint(1))).xyz;
    const r2: Any = softbody.bodyRotations.element(row.add(uint(2))).xyz;
    const up: Any = vec3(r0.y, r1.y, r2.y);
    const momentum: Any = vec3(0).toVar();
    const inertia: Any = vec3(0).toVar();
    const start: Any = softbody.bodyStart.element(body);
    Loop(
      { start: uint(0), end: softbody.bodyCount.element(body), type: 'uint', condition: '<' },
      ({ i }: { i: Any }) => {
        const index: Any = start.add(i);
        const rest: Any = softbody.restOffsets.element(index).xyz;
        const r: Any = vec3(r0.dot(rest), r1.dot(rest), r2.dot(rest));
        momentum.addAssign(cross(r, particles.velocities.element(index).xyz));
        inertia.addAssign(vec3(r.y.mul(r.y).add(r.z.mul(r.z)), 1, r.x.mul(r.x).add(r.y.mul(r.y))));
      },
    );
    const omega: Any = momentum.div(inertia.max(1e-6)).mul(vec3(1, 0, 1));
    const acceleration: Any = cross(up, vec3(0, 1, 0))
      .mul(balanceGravity)
      .mul(8)
      .sub(omega.mul(4));
    angularAcceleration.element(body).assign(vec4(acceleration, 0));
  })().compute(bodies.length);
  const applyBalance = Fn(() => {
    const i: Any = instanceIndex.add(uint(waterCount));
    const body: Any = particles.phase.element(i).shiftRight(uint(16)).sub(uint(1));
    const row: Any = body.mul(uint(3));
    const rest: Any = softbody.restOffsets.element(i).xyz;
    const r: Any = vec3(
      softbody.bodyRotations.element(row).xyz.dot(rest),
      softbody.bodyRotations.element(row.add(uint(1))).xyz.dot(rest),
      softbody.bodyRotations.element(row.add(uint(2))).xyz.dot(rest),
    );
    const v: Any = particles.velocities.element(i);
    v.assign(
      vec4(v.xyz.add(cross(angularAcceleration.element(body).xyz, r).mul(balanceDt)), float(0)),
    );
  })().compute(initial.length - waterCount);
  const meshes = geometries.map(
    (geometry, i) =>
      new SoftbodyMesh({
        geometry,
        softbody,
        bodyIndex: i,
        sourceMaterial: source.material,
        reachRadius: radius * 3.5,
      }),
  );
  const visual = liquidVisual(ctx, fluid, 0xbde9ff, values['roughness']);
  visual.surface.params.surface.attenuationDistance.value = 1.8;
  visual.surface.params.surface.envIntensity.value = 0.4;
  const waterMaterial = visual.surface.mesh.material as MeshPhysicalNodeMaterial;
  waterMaterial.ior = 1.333;
  waterMaterial.metalness = 0;
  waterMaterial.transmission = 1;
  waterMaterial.clearcoat = 0;
  const tray = basin(1.65, 1.15, 0.39);
  // Ceramic pool tiles make the water's transparency and refraction readable.
  const tile = document.createElement('canvas');
  tile.width = tile.height = 128;
  const paint = tile.getContext('2d')!;
  paint.fillStyle = '#b8d3df';
  paint.fillRect(0, 0, 128, 128);
  paint.strokeStyle = '#87acbf';
  paint.lineWidth = 2;
  paint.strokeRect(0, 0, 128, 128);
  const tileTexture = new CanvasTexture(tile);
  tileTexture.colorSpace = SRGBColorSpace;
  tileTexture.wrapS = tileTexture.wrapT = RepeatWrapping;
  tileTexture.repeat.set(7, 7);
  const deck = tray.getObjectByName('PlatformDeck') as Mesh<BufferGeometry, MeshStandardMaterial>;
  deck.material.map = tileTexture;
  deck.material.color.set(0xffffff);
  deck.material.metalness = 0;
  const dots = createParticleMesh({
    particles,
    radius,
    color: 0xdabec8,
    widthSegments: 8,
    heightSegments: 6,
    castShadow: false,
  });
  dots.visible = false;
  return {
    particles,
    loop,
    objects: [tray, ...meshes, visual.surface.mesh, visual.dots, dots],
    particleCount: initial.length,
    substeps,
    iterations,
    prepareRender: () => visual.prepareRender(),
    async update(dt) {
      balanceDt.value = dt;
      await ctx.renderer.computeAsync([balance, applyBalance]);
    },
    setParticleView(enabled) {
      for (const mesh of meshes) mesh.visible = !enabled;
      visual.setParticleView(enabled);
      visual.dots.visible = false;
      dots.visible = enabled;
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') {
        loop.gravity.y = -value;
        balanceGravity.value = value;
      }
      if (key === 'viscosity' && fluid.xsphCUniform) fluid.xsphCUniform.value = value;
      if (key === 'tension') fluid.cohesion?.setGamma(value);
      if (key === 'roughness') visual.surface.params.surface.roughness.value = value;
    },
    dispose() {
      tileTexture.dispose();
      const textures = new Set<Texture>();
      for (const value of Object.values(source.material))
        if (value && typeof value === 'object' && 'isTexture' in value)
          textures.add(value as Texture);
      for (const texture of textures) texture.dispose();
      source.material.dispose();
      source.geometry.dispose();
      visual.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}
