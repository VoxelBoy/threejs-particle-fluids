import { Color, CylinderGeometry, Mesh, TorusGeometry, Vector3 } from 'three';
import { Fn, If, float, instanceIndex, uniform, vec3, vec4 } from 'three/tsl';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { HashGrid, ParticleSystem, SimLoop, createXpbdUniforms } from '../../src/core/index.js';
import { FluidSurfaceRenderer, FluidSystem, ViscositySolver } from '../../src/fluids/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { basin, block, material, pedestal } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { lattice, tank } from './shared.js';

// TSL's generated operator chains need the broad node type at graph boundaries.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export function liquidVisual(
  ctx: BuildContext,
  fluid: FluidSystem,
  color: number,
  roughness = 0.1,
  metal = false,
) {
  const surface = new FluidSurfaceRenderer({
    fluidSystem: fluid,
    renderer: ctx.renderer,
    scene: ctx.scene,
    camera: ctx.camera,
    defaults: {
      smoothingResolution: 'full',
      imposterRadiusMul: 1.8,
      sigmaMul: 2.5,
      deltaMul: 10,
      muMul: 0.5,
      roughness,
      attenuationDistance: 0.7,
      thicknessScale: 0.015,
      envIntensity: 0.8,
    },
  });
  surface.params.surface.color.value.set(color);
  surface.params.thickness.splatRadius.value = 3.5;
  surface.params.anisotropy.enabled.value = true;
  surface.params.anisotropy.lambda.value = 0.85;
  const mat = surface.mesh.material as MeshPhysicalNodeMaterial;
  mat.fog = false;
  mat.transmission = 1;

  mat.clearcoat = 0.15;
  if (metal) {
    mat.metalness = 0.92;
    mat.transmission = 0.08;
    mat.color = new Color(0xa4becd);
  }
  const dots = createParticleMesh({
    particles: fluid.particles,
    radius: fluid.particles.particleRadius,
    color,
    widthSegments: 8,
    heightSegments: 6,
    castShadow: false,
  });
  dots.count = fluid.fluidParticles.count;
  dots.visible = false;
  let particleView = false;
  return {
    surface,
    dots,
    prepareRender() {
      if (!particleView) surface.prepareRender();
    },
    setParticleView(enabled: boolean) {
      particleView = enabled;
      dots.visible = enabled;
      surface.mesh.visible = !enabled;
    },
    dispose() {
      surface.dispose();
    },
  };
}

export async function buildFluid(
  ctx: BuildContext,
  values: Values,
  kind: 'tidal' | 'impact' | 'marble' | 'amber',
): Promise<Experiment> {
  const radius = ctx.quality === 'high' ? 0.014 : 0.018;
  const spacing = radius * 2;
  const halfX = 0.8,
    halfZ = 0.55;
  const tidalWalls = [
    { x: -0.2, z: 0, height: 0.11, length: 1.01 },
    { x: 0.07, z: -0.195, height: 0.31, length: 0.65 },
    { x: 0.37, z: 0.19, height: 0.36, length: 0.66 },
    { x: 0.64, z: 0, height: 0.075, length: 0.9 },
  ];
  let initial;
  if (kind === 'tidal') {
    initial = lattice([-0.75, radius, -0.45], [-0.34, 0.82, 0.45], spacing);
    initial.push(
      ...lattice(
        [-0.3, radius, -0.45],
        [0.74, 0.105, 0.45],
        spacing,
        (x, y, z) =>
          !tidalWalls.some(
            (wall) =>
              Math.abs(x - wall.x) < 0.0325 + radius &&
              Math.abs(z - wall.z) < wall.length / 2 + radius &&
              y < wall.height + radius,
          ),
      ),
    );
  } else if (kind === 'impact') {
    initial = lattice([-0.72, radius, -0.46], [0.72, 0.17, 0.46], spacing);
    const center = values['height'] ?? 1.1;
    initial.push(
      ...lattice(
        [-0.24, center - 0.24, -0.24],
        [0.24, center + 0.24, 0.24],
        spacing,
        (x, y, z) => x * x + (y - center) ** 2 + z * z < 0.24 ** 2,
      ),
    );
  } else if (kind === 'marble') {
    initial = lattice(
      [-0.31, 0.2, -0.31],
      [0.31, 0.82, 0.31],
      spacing,
      (x, y, z) => x * x + (y - 0.51) ** 2 + z * z < 0.3 ** 2,
    );
    initial = initial.map((p) => ({
      ...p,
      velocity: [-p.position[2] * values['spin']!, 0, p.position[0] * values['spin']!] as const,
    }));
  } else {
    initial = lattice([-0.7, radius, -0.43], [0.7, 0.13, 0.43], spacing);
  }
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const hashGrid = new HashGrid(particles, { cellSize: radius * 4 });
  const xpbd = createXpbdUniforms(1 / 60);
  const fluid = new FluidSystem({
    particles,
    hashGrid,
    xpbd,
    restDensity: 1000,
    h: radius * 4,
    particleSpacing: spacing,
    compliance: 1e-4,
    fluidParticles: { start: 0, count: initial.length },
    xsph: { c: kind === 'amber' ? 0.03 : values['viscosity']! },
    surfaceTension: Math.max(0.001, values['tension']!),
    vorticity: { strength: kind === 'marble' || kind === 'amber' ? 0 : 0.025 },
  });
  const colliders = tank(particles, halfX, halfZ);
  const objects = [kind === 'marble' ? pedestal(0.62) : basin(halfX * 2, halfZ * 2)];
  if (kind === 'tidal') {
    for (const wall of tidalWalls) {
      const center = new Vector3(wall.x, wall.height / 2, wall.z);
      colliders.addBox(center, new Vector3(0.0325, wall.height / 2, wall.length / 2));
      const obstacle = block([0.065, wall.height, wall.length], center.toArray(), 0x8da8ac, 0.009);
      obstacle.name = 'TidalWall';
      objects.push(obstacle);
    }
  }
  const nozzle =
    kind === 'amber'
      ? new Mesh(
          new CylinderGeometry(0.105, 0.105, 0.16, 48, 1, true),
          material(0xb2a896, 0.23, 0.8),
        )
      : undefined;
  if (nozzle) {
    nozzle.position.set(-0.22, values['height']! + 0.08, 0);
    const lip = new Mesh(new TorusGeometry(0.105, 0.012, 12, 64), nozzle.material);
    lip.rotation.x = Math.PI / 2;
    lip.position.y = -0.08;
    nozzle.add(lip);
    objects.push(nozzle);
  }
  const viscosity =
    kind === 'amber'
      ? new ViscositySolver({ fluid, viscosity: values['viscosity']!, iterations: 12 })
      : undefined;
  colliders.upload();
  const substeps = 3,
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    hashGrid,
    colliders: { colliders },
    materials: viscosity ? [fluid, viscosity] : [fluid],
  });
  loop.kernels.floorY.value = -1e9;
  loop.gravity.set(0, kind === 'marble' ? 0 : -(values['gravity'] ?? 0), 0);
  const colors = { tidal: 0x298fac, impact: 0x627aca, marble: 0x8bb2c5, amber: 0xb66c19 };
  const visual = liquidVisual(ctx, fluid, colors[kind], values['roughness'], kind === 'marble');
  if (kind === 'amber') {
    const material = visual.surface.mesh.material as MeshPhysicalNodeMaterial;
    material.transmission = 0.65;
    material.color.set(0xbd6e20);
    visual.surface.params.surface.attenuationDistance.value = 0.18;
    visual.surface.params.surface.roughness.value = 0.16;
  }
  objects.push(visual.surface.mesh, visual.dots);
  const source = uniform(new Vector3(-0.22, values['height'] ?? 1, 0));
  const feedStart = uniform(0),
    feedCount = uniform(0),
    feedSpeed = uniform(0.55);
  const feed = Fn(() => {
    const i: Any = instanceIndex;
    const slot: Any = i.toFloat().sub(feedStart).add(initial.length).mod(initial.length);
    If(slot.lessThan(feedCount), () => {
      const angle: Any = i.toFloat().mul(2.399963);
      const r: Any = i.toFloat().mul(0.618034).fract().sqrt().mul(0.077);
      const offset: Any = vec3(
        angle.cos().mul(r),
        slot
          .div(feedCount.max(1))
          .mul(feedSpeed)
          .mul(-1 / 60),
        angle.sin().mul(r),
      );
      const position: Any = vec4(source.add(offset), 1);
      particles.positions.element(i).assign(position);
      particles.predictedPositions.element(i).assign(position);
      particles.velocities.element(i).assign(vec4(0, feedSpeed.negate(), 0, 0));
    });
  })().compute(initial.length);
  let feedCursor = 0,
    feedCarry = 0;
  const attractor = uniform(new Vector3(0, 0.51, 0));
  const hit = uniform(new Vector3());
  const pull = uniform(values['gravity'] ?? 0);
  const frameDt = uniform(1 / 60);
  const attraction = Fn(() => {
    const i: Any = instanceIndex;
    const position: Any = particles.positions.element(i).xyz;
    const velocity: Any = particles.velocities.element(i);
    const delta: Any = attractor.sub(position);
    const force: Any = delta.div(delta.length().max(0.25)).mul(pull).mul(frameDt);
    velocity.assign(vec4(velocity.xyz.add(force), velocity.w));
  })().compute(initial.length);
  const impulse = Fn(() => {
    const i: Any = instanceIndex;
    const pos: Any = particles.positions.element(i).xyz;
    const distance: Any = pos.sub(hit).length();
    If(distance.lessThan(0.22), () => {
      const velocity: Any = particles.velocities.element(i);
      const falloff: Any = float(1).sub(distance.div(0.22)).pow(2);
      // Pull a small cap outwards; the lower-speed neck stretches then separates.
      const direction: Any = hit.sub(attractor).normalize();
      velocity.assign(vec4(velocity.xyz.add(direction.mul(falloff.mul(3.8))), velocity.w));
    });
  })().compute(initial.length);
  return {
    particles,
    loop,
    objects,
    particleCount: initial.length,
    substeps,
    iterations,
    async update(dt, time) {
      if (nozzle) {
        source.value.set(-0.22 + Math.sin(time * 0.45) * 0.18, values['height']!, 0);
        nozzle.position.copy(source.value).y += 0.08;
        feedCarry += (dt * 0.012 * values['flow']!) / spacing ** 3;
        feedCount.value = Math.floor(feedCarry);
        feedCarry -= feedCount.value;
        feedStart.value = feedCursor;
        feedCursor = (feedCursor + feedCount.value) % initial.length;
        feedSpeed.value = Math.max(0.1, values['flow']! * 0.55);
        if (feedCount.value > 0) await ctx.renderer.computeAsync(feed);
      }
      if (kind === 'tidal') loop.gravity.x = Math.sin(time * 1.15) * (values['wave'] ?? 0);
      if (kind === 'marble') {
        frameDt.value = dt;
        await ctx.renderer.computeAsync(attraction);
      }
    },
    prepareRender: () => visual.prepareRender(),
    setParticleView: (enabled) => visual.setParticleView(enabled),
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') {
        if (kind === 'marble') pull.value = value;
        else loop.gravity.y = -value;
      }
      if (key === 'viscosity') {
        if (viscosity) viscosity.viscosity.value = value;
        else if (fluid.xsphCUniform) fluid.xsphCUniform.value = value;
      }
      if (key === 'tension') fluid.cohesion?.setGamma(value);
      if (key === 'roughness') visual.surface.params.surface.roughness.value = value;
    },
    async interact(uv) {
      const point = await visual.surface.pick(uv);
      if (!point) return false;
      hit.value.copy(point);
      if (kind !== 'marble') attractor.value.copy(point).add(new Vector3(0, -0.2, 0));
      await ctx.renderer.computeAsync(impulse);
      return true;
    },
    dispose() {
      visual.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}
