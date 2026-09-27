import { Box3, Vector3 } from 'three';
import { Fn, If, float, instanceIndex, uniform, vec4 } from 'three/tsl';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  type ParticleInit,
  type ParticleRange,
  SimLoop,
  createXpbdUniforms,
} from '../../src/core/index.js';
import {
  FluidSystem,
  FluidVolumeRenderer,
  type FluidAppearance,
  type FluidVolumeRendererOptions,
} from '../../src/fluids/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { basin, block, pedestal } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { fitRadius, lattice, scaledSubsteps, tank } from './shared.js';

// TSL's generated operator chains need the broad node type at graph boundaries.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface LiquidVisualOptions {
  readonly bounds: Box3;
  readonly color: number;
  readonly colliders?: PrimitiveSet;
  readonly solids?: ParticleRange;
  readonly appearance?: FluidVolumeRendererOptions['appearance'];
  readonly cavities?: FluidVolumeRendererOptions['cavities'];
  readonly sdfColliders?: FluidVolumeRendererOptions['sdfColliders'];
  readonly carve?: FluidVolumeRendererOptions['carve'];
  readonly motionStretch?: number;
}

export function liquidVisual(ctx: BuildContext, fluid: FluidSystem, options: LiquidVisualOptions) {
  const surface = new FluidVolumeRenderer({
    fluidSystem: fluid,
    renderer: ctx.renderer,
    scene: ctx.scene,
    camera: ctx.camera,
    bounds: options.bounds,
    colliders: options.colliders,
    solids: options.solids,
    // Finer particles earn a finer surface grid, within a memory ceiling.
    voxelBudget:
      fluid.fluidParticles.count >= 50_000
        ? 1_200_000
        : fluid.fluidParticles.count >= 20_000
          ? 900_000
          : 600_000,
    appearance: { color: options.color, ...options.appearance },
    cavities: options.cavities,
    sdfColliders: options.sdfColliders,
    carve: options.carve,
    motionStretch: options.motionStretch,
  });
  const dots = createParticleMesh({
    particles: fluid.particles,
    radius: fluid.particles.particleRadius,
    color: options.color,
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
    setReflections(enabled: boolean) {
      surface.reflections.value = enabled ? 1 : 0;
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
  kind: 'tidal' | 'impact' | 'marble',
): Promise<Experiment> {
  const halfX = 0.8,
    halfZ = 0.55;
  const tidalWalls = [
    { x: -0.2, z: 0, height: 0.11, length: 1.01 },
    { x: 0.07, z: -0.195, height: 0.31, length: 0.65 },
    { x: 0.37, z: 0.19, height: 0.36, length: 0.66 },
    { x: 0.64, z: 0, height: 0.075, length: 0.9 },
  ];
  const fill = (radius: number): ParticleInit[] => {
    const spacing = radius * 2;
    let initial: ParticleInit[];
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
    } else {
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
    }
    return initial;
  };
  // Size particles so the preset fills its volume with the requested count.
  const radius = fitRadius(fill, ctx.particles, 0.018);
  const spacing = radius * 2;
  const initial = fill(radius);
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
    xsph: { c: values['viscosity']! },
    surfaceTension: Math.max(0.001, values['tension']!),
    vorticity: { strength: kind === 'marble' ? 0 : 0.025 },
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
  colliders.upload();
  const substeps = scaledSubsteps(3, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    hashGrid,
    colliders: { colliders },
    materials: [fluid],
  });
  loop.kernels.floorY.value = -1e9;
  loop.gravity.set(0, kind === 'marble' ? 0 : -(values['gravity'] ?? 0), 0);
  const top = { tidal: 1.05, impact: (values['height'] ?? 1.1) + 0.35, marble: 1.15 };
  const reach = kind === 'marble' ? 0.66 : halfX + 0.03;
  const bounds = new Box3(
    new Vector3(-reach, -0.02, kind === 'marble' ? -reach : -halfZ - 0.03),
    new Vector3(reach, top[kind], kind === 'marble' ? reach : halfZ + 0.03),
  );
  const looks: Record<typeof kind, { color: number; appearance: Partial<FluidAppearance> }> = {
    tidal: { color: 0x4fb4cf, appearance: { attenuationDistance: 0.55, scattering: 0.06 } },
    impact: { color: 0x5aa6cf, appearance: { attenuationDistance: 0.6, scattering: 0.04 } },
    marble: {
      color: 0x7fc4d8,
      appearance: { attenuationDistance: 0.8, scattering: 0.03, envIntensity: 1.2 },
    },
  };
  const visual = liquidVisual(ctx, fluid, {
    bounds,
    colliders,
    color: looks[kind].color,
    appearance: { roughness: values['roughness'], ...looks[kind].appearance },
  });
  objects.push(visual.surface.mesh, visual.dots);
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
  // The marble takes a broad pull; pools get a tighter splash.
  const pullRadius = kind === 'marble' ? 0.4 : 0.22;
  const strength = kind === 'marble' ? 0.8 : 3.8;
  const impulse = Fn(() => {
    const i: Any = instanceIndex;
    const pos: Any = particles.positions.element(i).xyz;
    const distance: Any = pos.sub(hit).length();
    If(distance.lessThan(pullRadius), () => {
      const velocity: Any = particles.velocities.element(i);
      // Smooth, wide falloff pulls out a broad lobe rather than a thin thread.
      const falloff: Any = float(1).sub(distance.div(pullRadius).pow(2)).pow(2);
      const direction: Any = hit.sub(attractor).normalize();
      velocity.assign(vec4(velocity.xyz.add(direction.mul(falloff.mul(strength))), velocity.w));
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
      if (kind === 'tidal') loop.gravity.x = Math.sin(time * 1.15) * (values['wave'] ?? 0);
      if (kind === 'marble') {
        frameDt.value = dt;
        await ctx.renderer.computeAsync(attraction);
      }
    },
    setReflections: (enabled) => visual.setReflections(enabled),
    prepareRender: () => visual.prepareRender(),
    setParticleView: (enabled) => visual.setParticleView(enabled),
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') {
        if (kind === 'marble') pull.value = value;
        else loop.gravity.y = -value;
      }
      if (key === 'viscosity' && fluid.xsphCUniform) fluid.xsphCUniform.value = value;
      if (key === 'tension') fluid.cohesion?.setGamma(value);
      if (key === 'roughness') visual.surface.appearance.roughness.value = value;
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
