import { Box3, Group, Object3D, Vector3 } from 'three';
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
import { basin, block, glassTank, pedestal, pivotStand } from '../runtime/stage.js';
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
  /** Multiplies the surface grid's voxel budget, for bounds that sweep a large volume. */
  readonly voxelScale?: number;
  readonly refraction?: FluidVolumeRendererOptions['refraction'];
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
      (options.voxelScale ?? 1) *
      (fluid.fluidParticles.count >= 50_000
        ? 1_200_000
        : fluid.fluidParticles.count >= 20_000
          ? 900_000
          : 600_000),
    appearance: { color: options.color, ...options.appearance },
    cavities: options.cavities,
    sdfColliders: options.sdfColliders,
    carve: options.carve,
    motionStretch: options.motionStretch,
    refraction: options.refraction,
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

const WAVE_CEILING = 0.7;
const WAVE_WALL_HALF = 0.0325;
// Tall enough that the tank's corners clear the stand through a full turn.
const WAVE_PIVOT = 1.05;
const WAVE_LIFT = WAVE_PIVOT - WAVE_CEILING / 2;
// Walls alternate between rising from the floor and hanging from the lid,
// each spanning half the height, so water snakes over and under them.
const WAVE_WALLS = [-0.48, -0.16, 0.16, 0.48].map((x, i) => {
  const fromFloor = i % 2 === 0;
  return {
    x,
    bottom: fromFloor ? 0 : WAVE_CEILING / 2,
    top: fromFloor ? WAVE_CEILING / 2 : WAVE_CEILING,
  };
});

/**
 * The sealed wave tank on its pivot. Every wall is an oriented box collider
 * attached to the tipping group, so the particles see the tank itself move.
 */
function waveTank(particles: ParticleSystem, halfX: number, halfZ: number) {
  const pivot = new Group();
  pivot.position.y = WAVE_PIVOT;
  const body = new Group();
  body.position.y = -WAVE_CEILING / 2;
  pivot.add(body);
  body.add(glassTank(halfX * 2, halfZ * 2, WAVE_CEILING));
  const colliders = new PrimitiveSet(particles, { capacity: 12 });
  const friction = { muS: 0.08, muK: 0.04 };
  const addWall = (center: Vector3, half: Vector3) => {
    const anchor = new Object3D();
    anchor.position.copy(center);
    body.add(anchor);
    const slot = colliders.addBox(center.clone().setY(center.y + WAVE_LIFT), half, friction);
    colliders.attachToObject3D(slot, anchor);
  };
  // Thick slabs just outside the interior, so fast particles can't tunnel out.
  const t = 0.1,
    h = WAVE_CEILING / 2;
  addWall(new Vector3(0, -t, 0), new Vector3(halfX + t, t, halfZ + t));
  addWall(new Vector3(0, WAVE_CEILING + t, 0), new Vector3(halfX + t, t, halfZ + t));
  for (const side of [-1, 1]) {
    addWall(new Vector3(side * (halfX + t), h, 0), new Vector3(t, h + t, halfZ + t));
    addWall(new Vector3(0, h, side * (halfZ + t)), new Vector3(halfX + t, h + t, t));
  }
  for (const wall of WAVE_WALLS) {
    const height = wall.top - wall.bottom;
    const center = new Vector3(wall.x, wall.bottom + height / 2, 0);
    addWall(center, new Vector3(WAVE_WALL_HALF, height / 2, halfZ));
    body.add(block([WAVE_WALL_HALF * 2, height, halfZ * 2], center.toArray(), 0x8da8ac, 0.009));
  }
  pivot.updateMatrixWorld(true);
  colliders.updateKinematics(1 / 60);
  colliders.upload();
  return { pivot, stand: pivotStand(halfZ * 2 + 0.1, WAVE_PIVOT), colliders };
}

export async function buildFluid(
  ctx: BuildContext,
  values: Values,
  kind: 'wave' | 'impact' | 'marble',
): Promise<Experiment> {
  const halfX = 0.8,
    halfZ = 0.55;
  const fill = (radius: number): ParticleInit[] => {
    const spacing = radius * 2;
    let initial: ParticleInit[];
    if (kind === 'wave') {
      // A layer held against the lid, which drops through the walls at the start.
      initial = lattice(
        [-halfX, WAVE_LIFT + WAVE_CEILING - 0.25, -halfZ],
        [halfX, WAVE_LIFT + WAVE_CEILING - radius, halfZ],
        spacing,
        (x, y) =>
          !WAVE_WALLS.some(
            (wall) =>
              Math.abs(x - wall.x) < WAVE_WALL_HALF + radius &&
              y - WAVE_LIFT > wall.bottom - radius &&
              y - WAVE_LIFT < wall.top + radius,
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
  const wave = kind === 'wave' ? waveTank(particles, halfX, halfZ) : null;
  const colliders = wave?.colliders ?? tank(particles, halfX, halfZ);
  if (!wave) colliders.upload();
  const objects = wave
    ? [wave.stand, wave.pivot]
    : [kind === 'marble' ? pedestal(0.62) : basin(halfX * 2, halfZ * 2)];
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
  // The wave tank's interior sweeps a disc of radius |(halfX, ceiling / 2)|.
  const sweep = Math.hypot(halfX, WAVE_CEILING / 2) + 0.03;
  const bounds =
    kind === 'wave'
      ? new Box3(
          new Vector3(-sweep, WAVE_PIVOT - sweep, -halfZ - 0.03),
          new Vector3(sweep, WAVE_PIVOT + sweep, halfZ + 0.03),
        )
      : kind === 'marble'
        ? new Box3(new Vector3(-0.66, -0.02, -0.66), new Vector3(0.66, 1.15, 0.66))
        : new Box3(
            new Vector3(-halfX - 0.03, -0.02, -halfZ - 0.03),
            new Vector3(halfX + 0.03, (values['height'] ?? 1.1) + 0.35, halfZ + 0.03),
          );
  const looks: Record<typeof kind, { color: number; appearance: Partial<FluidAppearance> }> = {
    wave: { color: 0xc8102e, appearance: { attenuationDistance: 0.55, scattering: 0.06 } },
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
    // Keep surface detail while the grid covers the tank's whole swing.
    voxelScale: kind === 'wave' ? 1.6 : 1,
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
  // The marble bursts outward from its center; pools get a tighter splash.
  const pullRadius = kind === 'marble' ? 0.4 : 0.22;
  const strength = kind === 'marble' ? 3.2 : 3.8;
  const impulse = Fn(() => {
    const i: Any = instanceIndex;
    const pos: Any = particles.positions.element(i).xyz;
    const distance: Any = pos.sub(hit).length();
    If(distance.lessThan(pullRadius), () => {
      const velocity: Any = particles.velocities.element(i);
      // Smooth, wide falloff pulls out a broad lobe rather than a thin thread.
      const falloff: Any = float(1).sub(distance.div(pullRadius).pow(2)).pow(2);
      const direction: Any =
        kind === 'marble' ? pos.sub(hit).div(distance.max(1e-4)) : hit.sub(attractor).normalize();
      velocity.assign(vec4(velocity.xyz.add(direction.mul(falloff.mul(strength))), velocity.w));
    });
  })().compute(initial.length);
  let angle = 0;
  return {
    particles,
    loop,
    objects,
    particleCount: initial.length,
    substeps,
    iterations,
    async update(dt) {
      if (wave && dt > 0) {
        // Accumulate the angle so speed changes don't make the tank jump.
        angle = (angle + dt * (values['speed'] ?? 0)) % (Math.PI * 2);
        wave.pivot.rotation.z = angle;
        wave.pivot.updateMatrixWorld(true);
        colliders.updateKinematics(dt);
        colliders.upload();
        visual.surface.refreshWalls();
      }
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
      if (kind === 'marble') hit.value.copy(attractor.value);
      else {
        hit.value.copy(point);
        attractor.value.copy(point).add(new Vector3(0, -0.2, 0));
      }
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
