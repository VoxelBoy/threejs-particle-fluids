import { Color, Mesh, SphereGeometry, Vector3 } from 'three';
import { Fn, If, float, instanceIndex, uniform, vec4 } from 'three/tsl';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { HashGrid, ParticleSystem, SimLoop, createXpbdUniforms } from '../../src/core/index.js';
import { FluidSurfaceRenderer, FluidSystem } from '../../src/fluids/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { basin, block, pedestal } from '../runtime/stage.js';
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
      imposterRadiusMul: 1.6,
      sigmaMul: 1.5,
      deltaMul: 10,
      muMul: 1,
      roughness,
      attenuationDistance: 0.3,
      thicknessScale: 0.025,
      envIntensity: 1.25,
    },
  });
  surface.params.surface.color.value.set(color);
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
  let initial;
  if (kind === 'tidal') {
    initial = lattice([-0.75, radius, -0.45], [-0.34, 0.82, 0.45], spacing);
    initial.push(
      ...lattice(
        [-0.3, radius, -0.45],
        [0.74, 0.105, 0.45],
        spacing,
        (x, _y, z) => Math.hypot(x - 0.15, z) > 0.19,
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
    initial = lattice(
      [-0.65, 0.53, -0.28],
      [-0.23, 0.53 + (values['column'] ?? 0.9), 0.28],
      spacing,
    );
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
    xsph: { c: values['viscosity']! },
    surfaceTension: Math.max(0.001, values['tension']!),
    vorticity: { strength: kind === 'marble' ? 0 : 0.025 },
  });
  const colliders = tank(particles, halfX, halfZ);
  const objects = [kind === 'marble' ? pedestal(0.62) : basin(halfX * 2, halfZ * 2)];
  if (kind === 'tidal') {
    colliders.addSphere(new Vector3(0.15, 0.15, 0), 0.18);
    const orb = new Mesh(
      new SphereGeometry(0.18, 48, 32),
      new MeshPhysicalNodeMaterial({ color: 0xc4ceca, metalness: 0.8, roughness: 0.19 }),
    );
    orb.position.set(0.15, 0.15, 0);
    orb.castShadow = true;
    objects.push(orb);
  }
  if (kind === 'amber') {
    for (let i = 0; i < 3; i++) {
      const height = 0.46 - i * 0.14;
      const x = -0.46 + i * 0.31;
      colliders.addBox(new Vector3(x, height / 2, 0), new Vector3(0.155, height / 2, 0.4));
      objects.push(block([0.31, height, 0.8], [x, height / 2, 0], 0xbba594, 0.015));
    }
  }
  colliders.upload();
  const substeps = 3,
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
  loop.gravity.set(0, -(values['gravity'] ?? 0), 0);
  const colors = { tidal: 0x298fac, impact: 0x627aca, marble: 0x8bb2c5, amber: 0xa45817 };
  const visual = liquidVisual(ctx, fluid, colors[kind], values['roughness'], kind === 'marble');
  objects.push(visual.surface.mesh, visual.dots);
  let kick = false;
  const center = uniform(new Vector3(0, kind === 'marble' ? 0.51 : 0.1, 0));
  const impulse = Fn(() => {
    const i: Any = instanceIndex;
    const pos: Any = particles.positions.element(i).xyz;
    const delta: Any = pos.sub(center);
    const distance: Any = delta.length();
    If(distance.lessThan(0.75), () => {
      const velocity: Any = particles.velocities.element(i);
      const strength: Any = float(1).sub(distance.div(0.75)).mul(0.6);
      const direction: Any = delta.add(new Vector3(0.1, 0.7, 0)).normalize();
      velocity.assign(vec4(velocity.xyz.add(direction.mul(strength)), velocity.w));
    });
  })().compute(initial.length);
  return {
    particles,
    loop,
    objects,
    particleCount: initial.length,
    substeps,
    iterations,
    async update(_dt, time) {
      if (kind === 'tidal') loop.gravity.x = Math.sin(time * 1.15) * (values['wave'] ?? 0);
      if (kick) {
        kick = false;
        await ctx.renderer.computeAsync(impulse);
      }
    },
    prepareRender: () => visual.prepareRender(),
    setParticleView: (enabled) => visual.setParticleView(enabled),
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'viscosity' && fluid.xsphCUniform) fluid.xsphCUniform.value = value;
      if (key === 'tension') fluid.cohesion?.setGamma(value);
      if (key === 'roughness') visual.surface.params.surface.roughness.value = value;
    },
    disturb() {
      kick = true;
    },
    dispose() {
      visual.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}
