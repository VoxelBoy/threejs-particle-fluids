import { BoxGeometry, CylinderGeometry, Group, Mesh, Vector3 } from 'three';
import { Fn, If, color, float, instanceIndex, mix, sin, cos, uniform, vec3, vec4 } from 'three/tsl';
import { HashGrid, ParticleSystem, SimLoop, createXpbdUniforms } from '../../src/core/index.js';
import { FluidSystem } from '../../src/fluids/index.js';
import { GasSystem, PointSpritesGasRenderer, VolumetricGasRenderer } from '../../src/gas/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { material, pedestal } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { fitRadius, lattice, scaledSubsteps, tank } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export function buildVortex(ctx: BuildContext, values: Values): Experiment {
  const fill = (r: number) => lattice([-0.5, r, -0.5], [0.5, 1.9, 0.5], r * 2);
  const radius = fitRadius(fill, ctx.particles, 0.035);
  const initial = fill(radius);
  const detailed = ctx.particles >= 25000;
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const hashGrid = new HashGrid(particles, { cellSize: radius * 4 });
  const xpbd = createXpbdUniforms(1 / 60);
  const fluid = new FluidSystem({
    particles,
    hashGrid,
    xpbd,
    restDensity: 1000,
    particleSpacing: radius * 2,
    h: radius * 4,
    fluidParticles: { start: 0, count: initial.length },
    compliance: 1e-4,
    xsph: { c: 0.03 },
  });
  const gas = new GasSystem({
    capacity: detailed ? 7200 : 4800,
    fluidParticles: particles,
    fluidRange: fluid.fluidParticles,
    hashGrid,
    h: radius * 4,
    xpbd,
    lifetime: 8,
  });
  const colliders = tank(particles, 0.52, 0.52);
  colliders.upload();
  const substeps = scaledSubsteps(2, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    hashGrid,
    colliders: { colliders },
    materials: [gas, fluid],
  });
  loop.gravity.set(0, 0, 0);
  loop.kernels.floorY.value = -1e9;
  const swirl = uniform(values['swirl']!);
  const rise = uniform(values['rise']!);
  const flowTime = uniform(0);
  const stir = Fn(() => {
    const i: Any = instanceIndex;
    const particle: Any = particles.positions.element(i);
    const p: Any = particle.xyz.toVar();
    // Recycle the carrier flow through an open top, preventing an artificial
    // ceiling from collecting the smoke into a flat, mushroom-shaped cap.
    If(p.y.greaterThan(1.96), () => {
      p.y.assign(radius);
      particle.assign(vec4(p, particle.w));
    });
    const velocity: Any = particles.velocities.element(i);
    const r2: Any = p.x.mul(p.x).add(p.z.mul(p.z));
    const lift: Any = float(1).sub(r2.mul(2)).max(0.25).mul(rise);
    const phase: Any = p.y.mul(5).sub(flowTime.mul(0.65));
    const curl: Any = sin(p.x.mul(9).add(flowTime.mul(0.4))).mul(cos(p.z.mul(9).sub(phase)));
    const goal: Any = vec3(
      p.z.negate().mul(swirl).add(sin(phase).mul(0.08)),
      lift.add(curl.mul(0.08)),
      p.x.mul(swirl).add(cos(phase).mul(0.08)),
    );
    velocity.assign(vec4(mix(velocity.xyz, goal, 0.12), velocity.w));
  })().compute(initial.length);
  const smoke = new PointSpritesGasRenderer({
    gas,
    size: 0.052,
    initialOpacity: 0.45,
    opacityTau: 4.5,
    colorNode: (pos: Any) => mix(color(0xe8a266), color(0xa89cdf), pos.y.div(1.5).clamp(0, 1)),
  });
  smoke.object.visible = false;
  const volume = new VolumetricGasRenderer({
    gas,
    min: new Vector3(-0.62, 0, -0.62),
    max: new Vector3(0.62, 1.72, 0.62),
    resolution: detailed ? [80, 128, 80] : [64, 96, 64],
    steps: detailed ? 96 : 72,
    density: values['density']!,
  });
  const dots = createParticleMesh({
    particles,
    radius: radius * 0.3,
    color: 0xa9cbd9,
    castShadow: false,
    widthSegments: 6,
    heightSegments: 4,
  });
  dots.visible = false;
  const vent = new Group();
  const recess = new Mesh(new CylinderGeometry(0.19, 0.19, 0.006, 64), material(0x0b141d, 0.8, 0));
  recess.position.y = 0.005;
  vent.add(recess);
  const grilleMaterial = material(0x738894, 0.35, 0.7);
  for (let i = -5; i <= 5; i++) {
    const x = i * 0.031;
    const length = Math.sqrt(0.18 ** 2 - x ** 2) * 2;
    const slat = new Mesh(new BoxGeometry(0.009, 0.008, length), grilleMaterial);
    slat.position.set(x, 0.012, 0);
    vent.add(slat);
  }
  let emissionCarry = 0;
  // Fill a gently twisting column; the density filter joins the samples into smoke.
  for (let i = 0; i < 1600; i++) {
    const h = (i * 0.61803398875) % 1;
    const angle = i * 2.399963;
    const radius = Math.sqrt((i * 0.41421356) % 1) * (0.08 + h * 0.17);
    const centerX = Math.sin(h * 9) * h * 0.11;
    const centerZ = Math.cos(h * 9) * h * 0.11;
    gas.emit(
      [centerX + Math.cos(angle) * radius, 0.07 + h * 1.35, centerZ + Math.sin(angle) * radius],
      [0, 0, 0],
      1,
      0,
    );
  }
  return {
    particles,
    loop,
    objects: [pedestal(0.58), vent, volume.object, smoke.object, dots],
    get particleCount() {
      const alive = gas.smokeAlive.value.array as Uint32Array;
      let count = initial.length;
      for (const flag of alive) count += flag;
      return count;
    },
    substeps,
    iterations,
    prepareRender: () => volume.update(ctx.renderer),
    async update(dt, time) {
      volume.time.value = time;
      flowTime.value = time;
      await ctx.renderer.computeAsync(stir);
      emissionCarry += values['emission']! * dt;
      const count = Math.floor(emissionCarry);
      if (count > 0) {
        emissionCarry -= count;
        for (let i = 0; i < count; i++) {
          const angle = time * 5 + i * 2.39996;
          const r = 0.12 * Math.sqrt((i * 0.61803398875 + time * 0.3) % 1);
          gas.emit(
            [Math.cos(angle) * r, 0.08 + (i % 5) * 0.008, Math.sin(angle) * r],
            [0, values['rise']!, 0],
            1,
            time,
          );
        }
      }
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'swirl') swirl.value = value;
      if (key === 'rise') rise.value = value;
      if (key === 'density') volume.density.value = value;
    },
    setParticleView(enabled) {
      dots.visible = enabled;
      smoke.object.visible = enabled;
      volume.object.visible = !enabled;
    },
    dispose() {
      smoke.dispose();
      volume.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}
