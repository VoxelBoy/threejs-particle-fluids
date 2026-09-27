import { BoxGeometry, CylinderGeometry, Group, Mesh, Vector3 } from 'three';
import { Fn, If, color, cos, float, instanceIndex, mix, sin, uniform, vec3, vec4 } from 'three/tsl';
import {
  FluidSystem,
  GasSpriteRenderer,
  GasSystem,
  GasVolumeRenderer,
  ParticleSystem,
  SimLoop,
} from '../../src/index.js';
import { material, pedestal } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { fitRadius, lattice, particleView, scaledSubsteps, tank } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export function buildVortex(ctx: BuildContext, values: Values): Experiment {
  // The air is a fluid filling the column; the smoke is carried by it.
  const fill = (r: number) => lattice([-0.5, r, -0.5], [0.5, 1.9, 0.5], r * 2);
  const radius = fitRadius(fill, ctx.particles, 0.035);
  const initial = fill(radius);
  const detailed = ctx.particles >= 25000;
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const air = new FluidSystem(particles, { viscosity: 0.03 });
  const smoke = new GasSystem(air, { capacity: detailed ? 7200 : 4800, lifetime: 8 });
  const substeps = scaledSubsteps(2, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(),
    colliders: [tank(particles, 0.52, 0.52)],
    materials: [smoke, air],
  });

  // Drive the air toward a rising, twisting flow.
  const swirl = uniform(values['swirl']!);
  const rise = uniform(values['rise']!);
  const flowTime = uniform(0);
  const stir = Fn(() => {
    const particle: Any = particles.positions.element(instanceIndex);
    const p: Any = particle.xyz.toVar();
    // Recycle the air through an open top; a ceiling would collect the smoke
    // into a flat, mushroom-shaped cap.
    If(p.y.greaterThan(1.96), () => {
      p.y.assign(radius);
      particle.assign(vec4(p, particle.w));
    });
    const velocity: Any = particles.velocities.element(instanceIndex);
    const lift: Any = float(1)
      .sub(p.x.mul(p.x).add(p.z.mul(p.z)).mul(2))
      .max(0.25)
      .mul(rise);
    const phase: Any = p.y.mul(5).sub(flowTime.mul(0.65));
    const curl: Any = sin(p.x.mul(9).add(flowTime.mul(0.4))).mul(cos(p.z.mul(9).sub(phase)));
    const goal: Any = vec3(
      p.z.negate().mul(swirl).add(sin(phase).mul(0.08)),
      lift.add(curl.mul(0.08)),
      p.x.mul(swirl).add(cos(phase).mul(0.08)),
    );
    velocity.assign(vec4(mix(velocity.xyz, goal, 0.12), velocity.w));
  })().compute(initial.length);

  const sprites = new GasSpriteRenderer(smoke, {
    size: 0.052,
    initialOpacity: 0.45,
    opacityTau: 4.5,
    colorNode: (position: Any) =>
      mix(color(0xe8a266), color(0xa89cdf), position.y.div(1.5).clamp(0, 1)),
  });
  sprites.object.visible = false;
  const volume = new GasVolumeRenderer(smoke, {
    renderer: ctx.renderer,
    min: new Vector3(-0.62, 0, -0.62),
    max: new Vector3(0.62, 1.72, 0.62),
    resolution: detailed ? [80, 128, 80] : [64, 96, 64],
    steps: detailed ? 96 : 72,
    density: values['density']!,
  });
  const dots = particleView(particles, { radius: radius * 0.27, color: 0xa9cbd9 });

  const vent = new Group();
  const recess = new Mesh(new CylinderGeometry(0.19, 0.19, 0.006, 64), material(0x0b141d, 0.8, 0));
  recess.position.y = 0.005;
  vent.add(recess);
  const grilleMaterial = material(0x738894, 0.35, 0.7);
  for (let i = -5; i <= 5; i++) {
    const x = i * 0.031;
    const slat = new Mesh(
      new BoxGeometry(0.009, 0.008, Math.sqrt(0.18 ** 2 - x ** 2) * 2),
      grilleMaterial,
    );
    slat.position.set(x, 0.012, 0);
    vent.add(slat);
  }

  // Start with a gently twisting column; the density filter joins the samples into smoke.
  for (let i = 0; i < 1600; i++) {
    const h = (i * 0.61803398875) % 1;
    const angle = i * 2.399963;
    const r = Math.sqrt((i * 0.41421356) % 1) * (0.08 + h * 0.17);
    const cx = Math.sin(h * 9) * h * 0.11;
    const cz = Math.cos(h * 9) * h * 0.11;
    smoke.emit([cx + Math.cos(angle) * r, 0.07 + h * 1.35, cz + Math.sin(angle) * r]);
  }
  let emissionCarry = 0;
  return {
    particles,
    loop,
    objects: [pedestal(0.58), vent, volume.object, sprites.object, dots],
    get particleCount() {
      return initial.length + smoke.aliveCount;
    },
    substeps,
    iterations,
    prepareRender: () => volume.update(),
    async update(dt, time) {
      volume.time = time;
      flowTime.value = time;
      await ctx.renderer.computeAsync(stir);
      emissionCarry += values['emission']! * dt;
      const count = Math.floor(emissionCarry);
      emissionCarry -= count;
      for (let i = 0; i < count; i++) {
        const angle = time * 5 + i * 2.39996;
        const r = 0.12 * Math.sqrt((i * 0.61803398875 + time * 0.3) % 1);
        smoke.emit([Math.cos(angle) * r, 0.08 + (i % 5) * 0.008, Math.sin(angle) * r]);
      }
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'swirl') swirl.value = value;
      if (key === 'rise') rise.value = value;
      if (key === 'density') volume.density = value;
    },
    setParticleView(enabled) {
      dots.visible = enabled;
      sprites.object.visible = enabled;
      volume.object.visible = !enabled;
    },
    dispose() {
      sprites.dispose();
      volume.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
