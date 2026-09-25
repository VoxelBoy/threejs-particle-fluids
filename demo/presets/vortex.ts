import { Mesh, TorusGeometry, Vector3 } from 'three';
import { Fn, color, float, instanceIndex, mix, uniform, vec3, vec4 } from 'three/tsl';
import { HashGrid, ParticleSystem, SimLoop, createXpbdUniforms } from '../../src/core/index.js';
import { FluidSystem } from '../../src/fluids/index.js';
import { GasSystem, PointSpritesGasRenderer } from '../../src/gas/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { material, pedestal } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { lattice, tank } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export function buildVortex(ctx: BuildContext, values: Values): Experiment {
  const radius = ctx.quality === 'high' ? 0.029 : 0.035;
  const initial = lattice([-0.5, radius, -0.5], [0.5, 1.5, 0.5], radius * 2);
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
    capacity: ctx.quality === 'high' ? 7200 : 4800,
    fluidParticles: particles,
    fluidRange: fluid.fluidParticles,
    hashGrid,
    h: radius * 4,
    xpbd,
    lifetime: 7,
  });
  const colliders = tank(particles, 0.52, 0.52);
  colliders.addPlane(new Vector3(0, -1, 0), new Vector3(0, 1.55, 0));
  colliders.upload();
  const substeps = 2,
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
  const stir = Fn(() => {
    const i: Any = instanceIndex;
    const p: Any = particles.positions.element(i).xyz;
    const velocity: Any = particles.velocities.element(i);
    const r2: Any = p.x.mul(p.x).add(p.z.mul(p.z));
    const lift: Any = float(1).sub(r2.mul(8)).mul(rise);
    const goal: Any = vec3(p.z.negate().mul(swirl), lift, p.x.mul(swirl));
    velocity.assign(vec4(mix(velocity.xyz, goal, 0.12), velocity.w));
  })().compute(initial.length);
  const smoke = new PointSpritesGasRenderer({
    gas,
    size: 0.052,
    initialOpacity: 0.45,
    opacityTau: 4.5,
    colorNode: (pos: Any) => mix(color(0xe8a266), color(0xa89cdf), pos.y.div(1.5).clamp(0, 1)),
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
  const ring = new Mesh(new TorusGeometry(0.47, 0.012, 10, 80), material(0xbcb3a3, 0.25, 0.8));
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.02;
  let emissionCarry = 0;
  // A helical seed makes the velocity field legible from the first frame.
  for (let i = 0; i < 360; i++) {
    const y = (i / 360) * 1.35 + 0.06,
      angle = i * 0.09;
    const r = 0.2 + 0.045 * Math.sin(i * 0.17);
    gas.emit([Math.cos(angle) * r, y, Math.sin(angle) * r], [0, 0, 0], 2, 0);
  }
  return {
    particles,
    loop,
    objects: [pedestal(0.68), ring, smoke.object, dots],
    get particleCount() {
      const alive = gas.smokeAlive.value.array as Uint32Array;
      let count = initial.length;
      for (const flag of alive) count += flag;
      return count;
    },
    substeps,
    iterations,
    async update(dt, time) {
      await ctx.renderer.computeAsync(stir);
      emissionCarry += values['emission']! * dt;
      const count = Math.floor(emissionCarry);
      if (count > 0) {
        emissionCarry -= count;
        for (let i = 0; i < count; i++) {
          const angle = time * 5 + i * 2.39996;
          const r = 0.09 + 0.08 * (i / count);
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
    },
    setParticleView(enabled) {
      dots.visible = enabled;
    },
    dispose() {
      smoke.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}
