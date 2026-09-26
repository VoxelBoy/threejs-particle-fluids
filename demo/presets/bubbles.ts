import { Box3, CylinderGeometry, Group, Mesh, Vector3 } from 'three';
import {
  Fn,
  If,
  cos,
  float,
  instanceIndex,
  instancedArray,
  sin,
  uint,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import { HashGrid, ParticleSystem, SimLoop, createXpbdUniforms } from '../../src/core/index.js';
import { FluidSystem } from '../../src/fluids/index.js';
import { VolumetricGasRenderer, type SmokeTracers } from '../../src/gas/index.js';
import { basin, material } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { liquidVisual } from './liquids.js';
import { lattice, tank } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const HALF_X = 0.5,
  HALF_Z = 0.35,
  LEVEL = 0.4;
const VENTS: readonly [number, number][] = [
  [-0.3, -0.12],
  [-0.12, 0.14],
  [0.05, -0.16],
  [0.2, 0.1],
  [0.34, -0.05],
  [-0.36, 0.18],
];
const PUFF = 320;
const SMOKE_LIFETIME = 5.5;

interface Bubble {
  readonly vent: readonly [number, number];
  readonly position: Vector3;
  readonly velocity: Vector3;
  readonly slot: number;
  state: 'wait' | 'grow' | 'rise' | 'pop';
  timer: number;
  radius: number;
  size: number;
  phase: number;
}

/** Smoke tracers released by bursting bubbles, advected by a procedural updraft. */
function smokePuffs(count: number) {
  const capacity = count * PUFF;
  const tracers: SmokeTracers = {
    capacity,
    lifetime: SMOKE_LIFETIME,
    smokePositions: instancedArray(capacity, 'vec4'),
    smokeAge: instancedArray(capacity, 'float'),
    smokeAlive: instancedArray(capacity, 'uint'),
  };
  const velocities = instancedArray(capacity, 'vec4');
  const at = uniform(new Vector3());
  const radius = uniform(0.05, 'float');
  const bank = uniform(0, 'float');
  const seed = uniform(0, 'float');
  const dt = uniform(1 / 60, 'float');
  const time = uniform(0, 'float');
  const rise = uniform(0.3, 'float');
  const random = (i: Any, k: number): Any =>
    sin(
      i
        .toFloat()
        .mul(12.9898)
        .add(seed.mul(78.233))
        .add(k * 37.719),
    )
      .mul(43758.5453)
      .fract();
  const release = Fn(() => {
    const i: Any = instanceIndex;
    const local: Any = i.toFloat().sub(bank);
    If(local.greaterThanEqual(0).and(local.lessThan(PUFF)), () => {
      // Uniform point in the bubble, biased to its upper half as it bursts.
      const u: Any = random(i, 1).mul(2).sub(1);
      const angle: Any = random(i, 2).mul(Math.PI * 2);
      const s: Any = u.mul(u).oneMinus().sqrt();
      const direction: Any = vec3(s.mul(cos(angle)), u.abs().mul(0.7).add(0.3), s.mul(sin(angle)));
      const offset: Any = direction.mul(
        random(i, 3)
          .pow(1 / 3)
          .mul(radius),
      );
      tracers.smokePositions.element(i).assign(vec4(at.add(offset), 0));
      velocities.element(i).assign(vec4(offset.mul(1.5).add(vec3(0, rise.mul(1.2), 0)), 0));
      tracers.smokeAge.element(i).assign(random(i, 4).mul(0.4));
      tracers.smokeAlive.element(i).assign(uint(1));
    });
  })().compute(capacity);
  const advect = Fn(() => {
    const i: Any = instanceIndex;
    If(tracers.smokeAlive.element(i).greaterThan(uint(0)), () => {
      const age: Any = tracers.smokeAge.element(i).add(dt);
      tracers.smokeAge.element(i).assign(age);
      If(age.greaterThanEqual(SMOKE_LIFETIME), () => {
        tracers.smokeAlive.element(i).assign(uint(0));
      });
      const p: Any = tracers.smokePositions.element(i).xyz.toVar();
      const v: Any = velocities.element(i).xyz.toVar();
      // Divergence-light swirl: two crossed shear waves that drift upward.
      const phase: Any = p.y.mul(6).sub(time.mul(0.9));
      const swirl: Any = vec3(
        sin(phase.add(p.z.mul(7))).mul(0.09),
        sin(p.x.mul(9).add(time))
          .mul(cos(p.z.mul(8)))
          .mul(0.05),
        cos(phase.add(p.x.mul(7))).mul(0.09),
      );
      const goal: Any = swirl.add(vec3(0, rise.mul(float(1).sub(age.div(SMOKE_LIFETIME))), 0));
      v.assign(v.add(goal.sub(v).mul(dt.mul(1.6))));
      const next: Any = p.add(v.mul(dt)).toVar();
      next.y.assign(next.y.max(LEVEL + 0.01));
      tracers.smokePositions.element(i).assign(vec4(next, 0));
      velocities.element(i).assign(vec4(v, 0));
    });
  })().compute(capacity);
  return { tracers, release, advect, at, radius, bank, seed, dt, time, rise };
}

export async function buildBubbles(ctx: BuildContext, values: Values): Promise<Experiment> {
  const radius = ctx.quality === 'high' ? 0.013 : 0.017;
  const spacing = radius * 2;
  const initial = lattice([-HALF_X, radius, -HALF_Z], [HALF_X, LEVEL, HALF_Z], spacing);
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
    xsph: { c: 0.04 },
    surfaceTension: 0.06,
    vorticity: { strength: 0.03 },
  });
  const colliders = tank(particles, HALF_X, HALF_Z, 12);
  // Walls wet by the water; bubbles are left out so their pockets stay round.
  const walls = tank(particles, HALF_X, HALF_Z, 5);
  walls.upload();
  const bubbles: Bubble[] = VENTS.map((vent, i) => ({
    vent,
    position: new Vector3(vent[0], -1, vent[1]),
    velocity: new Vector3(),
    slot: colliders.addSphere(new Vector3(vent[0], -1, vent[1]), 0.002, { muS: 0, muK: 0 }),
    state: 'wait',
    timer: 0.3 + i * 0.45,
    radius: 0.002,
    size: 0,
    phase: i * 1.7,
  }));
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
  loop.gravity.set(0, -9.81, 0);

  const visual = liquidVisual(ctx, fluid, {
    bounds: new Box3(
      new Vector3(-HALF_X - 0.03, -0.02, -HALF_Z - 0.03),
      new Vector3(HALF_X + 0.03, 0.72, HALF_Z + 0.03),
    ),
    colliders: walls,
    carve: colliders,
    color: 0x5fb3c9,
    appearance: { attenuationDistance: 0.55, scattering: 0.05, roughness: 0.03 },
    cavities: { smokeColor: 0xa9adb3, smokeDensity: values['density']! * 45 },
  });

  const puffs = smokePuffs(bubbles.length);
  const smoke = new VolumetricGasRenderer({
    gas: puffs.tracers,
    min: new Vector3(-HALF_X - 0.1, LEVEL - 0.02, -HALF_Z - 0.1),
    max: new Vector3(HALF_X + 0.1, 1.75, HALF_Z + 0.1),
    resolution: ctx.quality === 'high' ? [72, 112, 56] : [56, 88, 44],
    steps: ctx.quality === 'high' ? 96 : 72,
    density: values['density']! * 0.6,
  });

  const vents = new Group();
  const ventMaterial = material(0x1a2530, 0.6, 0.4);
  for (const [x, z] of VENTS) {
    const ring = new Mesh(new CylinderGeometry(0.03, 0.034, 0.012, 32), ventMaterial);
    ring.position.set(x, 0.006, z);
    ring.receiveShadow = true;
    vents.add(ring);
  }

  let seed = 0;
  const center = new Vector3();
  return {
    particles,
    loop,
    objects: [
      basin(HALF_X * 2, HALF_Z * 2, 0.56),
      vents,
      visual.surface.mesh,
      visual.dots,
      smoke.object,
    ],
    particleCount: initial.length,
    substeps,
    iterations,
    async update(dt, time) {
      const rate = values['rate']!;
      for (const b of bubbles) {
        b.timer -= dt;
        if (b.state === 'wait' && b.timer <= 0) {
          b.state = 'grow';
          b.size = values['size']! * (0.8 + 0.4 * Math.abs(Math.sin(time * 3.1 + b.phase)));
          b.radius = 0.004;
          b.position.set(b.vent[0], 0.012, b.vent[1]);
        }
        if (b.state === 'grow') {
          // Swell at the vent before detaching, as real bubbles do.
          b.radius = Math.min(b.size, b.radius + (b.size * dt) / 0.35);
          b.position.y = b.radius + 0.012;
          if (b.radius >= b.size) b.state = 'rise';
        }
        const previous = center.copy(b.position);
        if (b.state === 'rise') {
          const speed = values['rise']! * (0.7 + b.size * 6);
          b.phase += dt * 7;
          const wobble = 0.02 + b.size * 0.25;
          b.position.set(
            b.vent[0] + Math.sin(b.phase) * wobble,
            b.position.y + speed * dt,
            b.vent[1] + Math.cos(b.phase * 0.8) * wobble,
          );
          if (b.position.y > LEVEL - b.size * 0.1) {
            b.state = 'pop';
            b.timer = 0.12;
            puffs.at.value.set(b.position.x, LEVEL + b.size * 0.4, b.position.z);
            puffs.radius.value = b.size;
            puffs.bank.value = bubbles.indexOf(b) * PUFF;
            puffs.seed.value = ++seed;
            await ctx.renderer.computeAsync(puffs.release);
          }
        }
        if (b.state === 'pop') {
          b.radius = Math.max(0.002, b.radius - (b.size * dt) / 0.12);
          if (b.timer <= 0) {
            b.state = 'wait';
            b.timer = (0.4 + Math.abs(Math.sin(time * 1.3 + b.phase)) * 1.6) / Math.max(0.1, rate);
            b.position.set(b.vent[0], -1, b.vent[1]);
            b.radius = 0.002;
          }
        }
        b.velocity.copy(b.position).sub(previous).divideScalar(Math.max(dt, 1e-4));
        colliders.setSphere(b.slot, b.position, b.radius, b.velocity);
      }
      colliders.upload();
      puffs.dt.value = dt;
      puffs.time.value = time;
      puffs.rise.value = 0.05 + values['rise']! * 0.15;
      smoke.time.value = time;
      await ctx.renderer.computeAsync(puffs.advect);
    },
    async prepareRender() {
      visual.prepareRender();
      await smoke.update(ctx.renderer);
    },
    setParticleView: (enabled) => visual.setParticleView(enabled),
    setParameter(key, value) {
      values[key] = value;
      if (key === 'density') smoke.density.value = value * 0.6;
    },
    dispose() {
      visual.dispose();
      smoke.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
      walls.destroy();
    },
  };
}
