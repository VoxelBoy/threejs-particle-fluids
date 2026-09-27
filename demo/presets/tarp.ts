import {
  Box3,
  CylinderGeometry,
  DoubleSide,
  Group,
  Mesh,
  PlaneGeometry,
  SphereGeometry,
  TorusGeometry,
  Vector3,
} from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { Fn, If, instanceIndex, instancedArray, int, uniform, vec4 } from 'three/tsl';
import {
  ClothSystem,
  FluidSystem,
  ParticleSystem,
  SimLoop,
  createClothGraph,
  createClothSurface,
} from '../../src/index.js';
import { basin, material } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { liquidVisual } from './liquids.js';
import { particleView, scaledSubsteps, tank } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** The tarp's footprint: `WIDTH` across, `DEPTH` from its back posts to its front posts. */
const WIDTH = 0.9;
const DEPTH = 0.72;
const FRONT_HEIGHT = 0.5;
/** Share of the particle budget spent on the tarp; the rest is liquid. */
const TARP_SHARE = 0.4;
/** Where the stream lands, as a fraction of the way from the back edge to the front. */
const POUR_AT = 0.2;
const NOZZLE_LENGTH = 0.14;
/** Liquid is released this far up inside the nozzle, so the start of the flow is hidden. */
const RELEASE_DEPTH = 0.03;

/** A point on the flat tarp, `u` across from left to right, `v` from back to front. */
function tarpPoint(u: number, v: number, backHeight: number, out = new Vector3()): Vector3 {
  return out.set(
    (u - 0.5) * WIDTH,
    backHeight + (FRONT_HEIGHT - backHeight) * v,
    (v - 0.5) * DEPTH,
  );
}

/** A post from the deck up to a tarp corner, capped with a small ball. */
function post(corner: Vector3): Group {
  const group = new Group();
  const metal = material(0xb1bdc3, 0.3, 0.8);
  const shaft = new Mesh(new CylinderGeometry(0.011, 0.013, corner.y, 16), metal);
  shaft.position.set(corner.x, corner.y / 2, corner.z);
  const cap = new Mesh(new SphereGeometry(0.018, 20, 14), metal);
  cap.position.copy(corner);
  for (const part of [shaft, cap]) part.castShadow = part.receiveShadow = true;
  group.add(shaft, cap);
  return group;
}

/** A metal nozzle whose open mouth is at its origin, with a bore of `radius`. */
function nozzle(radius: number): Mesh {
  const metal = material(0xb2a896, 0.23, 0.8);
  metal.side = DoubleSide;
  const pipe = new Mesh(
    new CylinderGeometry(radius + 0.006, radius + 0.006, NOZZLE_LENGTH, 48, 1, true),
    metal,
  );
  pipe.geometry.translate(0, NOZZLE_LENGTH / 2, 0);
  const lip = new Mesh(new TorusGeometry(radius + 0.006, 0.005, 12, 64), metal);
  lip.rotation.x = Math.PI / 2;
  pipe.add(lip);
  pipe.castShadow = true;
  return pipe;
}

/**
 * Red liquid pours from a nozzle onto a canvas tarp pinned to four
 * posts, the back pair higher than the front, so it runs down the fabric and
 * spills off the front edge into a shallow basin. The nozzle reuses the
 * oldest liquid once every particle has been poured, so it can run
 * indefinitely.
 */
export function buildTarp(ctx: BuildContext, values: Values): Experiment {
  const backHeight = FRONT_HEIGHT + values['slope']!;
  // Square cells on the sloped fabric: the back-to-front span is its slope length.
  const along = Math.hypot(DEPTH, values['slope']!);
  const across = Math.max(
    16,
    Math.round(Math.sqrt((ctx.particles * TARP_SHARE * WIDTH) / along)) - 1,
  );
  const rows = Math.max(12, Math.round((across * along) / WIDTH));
  const cellSize = WIDTH / across;
  // Neighboring tarp particles overlap, and no liquid particle fits through
  // a cell until the fabric stretches by more than half.
  const radius = 0.55 * cellSize;

  // A flat sheet spanning the posts, which sags under its own weight.
  const geometry = new PlaneGeometry(1, 1, across, rows);
  const vertices = geometry.getAttribute('position');
  const point = new Vector3();
  for (let row = 0, i = 0; row <= rows; row++)
    for (let column = 0; column <= across; column++, i++) {
      tarpPoint(column / across, row / rows, backHeight, point);
      vertices.setXYZ(i, point.x, point.y, point.z);
    }
  // Each corner hangs from a 2 × 2 patch, like a grommet, rather than one particle.
  const corner = (column: number, row: number) => row * (across + 1) + column;
  const pinned = [0, across - 1].flatMap((column) =>
    [0, rows - 1].flatMap((row) => [
      corner(column, row),
      corner(column + 1, row),
      corner(column, row + 1),
      corner(column + 1, row + 1),
    ]),
  );
  // Heavy enough that pooled liquid can't push the canvas's particles apart and leak through.
  const surfaceDensity = 10;
  const graph = createClothGraph(geometry, { surfaceDensity, pinnedIndices: pinned });
  geometry.dispose();

  const tarpCount = graph.positions.length;
  const liquidCount = Math.max(1000, ctx.particles - tarpCount);
  const particles = new ParticleSystem(ctx.renderer, tarpCount + liquidCount, radius);
  const tarp = new ClothSystem(particles, {
    graph,
    stretchCompliance: 1e-8,
    // Scaled like the velvet's, so the canvas behaves the same at every particle level.
    bendCompliance: (across / 30) ** 4 * (0.35 / surfaceDensity),
    drag: 0.2,
    lift: 0.02,
  });
  // The tarp's neighboring particles overlap, so it must not collide with itself.
  particles.setCollisionGroup(tarp.range, 1);

  const liquidRange = { start: tarpCount, count: liquidCount };
  // Liquid waits pinned in a sparse grid far below the floor until the nozzle releases it.
  particles.uploadParticles(
    Array.from({ length: liquidCount }, (_, k) => ({
      position: [(k % 150) * 0.25 - 18.75, -30, Math.floor(k / 150) * 0.25 - 12.5] as const,
    })),
    tarpCount,
  );
  const liquid = new FluidSystem(particles, {
    range: liquidRange,
    viscosity: 0.01,
    surfaceTension: values['tension']!,
    adhesion: 0.15,
  });
  particles.setInvMass(liquidRange, 0);
  // The liquid feels the tarp as a moving wall and pushes it back.
  liquid.addBoundary(tarp.range);

  const walls = tank(particles, 0.8, 0.55);
  const substeps = scaledSubsteps(6, ctx.particles, 2),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    materials: [liquid, tarp],
    colliders: [walls],
    contact: { muS: 0.05, muK: 0.03 },
  });

  const surface = createClothSurface(tarp, {
    columns: across + 1,
    rows: rows + 1,
    material: new MeshPhysicalNodeMaterial({
      color: 0xe0a83a,
      side: DoubleSide,
      roughness: 0.8,
      metalness: 0,
      sheen: 0.3,
      sheenColor: 0xffe2a6,
      sheenRoughness: 0.6,
    }),
  });
  const tarpDots = particleView(particles, { range: tarp.range, color: 0xe0a83a });
  const visual = liquidVisual(ctx, liquid, {
    bounds: new Box3(new Vector3(-0.83, -0.02, -0.58), new Vector3(0.83, 1.3, 0.58)),
    colliders: [walls],
    solids: tarp.range,
    motionStretch: 0.04,
    appearance: {
      color: 0xd81b2c,
      attenuationDistance: 0.05,
      scattering: 0.05,
      roughness: 0.08,
    },
  });
  const posts = [0, 1].flatMap((u) => [0, 1].map((v) => post(tarpPoint(u, v, backHeight))));

  // Each release is one horizontal layer of a square lattice clipped to the
  // nozzle's bore, placed at rest.
  const spacing = liquid.particleSpacing;
  const bore = values['nozzle']! / 2;
  const reach = Math.max(1, bore / spacing);
  const disc: number[] = [];
  for (let i = -Math.floor(reach); i <= reach; i++)
    for (let j = -Math.floor(reach); j <= reach; j++)
      if (i * i + j * j <= reach * reach) disc.push(i * spacing, 0, j * spacing, 0);
  const discOffsets = instancedArray(new Float32Array(disc), 'vec4');
  const perLayer = disc.length / 4;
  const release = uniform(new Vector3());
  const emitStart = uniform(0, 'float');
  const emit = Fn(() => {
    // Slots wrap around the liquid's range, so the oldest liquid is poured again.
    const slot: Any = int(instanceIndex)
      .sub(emitStart.toInt())
      .add(liquidCount)
      .mod(liquidCount)
      .toVar();
    If(slot.lessThan(perLayer), () => {
      const i: Any = instanceIndex.add(tarpCount);
      const position: Any = vec4(release.add(discOffsets.element(slot).xyz), 0);
      particles.positions.element(i).assign(position);
      particles.predictedPositions.element(i).assign(position);
      particles.velocities.element(i).assign(vec4(0, 0, 0, 0));
      particles.invMass.element(i).assign(1 / liquid.mass);
      // Waiting and recycled particles carry a stale density.
      liquid.density.element(i).assign(liquid.restDensity);
    });
  })().compute(liquidCount);

  const spout = nozzle(bore);
  const placeNozzle = () => {
    tarpPoint(0.5, POUR_AT, backHeight, spout.position).y += values['height']!;
    release.value.copy(spout.position).y += RELEASE_DEPTH;
  };
  placeNozzle();
  let released = 0,
    sinceRelease = Infinity;

  return {
    particles,
    loop,
    objects: [
      basin(1.65, 1.15, 0.14),
      ...posts,
      surface,
      tarpDots,
      spout,
      visual.surface.mesh,
      visual.dots,
    ],
    particleCount: particles.capacity,
    substeps,
    iterations,
    async update(dt) {
      // Each layer leaves at rest, and the next follows once gravity has
      // pulled it one particle spacing clear.
      sinceRelease += dt;
      const interval = Math.sqrt((2 * spacing) / Math.max(values['gravity']!, 0.5));
      if (sinceRelease < interval) return;
      sinceRelease = Math.min(sinceRelease - interval, dt);
      emitStart.value = released;
      released = (released + perLayer) % liquidCount;
      await ctx.renderer.computeAsync(emit);
    },
    prepareRender: () => visual.update(),
    setReflections: (enabled) => (visual.surface.reflections = enabled),
    setParticleView(enabled) {
      visual.setParticleView(enabled);
      surface.visible = !enabled;
      tarpDots.visible = enabled;
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'height') placeNozzle();
      if (key === 'tension') liquid.surfaceTension = value;
    },
    dispose() {
      visual.surface.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
