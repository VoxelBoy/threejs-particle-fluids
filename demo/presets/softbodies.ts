import {
  Box3,
  BufferGeometry,
  Color,
  Euler,
  Float32BufferAttribute,
  Group,
  MeshStandardMaterial,
  Quaternion,
  SRGBColorSpace,
  TextureLoader,
  Vector3,
} from 'three';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createXpbdUniforms,
  type ParticleInit,
  type SDFData,
} from '../../src/core/index.js';
import { FluidSystem } from '../../src/fluids/index.js';
import {
  SoftbodyMesh,
  SoftbodySystem,
  voxelize,
  type SoftbodyDef,
} from '../../src/softbody/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { basin, block, glassTank, platform } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import type { ElasticAsset } from './elastic.js';
import { loadBunny } from './honey.js';
import { liquidVisual } from './liquids.js';
import { lattice, scaledSubsteps, triangleMesh } from './shared.js';

/** Shape-matching compliance for a body of `count` particles; 0 = firm, 1 = jelly. */
const compliance = (softness: number, count: number) => 10 ** (-6 + softness * 3) * (count / 200);

/** Trilinear lookup into a baked SDF; voxel `i`'s centre sits at `origin + (i + ½)·voxel`. */
function sdfSampler(sdf: SDFData) {
  const [nx, ny, nz] = sdf.resolution;
  return (x: number, y: number, z: number) => {
    const g = [x, y, z].map((v, k) =>
      Math.min(
        sdf.resolution[k]! - 1.001,
        Math.max(0, (v - sdf.origin[k]!) / sdf.voxelSize[k]! - 0.5),
      ),
    ) as [number, number, number];
    const [i, j, k] = g.map(Math.floor) as [number, number, number];
    const [fx, fy, fz] = [g[0] - i, g[1] - j, g[2] - k];
    const at = (a: number, b: number, c: number) =>
      sdf.data[Math.min(nx - 1, a) + Math.min(ny - 1, b) * nx + Math.min(nz - 1, c) * nx * ny]!;
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    return lerp(
      lerp(
        lerp(at(i, j, k), at(i + 1, j, k), fx),
        lerp(at(i, j + 1, k), at(i + 1, j + 1, k), fx),
        fy,
      ),
      lerp(
        lerp(at(i, j, k + 1), at(i + 1, j, k + 1), fx),
        lerp(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), fx),
        fy,
      ),
      fz,
    );
  };
}

/**
 * Fill a baked SDF with a cubic lattice of about `count` particles, keeping
 * the largest face-connected piece. Surface particles come first, as
 * `SoftbodyDef` requires.
 */
function sampleSdfBody(sdf: SDFData, count: number) {
  const phi = sdfSampler(sdf);
  const min = sdf.origin;
  const size = sdf.resolution.map((n, k) => n * sdf.voxelSize[k]!);
  const lattice = (spacing: number) => {
    const dims = size.map((s) => Math.floor(s / spacing)) as [number, number, number];
    const cell = new Int32Array(dims[0] * dims[1] * dims[2]).fill(-1);
    const points: [number, number, number][] = [];
    for (let k = 0; k < dims[2]; k++)
      for (let j = 0; j < dims[1]; j++)
        for (let i = 0; i < dims[0]; i++) {
          const p: [number, number, number] = [
            min[0]! + (i + 0.5) * spacing,
            min[1]! + (j + 0.5) * spacing,
            min[2]! + (k + 0.5) * spacing,
          ];
          // A slight dilation keeps thin features such as ears attached.
          if (phi(...p) < 0.2 * spacing) {
            cell[i + j * dims[0] + k * dims[0] * dims[1]] = points.length;
            points.push(p);
          }
        }
    const neighbors = points.map((p) => {
      const [i, j, k] = p.map((v, a) => Math.round((v - min[a]!) / spacing - 0.5));
      const out: number[] = [];
      for (const [di, dj, dk] of [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
      ] as const) {
        const a = i! + di,
          b = j! + dj,
          c = k! + dk;
        if (a < 0 || b < 0 || c < 0 || a >= dims[0] || b >= dims[1] || c >= dims[2]) continue;
        const n = cell[a + b * dims[0] + c * dims[0] * dims[1]]!;
        if (n >= 0) out.push(n);
      }
      return out;
    });
    // Largest connected component.
    const component = new Int32Array(points.length).fill(-1);
    let best: number[] = [];
    for (let s = 0; s < points.length; s++) {
      if (component[s] !== -1) continue;
      const members = [s];
      component[s] = s;
      for (let q = 0; q < members.length; q++)
        for (const n of neighbors[members[q]!]!)
          if (component[n] === -1) {
            component[n] = s;
            members.push(n);
          }
      if (members.length > best.length) best = members;
    }
    return { points, neighbors, members: best };
  };
  // Lattice counts fall as spacing grows; bisect for the requested count.
  let low = 0.003,
    high = 0.1;
  for (let step = 0; step < 24; step++) {
    const mid = (low + high) / 2;
    if (lattice(mid).members.length > count) low = mid;
    else high = mid;
  }
  const spacing = low;
  const { points, neighbors, members } = lattice(spacing);
  const keep = new Set(members);
  const degree = (i: number) => neighbors[i]!.filter((n) => keep.has(n)).length;
  const ordered = [...members].sort((a, b) => Number(degree(b) < 6) - Number(degree(a) < 6));
  const local = new Map(ordered.map((i, n) => [i, n]));
  const edges: number[] = [];
  for (const i of ordered)
    for (const n of neighbors[i]!)
      if (keep.has(n) && local.get(i)! < local.get(n)!) edges.push(local.get(i)!, local.get(n)!);
  return {
    spacing,
    positions: ordered.map((i) => points[i]!),
    surface: ordered.map((i) => Number(degree(i) < 6)),
    edges,
  };
}

const LINEUP = 5;
// Faces the camera, which looks down -z at the row.
const LINEUP_YAW = Math.PI / 2;
const FIRM = new Color(0x9fb3c8);
const JELLY = new Color(0xe8607e);

export async function buildBunnyLineup(ctx: BuildContext, values: Values): Promise<Experiment> {
  const bunny = await loadBunny();
  const perBody = Math.round(ctx.particles / LINEUP);
  const shape = sampleSdfBody(bunny.sdf, perBody);
  const radius = shape.spacing / 2;
  const yaw = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), LINEUP_YAW);
  const initial: ParticleInit[] = [];
  const bodies: SoftbodyDef[] = [];
  const geometries: BufferGeometry[] = [];
  for (let n = 0; n < LINEUP; n++) {
    const offset = new Vector3((n - (LINEUP - 1) / 2) * 0.56, values['height']!, 0);
    const start = initial.length;
    const rest = new Float32Array(shape.positions.length * 3);
    shape.positions.forEach((local, i) => {
      const p = new Vector3(...local).applyQuaternion(yaw).add(offset);
      rest.set(p.toArray(), i * 3);
      initial.push({
        position: [p.x, p.y, p.z],
        velocity: [0, 0, 0],
        invMass: shape.positions.length / 1000,
        phase: n,
      });
    });
    bodies.push({
      particleRange: { start, count: shape.positions.length },
      restPositions: rest,
      surfaceFlag: Uint8Array.from(shape.surface),
      phaseId: n,
      // Left to right: firm rubber through to loose jelly.
      // Doubles left to right, up to 5e-5 on the right; scaled by the softness slider.
      matchCompliance: 5e-5 * 2 ** (n - (LINEUP - 1)) * values['softness']!,
      edges: Uint32Array.from(shape.edges),
    });
    geometries.push(
      bunny.mesh.geometry.clone().applyQuaternion(yaw).translate(offset.x, offset.y, offset.z),
    );
  }
  bunny.mesh.geometry.dispose();
  (bunny.mesh.material as MeshStandardMaterial).dispose();
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const xpbd = createXpbdUniforms(1 / 60);
  // Per-particle (implicit) shape matching keeps the skinned mesh from shearing.
  // It takes one compliance per system, so each bunny gets its own.
  const softbodies = bodies.map(
    (body) => new SoftbodySystem({ particles, xpbd, bodies: [body], shapeMatchMode: 'implicit' }),
  );
  const colliders = new PrimitiveSet(particles, { capacity: 1 });
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(), { muS: 0.6, muK: 0.5 });
  colliders.upload();
  const substeps = scaledSubsteps(6, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    materials: softbodies,
    colliders: { colliders },
  });
  loop.kernels.floorY.value = -1e9;
  loop.gravity.set(0, -values['gravity']!, 0);
  const materials = geometries.map(
    (_, n) =>
      new MeshStandardMaterial({
        color: FIRM.clone().lerp(JELLY, n / (LINEUP - 1)),
        roughness: 0.35 - (0.2 * n) / (LINEUP - 1),
        metalness: 0,
      }),
  );
  const meshes = geometries.map(
    (geometry, n) =>
      new SoftbodyMesh({
        geometry,
        softbody: softbodies[n]!,
        bodyIndex: 0,
        sourceMaterial: materials[n]!,
        reachRadius: radius * 3.3,
      }),
  );
  for (const mesh of meshes) mesh.castShadow = mesh.receiveShadow = true;
  const dots = createParticleMesh({
    particles,
    radius,
    color: 0xd8bdd2,
    widthSegments: 6,
    heightSegments: 4,
    castShadow: false,
  });
  dots.visible = false;
  return {
    particles,
    loop,
    objects: [basin(2.9, 0.9), ...meshes, dots],
    particleCount: initial.length,
    substeps,
    iterations,
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
    },
    setParticleView(enabled) {
      for (const mesh of meshes) mesh.visible = !enabled;
      dots.visible = enabled;
    },
    dispose() {
      for (const material of materials) material.dispose();
      particles.destroy();
      colliders.destroy();
    },
  };
}

const BANANAS = 8;
const BANANA_LENGTH = 0.41;
const JAR_HALF = 0.35;
const JAR_HEIGHT = 1.0;
const JAR_FLOOR = 0.26;
const FILL = 0.75;
// Short paddles up the shaft, each pointing opposite the one below it.
const PADDLE_REACH = 0.29;
const PADDLE_HALF_HEIGHT = 0.05;
const PADDLES = [0.1, 0.24, 0.38, 0.52, 0.66, 0.8];

export async function buildBlender(ctx: BuildContext, values: Values): Promise<Experiment> {
  const response = await fetch(`${import.meta.env.BASE_URL}models/elastic/banana.json`);
  if (!response.ok) throw new Error('Could not load the banana soft-body asset.');
  const asset = (await response.json()) as ElasticAsset;
  const texture = await new TextureLoader().loadAsync(
    `${import.meta.env.BASE_URL}models/elastic/colormap.png`,
  );
  texture.colorSpace = SRGBColorSpace;
  texture.anisotropy = Math.min(8, ctx.renderer.getMaxAnisotropy());
  const sourceMaterial = new MeshStandardMaterial({
    color: 0xffffff,
    map: texture,
    roughness: 0.62,
    metalness: 0,
  });
  // Liquid and bananas together fill 3/4 of the jar with the level's particles.
  // Every particle shares one size, so the bananas are voxelized at it too.
  const spacing = Math.cbrt(((JAR_HALF * 2) ** 2 * JAR_HEIGHT * FILL) / ctx.particles);
  const radius = spacing / 2;
  const template = new BufferGeometry();
  template.setAttribute('position', new Float32BufferAttribute(asset.positions, 3));
  template.setAttribute('normal', new Float32BufferAttribute(asset.normals, 3));
  template.setAttribute('uv', new Float32BufferAttribute(asset.uvs, 2));
  template.setIndex(asset.indices);
  template.computeBoundingBox();
  const size = template.boundingBox!.getSize(new Vector3());
  const scale = BANANA_LENGTH / Math.max(size.x, size.y, size.z);
  template.scale(scale, scale, scale).center();
  const voxels = voxelize(triangleMesh(template), { particleRadius: radius });
  const count = voxels.count;

  // Keep bananas inside the liquid and clear of the shaft and paddles.
  const fillTop = JAR_FLOOR + JAR_HEIGHT * FILL;
  const blocked = (x: number, y: number, z: number, margin: number) =>
    Math.abs(x) > JAR_HALF - margin ||
    Math.abs(z) > JAR_HALF - margin ||
    y < JAR_FLOOR + margin ||
    y > fillTop - margin ||
    Math.hypot(x, z) < 0.03 + margin ||
    PADDLES.some(
      (height, k) =>
        Math.abs(z) < 0.012 + margin &&
        Math.abs(y - JAR_FLOOR - height) < PADDLE_HALF_HEIGHT + margin &&
        (k % 2 ? -x : x) > -margin &&
        Math.abs(x) < PADDLE_REACH + margin,
    );
  const cell = (x: number, y: number, z: number) =>
    `${Math.floor(x / spacing)},${Math.floor(y / spacing)},${Math.floor(z / spacing)}`;
  const occupied = new Set<string>();
  const nearBanana = (x: number, y: number, z: number) => {
    for (let a = -1; a <= 1; a++)
      for (let b = -1; b <= 1; b++)
        for (let c = -1; c <= 1; c++)
          if (occupied.has(cell(x + a * spacing, y + b * spacing, z + c * spacing))) return true;
    return false;
  };
  // Seeded so every run starts from the same scatter.
  let seed = 0x2545f491;
  const random = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const bananas: Vector3[][] = [];
  const geometries: BufferGeometry[] = [];
  for (let n = 0; n < BANANAS; n++) {
    // Scatter at random, rejecting placements that touch the jar, the rotor,
    // or a banana already placed.
    for (let attempt = 0; attempt < 400; attempt++) {
      const center = new Vector3(
        (random() - 0.5) * JAR_HALF * 1.6,
        JAR_FLOOR + 0.1 + random() * (fillTop - JAR_FLOOR - 0.2),
        (random() - 0.5) * JAR_HALF * 1.6,
      );
      const rotation = new Quaternion().setFromEuler(
        new Euler(random() * Math.PI * 2, random() * Math.PI * 2, random() * Math.PI * 2),
      );
      const points = Array.from({ length: count }, (_, i) =>
        new Vector3()
          .fromArray(voxels.positions, i * 3)
          .applyQuaternion(rotation)
          .add(center),
      );
      if (points.some((p) => blocked(p.x, p.y, p.z, radius) || nearBanana(p.x, p.y, p.z))) continue;
      for (const p of points) occupied.add(cell(p.x, p.y, p.z));
      bananas.push(points);
      geometries.push(
        template.clone().applyQuaternion(rotation).translate(center.x, center.y, center.z),
      );
      break;
    }
  }
  template.dispose();

  const water = lattice(
    [-JAR_HALF, JAR_FLOOR, -JAR_HALF],
    [JAR_HALF, fillTop, JAR_HALF],
    spacing,
    (x, y, z) => !nearBanana(x, y, z) && !blocked(x, y, z, -radius),
  );
  const waterCount = water.length;
  const initial: ParticleInit[] = [...water];
  const bodies: SoftbodyDef[] = bananas.map((banana, n) => {
    const start = initial.length;
    const phaseId = (n + 1) << 16;
    const rest = new Float32Array(count * 3);
    banana.forEach((p, i) => {
      rest.set(p.toArray(), i * 3);
      // Same density as the liquid, so the bananas hang in suspension.
      initial.push({
        position: [p.x, p.y, p.z],
        velocity: [0, 0, 0],
        invMass: 1 / (1000 * spacing ** 3),
        phase: phaseId,
      });
    });
    return {
      particleRange: { start, count },
      restPositions: rest,
      surfaceFlag: voxels.surfaceFlag,
      phaseId,
      matchCompliance: compliance(values['softness']!, count),
      edges: voxels.edges,
    };
  });
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const xpbd = createXpbdUniforms(1 / 60);
  const hashGrid = new HashGrid(particles, { cellSize: radius * 4 });
  const softbody = new SoftbodySystem({ particles, xpbd, bodies, shapeMatchMode: 'implicit' });
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
    surfaceTension: 0.05,
  });
  // Registration installs the pressure reaction kernels. Complete it before SimLoop snapshots them.
  for (let i = 0; i < bodies.length; i++)
    await fluid.registerBoundaryParticles(softbody.surfaceRange(i));

  const colliders = new PrimitiveSet(particles, { capacity: 16 });
  const wall = { muS: 0.3, muK: 0.2 };
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, JAR_FLOOR, 0), wall);
  colliders.addPlane(new Vector3(0, -1, 0), new Vector3(0, JAR_FLOOR + JAR_HEIGHT, 0), wall);
  for (const axis of [new Vector3(1, 0, 0), new Vector3(0, 0, 1)])
    for (const side of [-1, 1])
      colliders.addPlane(
        axis.clone().multiplyScalar(-side),
        axis.clone().multiplyScalar(side * JAR_HALF),
        wall,
      );
  // Each paddle is an oriented box that follows its mesh on the spinning
  // rotor; the shaft is a fixed capsule on the axis.
  const rotor = new Group();
  rotor.position.y = JAR_FLOOR;
  const steel = new MeshStandardMaterial({ color: 0xc9d1d6, roughness: 0.22, metalness: 0.9 });
  const shaft = block([0.06, JAR_HEIGHT, 0.06], [0, JAR_HEIGHT / 2, 0], 0x8b979e, 0.025);
  shaft.material = steel;
  rotor.add(shaft);
  const paddles = PADDLES.map((height, k) => {
    const paddle = block(
      [PADDLE_REACH, PADDLE_HALF_HEIGHT * 2, 0.024],
      [((k % 2 ? -1 : 1) * PADDLE_REACH) / 2, height, 0],
      0xc9d1d6,
      0.008,
    );
    paddle.material = steel;
    rotor.add(paddle);
    return paddle;
  });
  rotor.updateMatrixWorld(true);
  for (const paddle of paddles) {
    const slot = colliders.addBox(
      paddle.getWorldPosition(new Vector3()),
      new Vector3(PADDLE_REACH / 2, PADDLE_HALF_HEIGHT, 0.012),
      wall,
    );
    colliders.attachToObject3D(slot, paddle);
  }
  colliders.addCapsule(
    new Vector3(0, JAR_FLOOR, 0),
    new Vector3(0, JAR_FLOOR + JAR_HEIGHT, 0),
    0.03,
    wall,
  );
  colliders.upload();

  const substeps = scaledSubsteps(5, ctx.particles),
    iterations = 2;
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
      friction: { muS: 0.2, muK: 0.1 },
      emittingRanges: bodies.map((body) => body.particleRange),
    },
  });
  loop.kernels.floorY.value = -1e9;
  loop.gravity.set(0, -values['gravity']!, 0);
  const meshes = geometries.map(
    (geometry, i) =>
      new SoftbodyMesh({
        geometry,
        softbody,
        bodyIndex: i,
        sourceMaterial,
        reachRadius: radius * 3.5,
      }),
  );
  const visual = liquidVisual(ctx, fluid, {
    bounds: new Box3(
      new Vector3(-JAR_HALF - 0.03, JAR_FLOOR - 0.02, -JAR_HALF - 0.03),
      new Vector3(JAR_HALF + 0.03, JAR_FLOOR + JAR_HEIGHT + 0.02, JAR_HALF + 0.03),
    ),
    colliders,
    solids: { start: waterCount, count: initial.length - waterCount },
    color: 0x9fd8ec,
    appearance: { attenuationDistance: 1.2, scattering: 0.04, roughness: 0.06 },
    // Bent rays smear dark patches where bananas cross the surface.
    refraction: false,
  });
  const base = platform(1.1, 1.1);
  const motor = block(
    [0.86, JAR_FLOOR - 0.01, 0.86],
    [0, (JAR_FLOOR - 0.01) / 2, 0],
    0x26343f,
    0.06,
  );
  const jar = glassTank(JAR_HALF * 2, JAR_HALF * 2, JAR_HEIGHT);
  jar.position.y = JAR_FLOOR;
  let angle = 0;
  return {
    particles,
    loop,
    objects: [base, motor, jar, rotor, ...meshes, visual.surface.mesh, visual.dots],
    particleCount: initial.length,
    substeps,
    iterations,
    update(dt) {
      if (dt <= 0) return;
      angle = (angle + dt * (values['rpm']! / 60) * Math.PI * 2) % (Math.PI * 2);
      rotor.rotation.y = angle;
      rotor.updateMatrixWorld(true);
      colliders.updateKinematics(dt);
      colliders.upload();
      visual.surface.refreshWalls();
    },
    setReflections: (enabled) => visual.setReflections(enabled),
    prepareRender: () => visual.prepareRender(),
    setParticleView(enabled) {
      visual.setParticleView(enabled);
      for (const mesh of meshes) mesh.visible = !enabled;
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'viscosity' && fluid.xsphCUniform) fluid.xsphCUniform.value = value;
    },
    dispose() {
      visual.dispose();
      sourceMaterial.dispose();
      texture.dispose();
      steel.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}
