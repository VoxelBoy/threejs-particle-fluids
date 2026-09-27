import {
  BufferGeometry,
  Euler,
  Float32BufferAttribute,
  Group,
  MeshStandardMaterial,
  Quaternion,
  SRGBColorSpace,
  TextureLoader,
  Vector3,
} from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import {
  ParticleSystem,
  SimLoop,
  SoftbodyMesh,
  SoftbodySystem,
  type ParticleInit,
  type SoftbodyDef,
} from '../../src/index.js';
import { basin, block, panelFrame } from '../runtime/stage.js';
import { ELASTIC_BODY_BUDGETS, type BuildContext, type Experiment, type Values } from '../types.js';
import { particleView, scaledSubsteps, seededRandom, tank } from './shared.js';

/** A baked particle template, written by `npm run assets:elastic`. */
export interface SampledBody {
  /** Scale that maps the render mesh onto the template. */
  scale: number;
  positions: number[];
  /** 1 for surface particles, which come first. */
  surface: number[];
  edges: number[];
}

/** A smoothed render mesh, written by `npm run assets:elastic`. */
export interface ElasticAsset {
  positions: number[];
  normals: number[];
  uvs: number[];
  indices: number[];
}

const base = `${import.meta.env.BASE_URL}models/elastic/`;

async function loadJson<T>(file: string): Promise<T> {
  const response = await fetch(`${base}${file}.json`);
  if (!response.ok) throw new Error(`Could not load the ${file} soft-body asset.`);
  return (await response.json()) as T;
}

/** One of the bundled Kenney food meshes. */
export async function loadElasticMesh(name: string): Promise<BufferGeometry> {
  const asset = await loadJson<ElasticAsset>(name);
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(asset.positions, 3));
  geometry.setAttribute('normal', new Float32BufferAttribute(asset.normals, 3));
  geometry.setAttribute('uv', new Float32BufferAttribute(asset.uvs, 2));
  geometry.setIndex(asset.indices);
  return geometry;
}

/**
 * The atlas-textured material the Kenney meshes share. Its UV colors (banana
 * peel and tips, bread crust, and so on) survive smoothing and skinning.
 */
export async function loadElasticMaterial(renderer: WebGPURenderer): Promise<MeshStandardMaterial> {
  const texture = await new TextureLoader().loadAsync(`${base}colormap.png`);
  texture.colorSpace = SRGBColorSpace;
  texture.anisotropy = Math.min(8, renderer.getMaxAnisotropy());
  return new MeshStandardMaterial({ map: texture, roughness: 0.62, metalness: 0 });
}

/** Shape-matching compliance for a body of `count` particles; 0 = firm, 1 = jelly. */
export const compliance = (softness: number, count: number) =>
  10 ** (-6 + softness * 3) * (count / 200);

const names = ['donut', 'croissant', 'banana', 'ginger-bread'] as const;
const BODIES = 20;

export async function buildElastic(ctx: BuildContext, values: Values): Promise<Experiment> {
  // Local CC0 assets are smoothed and voxelized ahead of time, one template
  // per particle level. Twenty bodies share the budget; use the closest one.
  const perBody = ELASTIC_BODY_BUDGETS.reduce((best, b) =>
    Math.abs(b - ctx.particles / BODIES) < Math.abs(best - ctx.particles / BODIES) ? b : best,
  );
  const assets = await Promise.all(
    names.map(async (name) => ({
      geometry: await loadElasticMesh(name),
      shape: await loadJson<SampledBody>(`${name}-${perBody}`),
    })),
  );
  const material = await loadElasticMaterial(ctx.renderer);
  const radius = 0.015 * Math.cbrt(200 / perBody);
  const initial: ParticleInit[] = [];
  const random = seededRandom(0x9e3779b9);
  const bodies: SoftbodyDef[] = [];
  const geometries: BufferGeometry[] = [];
  for (let n = 0; n < BODIES; n++) {
    const { geometry, shape } = assets[n % assets.length]!;
    const count = shape.positions.length / 3;
    const start = initial.length;
    // Stagger the grid, jitter each form, and tumble it so rings (donuts)
    // don't stack in register and thread through each other as they fall.
    const layer = Math.floor(n / 12);
    const center = new Vector3(
      ((n % 4) - 1.5) * 0.44 + (layer ? 0.22 : 0) + (random() - 0.5) * 0.12,
      values['height']! + layer * 0.55 + random() * 0.14,
      ((Math.floor(n / 4) % 3) - 1) * 0.47 + (random() - 0.5) * 0.12,
    );
    const rotation = new Quaternion().setFromEuler(
      new Euler(random() * Math.PI * 2, random() * Math.PI * 2, random() * Math.PI * 2),
    );
    for (let i = 0; i < count; i++) {
      const p = new Vector3()
        .fromArray(shape.positions, i * 3)
        .applyQuaternion(rotation)
        .add(center);
      // Every body weighs the same at every particle level.
      initial.push({ position: [p.x, p.y, p.z], invMass: count / 20 });
    }
    geometries.push(
      geometry
        .clone()
        .scale(shape.scale, shape.scale, shape.scale)
        .applyQuaternion(rotation)
        .translate(center.x, center.y, center.z),
    );
    bodies.push({
      range: { start, count },
      surfaceCount: shape.surface.filter(Boolean).length,
      compliance: compliance(values['softness']!, perBody),
      edges: Uint32Array.from(shape.edges),
    });
  }
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  // Self collision lets rings and limbs fold onto themselves without passing through.
  const forms = new SoftbodySystem(particles, {
    bodies,
    shapeMatching: 'local',
    selfCollision: true,
  });
  const walls = tank(particles, 1.05, 0.87);
  // A transparent panel on each side makes the squeeze visible without hiding the forms.
  const plates = [-1, 1].map((side) => {
    const plate = new Group();
    plate.position.x = side * 1.06;
    const panel = block([0.04, 0.42, 1.74], [0, 0.2, 0], 0x9fc3cd, 0.008);
    const panelMaterial = panel.material as MeshStandardMaterial;
    panelMaterial.transparent = true;
    panelMaterial.opacity = 0.15;
    panelMaterial.depthWrite = false;
    panel.castShadow = false;
    plate.add(panel);
    const frame = panelFrame(1.78, 0.46);
    frame.rotation.y = Math.PI / 2;
    frame.position.y = 0.215;
    plate.add(frame);
    plate.updateMatrixWorld(true);
    walls.attach(
      walls.addBox(new Vector3(), new Vector3(0.02, 0.21, 0.87), { muS: 0.1, muK: 0.05 }),
      panel,
    );
    return plate;
  });
  const substeps = scaledSubsteps(6, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    materials: [forms],
    colliders: [walls],
    contact: { muS: 0.12, muK: 0.06, maxContacts: initial.length * 20 },
  });
  const meshes = geometries.map((geometry, i) => new SoftbodyMesh(forms, i, geometry, material));
  for (const { geometry } of assets) geometry.dispose();
  const dots = particleView(particles, { color: 0xd8bdd2 });
  return {
    particles,
    loop,
    objects: [basin(2.15, 1.8), ...plates, ...meshes, dots],
    particleCount: initial.length,
    substeps,
    iterations,
    update(_dt, time) {
      const cycle = (1 - Math.cos(Math.max(0, time - 1) * 1.05)) / 2;
      // At the default compression (0.8) the plates close to 0.19 m from centre.
      const distance = Math.max(0.1, 1.06 - cycle * values['compression']! * 1.0875);
      plates[0]!.position.x = -distance;
      plates[1]!.position.x = distance;
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
    },
    setParticleView(enabled) {
      for (const mesh of meshes) mesh.visible = !enabled;
      dots.visible = enabled;
    },
    dispose() {
      material.map?.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
