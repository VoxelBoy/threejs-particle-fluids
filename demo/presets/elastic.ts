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
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createXpbdUniforms,
  type ParticleInit,
} from '../../src/core/index.js';
import { SoftbodyMesh, SoftbodySystem, type SoftbodyDef } from '../../src/softbody/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { basin, block, panelFrame } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { tank } from './shared.js';

interface SampledBody {
  scale: number;
  positions: number[];
  surface: number[];
  edges: number[];
}
export interface ElasticAsset {
  positions: number[];
  normals: number[];
  uvs: number[];
  indices: number[];
  /** Particle templates keyed by per-body budget (see prepare-elastic-assets). */
  templates: Record<string, SampledBody>;
}

const names = ['donut', 'croissant', 'banana', 'ginger-bread'] as const;

export async function buildElastic(ctx: BuildContext, values: Values): Promise<Experiment> {
  // Local CC0 assets are smoothed and voxelized ahead of time. No runtime asset
  // service or CPU voxelization is needed when switching presets or quality.
  const assets = await Promise.all(
    names.map(async (name) => {
      const response = await fetch(`${import.meta.env.BASE_URL}models/elastic/${name}.json`);
      if (!response.ok) throw new Error(`Could not load the ${name} soft-body mesh.`);
      return (await response.json()) as ElasticAsset;
    }),
  );
  const texture = await new TextureLoader().loadAsync(
    `${import.meta.env.BASE_URL}models/elastic/colormap.png`,
  );
  texture.colorSpace = SRGBColorSpace;
  texture.anisotropy = Math.min(8, ctx.renderer.getMaxAnisotropy());
  // Kenney's source MTL uses white Kd with this atlas. Preserve its UV colors
  // (banana peel and tips, bread crust, etc.) through smoothing and skinning.
  const sourceMaterial = new MeshStandardMaterial({
    color: 0xffffff,
    map: texture,
    roughness: 0.62,
    metalness: 0,
  });
  // Twenty bodies share the budget; use the closest baked template.
  const budgets = Object.keys(assets[0]!.templates).map(Number);
  const perBody = budgets.reduce((best, b) =>
    Math.abs(b - ctx.particles / 20) < Math.abs(best - ctx.particles / 20) ? b : best,
  );
  const radius = 0.015 * Math.cbrt(200 / perBody);
  const initial: ParticleInit[] = [];
  const bodies: SoftbodyDef[] = [];
  const geometries: BufferGeometry[] = [];
  for (let n = 0; n < 20; n++) {
    const asset = assets[n % assets.length]!;
    const shape = asset.templates[perBody]!;
    const count = shape.positions.length / 3;
    const start = initial.length;
    const center = new Vector3(
      ((n % 4) - 1.5) * 0.44,
      values['height']! + Math.floor(n / 12) * 0.48,
      ((Math.floor(n / 4) % 3) - 1) * 0.47,
    );
    const rotation = new Euler(0.4 + (n % 3) * 0.32, (n % 5) * 0.42 - 0.8, ((n % 4) - 1.5) * 0.23);
    const rest = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const p = new Vector3()
        .fromArray(shape.positions, i * 3)
        .applyEuler(rotation)
        .add(center);
      rest.set(p.toArray(), i * 3);
      // Equal body masses at both resolutions; phase 0 retains self-collision.
      initial.push({
        position: [p.x, p.y, p.z],
        velocity: [0, 0, 0],
        invMass: count / 20,
        phase: 0,
      });
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(asset.positions, 3));
    geometry.setAttribute('normal', new Float32BufferAttribute(asset.normals, 3));
    geometry.setAttribute('uv', new Float32BufferAttribute(asset.uvs, 2));
    geometry.setIndex(asset.indices);
    geometry.scale(shape.scale, shape.scale, shape.scale);
    geometry.applyQuaternion(new Quaternion().setFromEuler(rotation));
    geometry.translate(center.x, center.y, center.z);
    geometries.push(geometry);
    bodies.push({
      particleRange: { start, count },
      restPositions: rest,
      surfaceFlag: Uint8Array.from(shape.surface),
      phaseId: 0,
      matchCompliance: 10 ** (-6 + values['softness']! * 3) * (perBody / 200),
      edges: Uint32Array.from(shape.edges),
    });
  }
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const xpbd = createXpbdUniforms(1 / 60);
  const hashGrid = new HashGrid(particles, { cellSize: radius * 4 });
  const softbody = new SoftbodySystem({ particles, xpbd, bodies, shapeMatchMode: 'implicit' });
  const colliders = tank(particles, 1.05, 0.87);
  const plates = [-1, 1].map((side) => {
    const plate = new Group();
    plate.position.x = side * 1.06;
    const panel = block([0.04, 0.42, 1.74], [0, 0.2, 0], 0x9fc3cd, 0.008);
    const material = panel.material as MeshStandardMaterial;
    material.transparent = true;
    material.opacity = 0.15;
    material.depthWrite = false;
    panel.castShadow = false;
    plate.add(panel);
    const frame = panelFrame(1.78, 0.46);
    frame.rotation.y = Math.PI / 2;
    frame.position.y = 0.215;
    plate.add(frame);
    // A transparent panel makes the squeeze visible without hiding the forms.
    const collider = colliders.addBox(
      new Vector3(side * 1.06, 0.2, 0),
      new Vector3(0.02, 0.21, 0.87),
      {
        muS: 0.1,
        muK: 0.05,
      },
    );
    colliders.attachToObject3D(collider, panel);
    return plate;
  });
  colliders.upload();
  const substeps = 4,
    iterations = 4;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    hashGrid,
    materials: [softbody],
    colliders: { colliders },
    contact: { hashGrid, maxContacts: initial.length * 20, friction: { muS: 0.12, muK: 0.06 } },
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
        reachRadius: radius * 3.3,
      }),
  );
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
    objects: [basin(2.15, 1.8), ...plates, ...meshes, dots],
    particleCount: initial.length,
    substeps,
    iterations,
    update(dt, time) {
      const cycle = (1 - Math.cos(Math.max(0, time - 1) * 1.05)) / 2;
      const distance = 1.06 - cycle * values['compression']! * 0.85;
      plates[0]!.position.x = -distance;
      plates[1]!.position.x = distance;
      colliders.updateKinematics(dt);
      colliders.upload();
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
      sourceMaterial.dispose();
      texture.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}
