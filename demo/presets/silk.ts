import { DoubleSide, Mesh, PlaneGeometry, Quaternion, SphereGeometry, Vector3 } from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import {
  ClothSystem,
  ParticleSystem,
  PrimitiveSet,
  SDFCollider,
  SimLoop,
  createClothGraph,
  createClothSurface,
  type ClothSystemOptions,
} from '../../src/index.js';
import { clothStand, platform } from '../runtime/stage.js';
import { BUNNY_YAW, loadBunny } from './honey.js';
import { particleView, scaledSubsteps } from './shared.js';
import type { BuildContext, Experiment, Values } from '../types.js';

function velvet(): MeshPhysicalNodeMaterial {
  return new MeshPhysicalNodeMaterial({
    color: 0x870b21,
    side: DoubleSide,
    roughness: 0.9,
    metalness: 0,
    sheen: 1,
    sheenColor: 0xd54b5c,
    sheenRoughness: 0.7,
  });
}

/**
 * A square grid of velvet: `segments` cells per side, so about `particles`
 * particles in total. `place` positions the plane before it becomes cloth.
 */
function velvetCloth(
  ctx: BuildContext,
  size: [number, number],
  place: (geometry: PlaneGeometry) => void,
  options: Omit<ClothSystemOptions, 'graph'> & {
    readonly softness: number;
    readonly pinned?: (segments: number) => number[];
    readonly radius: (spacing: number) => number;
  },
) {
  const segments = Math.max(12, Math.round(Math.sqrt(ctx.particles)) - 1);
  const geometry = new PlaneGeometry(size[0], size[1], segments, segments);
  place(geometry);
  const graph = createClothGraph(geometry, {
    surfaceDensity: 0.08,
    pinnedIndices: options.pinned?.(segments) ?? [],
  });
  geometry.dispose();
  const spacing = size[0] / segments;
  const particles = new ParticleSystem(
    ctx.renderer,
    graph.positions.length,
    options.radius(spacing),
  );
  // Bending gradients grow as the grid refines and particle masses shrink with
  // area, so scale the compliance to keep the drape the same at every level.
  const bendCompliance = (softness: number) =>
    10 ** (-1 + softness * 5) * (segments / 30) ** 4 * (0.35 / 0.08);
  const cloth = new ClothSystem(particles, {
    graph,
    stretchTolerance: 0.06,
    ...options,
    bendCompliance: bendCompliance(options.softness),
  });
  const mesh = createClothSurface(cloth, {
    columns: segments + 1,
    rows: segments + 1,
    material: velvet(),
  });
  // The dots are drawn a little smaller than the collision radius, like the fabric.
  const dots = particleView(particles, { radius: 0.0108, color: 0xd54b5c });
  return {
    particles,
    cloth,
    mesh,
    dots,
    spacing,
    setSoftness: (softness: number) => (cloth.bendCompliance = bendCompliance(softness)),
    setParticleView(enabled: boolean) {
      mesh.visible = !enabled;
      dots.visible = enabled;
    },
  };
}

export function buildCloth(ctx: BuildContext, values: Values): Experiment {
  const { particles, cloth, mesh, dots, setSoftness, setParticleView } = velvetCloth(
    ctx,
    [1.26, 1.2],
    (geometry) => {
      // Hang it with a gentle wave so the first folds aren't perfectly regular.
      geometry.translate(0, 1.0, 0);
      const vertices = geometry.getAttribute('position');
      for (let i = 0; i < vertices.count; i++) {
        const drape = (1.6 - vertices.getY(i)) / 1.2;
        vertices.setZ(i, Math.sin(vertices.getX(i) * 18) * 0.018 * drape);
      }
    },
    {
      softness: values['bend']!,
      // Pin the whole top hem, spreading the load instead of hanging it from two corners.
      pinned: (segments) => Array.from({ length: segments + 1 }, (_, i) => i),
      // About 0.7 grid spacings, so contacts overlap smoothly.
      radius: (spacing) => Math.max(0.008, 0.7 * spacing),
      wind: new Vector3(0, 0, values['wind']!),
      drag: 0.18,
      lift: 0.02,
    },
  );
  const ball = new Mesh(
    new SphereGeometry(0.23, 64, 48),
    new MeshPhysicalNodeMaterial({
      color: 0xe3e7ec,
      metalness: 1,
      roughness: 0.035,
      envMapIntensity: 1.35,
    }),
  );
  ball.position.set(0, 0.68, 0.85);
  ball.castShadow = true;
  const colliders = new PrimitiveSet(particles);
  // The fabric's lowest particles rest 5 cm up, level with the stand's base.
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0.05 - particles.particleRadius, 0));
  // Enough grip that touching cloth moves with the ball instead of chattering over it.
  colliders.attach(colliders.addSphere(new Vector3(), 0.23, { muS: 0.4, muK: 0.3 }), ball);
  const substeps = scaledSubsteps(5, ctx.particles, 2),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    materials: [cloth],
    colliders: [colliders],
  });
  return {
    particles,
    loop,
    objects: [mesh, ball, dots, clothStand()],
    particleCount: particles.capacity,
    substeps,
    iterations,
    update(_dt, time) {
      ball.position.set(0, 0.68, Math.cos(time * values['speed']!) * 0.85);
      const gust = 1 + 0.25 * (Math.sin(time * 1.3) * 0.5 + Math.sin(time * 2.7) * 0.25);
      cloth.wind.set(Math.sin(time * 0.8) * values['wind']! * 0.45, 0.12, values['wind']! * gust);
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'bend') setSoftness(value);
    },
    setParticleView,
    dispose() {
      particles.dispose();
      loop.dispose();
    },
  };
}

export async function buildClothDrop(ctx: BuildContext, values: Values): Promise<Experiment> {
  const size = 1.1;
  const { particles, cloth, mesh, dots, spacing, setSoftness, setParticleView } = velvetCloth(
    ctx,
    [size, size],
    (geometry) => {
      // Lying flat and slightly turned, with a faint ripple so the first
      // contact isn't perfectly symmetric.
      geometry
        .rotateX(-Math.PI / 2)
        .rotateY(0.35)
        .translate(0, values['height']!, 0);
      const vertices = geometry.getAttribute('position');
      for (let i = 0; i < vertices.count; i++) {
        const x = vertices.getX(i),
          z = vertices.getZ(i);
        vertices.setY(i, vertices.getY(i) + Math.sin(x * 9 + 0.6) * Math.cos(z * 7) * 0.012);
      }
    },
    {
      softness: values['bend']!,
      // Under half the grid spacing, so neighboring particles never touch and
      // cloth-on-cloth contacts only fire between separate folds.
      radius: (spacing) => 0.45 * spacing,
      // Falling flat, the full drag turns the cloth into a parachute.
      drag: 0.04,
      lift: 0.005,
      // Bleeds off the jitter where stretched cloth fights the bunny, without
      // slowing the cloth's fall or slide as a whole.
      damping: values['damping']!,
    },
  );
  const floor = new PrimitiveSet(particles);
  floor.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0.005, 0), { muS: 0.35, muK: 0.3 });
  const bunny = await loadBunny();
  const bunnyCollider = new SDFCollider(particles, bunny.sdf, {
    rotation: new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), BUNNY_YAW),
    // Keep the gap between neighboring particles clear of the bunny, so its
    // thin ears can't slip between them.
    thickness: 1.3 * spacing - particles.particleRadius,
    muS: values['friction']!,
    muK: values['friction']! * 0.85,
  });
  // Thin ears need short steps and frequent collision solves to stay covered.
  const substeps = scaledSubsteps(12, ctx.particles, 2),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    materials: [cloth],
    colliders: [floor, bunnyCollider],
    contact: { muS: 0.3, muK: 0.2 },
  });
  return {
    particles,
    loop,
    objects: [mesh, bunny.mesh, dots, platform(1.7, 1.7)],
    particleCount: particles.capacity,
    substeps,
    iterations,
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'bend') setSoftness(value);
      if (key === 'damping') cloth.damping = value;
      if (key === 'friction') {
        bunnyCollider.muS = value;
        bunnyCollider.muK = value * 0.85;
      }
    },
    setParticleView,
    dispose() {
      bunnyCollider.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
