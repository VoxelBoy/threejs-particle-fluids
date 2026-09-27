import { DoubleSide, Mesh, PlaneGeometry, Quaternion, SphereGeometry, Vector3 } from 'three';
import { Fn, instanceIndex, instancedArray, mix, select, uniform, vec4 } from 'three/tsl';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import {
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SDFCollider,
  SimLoop,
  createXpbdUniforms,
} from '../../src/core/index.js';
import { ClothSystem, createClothSurface, fromBufferGeometry } from '../../src/cloth/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { clothStand, platform } from '../runtime/stage.js';
import { BUNNY_YAW, loadBunny } from './honey.js';
import { scaledSubsteps } from './shared.js';
import type { BuildContext, Experiment, Values } from '../types.js';

// TSL's generated operator chains need the broad node type at graph boundaries.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

function velvet(): MeshPhysicalNodeMaterial {
  return new MeshPhysicalNodeMaterial({
    color: 0x870b21,
    side: DoubleSide,
    roughness: 0.9,
    metalness: 0,
    sheen: 1,
    sheenColor: 0xd54b5c,
    sheenRoughness: 0.7,
    flatShading: false,
  });
}

export function buildCloth(ctx: BuildContext, values: Values): Experiment {
  // A (segments + 1)² grid of particles close to the requested count.
  const segments = Math.max(12, Math.round(Math.sqrt(ctx.particles)) - 1);
  const geometry = new PlaneGeometry(1.26, 1.2, segments, segments).translate(0, 1.0, 0);
  // Distribute the load across a supported hem instead of concentrating all
  // tension in two tiny corner patches. The rows below it remain free to fold.
  const pinnedIndices = Array.from({ length: segments + 1 }, (_, i) => i);
  const vertices = geometry.getAttribute('position');
  for (let i = 0; i < vertices.count; i++) {
    const drape = (1.6 - vertices.getY(i)) / 1.2;
    vertices.setZ(i, Math.sin(vertices.getX(i) * 18) * 0.018 * drape);
  }
  const graph = fromBufferGeometry(geometry, { surfaceDensity: 0.08, pinnedIndices });
  // Collision thickness: about 0.7 grid spacings, so contacts overlap smoothly.
  const particles = new ParticleSystem(
    ctx.renderer,
    graph.positions.length,
    Math.max(0.008, 0.7 * (1.26 / segments)),
  );
  particles.uploadParticles(
    graph.positions.map((position, i) => ({
      position,
      velocity: [0, 0, 0],
      invMass: graph.invMass[i]!,
      phase: 1,
    })),
  );
  const xpbd = createXpbdUniforms(1 / 60);
  // Angular-constraint gradients scale with inverse edge length, and particle
  // inverse mass scales with inverse area. Account for both when refining the grid.
  const bendCompliance = (softness: number) =>
    10 ** (-1 + softness * 5) * (segments / 30) ** 4 * (0.35 / 0.08);
  const cloth = new ClothSystem({
    particles,
    xpbd,
    graph,
    particleOffset: 0,
    stretchCompliance: 1e-7,
    bendCompliance: bendCompliance(values['bend']!),
    stretchTolerance: 0.06,
    wind: new Vector3(0, 0, values['wind']!),
    dragCoeff: 0.18,
    liftCoeff: 0.02,
  });
  const substeps = scaledSubsteps(5, ctx.particles, 2),
    iterations = 2;
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
  const colliders = new PrimitiveSet(particles, { capacity: 2 });
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0.01, 0));
  // Enough grip that contacting cloth moves with the ball instead of chattering over it.
  const sphere = colliders.addSphere(ball.position, 0.23, { muS: 0.4, muK: 0.3 });
  colliders.attachToObject3D(sphere, ball);
  colliders.upload();
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    materials: [cloth],
    colliders: { colliders },
  });
  loop.gravity.set(0, -values['gravity']!, 0);
  loop.kernels.floorY.value = 0.05;
  const mesh = createClothSurface({
    particles,
    columns: segments + 1,
    rows: segments + 1,
    subdivisions: 3,
    smooth: true,
    material: velvet(),
  });
  geometry.dispose();
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  const dots = createParticleMesh({ particles, radius: 0.012, color: 0xd54b5c, castShadow: false });
  dots.visible = false;
  const objects = [mesh, ball, dots, clothStand()];
  return {
    particles,
    loop,
    objects,
    particleCount: particles.capacity,
    substeps,
    iterations,
    update(dt, time) {
      const phase = time * values['speed']!;
      ball.position.set(0, 0.68, Math.cos(phase) * 0.85);
      ball.updateMatrixWorld();
      colliders.updateKinematics(dt);
      colliders.upload();
      const gust = 1 + 0.25 * (Math.sin(time * 1.3) * 0.5 + Math.sin(time * 2.7) * 0.25);
      cloth.wind.value.set(
        Math.sin(time * 0.8) * values['wind']! * 0.45,
        0.12,
        values['wind']! * gust,
      );
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'bend' && cloth.bending) {
        (cloth.bending.compliance.value.array as Float32Array).fill(bendCompliance(value));
        cloth.bending.compliance.value.needsUpdate = true;
      }
    },
    setParticleView(enabled) {
      mesh.visible = !enabled;
      dots.visible = enabled;
    },
    dispose() {
      particles.destroy();
      colliders.destroy();
    },
  };
}

export async function buildClothDrop(ctx: BuildContext, values: Values): Promise<Experiment> {
  // A loose square of velvet, lying flat and slightly turned, released above the bunny.
  const size = 1.1;
  const segments = Math.max(12, Math.round(Math.sqrt(ctx.particles)) - 1);
  const geometry = new PlaneGeometry(size, size, segments, segments)
    .rotateX(-Math.PI / 2)
    .rotateY(0.35)
    .translate(0, values['height']!, 0);
  // A faint ripple keeps the first contact from being perfectly symmetric.
  const vertices = geometry.getAttribute('position');
  for (let i = 0; i < vertices.count; i++) {
    const x = vertices.getX(i),
      z = vertices.getZ(i);
    vertices.setY(i, vertices.getY(i) + Math.sin(x * 9 + 0.6) * Math.cos(z * 7) * 0.012);
  }
  const graph = fromBufferGeometry(geometry, { surfaceDensity: 0.08 });
  const spacing = size / segments;
  // Under half the grid spacing, so neighbouring particles never touch and
  // cloth-on-cloth contacts only fire between separate folds. A particle from
  // another fold still can't pass through a grid cell's centre.
  const radius = 0.45 * spacing;
  const particles = new ParticleSystem(ctx.renderer, graph.positions.length, radius);
  particles.uploadParticles(
    graph.positions.map((position, i) => ({
      position,
      velocity: [0, 0, 0],
      invMass: graph.invMass[i]!,
      phase: 1,
    })),
  );
  const xpbd = createXpbdUniforms(1 / 60);
  const bendCompliance = (softness: number) =>
    10 ** (-1 + softness * 5) * (segments / 30) ** 4 * (0.35 / 0.08);
  const cloth = new ClothSystem({
    particles,
    xpbd,
    graph,
    particleOffset: 0,
    stretchCompliance: 1e-7,
    bendCompliance: bendCompliance(values['bend']!),
    stretchTolerance: 0.06,
    wind: new Vector3(),
    // Falling flat, the full drag turns the cloth into a parachute.
    dragCoeff: 0.04,
    liftCoeff: 0.005,
  });
  const colliders = new PrimitiveSet(particles, { capacity: 1 });
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0.005, 0), { muS: 0.35, muK: 0.3 });
  colliders.upload();
  const bunny = await loadBunny();
  const bunnyCollider = new SDFCollider(particles, bunny.sdf, {
    rotation: new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), BUNNY_YAW),
    // Keep the gap between neighbouring particles clear of the bunny, so its
    // thin ears can't slip between them.
    thickness: 1.3 * spacing - radius,
    muS: values['friction']!,
    muK: values['friction']! * 0.85,
  });
  // Thin ears need short steps and frequent collision solves to stay covered.
  // Pull each particle's velocity toward its four grid neighbours' after every
  // substep. This bleeds off the jitter where stretched cloth fights the
  // bunny, without slowing the cloth's fall or slide as a whole.
  const columns = segments + 1;
  const damping = uniform(values['damping']!);
  const smoothed = instancedArray(particles.capacity, 'vec4');
  const smooth = Fn(() => {
    const i: Any = instanceIndex;
    const column: Any = i.mod(columns);
    const row: Any = i.div(columns);
    const at = (valid: Any, j: Any): Any => particles.velocities.element(select(valid, j, i)).xyz;
    const v: Any = particles.velocities.element(i);
    const average: Any = at(column.greaterThan(0), i.sub(1))
      .add(at(column.lessThan(columns - 1), i.add(1)))
      .add(at(row.greaterThan(0), i.sub(columns)))
      .add(at(row.lessThan(columns - 1), i.add(columns)))
      .mul(0.25);
    smoothed.element(i).assign(vec4(mix(v.xyz, average, damping), v.w));
  })().compute(particles.capacity);
  const apply = Fn(() => {
    particles.velocities.element(instanceIndex).assign(smoothed.element(instanceIndex));
  })().compute(particles.capacity);
  const hashGrid = new HashGrid(particles, { cellSize: radius * 4 });
  const substeps = scaledSubsteps(12, ctx.particles, 2),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    hashGrid,
    materials: [cloth, { postAdvectKernels: [smooth, apply] }],
    contact: {
      hashGrid,
      maxContacts: particles.capacity * 8,
      friction: { muS: 0.3, muK: 0.2 },
    },
    colliders: { colliders, sdfColliders: [bunnyCollider] },
  });
  loop.gravity.set(0, -values['gravity']!, 0);
  loop.kernels.floorY.value = -1e9;
  const mesh = createClothSurface({
    particles,
    columns: segments + 1,
    rows: segments + 1,
    subdivisions: 3,
    smooth: true,
    material: velvet(),
  });
  geometry.dispose();
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  const dots = createParticleMesh({ particles, radius: 0.012, color: 0xd54b5c, castShadow: false });
  dots.visible = false;
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
      if (key === 'bend' && cloth.bending) {
        (cloth.bending.compliance.value.array as Float32Array).fill(bendCompliance(value));
        cloth.bending.compliance.value.needsUpdate = true;
      }
      if (key === 'friction') {
        bunnyCollider.muSUniform.value = value;
        bunnyCollider.muKUniform.value = value * 0.85;
      }
      if (key === 'damping') damping.value = value;
    },
    setParticleView(enabled) {
      mesh.visible = !enabled;
      dots.visible = enabled;
    },
    dispose() {
      particles.destroy();
      colliders.destroy();
      bunnyCollider.destroy();
      hashGrid.destroy();
    },
  };
}
