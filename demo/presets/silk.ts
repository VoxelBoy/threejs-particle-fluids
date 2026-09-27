import { DoubleSide, Mesh, PlaneGeometry, SphereGeometry, Vector3 } from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { ParticleSystem, PrimitiveSet, SimLoop, createXpbdUniforms } from '../../src/core/index.js';
import { ClothSystem, createClothSurface, fromBufferGeometry } from '../../src/cloth/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { clothStand } from '../runtime/stage.js';
import { scaledSubsteps } from './shared.js';
import type { BuildContext, Experiment, Values } from '../types.js';

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
  const mat = new MeshPhysicalNodeMaterial({
    color: 0x870b21,
    side: DoubleSide,
    roughness: 0.9,
    metalness: 0,
    sheen: 1,
    sheenColor: 0xd54b5c,
    sheenRoughness: 0.7,
    flatShading: false,
  });
  const mesh = createClothSurface({
    particles,
    columns: segments + 1,
    rows: segments + 1,
    subdivisions: 3,
    material: mat,
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
