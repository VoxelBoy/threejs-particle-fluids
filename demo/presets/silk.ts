import { DoubleSide, Mesh, PlaneGeometry, Vector3 } from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { color, mix, sin, uv, vertexIndex } from 'three/tsl';
import { ParticleSystem, SimLoop, createXpbdUniforms } from '../../src/core/index.js';
import { ClothSystem, fromBufferGeometry } from '../../src/cloth/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { block } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';

export function buildCloth(ctx: BuildContext, values: Values): Experiment {
  const segments = ctx.quality === 'high' ? 42 : 30;
  const geometry = new PlaneGeometry(1.28, 1.18, segments, segments).translate(0, 1.02, 0);
  const positions = geometry.getAttribute('position');
  for (let i = 0; i < positions.count; i++) {
    const v = (1.61 - positions.getY(i)) / 1.18;
    positions.setZ(i, Math.sin(positions.getX(i) * 18) * 0.035 * v);
  }
  geometry.computeVertexNormals();
  const pinnedIndices = Array.from({ length: segments + 1 }, (_, i) => i).filter(
    (i) => i < 3 || i > segments - 3,
  );
  const graph = fromBufferGeometry(geometry, { surfaceDensity: 0.35, pinnedIndices });
  const particles = new ParticleSystem(ctx.renderer, graph.positions.length, 0.018);
  particles.uploadParticles(
    graph.positions.map((position, i) => ({
      position,
      velocity: [0, 0, 0],
      invMass: graph.invMass[i]!,
      phase: 1,
    })),
  );
  const xpbd = createXpbdUniforms(1 / 60);
  const cloth = new ClothSystem({
    particles,
    xpbd,
    graph,
    particleOffset: 0,
    stretchCompliance: 1e-7,
    bendCompliance: 10 ** (-6 + values['bend']! * 3),
    stretchTolerance: 0.06,
    wind: new Vector3(0.5, 0, values['wind']!),
    dragCoeff: 0.18,
    liftCoeff: 0.02,
  });
  const substeps = 4,
    iterations = 3;
  const loop = new SimLoop(particles, { substeps, iterations, xpbd, materials: [cloth] });
  loop.gravity.set(0, -values['gravity']!, 0);
  loop.kernels.floorY.value = 0.05;
  const mat = new MeshPhysicalNodeMaterial({
    color: 0xbba4d5,
    side: DoubleSide,
    roughness: 0.36,
    metalness: 0.12,
    sheen: 1,
    sheenColor: 0xf3cfef,
    sheenRoughness: 0.45,
    flatShading: true,
  });
  mat.positionNode = particles.positions.element(vertexIndex).xyz;
  mat.colorNode = mix(color(0x9986b9), color(0xd7b9d5), sin(uv().y.mul(5)).mul(0.5).add(0.5));
  const mesh = new Mesh(geometry, mat);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  const dots = createParticleMesh({ particles, radius: 0.012, color: 0xcbbde8, castShadow: false });
  dots.visible = false;
  const objects = [
    mesh,
    dots,
    block([1.65, 0.1, 0.65], [0, -0.07, 0], 0x69717e),
    block([0.022, 1.7, 0.022], [-0.71, 0.8, 0], 0x9698ac, 0.007),
    block([0.022, 1.7, 0.022], [0.71, 0.8, 0], 0x9698ac, 0.007),
    block([1.46, 0.022, 0.022], [0, 1.62, 0], 0x9698ac, 0.007),
  ];
  return {
    particles,
    loop,
    objects,
    particleCount: particles.capacity,
    substeps,
    iterations,
    update(_dt, time) {
      const gust = 1 + values['gust']! * (Math.sin(time * 1.3) * 0.5 + Math.sin(time * 2.7) * 0.25);
      cloth.wind.value.set(
        Math.sin(time * 0.8) * values['wind']! * 0.45,
        0.12,
        values['wind']! * gust,
      );
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
    },
    setParticleView(enabled) {
      mesh.visible = !enabled;
      dots.visible = enabled;
    },
    dispose() {
      particles.destroy();
    },
  };
}
