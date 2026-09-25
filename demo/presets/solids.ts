import { SphereGeometry, TorusGeometry, type Object3D } from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import {
  HashGrid,
  ParticleSystem,
  SimLoop,
  createXpbdUniforms,
  type ParticleInit,
} from '../../src/core/index.js';
import { FluidSystem } from '../../src/fluids/index.js';
import {
  SoftbodyMesh,
  SoftbodySystem,
  voxelize,
  type SoftbodyDef,
} from '../../src/softbody/index.js';
import { createParticleMesh } from '../../src/render/particles.js';
import { basin, pedestal } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { liquidVisual } from './liquids.js';
import { lattice, tank, triangleMesh } from './shared.js';

export function buildBodies(ctx: BuildContext, values: Values, withWater: boolean): Experiment {
  const radius = ctx.quality === 'high' ? 0.016 : 0.021;
  const spacing = radius * 2;
  const initial: ParticleInit[] = withWater
    ? lattice([-0.76, radius, -0.5], [0.76, 0.34, 0.5], spacing)
    : [];
  const waterCount = initial.length;
  const height = withWater ? 0.65 : values['height']!;
  const geometries = [
    new SphereGeometry(0.19, 32, 24).translate(-0.46, height, 0.04),
    new RoundedBoxGeometry(0.32, 0.36, 0.32, 3, 0.065)
      .rotateY(0.3)
      .translate(0, height + 0.22, -0.09),
    new TorusGeometry(0.15, 0.075, 16, 36)
      .rotateX(-Math.PI / 2)
      .rotateZ(0.25)
      .translate(0.45, height + 0.4, 0.06),
  ];
  const bodies: SoftbodyDef[] = geometries.map((geometry, index) => {
    const voxels = voxelize(triangleMesh(geometry), { particleRadius: radius });
    const start = initial.length;
    const phaseId = (index + 1) << 16;
    for (let i = 0; i < voxels.count; i++) {
      initial.push({
        position: [
          voxels.positions[i * 3]!,
          voxels.positions[i * 3 + 1]!,
          voxels.positions[i * 3 + 2]!,
        ],
        velocity: [0, 0, 0],
        invMass: withWater ? 70 / values['weight']! : 10,
        phase: phaseId,
      });
    }
    return {
      particleRange: { start, count: voxels.count },
      restPositions: voxels.positions,
      surfaceFlag: voxels.surfaceFlag,
      phaseId,
      matchCompliance: withWater ? 0.0001 : 10 ** (-5 + values['softness']! * 3),
      edges: voxels.edges,
    };
  });
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const xpbd = createXpbdUniforms(1 / 60);
  const hashGrid = new HashGrid(particles, { cellSize: radius * 4 });
  const softbody = new SoftbodySystem({ particles, xpbd, bodies, shapeMatchMode: 'explicit' });
  const fluid = withWater
    ? new FluidSystem({
        particles,
        hashGrid,
        xpbd,
        restDensity: 1000,
        particleSpacing: spacing,
        h: radius * 4,
        compliance: 1e-4,
        fluidParticles: { start: 0, count: waterCount },
        xsph: { c: values['viscosity']! },
        surfaceTension: values['tension']!,
      })
    : undefined;
  if (fluid) bodies.forEach((_, i) => fluid.registerBoundaryParticles(softbody.surfaceRange(i)));
  const colliders = tank(particles, 0.8, 0.55);
  colliders.upload();
  const substeps = 4,
    iterations = 3;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    xpbd,
    hashGrid,
    materials: fluid ? [fluid, softbody] : [softbody],
    colliders: { colliders },
    contact: {
      hashGrid,
      maxContacts: initial.length * 8,
      friction: { muS: 0.35, muK: 0.2 },
      ...(fluid ? { emittingRanges: bodies.map((body) => body.particleRange) } : {}),
    },
  });
  loop.kernels.floorY.value = -1e9;
  loop.gravity.set(0, -values['gravity']!, 0);
  const colors = withWater ? [0xeacda0, 0xdca58c, 0xc8cec2] : [0xce7992, 0xb596cd, 0xdbaf8f];
  const meshes = geometries.map(
    (geometry, i) =>
      new SoftbodyMesh({
        geometry,
        softbody,
        bodyIndex: i,
        color: colors[i]!,
        roughness: 0.24,
        metalness: 0.12,
        reachRadius: radius * 2.5,
      }),
  );
  const objects: Object3D[] = [withWater ? basin() : pedestal(0.95), ...meshes];
  const visual = fluid ? liquidVisual(ctx, fluid, 0x358e80, values['roughness']) : undefined;
  if (visual) objects.push(visual.surface.mesh, visual.dots);
  const dots = createParticleMesh({
    particles,
    radius,
    color: 0xdabec8,
    widthSegments: 8,
    heightSegments: 6,
    castShadow: false,
  });
  dots.visible = false;
  objects.push(dots);
  return {
    particles,
    loop,
    objects,
    particleCount: initial.length,
    substeps,
    iterations,
    prepareRender: () => visual?.prepareRender(),
    setParticleView(enabled) {
      meshes.forEach((mesh) => {
        mesh.visible = !enabled;
      });
      visual?.setParticleView(enabled);
      // This mesh covers all particles, including the fluid range.
      if (visual) visual.dots.visible = false;
      dots.visible = enabled;
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'viscosity' && fluid?.xsphCUniform) fluid.xsphCUniform.value = value;
      if (key === 'tension') fluid?.cohesion?.setGamma(value);
      if (key === 'roughness' && visual) visual.surface.params.surface.roughness.value = value;
    },
    dispose() {
      visual?.dispose();
      particles.destroy();
      hashGrid.destroy();
      colliders.destroy();
    },
  };
}
