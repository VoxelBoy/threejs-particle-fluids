import { describe, expect, it } from 'vitest';
import {
  Box3,
  DataTexture,
  EquirectangularReflectionMapping,
  FloatType,
  PerspectiveCamera,
  RGBAFormat,
  Scene,
  Vector2,
  Vector3,
} from 'three';
import {
  FluidSurfaceRenderer,
  FluidSystem,
  ParticleSystem,
  SDFCollider,
  createParticleRenderer,
  type ParticleInit,
  type SDFData,
} from '../../../src/index.js';

// FluidSurfaceRenderer end to end: splat a cube of particles, then pick
// straight down the view axis to find where the drawn surface is.

const R = 0.01;
const SPACING = 2 * R;
const N = 8;

/** An N³ block of particles centred on the origin: it spans ±0.08 m. */
function block(): ParticleInit[] {
  const out: ParticleInit[] = [];
  const offset = ((N - 1) * SPACING) / 2;
  for (let z = 0; z < N; z++)
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++)
        out.push({ position: [x * SPACING - offset, y * SPACING - offset, z * SPACING - offset] });
  return out;
}

/** Analytic SDF of a sphere of `radius` about the local origin. */
function sphereSdf(radius: number): SDFData {
  const n = 24;
  const size = (3 * radius) / n;
  const origin = -1.5 * radius;
  const data = new Float32Array(n * n * n);
  for (let z = 0; z < n; z++)
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const p = [x, y, z].map((i) => origin + (i + 0.5) * size);
        data[x + n * (y + n * z)] = Math.hypot(p[0]!, p[1]!, p[2]!) - radius;
      }
  return {
    data,
    resolution: [n, n, n],
    origin: [origin, origin, origin],
    voxelSize: [size, size, size],
  };
}

async function setup() {
  const renderer = await createParticleRenderer();
  const particles = new ParticleSystem(renderer, N ** 3, R);
  particles.uploadParticles(block());
  const fluid = new FluidSystem(particles);
  const scene = new Scene();
  const camera = new PerspectiveCamera(40, 1, 0.01, 10);
  camera.position.set(0, 0, 1);
  camera.updateMatrixWorld();
  const bounds = new Box3(new Vector3(-0.3, -0.3, -0.3), new Vector3(0.3, 0.3, 0.3));
  return { renderer, particles, fluid, scene, camera, bounds };
}

const centre = new Vector2(0.5, 0.5);

describe('FluidSurfaceRenderer', () => {
  it('draws the surface with no colliders', async () => {
    const { renderer, particles, fluid, scene, camera, bounds } = await setup();
    try {
      const surface = new FluidSurfaceRenderer(fluid, { renderer, scene, camera, bounds });
      await surface.update();
      const hit = await surface.pick(centre);
      expect(hit).not.toBeNull();
      // Front face of the block (z = 0.07 + R) plus the surface radius.
      expect(hit!.z).toBeGreaterThan(0.06);
      expect(hit!.z).toBeLessThan(0.12);
      surface.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('re-wets after an SDFCollider moves', async () => {
    const { renderer, particles, fluid, scene, camera, bounds } = await setup();
    try {
      const collider = new SDFCollider(particles, sphereSdf(0.1), {
        position: new Vector3(0, 0, 2),
      });
      const surface = new FluidSurfaceRenderer(fluid, {
        renderer,
        scene,
        camera,
        bounds,
        colliders: [collider],
      });
      await surface.update();
      const before = await surface.pick(centre);
      expect(before).not.toBeNull();
      expect(before!.z).toBeGreaterThan(0.06);

      // The sphere now covers z ∈ [−0.02, 0.18] on the axis, so the liquid
      // is cut back to the part of the block behind it.
      collider.setPosition(new Vector3(0, 0, 0.08));
      await surface.update();
      const after = await surface.pick(centre);
      expect(after).not.toBeNull();
      expect(after!.z).toBeLessThan(0);
      surface.dispose();
      collider.dispose();
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('follows scene.environment after construction', async () => {
    const { renderer, particles, fluid, scene, camera, bounds } = await setup();
    try {
      const surface = new FluidSurfaceRenderer(fluid, { renderer, scene, camera, bounds });
      scene.add(surface.mesh);
      const material = surface.mesh.material as unknown as { fragmentNode: unknown };
      await surface.update();
      renderer.render(scene, camera);
      const gradient = material.fragmentNode;

      const environment = new DataTexture(new Float32Array(16 * 8 * 4).fill(1), 16, 8, RGBAFormat);
      environment.type = FloatType;
      environment.mapping = EquirectangularReflectionMapping;
      environment.needsUpdate = true;
      scene.environment = environment;
      await surface.update();
      expect(material.fragmentNode).not.toBe(gradient);
      renderer.render(scene, camera);

      // Swapping one map for another keeps the shader.
      const withMap = material.fragmentNode;
      scene.environment = environment.clone();
      await surface.update();
      expect(material.fragmentNode).toBe(withMap);
      renderer.render(scene, camera);

      surface.dispose();
      expect(surface.mesh.parent).toBeNull();
      await expect(surface.update()).rejects.toThrow('FluidSurfaceRenderer: already disposed');
      surface.dispose(); // idempotent
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);

  it('validates its settings', async () => {
    const { renderer, particles, fluid, scene, camera, bounds } = await setup();
    try {
      const plain = new FluidSurfaceRenderer(fluid, { renderer, scene, camera, bounds });
      expect(() => (plain.smokeDensity = 1)).toThrow(/pass `cavities` in the options/);
      expect(() => plain.setAppearance({ ior: NaN })).toThrow(
        'FluidSurfaceRenderer: appearance.ior must be finite',
      );
      plain.dispose();

      const smoky = new FluidSurfaceRenderer(fluid, {
        renderer,
        scene,
        camera,
        bounds,
        cavities: { smokeColor: 0x808080, smokeDensity: 2 },
      });
      expect(smoky.smokeDensity).toBe(2);
      smoky.smokeDensity = 3;
      expect(smoky.smokeDensity).toBe(3);
      expect(() => (smoky.smokeDensity = -1)).toThrow(
        'FluidSurfaceRenderer: smokeDensity must be ≥ 0',
      );
      smoky.dispose();

      expect(
        () =>
          new FluidSurfaceRenderer(fluid, { renderer, scene, camera, bounds, motionStretch: -1 }),
      ).toThrow('FluidSurfaceRenderer: motionStretch must be ≥ 0');
      expect(
        () =>
          new FluidSurfaceRenderer(fluid, {
            renderer,
            scene,
            camera,
            bounds: new Box3(new Vector3(0, 0, 0), new Vector3(1, 0, 1)),
          }),
      ).toThrow('FluidSurfaceRenderer: bounds must have positive extent');
      particles.dispose();
    } finally {
      renderer.dispose();
    }
  }, 60_000);
});
