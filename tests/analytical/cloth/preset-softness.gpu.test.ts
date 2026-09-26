import { expect, it } from 'vitest';
import { PerspectiveCamera, Scene, Vector3 } from 'three';
import { createParticleRenderer } from '../../../src/core/index.js';
import { buildCloth } from '../../../demo/presets/silk.js';
import { disposeObjects } from '../../../demo/runtime/stage.js';

it('the fabric softness control changes folding while the supported hem stays fixed', async () => {
  const renderer = await createParticleRenderer();
  const folds: number[] = [];
  try {
    for (const bend of [0, 1]) {
      const experiment = buildCloth(
        { renderer, scene: new Scene(), camera: new PerspectiveCamera(), particles: 961 },
        { bend, wind: 0.3, speed: 0.8, gravity: 6 },
      );
      try {
        const initial = Float32Array.from(experiment.particles.positions.value.array);
        for (let frame = 0; frame < 150; frame++) {
          await experiment.update!(1 / 60, frame / 60);
          await experiment.loop.step(1 / 60);
        }
        const { positions } = await experiment.particles.readback();
        expect(Array.from(positions).every(Number.isFinite)).toBe(true);
        for (let i = 0; i < 31; i++)
          for (let axis = 0; axis < 3; axis++)
            expect(positions[i * 4 + axis]).toBeCloseTo(initial[i * 4 + axis]!, 5);
        const normals: Vector3[] = [];
        for (let y = 0; y < 30; y++)
          for (let x = 0; x < 30; x++) {
            const index = (y * 31 + x) * 4;
            const origin = new Vector3().fromArray(positions, index);
            normals.push(
              new Vector3()
                .fromArray(positions, index + 4)
                .sub(origin)
                .cross(new Vector3().fromArray(positions, index + 124).sub(origin))
                .normalize(),
            );
          }
        let total = 0,
          count = 0;
        for (let y = 0; y < 29; y++)
          for (let x = 0; x < 29; x++)
            for (const neighbor of [y * 30 + x + 1, (y + 1) * 30 + x]) {
              total += normals[y * 30 + x]!.angleTo(normals[neighbor]!) ** 2;
              count++;
            }
        folds.push(Math.sqrt(total / count));
      } finally {
        experiment.dispose();
        disposeObjects(experiment.objects);
      }
    }
    // RMS angles between adjacent patches measure folds independent of camera.
    expect(folds[1]!).toBeGreaterThan(folds[0]! * 1.6);
  } finally {
    renderer.dispose();
  }
}, 60_000);
