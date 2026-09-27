import { describe, expect, it } from 'vitest';
import { Fn, instanceIndex, uniform, vec3, vec4 } from 'three/tsl';
import {
  ParticleSystem,
  SoftbodySystem,
  createParticleRenderer,
  type SolverContext,
} from '../../../src/index.js';

/** The solver state SimLoop hands a material, so its kernels can run without a loop. */
function solverContext(particles: ParticleSystem, dt: number): SolverContext {
  let group = 0;
  return {
    particles,
    dt: uniform(dt, 'float'),
    get hashGrid(): never {
      throw new Error('soft bodies do not use the neighbor grid');
    },
    allocateCollisionGroup: () => ++group,
  };
}

// Internal shape constraints must not undo a translation introduced by another
// material between iterations (for example, the reaction from fluid pressure).
describe('soft-body coupling preserves external translations', () => {
  for (const mode of ['global', 'local'] as const)
    it(mode, async () => {
      const renderer = await createParticleRenderer();
      const particles = new ParticleSystem(renderer, 27, 0.02);
      try {
        const rest = new Float32Array(81),
          edges: number[] = [];
        for (let z = 0; z < 3; z++)
          for (let y = 0; y < 3; y++)
            for (let x = 0; x < 3; x++) {
              const i = x + y * 3 + z * 9;
              rest.set([(x - 1) * 0.04, (y - 1) * 0.04, (z - 1) * 0.04], i * 3);
              if (x < 2) edges.push(i, i + 1);
              if (y < 2) edges.push(i, i + 3);
              if (z < 2) edges.push(i, i + 9);
            }
        particles.uploadParticles(
          Array.from({ length: 27 }, (_, i) => ({
            position: [rest[i * 3]!, rest[i * 3 + 1]!, rest[i * 3 + 2]!] as const,
            invMass: 10,
          })),
        );
        const body = new SoftbodySystem(particles, {
          shapeMatching: mode,
          selfCollision: true,
          bodies: [
            {
              range: { start: 0, count: 27 },
              restPositions: rest,
              compliance: 1e-9,
              edges: Uint32Array.from(edges),
            },
          ],
        });
        const kernels = body.build(solverContext(particles, 1 / 60));
        await renderer.computeAsync([...(kernels.preSolve ?? [])]);
        const translate = Fn(() => {
          const p = particles.predictedPositions.element(instanceIndex);
          p.assign(vec4(p.xyz.add(vec3(0.12, 0.23, -0.08)), p.w));
        })().compute(27);
        await renderer.computeAsync(translate);
        for (let i = 0; i < 4; i++) await renderer.computeAsync([...(kernels.solve ?? [])]);
        const data = new Float32Array(
          await renderer.getArrayBufferAsync(particles.predictedPositions.value),
        );
        const mean = [0, 0, 0];
        for (let i = 0; i < 27; i++) for (let k = 0; k < 3; k++) mean[k]! += data[i * 4 + k]! / 27;
        expect(mean[0]).toBeCloseTo(0.12, 4);
        expect(mean[1]).toBeCloseTo(0.23, 4);
        expect(mean[2]).toBeCloseTo(-0.08, 4);
      } finally {
        particles.dispose();
        renderer.dispose();
      }
    });
});
