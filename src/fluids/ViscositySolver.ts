import { Fn, float, instanceIndex, instancedArray, uniform, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import { emitForEachPair, type Material } from '../core/index.js';
import type { FluidSystem } from './FluidSystem.js';
import { emitPoly6FromRSq } from './sim/kernels.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface ViscositySolverOptions {
  readonly fluid: FluidSystem;
  /** Kinematic diffusion coefficient. Higher values resist relative motion. */
  readonly viscosity: number;
  /** Fixed Jacobi budget for the implicit diffusion solve. */
  readonly iterations?: number;
}

/** Stable velocity diffusion for thick liquids. Register after the fluid material. */
export class ViscositySolver implements Material {
  readonly viscosity: ReturnType<typeof uniform<'float', number>>;
  readonly postAdvectKernels: readonly ComputeNode[];

  constructor({ fluid, viscosity, iterations = 12 }: ViscositySolverOptions) {
    if (
      !Number.isFinite(viscosity) ||
      viscosity < 0 ||
      !Number.isInteger(iterations) ||
      iterations < 1 ||
      iterations > 64
    )
      throw new Error(
        'Viscosity must be nonnegative and iterations must be an integer from 1 to 64.',
      );
    const { particles, fluidParticles, sph, pairList, pairCount, particleVolumeUniform, xpbd } =
      fluid;
    this.viscosity = uniform(viscosity, 'float');
    const rhs = instancedArray(particles.capacity, 'vec4');
    const ping = instancedArray(particles.capacity, 'vec4');
    const pong = instancedArray(particles.capacity, 'vec4');
    const seed = Fn(() => {
      const i: Any = instanceIndex.add(fluidParticles.start);
      const v: Any = particles.velocities.element(i);
      rhs.element(i).assign(v);
      ping.element(i).assign(v);
    })().compute(fluidParticles.count);
    // Jacobi sweep reading one iterate and writing the other, so every
    // iteration is a single dispatch.
    const sweep = (source: Any, target: Any): ComputeNode =>
      Fn(() => {
        const i: Any = instanceIndex.add(fluidParticles.start);
        const position: Any = particles.positions.element(i).xyz;
        const sum: Any = vec3(0).toVar();
        const weight: Any = float(0).toVar();
        emitForEachPair({
          pairList,
          pairCount,
          queryIdx: i,
          fluidStart: fluidParticles.start,
          fluidCount: fluidParticles.count,
          onCandidate: (j: Any) => {
            const delta: Any = position.sub(particles.positions.element(j).xyz);
            const w: Any = emitPoly6FromRSq(delta.dot(delta), sph).mul(particleVolumeUniform);
            sum.addAssign(source.element(j).xyz.mul(w));
            weight.addAssign(w);
          },
        });
        // Backward Euler: (I - dt ν L)v = v₀.
        const rate: Any = this.viscosity
          .mul(xpbd.dt)
          .mul(10 / (fluid.h * fluid.h))
          .max(0);
        const velocity: Any = rhs.element(i).xyz.add(sum.mul(rate)).div(weight.mul(rate).add(1));
        target.element(i).assign(vec4(velocity, 0));
      })().compute(fluidParticles.count);
    const forward = sweep(ping, pong);
    const backward = sweep(pong, ping);
    const last = iterations % 2 === 1 ? pong : ping;
    const apply = Fn(() => {
      const i: Any = instanceIndex.add(fluidParticles.start);
      const velocity: Any = particles.velocities.element(i);
      velocity.assign(vec4(last.element(i).xyz, velocity.w));
    })().compute(fluidParticles.count);
    this.postAdvectKernels = [
      seed,
      ...Array.from({ length: iterations }, (_, k) => (k % 2 === 0 ? forward : backward)),
      apply,
    ];
  }
}
