import { Fn, float, instancedArray, uniform, vec3, vec4 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import { emitPoly6FromRSq, type Material, type MaterialKernels } from '../core/index.js';
import type { FluidSystem } from './FluidSystem.js';
import { emitFluidIndex } from './sim/shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface ViscositySolverOptions {
  /** Kinematic viscosity. Higher values resist relative motion more. Honey is around 20. */
  readonly viscosity: number;
  /** Jacobi sweeps per substep, 1–64. Default 12. Finer particles need more. */
  readonly iterations?: number;
}

/**
 * Implicit viscosity for thick liquids such as honey. Each substep solves
 * `(I − dt ν L) v = v₀` for the fluid's velocities with Jacobi sweeps, which
 * stays stable at viscosities where {@link FluidSystem}'s explicit
 * `viscosity` would blow up.
 *
 * List it after its fluid in the {@link SimLoop}'s `materials`.
 */
export class ViscositySolver implements Material {
  readonly fluid: FluidSystem;
  private readonly viscosityUniform: UniformNode<'float', number>;
  private readonly iterations: number;

  constructor(fluid: FluidSystem, options: ViscositySolverOptions) {
    const iterations = options.iterations ?? 12;
    if (!(options.viscosity >= 0) || !Number.isFinite(options.viscosity)) {
      throw new Error(`ViscositySolver: viscosity must be ≥ 0, got ${options.viscosity}`);
    }
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 64) {
      throw new Error(
        `ViscositySolver: iterations must be an integer from 1 to 64, got ${iterations}`,
      );
    }
    this.fluid = fluid;
    this.viscosityUniform = uniform(options.viscosity, 'float');
    this.iterations = iterations;
  }

  get viscosity(): number {
    return this.viscosityUniform.value;
  }
  set viscosity(value: number) {
    this.viscosityUniform.value = value;
  }

  build(): MaterialKernels {
    let context;
    try {
      context = this.fluid.kernelContext;
    } catch {
      throw new Error('ViscositySolver: list it after its FluidSystem in `materials`');
    }
    const { particles, range, neighbors, sph, particleVolume, dt } = context;
    const h = this.fluid.smoothingRadius;
    const rhs = instancedArray(particles.capacity, 'vec4');
    const ping = instancedArray(particles.capacity, 'vec4');
    const pong = instancedArray(particles.capacity, 'vec4');

    const seed = Fn(() => {
      const i: Any = emitFluidIndex(context);
      const v: Any = particles.velocities.element(i);
      rhs.element(i).assign(v);
      ping.element(i).assign(v);
    })().compute(range.count);

    // Each sweep reads one iterate and writes the other, so a sweep is one dispatch.
    const sweep = (source: Any, target: Any): ComputeNode =>
      Fn(() => {
        const i: Any = emitFluidIndex(context);
        const position: Any = particles.positions.element(i).xyz;
        const sum: Any = vec3(0).toVar();
        const weight: Any = float(0).toVar();
        neighbors.forEach(i, (j: Any) => {
          const offset: Any = position.sub(particles.positions.element(j).xyz);
          const w: Any = emitPoly6FromRSq(offset.dot(offset), sph).mul(particleVolume);
          sum.addAssign(source.element(j).xyz.mul(w));
          weight.addAssign(w);
        });
        const rate: Any = this.viscosityUniform
          .mul(dt)
          .mul(10 / (h * h))
          .max(0);
        target
          .element(i)
          .assign(vec4(rhs.element(i).xyz.add(sum.mul(rate)).div(weight.mul(rate).add(1)), 0));
      })().compute(range.count);
    const forward = sweep(ping, pong);
    const backward = sweep(pong, ping);
    const result = this.iterations % 2 === 1 ? pong : ping;

    const apply = Fn(() => {
      const i: Any = emitFluidIndex(context);
      const velocity: Any = particles.velocities.element(i);
      velocity.assign(vec4(result.element(i).xyz, velocity.w));
    })().compute(range.count);

    return {
      postSolve: [
        seed,
        ...Array.from({ length: this.iterations }, (_, k) => (k % 2 === 0 ? forward : backward)),
        apply,
      ],
    };
  }
}
