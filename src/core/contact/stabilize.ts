import { Fn, If, Return, atomicLoad, bool, float, instanceIndex, uint, vec3 } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';

import type { ParticleSystem } from '../particles.js';
import type { ContactBuffer } from './ContactBuffer.js';
import { type ContactAccumulator, emitAccumulateDelta } from './accumulator.js';
import { emitGeometrySelection, type ContactGeometryExtension } from './extension.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Build the scatter-mode contact stabilization kernel (paper Macklin 2014
 * §4.4 "Initial Conditions" pre-stabilization pass).
 *
 * Differs from the solve kernel in three ways:
 *   1. Reads `particles.positions` (x) instead of `predictedPositions`
 *      (x*) for the constraint-distance calc — paper §4.4: "solving the
 *      resulting contact constraints with the **original, rather than
 *      predicted** positions."
 *   2. No friction — stabilization only resolves normal interpenetration.
 *   3. Caller applies the accumulator to **both** `positions` and
 *      `predictedPositions` via `buildApplyAccumulatorToBothKernel`.
 *
 * Phase 15a — accepts the same {@link ContactGeometryExtension} list as
 * the solve kernel. Stabilization needs the geometry-mode hooks too:
 * a first-iter rigid-rigid interpenetration recovery should use the SDF
 * `(n, d)` (eqs. 17–20), not the spherical eq. 22 normal — otherwise
 * stabilize scatters a wrong-direction correction that the iter-loop
 * SDF solve has to undo.
 *
 * Same scatter pattern as solve: one thread per contact, `atomicAdd` into
 * the per-particle fixed-point accumulator.
 */
export function buildContactStabilizeKernel(args: {
  readonly particles: ParticleSystem;
  readonly contacts: ContactBuffer;
  readonly accumulator: ContactAccumulator;
  readonly geometryExtensions?: readonly ContactGeometryExtension[];
}): ComputeNode {
  const { particles, contacts, accumulator } = args;
  const geometryExtensions = args.geometryExtensions ?? [];
  const twoR = 2 * particles.particleRadius;
  const maxContacts = contacts.maxContacts;

  return Fn(() => {
    const c: Any = instanceIndex;
    const nRaw: Any = atomicLoad(contacts.counter.element(uint(0)));
    If(c.greaterThanEqual(nRaw).or(c.greaterThanEqual(uint(maxContacts))), () => {
      Return();
    });

    const rec: Any = contacts.records.element(c);
    const i: Any = rec.get('i').toVar();
    const j: Any = rec.get('j').toVar();

    // Geometry selection — same protocol as solve. The spherical default
    // here reads ORIGINAL positions (paper §4.4) instead of
    // `predictedPositions`, and uses `invMass` (not `contactInvMass`) so
    // §4.4 stabilization sees physical mass, not the §5.2 stack-scaled
    // override that has no place in initial-condition repair.
    const outN: Any = vec3(0, 0, 0).toVar();
    const outD: Any = float(0).toVar();
    const outHandled: Any = bool(false).toVar();

    emitGeometrySelection(
      {
        i,
        j,
        c,
        particles,
        outN,
        outD,
        outHandled,
      },
      geometryExtensions,
      () => {
        const xi: Any = particles.positions.element(i).xyz.toVar();
        const xj: Any = particles.positions.element(j).xyz.toVar();
        const diff: Any = xi.sub(xj).toVar();
        const len: Any = diff.length().toVar();
        const C: Any = len.sub(float(twoR)).toVar();
        const wSum: Any = particles.invMass.element(i).add(particles.invMass.element(j)).toVar();
        const valid: Any = len
          .greaterThan(float(1e-8))
          .and(C.lessThan(float(0.0)))
          .and(wSum.greaterThan(float(0.0)));
        If(valid, () => {
          outN.assign(diff.div(len));
          outD.assign(C.negate());
          outHandled.assign(bool(true));
        });
      },
    );

    If(outHandled.not(), () => {
      Return();
    });

    // Stabilize scatter: pure normal-direction correction. Δλ_n = d / wSum
    // with `wSum` re-read from `invMass` so §4.4 stays on physical mass
    // even when extensions write the geometry-mode `(n, d)` (the rigid
    // extension uses `contactInvMass` for solve; stabilize keeps physical
    // mass uniformly across all geometry modes).
    const wi: Any = particles.invMass.element(i).toVar();
    const wj: Any = particles.invMass.element(j).toVar();
    const wSum: Any = wi.add(wj).toVar();
    const dLambda: Any = outD.div(wSum).toVar();
    const dxI: Any = outN.mul(wi.mul(dLambda)).toVar();
    const dxJ: Any = outN.mul(wj.mul(dLambda)).negate().toVar();

    emitAccumulateDelta(accumulator, i, dxI);
    emitAccumulateDelta(accumulator, j, dxJ);
  })().compute(maxContacts);
}
