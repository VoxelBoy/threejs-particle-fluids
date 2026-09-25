import { If } from 'three/tsl';

import type { ParticleSystem } from '../particles.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Phase 15a — geometry-mode hook for the shared contact pipeline.
 *
 * The contact-solve and contact-stabilize kernels in core are
 * parameterised by a list of `ContactGeometryExtension`s. Per pair, each
 * extension is given a chance to compute the contact normal `n` and
 * penetration depth `d`; the first extension to set `outHandled = true`
 * claims the pair. If no extension claims, a default spherical computation
 * (Macklin 2014 §6.1 eq. 22) fires.
 *
 * Conceptually the extension is "the geometry mode for this pair". A
 * single extension typically:
 *
 *   1. Tests whether it owns the pair (e.g. softbody's rigid extension
 *      checks `flags[i] & flags[j] & FLAG_RIGID`). If not its mode,
 *      leaves `outHandled = false` so the next extension or the spherical
 *      default takes over.
 *   2. If it owns the pair, runs its geometry-mode-specific math
 *      (boundary tests, distance gates, mass guards). When valid, writes
 *      `outN`, `outD`, and sets `outHandled = true`. When the pair is
 *      owned but has no valid contact (degenerate, separating, kinematic
 *      pair) it leaves `outHandled = false` — the kernel's post-`If`
 *      gate then skips the pair entirely.
 *
 * The shared `emitContactSolveCorrection` helper (`correction.ts`) reads
 * the `(n, d, i, j, c)` produced here, accumulates Δλ_n, scatters the
 * normal correction, and runs the Macklin 2020 §3.5 static-friction
 * cone gate. Friction is therefore the same across all geometry modes —
 * the only thing an extension contributes is `(n, d)`.
 *
 * Cross-extension composition: extensions are tried in registration
 * order (the order materials appear in `SimLoopOptions.materials`). The
 * composer wraps each call in `If(outHandled.not(), ...)` so once a pair
 * is claimed, later extensions and the spherical default short-circuit.
 *
 * Determinism: extensions read freely from particle storage; outputs
 * flow through the kernel's per-pair atomicAdd scatter, so a deterministic
 * extension implementation gives a Tier 1 bit-exact contact solve.
 */
export interface ContactGeometryExtension {
  /**
   * Emit TSL inside the contact-solve / stabilize kernel for one pair.
   *
   * Inputs:
   *   - `i`, `j`: pair particle indices (TSL u32 nodes).
   *   - `c`: contact slot index (TSL u32 node).
   *   - `particles`: shared particle storage.
   *
   * Outputs (caller-provided TSL `.toVar()` cells):
   *   - `outN`: unit contact normal pointing from j toward i (vec3 var).
   *   - `outD`: positive penetration depth in metres (float var).
   *   - `outHandled`: bool var. Implementations set this to `true` when
   *     they have produced a valid `(outN, outD)` for the pair. Leaving
   *     it untouched (false) defers to the next extension or the
   *     spherical default.
   */
  emit(args: {
    readonly i: Any;
    readonly j: Any;
    readonly c: Any;
    readonly particles: ParticleSystem;
    readonly outN: Any;
    readonly outD: Any;
    readonly outHandled: Any;
  }): void;
}

export interface EmitGeometrySelectionArgs {
  readonly i: Any;
  readonly j: Any;
  readonly c: Any;
  readonly particles: ParticleSystem;
  readonly outN: Any;
  readonly outD: Any;
  readonly outHandled: Any;
}

/**
 * Compose a list of {@link ContactGeometryExtension}s with a default
 * spherical computation. Extensions are tried in registration order;
 * the first to set `outHandled = true` claims the pair. If none claim,
 * `emitDefault` runs.
 *
 * The chain is implemented with `If(outHandled.not(), ...)` guards, so
 * already-claimed pairs short-circuit at runtime. There is no per-pair
 * runtime dispatch beyond the chain of `If`s — the per-extension cost
 * is one bool branch.
 *
 * `emitDefault` is the kernel-specific spherical fallback (paper §6.1
 * eq. 22). Solve and stabilize both pass a closure here because they
 * differ in (a) which positions to read (`predictedPositions` vs
 * `positions`) and (b) what guards to apply.
 */
export function emitGeometrySelection(
  args: EmitGeometrySelectionArgs,
  extensions: readonly ContactGeometryExtension[],
  emitDefault: () => void,
): void {
  for (const ext of extensions) {
    If(args.outHandled.not(), () => {
      ext.emit(args);
    });
  }
  If(args.outHandled.not(), () => {
    emitDefault();
  });
}
