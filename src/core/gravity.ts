import { Vector3 } from 'three';

/**
 * Default gravitational acceleration (m/s²).
 *
 * `SimLoop` reads its `gravity: Vector3` field each step and feeds it to the
 * predict kernel as `f_ext`. Mutating the vector mutates the simulation —
 * there is no per-step snapshotting in v1. A scene that wants zero-g sets
 * this to `(0, 0, 0)`; a scene that wants a custom direction mutates in
 * place.
 */
export const DEFAULT_GRAVITY: Readonly<Vector3> = Object.freeze(new Vector3(0, -9.81, 0));
