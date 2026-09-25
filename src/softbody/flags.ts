/**
 *
 *
 * Read by softbody's {@link buildRigidGeometryExtension} (which claims
 * pairs whose participants both set this bit and emits the §5.1 SDF
 * `(n, d)` for the shared contact pipeline) and softbody's stiff-stacks
 * mass-scaling kernel (which writes `particles.contactInvMass` only for
 * rigid-flagged slots — paper §5.2 eq. 21).
 *
 */
export const FLAG_RIGID = 1 << 0;
