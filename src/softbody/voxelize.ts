/**
 * CPU voxelizer: indexed triangle mesh → particle positions for
 * {@link SoftbodySystem} bodies.
 *
 * Algorithm:
 *   1. Compute the mesh's axis-aligned bounding box.
 *   2. Build a voxel grid with spacing `2·particleRadius` (particles
 *      pack at spacing `2r` so neighbors touch without overlap — same
 *      convention as `src/fluids` particleSpacing).
 *   3. For each voxel centre, run a parity ray-cast against the mesh's
 *      triangles. Centres with an odd intersection count are
 *      occupied (inside the closed mesh).
 *   4. Flag surface voxels via 6-neighbor face occupancy: a voxel is
 *      "surface" iff any of its six axis-aligned neighbor cells is
 *      unoccupied (or leaves the voxel grid). This matches Akinci
 *      2012 §2.2 single-layer boundary sampling.
 *   5. Reorder the result so surface particles come first and
 *      interior particles come after, enabling
 *      {@link SoftbodySystem.surfaceRange} to expose a contiguous
 *      `ParticleRange` without per-particle indirection.
 *
 * Pure-CPU utility — no GPU or renderer required. Safe to call at
 * scene load time from the browser or from a Node CLI (see
 * `src/softbody/voxelize.ts`).
 *
 * Scope (Phase 10): closed triangle meshes only. Non-closed or self-
 * intersecting meshes give undefined-but-non-crashing output (the
 * ray-cast parity is unreliable on non-manifold topology). Phase 15's
 * rigid pipeline extends this with per-particle SDF data; that
 * extension builds on the same voxel grid.
 */

export interface TriangleMesh {
  /** Flat xyz per vertex, length `3 · V`. */
  readonly vertices: Float32Array;
  /** Flat triangle indices into {@link vertices}, length `3 · T`. */
  readonly indices: Uint32Array;
}

export interface VoxelizeOptions {
  /**
   * Scene-global particle radius in metres. The contact diameter is
   * `2·particleRadius` (paper §3 fixed-radius scene). Voxel grid spacing
   * is `2·particleRadius · spacingFactor` — see {@link spacingFactor}.
   */
  readonly particleRadius: number;
  /**
   * Phase 15 — when true, populate {@link VoxelizeResult.restSDF} with
   * per-particle `(φ, ∇φ)` for use by `RigidBodySystem` (Macklin 2014 §5.1
   * sparse SDF contact, eqs. 17–20). Computed by direct closest-point-on-
   * mesh per particle (Ericson RTCD §5.1.5) — paper §5.1: "we sample our
   * field function onto each particle." No volumetric grid intermediate.
   * Default `false` — Phase 10 softbody scenes don't pay the cost.
   */
  readonly bakeSdf?: boolean;
  /**
   * Phase 15 — voxel-grid spacing as a multiple of `2·particleRadius`.
   * Default `1.0` packs particles tangent (no overlap). Values `< 1.0`
   * pack the rest pose with overlap, addressing paper §5.1's
   * documented limitation:
   *
   *   *"A limitation of this approach is that particle surfaces are
   *   not entirely smooth. To reduce bumping and artificial sticking,
   *   we ensure some overlap between particles in the rest pose."*
   *
   * Without overlap, two voxelized rigid bodies in face-to-face contact
   * produce contact pairs whose `(n, d)` correction has lateral
   * components proportional to the misalignment between the two bodies'
   * particle grids — observable as rotational energy injection in the
   * upper body. With ~20 % overlap (`spacingFactor: 0.8`), the rest-pose
   * surface samples are dense enough that the lateral component is
   * substantially reduced.
   *
   * Cost: particle count scales as `(1/spacingFactor)³`. At 0.8 that's
   * ~2× the no-overlap count.
   *
   * Range: `(0, 1]`. Throws otherwise.
   */
  readonly spacingFactor?: number;
}

export interface VoxelizeResult {
  /**
   * Flat xyz positions in world space, length `3 · count`. Surface
   * particles come first in the array (indices `[0, surfaceCount)`),
   * followed by interior particles (`[surfaceCount, count)`).
   */
  readonly positions: Float32Array;
  /**
   * Per-particle surface marker — `1` for the first `surfaceCount`
   * particles, `0` for the rest. Length = `count`. Redundant with
   * {@link surfaceCount} but simplifies runtime code that reads the
   * flag per-particle.
   */
  readonly surfaceFlag: Uint8Array;
  readonly surfaceCount: number;
  /** Total particle count (= `positions.length / 3`). */
  readonly count: number;
  /**
   * Phase 12 — packed undirected edges between voxel-grid 6-face neighbors,
   * emitted as `[i0, j0, i1, j1, ...]` with `i < j` per pair. Length is
   * `2 · edgeCount` (call it `E`); pair `k` is `(edges[2k], edges[2k+1])`.
   * Indices are into the post-reorder particle array (surface-first), so
   * a kernel can read `positions[3·edges[2k]..]` directly.
   *
   * Source: Mueller, Chentanez 2011b §5.1 — "A group contains the
   * corresponding particle and all the particles connected to it via a
   * single edge." For volumetric voxel-grid bodies the natural edge set is
   * 6-face occupancy adjacency; this matches the topology graph the §5.1
   * implicit-shape-matching kernel walks at runtime. Empty (`length = 0`)
   * for single-particle or fully-disconnected bodies.
   */
  readonly edges: Uint32Array;
  /**
   * Phase 15 — per-particle signed distance and gradient `(φ, ∇φ_x, ∇φ_y,
   * ∇φ_z)`, populated only when {@link VoxelizeOptions.bakeSdf} is true.
   * Length `4 · count`; entry `k` lives at `[4k .. 4k+3]` and follows the
   * surface-first reorder (so `restSDF[4·i + 0]` matches `positions[3·i +
   * …]`).
   *
   * Sign convention is paper-faithful (Macklin 2014 §5.1, Figure 7): `φ ≤
   * 0` inside the closed mesh, `φ > 0` outside. Every voxelized particle
   * is interior by construction (the occupancy raycast classified them
   * inside before reaching this stage), so `φ` here is always `≤ 0`.
   * `∇φ` is unit-length by construction (the normalised offset from the
   * closest surface point to the particle), pointing outward from the
   * mesh — which is the eq. 17 contact-normal direction the runtime
   * kernel expects.
   *
   * `undefined` when `bakeSdf` was false. `RigidBodySystem` rejects
   * voxelize input without this block.
   */
  readonly restSDF?: Float32Array;
}

// ---------------------------------------------------------------------------
// Möller-Trumbore ray-triangle intersection.
//
// Given a ray `p + t·d` (d need not be a unit vector) and a triangle
// (v0, v1, v2), return the intersection parameter `t > EPS` or -1 if
// the ray misses / is parallel to the triangle / intersects behind the
// origin. The classical branchless form from Möller & Trumbore 1997.
// ---------------------------------------------------------------------------

const INTERSECT_EPS = 1e-9;

function rayTriangleIntersect(
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number,
): number {
  const e1x = bx - ax;
  const e1y = by - ay;
  const e1z = bz - az;
  const e2x = cx - ax;
  const e2y = cy - ay;
  const e2z = cz - az;
  // h = d × e2
  const hx = dy * e2z - dz * e2y;
  const hy = dz * e2x - dx * e2z;
  const hz = dx * e2y - dy * e2x;
  // a = e1 · h
  const a = e1x * hx + e1y * hy + e1z * hz;
  if (a > -INTERSECT_EPS && a < INTERSECT_EPS) return -1; // parallel
  const f = 1 / a;
  // s = origin − a
  const sx = ox - ax;
  const sy = oy - ay;
  const sz = oz - az;
  // u = f · (s · h)
  const u = f * (sx * hx + sy * hy + sz * hz);
  if (u < 0 || u > 1) return -1;
  // q = s × e1
  const qx = sy * e1z - sz * e1y;
  const qy = sz * e1x - sx * e1z;
  const qz = sx * e1y - sy * e1x;
  // v = f · (d · q)
  const v = f * (dx * qx + dy * qy + dz * qz);
  if (v < 0 || u + v > 1) return -1;
  // t = f · (e2 · q)
  const t = f * (e2x * qx + e2y * qy + e2z * qz);
  if (t < INTERSECT_EPS) return -1; // behind origin
  return t;
}

// ---------------------------------------------------------------------------
// Parity ray-cast point-in-mesh test.
//
// Fires a ray in a slightly-off-axis direction to minimize degenerate
// hits on mesh edges / vertices. Counts triangle intersections; odd =
// inside, even = outside.
// ---------------------------------------------------------------------------

function isInsideMesh(mesh: TriangleMesh, px: number, py: number, pz: number): boolean {
  // Off-axis direction — avoids axis-aligned edge degeneracies on the
  // most common test geometries (axis-aligned cubes / boxes). Not a
  // unit vector, which is fine: t is in this ray's own parameter
  // space.
  const dx = 1.0;
  const dy = 0.00137;
  const dz = 0.00241;
  let intersections = 0;
  const indices = mesh.indices;
  const vertices = mesh.vertices;
  const triCount = indices.length / 3;
  for (let t = 0; t < triCount; t++) {
    const i0 = indices[3 * t + 0]!;
    const i1 = indices[3 * t + 1]!;
    const i2 = indices[3 * t + 2]!;
    const hit = rayTriangleIntersect(
      px,
      py,
      pz,
      dx,
      dy,
      dz,
      vertices[3 * i0 + 0]!,
      vertices[3 * i0 + 1]!,
      vertices[3 * i0 + 2]!,
      vertices[3 * i1 + 0]!,
      vertices[3 * i1 + 1]!,
      vertices[3 * i1 + 2]!,
      vertices[3 * i2 + 0]!,
      vertices[3 * i2 + 1]!,
      vertices[3 * i2 + 2]!,
    );
    if (hit > 0) intersections++;
  }
  return (intersections & 1) === 1;
}

// ---------------------------------------------------------------------------
// Closest-point on triangle (Ericson, RTCD §5.1.5 — region table form).
//

interface ClosestPoint {
  qx: number;
  qy: number;
  qz: number;
  distSq: number;
}

function closestPointOnTriangle(
  px: number,
  py: number,
  pz: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  cx: number,
  cy: number,
  cz: number,
  out: ClosestPoint,
): void {
  // Edge vectors and AP.
  const abx = bx - ax,
    aby = by - ay,
    abz = bz - az;
  const acx = cx - ax,
    acy = cy - ay,
    acz = cz - az;
  const apx = px - ax,
    apy = py - ay,
    apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) {
    setOut(out, ax, ay, az, px, py, pz);
    return;
  }
  const bpx = px - bx,
    bpy = py - by,
    bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) {
    setOut(out, bx, by, bz, px, py, pz);
    return;
  }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    setOut(out, ax + v * abx, ay + v * aby, az + v * abz, px, py, pz);
    return;
  }
  const cpx = px - cx,
    cpy = py - cy,
    cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) {
    setOut(out, cx, cy, cz, px, py, pz);
    return;
  }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    setOut(out, ax + w * acx, ay + w * acy, az + w * acz, px, py, pz);
    return;
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    setOut(out, bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz), px, py, pz);
    return;
  }
  // Inside-face region (barycentric).
  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  setOut(out, ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w, px, py, pz);
}

function setOut(
  out: ClosestPoint,
  qx: number,
  qy: number,
  qz: number,
  px: number,
  py: number,
  pz: number,
): void {
  const rx = px - qx;
  const ry = py - qy;
  const rz = pz - qz;
  out.qx = qx;
  out.qy = qy;
  out.qz = qz;
  out.distSq = rx * rx + ry * ry + rz * rz;
}

// ---------------------------------------------------------------------------
// Top-level voxelize.
// ---------------------------------------------------------------------------

export function voxelize(mesh: TriangleMesh, options: VoxelizeOptions): VoxelizeResult {
  const { particleRadius } = options;
  if (!Number.isFinite(particleRadius) || particleRadius <= 0) {
    throw new Error(
      `voxelize: particleRadius must be a positive finite number, got ${particleRadius}`,
    );
  }
  const spacingFactor = options.spacingFactor ?? 1.0;
  if (!Number.isFinite(spacingFactor) || spacingFactor <= 0 || spacingFactor > 1) {
    throw new Error(`voxelize: spacingFactor must be in (0, 1], got ${spacingFactor}`);
  }
  if (mesh.indices.length % 3 !== 0) {
    throw new Error(`voxelize: mesh.indices length ${mesh.indices.length} is not a multiple of 3`);
  }
  if (mesh.vertices.length % 3 !== 0) {
    throw new Error(
      `voxelize: mesh.vertices length ${mesh.vertices.length} is not a multiple of 3`,
    );
  }

  // Voxel grid spacing — `2·r · spacingFactor`. Default `factor=1` keeps
  // backward-compatible non-overlapping packing for Phase 10 softbody
  // scenes; rigid bodies pass `factor < 1` per paper §5.1's "ensure some
  // overlap" guidance to smooth the contact surface.
  const spacing = 2 * particleRadius * spacingFactor;

  // --- Bounding box ---
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  const vcount = mesh.vertices.length / 3;
  if (vcount === 0) {
    throw new Error('voxelize: mesh has no vertices');
  }
  for (let v = 0; v < vcount; v++) {
    const x = mesh.vertices[3 * v + 0]!;
    const y = mesh.vertices[3 * v + 1]!;
    const z = mesh.vertices[3 * v + 2]!;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }

  // --- Grid dimensions ---
  // Number of voxel cells along each axis, rounded up so the mesh is
  // fully enclosed. Voxel centres start at bbox_min + spacing/2 so
  // each voxel's axis-aligned bbox [c − r, c + r] stays inside the
  // padded bounding box.
  const nx = Math.max(1, Math.ceil((maxX - minX) / spacing));
  const ny = Math.max(1, Math.ceil((maxY - minY) / spacing));
  const nz = Math.max(1, Math.ceil((maxZ - minZ) / spacing));
  const originX = minX + particleRadius;
  const originY = minY + particleRadius;
  const originZ = minZ + particleRadius;

  // --- Occupancy pass ---
  // occupancy[iz * nx * ny + iy * nx + ix] = 1 iff voxel centre lies
  // inside the mesh. Linear index matches the 3D-grid order used for
  // neighbor lookup below.
  const gridSize = nx * ny * nz;
  const occupancy = new Uint8Array(gridSize);
  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const cx = originX + ix * spacing;
        const cy = originY + iy * spacing;
        const cz = originZ + iz * spacing;
        if (isInsideMesh(mesh, cx, cy, cz)) {
          occupancy[iz * nx * ny + iy * nx + ix] = 1;
        }
      }
    }
  }

  // --- Surface classification ---
  // A voxel is "surface" iff it's occupied AND any of its six face-axis
  // neighbors is either (a) outside the grid, or (b) unoccupied.
  // Matches the paper-consistent single-layer sampling described in
  // plan §"Particle representation".
  const surface = new Uint8Array(gridSize);
  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const idx = iz * nx * ny + iy * nx + ix;
        if (occupancy[idx] === 0) continue;
        const neighborOffsets: readonly [number, number, number][] = [
          [-1, 0, 0],
          [1, 0, 0],
          [0, -1, 0],
          [0, 1, 0],
          [0, 0, -1],
          [0, 0, 1],
        ];
        let isSurface = 0;
        for (const [dx, dy, dz] of neighborOffsets) {
          const nxi = ix + dx;
          const nyi = iy + dy;
          const nzi = iz + dz;
          if (nxi < 0 || nxi >= nx || nyi < 0 || nyi >= ny || nzi < 0 || nzi >= nz) {
            isSurface = 1;
            break;
          }
          if (occupancy[nzi * nx * ny + nyi * nx + nxi] === 0) {
            isSurface = 1;
            break;
          }
        }
        surface[idx] = isSurface;
      }
    }
  }

  // --- Collect + reorder [surface..., interior...] ---
  // Two-pass so the grid → final-particle-index map is consistent with the
  // positions array. First pass counts surface vs interior to size the map
  // and the writer offsets; second pass fills positions and builds the map.
  let surfaceCount = 0;
  for (let i = 0; i < gridSize; i++) {
    if (occupancy[i] === 1 && surface[i] === 1) surfaceCount++;
  }
  let interiorCount = 0;
  for (let i = 0; i < gridSize; i++) {
    if (occupancy[i] === 1 && surface[i] === 0) interiorCount++;
  }
  const count = surfaceCount + interiorCount;

  const positions = new Float32Array(3 * count);
  const surfaceFlag = new Uint8Array(count);
  surfaceFlag.fill(1, 0, surfaceCount);

  // gridToParticle[gridIdx] = final particle index after surface-first reorder,
  // or 0xFFFFFFFF for unoccupied cells. Sized to gridSize so the edge walk
  // below can index it directly without a hash. Memory cost is 4·gridSize
  // bytes; for a 64³ MVP body that is 1 MB CPU-side and freed at function
  // return — non-load-bearing.
  const SENTINEL = 0xffffffff;
  const gridToParticle = new Uint32Array(gridSize).fill(SENTINEL);

  let surfaceWriteIdx = 0;
  let interiorWriteIdx = surfaceCount;
  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const idx = iz * nx * ny + iy * nx + ix;
        if (occupancy[idx] === 0) continue;
        const cx = originX + ix * spacing;
        const cy = originY + iy * spacing;
        const cz = originZ + iz * spacing;
        const writeIdx = surface[idx] === 1 ? surfaceWriteIdx++ : interiorWriteIdx++;
        positions[3 * writeIdx + 0] = cx;
        positions[3 * writeIdx + 1] = cy;
        positions[3 * writeIdx + 2] = cz;
        gridToParticle[idx] = writeIdx;
      }
    }
  }

  // --- Edge emission (Phase 12 — Mueller 2011 §5.1 neighborhood) ---
  // Per occupied cell, check the three positive-axis face neighbors (+x,
  // +y, +z); checking only positive directions avoids emitting both (i,j)
  // and (j,i) for the same edge. Each emitted pair is normalised to i<j.
  const edgeBuf: number[] = [];
  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const idx = iz * nx * ny + iy * nx + ix;
        if (occupancy[idx] === 0) continue;
        const i = gridToParticle[idx]!;
        if (ix + 1 < nx) {
          const nIdx = iz * nx * ny + iy * nx + (ix + 1);
          if (occupancy[nIdx] === 1) {
            const j = gridToParticle[nIdx]!;
            const lo = i < j ? i : j;
            const hi = i < j ? j : i;
            edgeBuf.push(lo, hi);
          }
        }
        if (iy + 1 < ny) {
          const nIdx = iz * nx * ny + (iy + 1) * nx + ix;
          if (occupancy[nIdx] === 1) {
            const j = gridToParticle[nIdx]!;
            const lo = i < j ? i : j;
            const hi = i < j ? j : i;
            edgeBuf.push(lo, hi);
          }
        }
        if (iz + 1 < nz) {
          const nIdx = (iz + 1) * nx * ny + iy * nx + ix;
          if (occupancy[nIdx] === 1) {
            const j = gridToParticle[nIdx]!;
            const lo = i < j ? i : j;
            const hi = i < j ? j : i;
            edgeBuf.push(lo, hi);
          }
        }
      }
    }
  }
  const edges = new Uint32Array(edgeBuf);

  // ---------------------------------------------------------------------
  // Per-particle SDF (Phase 15 — Macklin 2014 §5.1, eqs. 17–20).
  //
  // Direct closest-point-on-mesh per particle. φ_i = −|x_i − q*| (interior
  // particles only — every voxelized particle is inside the closed mesh
  // by construction); ∇φ_i = (x_i − q*) / |x_i − q*|, the outward unit
  // normal at the closest surface projection. Paper Figure 7.
  //

  let restSDF: Float32Array | undefined;
  if (options.bakeSdf) {
    restSDF = new Float32Array(4 * count);
    const triCount = mesh.indices.length / 3;
    const scratch: ClosestPoint = { qx: 0, qy: 0, qz: 0, distSq: 0 };
    const best: ClosestPoint = { qx: 0, qy: 0, qz: 0, distSq: 0 };
    for (let p = 0; p < count; p++) {
      const px = positions[3 * p + 0]!;
      const py = positions[3 * p + 1]!;
      const pz = positions[3 * p + 2]!;
      best.distSq = Infinity;
      for (let t = 0; t < triCount; t++) {
        const i0 = mesh.indices[3 * t + 0]!;
        const i1 = mesh.indices[3 * t + 1]!;
        const i2 = mesh.indices[3 * t + 2]!;
        closestPointOnTriangle(
          px,
          py,
          pz,
          mesh.vertices[3 * i0 + 0]!,
          mesh.vertices[3 * i0 + 1]!,
          mesh.vertices[3 * i0 + 2]!,
          mesh.vertices[3 * i1 + 0]!,
          mesh.vertices[3 * i1 + 1]!,
          mesh.vertices[3 * i1 + 2]!,
          mesh.vertices[3 * i2 + 0]!,
          mesh.vertices[3 * i2 + 1]!,
          mesh.vertices[3 * i2 + 2]!,
          scratch,
        );
        if (scratch.distSq < best.distSq) {
          best.distSq = scratch.distSq;
          best.qx = scratch.qx;
          best.qy = scratch.qy;
          best.qz = scratch.qz;
        }
      }
      const dist = Math.sqrt(best.distSq);
      // Sign — `φ ≥ 0 inside, < 0 outside, 0 on boundary` (paper-implicit
      // convention; see derivation below). Voxelize occupancy guarantees
      // every retained particle is interior; `φ_i = +dist`.
      const phi = dist;

      //
      // Therefore: `∇φ = (particle − closestSurface) / |…|` — direction
      // FROM the closest surface point INTO the particle, which for an
      // interior particle is the INWARD direction (toward the body's
      // interior).
      //
      // Degenerate case `dist == 0` would mean the particle sits exactly
      // on a triangle. Pick `x̂` as a safe fallback — paper Figure 7 says
      // arbitrary medial-axis gradients are acceptable; eq. 17's
      // `min(|φ_i|, |φ_j|)` always picks a non-zero-`|φ|` neighbour as
      // the normal-provider when one exists.
      const base = 4 * p;
      restSDF[base + 0] = phi;
      if (dist > 0) {
        const inv = 1 / dist;
        restSDF[base + 1] = (px - best.qx) * inv;
        restSDF[base + 2] = (py - best.qy) * inv;
        restSDF[base + 3] = (pz - best.qz) * inv;
      } else {
        restSDF[base + 1] = 1; // degenerate-fallback x̂; see comment above
        restSDF[base + 2] = 0;
        restSDF[base + 3] = 0;
      }
    }
  }

  return restSDF !== undefined
    ? { positions, surfaceFlag, surfaceCount, count, edges, restSDF }
    : { positions, surfaceFlag, surfaceCount, count, edges };
}

// ---------------------------------------------------------------------------
// Binary format encoder.
//
// v2 (Phase 15) layout:
//   u32 magic    = 0x32584F56  ('VOX2' little-endian)
//   u32 count
//   u32 surfaceCount
//   u32 edgeCount
//   u32 hasSdf                                     ← 0 or 1
//   count × 3 × f32 positions
//   count × u8 surfaceFlag
//   pad to 4-byte boundary
//   edgeCount × 2 × u32 edges
//   if hasSdf == 1: count × 4 × f32 restSDF        ← new in v2
//
// v1 (Phase 12) layout (read-only — decoder accepts v1 buffers and
// returns `restSDF: undefined`):
//   u32 magic    = 0x31584F56  ('VOX1' little-endian)
//   u32 count
//   u32 surfaceCount
//   u32 edgeCount
//   count × 3 × f32 positions
//   count × u8 surfaceFlag
//   pad to 4-byte boundary
//   edgeCount × 2 × u32 edges
//
// v0 (pre-Phase-12) layout (read-only — `decodeVoxelizeBinary` falls
// back when the magic is absent):
//   u32 count
//   u32 surfaceCount
//   count × 3 × f32 positions
//   count × u8 surfaceFlag
// v0 is detected by the first u32 not matching the v1/v2 magic; in
// that case `edges` is synthesized as an empty `Uint32Array` so
// consuming code that uses §5.3 explicit shape matching keeps working
// without a re-bake. v0 cached assets that need §5.1 or §5.1 + SDF
// must be re-baked.
//
// Inlined here so the runtime wrapper in `src/softbody` doesn't
// have to reach into a sibling package for serialization.
// ---------------------------------------------------------------------------

const VOX_MAGIC_V1 = 0x31584f56; // 'V', 'O', 'X', '1' as a little-endian u32
const VOX_MAGIC_V2 = 0x32584f56; // 'V', 'O', 'X', '2' as a little-endian u32
const V1_HEADER_BYTES = 4 * 4; // magic + count + surfaceCount + edgeCount
const V2_HEADER_BYTES = 5 * 4; // v1 fields + hasSdf
const V0_HEADER_BYTES = 2 * 4; // count + surfaceCount

export function encodeVoxelizeBinary(result: VoxelizeResult): ArrayBuffer {
  const positionBytes = 3 * 4 * result.count;
  const flagBytes = result.count;
  const flagPad = (4 - (flagBytes % 4)) % 4;
  const edgeBytes = result.edges.byteLength;
  const edgeCount = result.edges.length / 2;
  const hasSdf = result.restSDF !== undefined ? 1 : 0;
  const sdfBytes = hasSdf === 1 ? 4 * 4 * result.count : 0;
  if (hasSdf === 1 && result.restSDF!.length !== 4 * result.count) {
    throw new Error(
      `encodeVoxelizeBinary: restSDF length ${result.restSDF!.length} does not match 4 · count = ${4 * result.count}`,
    );
  }
  const byteLength = V2_HEADER_BYTES + positionBytes + flagBytes + flagPad + edgeBytes + sdfBytes;
  const buf = new ArrayBuffer(byteLength);
  const dvHeader = new DataView(buf, 0, V2_HEADER_BYTES);
  dvHeader.setUint32(0, VOX_MAGIC_V2, true);
  dvHeader.setUint32(4, result.count, true);
  dvHeader.setUint32(8, result.surfaceCount, true);
  dvHeader.setUint32(12, edgeCount, true);
  dvHeader.setUint32(16, hasSdf, true);
  const posView = new Float32Array(buf, V2_HEADER_BYTES, 3 * result.count);
  posView.set(result.positions);
  const flagView = new Uint8Array(buf, V2_HEADER_BYTES + positionBytes, flagBytes);
  flagView.set(result.surfaceFlag);
  if (edgeBytes > 0) {
    const edgeView = new Uint32Array(
      buf,
      V2_HEADER_BYTES + positionBytes + flagBytes + flagPad,
      result.edges.length,
    );
    edgeView.set(result.edges);
  }
  if (hasSdf === 1) {
    const sdfView = new Float32Array(
      buf,
      V2_HEADER_BYTES + positionBytes + flagBytes + flagPad + edgeBytes,
      4 * result.count,
    );
    sdfView.set(result.restSDF!);
  }
  return buf;
}

export function decodeVoxelizeBinary(buf: ArrayBuffer): VoxelizeResult {
  if (buf.byteLength < V0_HEADER_BYTES) {
    throw new Error(
      `decodeVoxelizeBinary: buffer too small (${buf.byteLength} bytes), need at least ${V0_HEADER_BYTES} for the header`,
    );
  }
  const dv = new DataView(buf);
  const first = dv.getUint32(0, true);
  if (first === VOX_MAGIC_V2) {
    if (buf.byteLength < V2_HEADER_BYTES) {
      throw new Error(
        `decodeVoxelizeBinary: buffer size ${buf.byteLength} below v2 header (${V2_HEADER_BYTES})`,
      );
    }
    const count = dv.getUint32(4, true);
    const surfaceCount = dv.getUint32(8, true);
    const edgeCount = dv.getUint32(12, true);
    const hasSdf = dv.getUint32(16, true);
    const positionBytes = 3 * 4 * count;
    const flagBytes = count;
    const flagPad = (4 - (flagBytes % 4)) % 4;
    const edgeBytes = edgeCount * 2 * 4;
    const sdfBytes = hasSdf === 1 ? 4 * 4 * count : 0;
    const expectedBytes =
      V2_HEADER_BYTES + positionBytes + flagBytes + flagPad + edgeBytes + sdfBytes;
    if (buf.byteLength < expectedBytes) {
      throw new Error(
        `decodeVoxelizeBinary: buffer size ${buf.byteLength} does not match v2 declared count=${count} edgeCount=${edgeCount} hasSdf=${hasSdf} (expected ≥ ${expectedBytes})`,
      );
    }
    const positions = new Float32Array(buf.slice(V2_HEADER_BYTES, V2_HEADER_BYTES + positionBytes));
    const surfaceFlag = new Uint8Array(
      buf.slice(V2_HEADER_BYTES + positionBytes, V2_HEADER_BYTES + positionBytes + flagBytes),
    );
    const edges =
      edgeBytes > 0
        ? new Uint32Array(
            buf.slice(
              V2_HEADER_BYTES + positionBytes + flagBytes + flagPad,
              V2_HEADER_BYTES + positionBytes + flagBytes + flagPad + edgeBytes,
            ),
          )
        : new Uint32Array(0);
    const restSDF =
      hasSdf === 1
        ? new Float32Array(
            buf.slice(
              V2_HEADER_BYTES + positionBytes + flagBytes + flagPad + edgeBytes,
              V2_HEADER_BYTES + positionBytes + flagBytes + flagPad + edgeBytes + sdfBytes,
            ),
          )
        : undefined;
    return restSDF !== undefined
      ? { positions, surfaceFlag, surfaceCount, count, edges, restSDF }
      : { positions, surfaceFlag, surfaceCount, count, edges };
  }
  if (first === VOX_MAGIC_V1) {
    if (buf.byteLength < V1_HEADER_BYTES) {
      throw new Error(
        `decodeVoxelizeBinary: buffer size ${buf.byteLength} below v1 header (${V1_HEADER_BYTES})`,
      );
    }
    const count = dv.getUint32(4, true);
    const surfaceCount = dv.getUint32(8, true);
    const edgeCount = dv.getUint32(12, true);
    const positionBytes = 3 * 4 * count;
    const flagBytes = count;
    const flagPad = (4 - (flagBytes % 4)) % 4;
    const edgeBytes = edgeCount * 2 * 4;
    const expectedBytes = V1_HEADER_BYTES + positionBytes + flagBytes + flagPad + edgeBytes;
    if (buf.byteLength < expectedBytes) {
      throw new Error(
        `decodeVoxelizeBinary: buffer size ${buf.byteLength} does not match v1 declared count=${count} edgeCount=${edgeCount} (expected ≥ ${expectedBytes})`,
      );
    }
    const positions = new Float32Array(buf.slice(V1_HEADER_BYTES, V1_HEADER_BYTES + positionBytes));
    const surfaceFlag = new Uint8Array(
      buf.slice(V1_HEADER_BYTES + positionBytes, V1_HEADER_BYTES + positionBytes + flagBytes),
    );
    const edges =
      edgeBytes > 0
        ? new Uint32Array(
            buf.slice(
              V1_HEADER_BYTES + positionBytes + flagBytes + flagPad,
              V1_HEADER_BYTES + positionBytes + flagBytes + flagPad + edgeBytes,
            ),
          )
        : new Uint32Array(0);
    return { positions, surfaceFlag, surfaceCount, count, edges };
  }
  // v0 fallback: `first` is treated as `count`. No edges in v0 — synthesize
  // empty so callers that only need positions+surface keep working.
  const count = first;
  const surfaceCount = dv.getUint32(4, true);
  const expectedBytes = V0_HEADER_BYTES + 3 * 4 * count + count;
  if (buf.byteLength < expectedBytes) {
    throw new Error(
      `decodeVoxelizeBinary: buffer size ${buf.byteLength} does not match declared count=${count} (expected ≥ ${expectedBytes})`,
    );
  }
  const positions = new Float32Array(buf.slice(V0_HEADER_BYTES, V0_HEADER_BYTES + 3 * 4 * count));
  const surfaceFlag = new Uint8Array(
    buf.slice(V0_HEADER_BYTES + 3 * 4 * count, V0_HEADER_BYTES + 3 * 4 * count + count),
  );
  return {
    positions,
    surfaceFlag,
    surfaceCount,
    count,
    edges: new Uint32Array(0),
  };
}
