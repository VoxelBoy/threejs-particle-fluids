/** Rebuild the bundled CC0 meshes and particle templates: npm run assets:elastic. */
export const BODY_BUDGETS = [50, 250, 500, 750, 1250] as const;
import { copyFile, readFile, writeFile } from 'node:fs/promises';
import { BufferGeometry, Float32BufferAttribute, Vector2, Vector3 } from 'three';
import { voxelize, type TriangleMesh, type VoxelizeResult } from '../src/softbody/voxelize.js';

interface MeshData {
  vertices: Vector3[];
  faces: number[][];
  faceUvs: Vector2[][];
}

function readObj(text: string): MeshData {
  const source: Vector3[] = [],
    faces: number[][] = [],
    textureCoords: Vector2[] = [],
    faceUvs: Vector2[][] = [];
  for (const line of text.split(/\r?\n/)) {
    const [kind, ...fields] = line.trim().split(/\s+/);
    if (kind === 'v')
      source.push(new Vector3(...(fields.slice(0, 3).map(Number) as [number, number, number])));
    if (kind === 'vt') textureCoords.push(new Vector2(Number(fields[0]), Number(fields[1])));
    if (kind === 'f') {
      const face = fields.map((v) => Number(v.split('/')[0]) - 1);
      const coords = fields.map((v) => textureCoords[Number(v.split('/')[1]) - 1]!);
      for (let i = 1; i < face.length - 1; i++) {
        faces.push([face[0]!, face[i]!, face[i + 1]!]);
        faceUvs.push([coords[0]!, coords[i]!, coords[i + 1]!]);
      }
    }
  }
  const vertices: Vector3[] = [],
    welded = new Map<string, number>();
  const ids = source.map((v) => {
    const key = v
      .toArray()
      .map((x) => x.toFixed(6))
      .join(',');
    if (!welded.has(key)) {
      welded.set(key, vertices.length);
      vertices.push(v);
    }
    return welded.get(key)!;
  });
  // These OBJ exports contain coincident, opposite-facing triangles. Keep one
  // shell: duplicated faces would cancel the voxelizer's inside/outside parity.
  const unique = new Map<string, { indices: number[]; uv: Vector2[] }>();
  for (const [index, face] of faces.entries()) {
    const mapped = face.map((i) => ids[i]!);
    unique.set([...mapped].sort((a, b) => a - b).join(','), {
      indices: mapped,
      uv: faceUvs[index]!,
    });
  }
  return {
    vertices,
    faces: [...unique.values()].map((f) => f.indices),
    faceUvs: [...unique.values()].map((f) => f.uv),
  };
}

/** Loop subdivision on the welded, closed shell; preserves holes and concavity. */
function subdivide(mesh: MeshData): MeshData {
  const neighbors = mesh.vertices.map(() => new Set<number>());
  const edges = new Map<string, { a: number; b: number; opposite: number[]; index: number }>();
  const key = (a: number, b: number) => (a < b ? `${a},${b}` : `${b},${a}`);
  for (const f of mesh.faces)
    for (let k = 0; k < 3; k++) {
      const a = f[k]!,
        b = f[(k + 1) % 3]!,
        c = f[(k + 2) % 3]!;
      neighbors[a]!.add(b);
      neighbors[b]!.add(a);
      const id = key(a, b);
      if (!edges.has(id))
        edges.set(id, { a, b, opposite: [], index: mesh.vertices.length + edges.size });
      edges.get(id)!.opposite.push(c);
    }
  const vertices = mesh.vertices.map((v, i) => {
    const ring = [...neighbors[i]!],
      n = ring.length;
    const beta = (5 / 8 - (3 / 8 + Math.cos((2 * Math.PI) / n) / 4) ** 2) / n;
    const result = v.clone().multiplyScalar(1 - n * beta);
    for (const j of ring) result.addScaledVector(mesh.vertices[j]!, beta);
    return result;
  });
  for (const edge of edges.values()) {
    if (edge.opposite.length !== 2) throw new Error('Expected a closed manifold CC0 mesh.');
    vertices.push(
      mesh.vertices[edge.a]!.clone()
        .add(mesh.vertices[edge.b]!)
        .multiplyScalar(3 / 8)
        .addScaledVector(mesh.vertices[edge.opposite[0]!]!, 1 / 8)
        .addScaledVector(mesh.vertices[edge.opposite[1]!]!, 1 / 8),
    );
  }
  const faces: number[][] = [];
  const faceUvs: Vector2[][] = [];
  for (const [index, [a, b, c]] of (mesh.faces as [number, number, number][]).entries()) {
    const ab = edges.get(key(a, b))!.index,
      bc = edges.get(key(b, c))!.index,
      ca = edges.get(key(c, a))!.index;
    faces.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    // Subdivide UVs per face, preserving atlas seams instead of blending colors
    // across unrelated islands when their positions share a welded vertex.
    const [ua, ub, uc] = mesh.faceUvs[index] as [Vector2, Vector2, Vector2];
    const uab = ua.clone().add(ub).multiplyScalar(0.5);
    const ubc = ub.clone().add(uc).multiplyScalar(0.5);
    const uca = uc.clone().add(ua).multiplyScalar(0.5);
    faceUvs.push([ua, uab, uca], [ub, ubc, uab], [uc, uca, ubc], [uab, ubc, uca]);
  }
  return { vertices, faces, faceUvs };
}

function renderMesh(mesh: MeshData) {
  // Calculate smooth normals on the welded shell BEFORE splitting UV seams.
  // Duplicating vertices first would introduce hard edges along the texture atlas.
  const welded = new BufferGeometry();
  welded.setAttribute(
    'position',
    new Float32BufferAttribute(
      mesh.vertices.flatMap((v) => v.toArray()),
      3,
    ),
  );
  welded.setIndex(mesh.faces.flat());
  welded.computeVertexNormals();
  const normal = welded.getAttribute('normal');
  const positions: number[] = [],
    normals: number[] = [],
    uvs: number[] = [],
    indices: number[] = [];
  const vertices = new Map<string, number>();
  mesh.faces.forEach((face, i) =>
    face.forEach((v, corner) => {
      const uv = mesh.faceUvs[i]![corner]!;
      const key = `${v}/${uv.x.toFixed(7)}/${uv.y.toFixed(7)}`;
      if (!vertices.has(key)) {
        vertices.set(key, positions.length / 3);
        positions.push(...mesh.vertices[v]!.toArray());
        normals.push(normal.getX(v), normal.getY(v), normal.getZ(v));
        uvs.push(uv.x, uv.y);
      }
      indices.push(vertices.get(key)!);
    }),
  );
  welded.dispose();
  return { positions, normals, uvs, indices };
}

function adjacency(voxels: VoxelizeResult) {
  const graph = Array.from({ length: voxels.count }, () => new Set<number>());
  for (let i = 0; i < voxels.edges.length; i += 2) {
    const a = voxels.edges[i]!,
      b = voxels.edges[i + 1]!;
    graph[a]!.add(b);
    graph[b]!.add(a);
  }
  return graph;
}
function connected(graph: Set<number>[], active: Set<number>, omit = -1) {
  const start = [...active].find((i) => i !== omit);
  const seen = new Set<number>();
  if (start === undefined) return seen;
  const queue = [start];
  seen.add(start);
  for (const i of queue)
    for (const j of graph[i]!)
      if (j !== omit && active.has(j) && !seen.has(j)) {
        seen.add(j);
        queue.push(j);
      }
  return seen;
}

function sample(mesh: TriangleMesh, count: number) {
  // Find a connected lattice just above the requested count. Prune only boundary
  // samples whose removal preserves connectivity; never fill a mesh's holes.
  let low = 0.001,
    high = 0.08;
  let best: { voxels: VoxelizeResult; radius: number } | undefined;
  for (let k = 0; k < 19; k++) {
    const radius = (low + high) / 2;
    const voxels = voxelize(mesh, { particleRadius: radius });
    const graph = adjacency(voxels),
      all = new Set(graph.map((_, i) => i));
    const isConnected = connected(graph, all).size === voxels.count;
    if (voxels.count >= count) {
      low = radius;
      if (isConnected && (!best || voxels.count < best.voxels.count)) best = { voxels, radius };
    } else high = radius;
  }
  // Coarse lattices can split thin features; scan from coarse to fine for the
  // first connected one. Small budgets may overshoot and are trimmed below.
  for (let radius = 0.08; !best && radius > 0.001; radius *= 0.97) {
    const voxels = voxelize(mesh, { particleRadius: radius });
    if (voxels.count < count) continue;
    const graph = adjacency(voxels);
    if (connected(graph, new Set(graph.map((_, i) => i))).size === voxels.count)
      best = { voxels, radius };
  }
  if (!best) throw new Error(`No suitable connected ${count}-particle lattice.`);
  const { voxels, radius } = best,
    graph = adjacency(voxels);
  const active = new Set(graph.map((_, i) => i));
  while (active.size > count) {
    // Favor broad surfaces over thin tips (low-degree vertices), preserving limbs.
    const candidates = [...active]
      .filter((i) => voxels.surfaceFlag[i] === 1)
      .sort((a, b) => graph[b]!.size - graph[a]!.size);
    const remove = candidates.find((i) => connected(graph, active, i).size === active.size - 1);
    // Thin limbs at tiny budgets: keep the few extra samples rather than split.
    if (remove === undefined) break;
    active.delete(remove);
    for (const j of graph[remove]!) graph[j]!.delete(remove);
  }
  const ordered = [...active].sort(
    (a, b) => Number(graph[b]!.size < 6) - Number(graph[a]!.size < 6),
  );
  const ids = new Map(ordered.map((i, j) => [i, j]));
  const edges: number[] = [];
  for (const i of ordered)
    for (const j of graph[i]!)
      if (active.has(j) && ids.get(i)! < ids.get(j)!) edges.push(ids.get(i)!, ids.get(j)!);
  // Every asset shares a collision radius. Match its lattice spacing exactly,
  // and scale the render shell with it so every level has the same mass.
  const targetRadius = 0.015 * Math.cbrt(200 / count),
    scale = targetRadius / radius;
  const positions = ordered.flatMap((i) =>
    [0, 1, 2].map((k) => voxels.positions[i * 3 + k]! * scale),
  );
  return { scale, positions, edges, surface: ordered.map((i) => Number(graph[i]!.size < 6)) };
}

const round = (v: number) => Number(v.toFixed(7));
for (const name of ['donut', 'croissant', 'banana', 'ginger-bread']) {
  let mesh = readObj(
    await readFile(new URL(`../demo/assets/elastic/${name}.obj`, import.meta.url), 'utf8'),
  );
  // Give the flat cookie enough thickness for volumetric bending, and make the
  // banana's tips substantial enough to survive the Balanced particle sampling.
  if (name === 'ginger-bread') for (const v of mesh.vertices) v.y *= 2.8;
  if (name === 'banana')
    for (const v of mesh.vertices) {
      v.x *= 1.3;
      v.y *= 1.15;
    }
  mesh = subdivide(subdivide(mesh));
  const min = new Vector3(Infinity, Infinity, Infinity),
    max = new Vector3(-Infinity, -Infinity, -Infinity);
  for (const v of mesh.vertices) {
    min.min(v);
    max.max(v);
  }
  const center = min.add(max).multiplyScalar(0.5);
  for (const v of mesh.vertices) v.sub(center);
  // Repair the winding if the source's last duplicate faces pointed inwards.
  const signedVolume = mesh.faces.reduce(
    (sum, [a, b, c]) =>
      sum + mesh.vertices[a!]!.dot(mesh.vertices[b!]!.clone().cross(mesh.vertices[c!]!)) / 6,
    0,
  );
  if (signedVolume < 0) {
    for (const face of mesh.faces) face.reverse();
    for (const uv of mesh.faceUvs) uv.reverse();
  }
  const triangles = {
    vertices: Float32Array.from(mesh.vertices.flatMap((v) => v.toArray())),
    indices: Uint32Array.from(mesh.faces.flat()),
  };
  // One template per particle-count level: 20 bodies share each level's budget.
  const templates = Object.fromEntries(
    BODY_BUDGETS.map((count) => [count, sample(triangles, count)]),
  );
  const data = { ...renderMesh(mesh), templates };
  await writeFile(
    new URL(`../public/models/elastic/${name}.json`, import.meta.url),
    JSON.stringify(data, (_key, v: unknown) => (typeof v === 'number' ? round(v) : v)) + '\n',
  );
  console.log(`${name}: ${mesh.vertices.length} vertices, ${BODY_BUDGETS.join(' / ')} particles`);
}
await copyFile(
  new URL('../demo/assets/elastic/Textures/colormap.png', import.meta.url),
  new URL('../public/models/elastic/colormap.png', import.meta.url),
);
