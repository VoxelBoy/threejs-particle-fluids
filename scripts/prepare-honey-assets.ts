/**
 * Rebuild the honey preset's bunny assets: npm run assets:honey.
 *
 * Source: the Stanford bunny (demo/assets/honey/bunny.drc, from the three.js
 * examples). Outputs, all in public/models/honey/:
 *   bunny.mesh.bin  quantized display mesh (uint16 positions and indices)
 *   bunny.sdf.bin   signed distance field baked from a simplified copy
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { bakeMeshToSdf } from '../src/sdf/bake.js';
import { encodeSdfBinary } from '../src/sdf/writeBinary.js';

const HEIGHT = 0.42;
const CLUSTER = 0.009;
const SDF_RESOLUTION = 48;
export const MESH_MAGIC = 0x4e554242; // "BBUN"

interface Mesh {
  positions: Float32Array;
  indices: Uint32Array;
}

async function decodeDraco(path: string): Promise<Mesh> {
  const decoderPath = new URL(
    '../node_modules/three/examples/jsm/libs/draco/draco_decoder.js',
    import.meta.url,
  );
  const source = await readFile(decoderPath, 'utf8');
  const require = createRequire(import.meta.url);
  // three.js vendors this emscripten build as a classic script (no module
  // exports), so evaluate the local file with Node's require in scope.
  const factory = new Function(
    'require',
    '__dirname',
    'process',
    `${source}; return DracoDecoderModule;`,
  )(require, '/', process);
  const draco = await factory();
  const bytes = await readFile(path);
  const buffer = new draco.DecoderBuffer();
  buffer.Init(new Int8Array(bytes), bytes.length);
  const decoder = new draco.Decoder();
  const mesh = new draco.Mesh();
  const status = decoder.DecodeBufferToMesh(buffer, mesh);
  if (!status.ok()) throw new Error(`Draco decode failed: ${status.error_msg()}`);
  const attribute = decoder.GetAttribute(mesh, decoder.GetAttributeId(mesh, draco.POSITION));
  const values = new draco.DracoFloat32Array();
  decoder.GetAttributeFloatForAllPoints(mesh, attribute, values);
  const positions = new Float32Array(mesh.num_points() * 3);
  for (let i = 0; i < positions.length; i++) positions[i] = values.GetValue(i);
  const face = new draco.DracoInt32Array();
  const indices = new Uint32Array(mesh.num_faces() * 3);
  for (let f = 0; f < mesh.num_faces(); f++) {
    decoder.GetFaceFromMesh(mesh, f, face);
    for (let k = 0; k < 3; k++) indices[f * 3 + k] = face.GetValue(k);
  }
  return { positions, indices };
}

/** Scale to HEIGHT, centre on x/z, and rest on y = 0. */
function normalize(mesh: Mesh): void {
  const min = [Infinity, Infinity, Infinity],
    max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i++) {
    min[i % 3] = Math.min(min[i % 3]!, mesh.positions[i]!);
    max[i % 3] = Math.max(max[i % 3]!, mesh.positions[i]!);
  }
  const scale = HEIGHT / (max[1]! - min[1]!);
  const offset = [-(min[0]! + max[0]!) / 2, -min[1]!, -(min[2]! + max[2]!) / 2];
  for (let i = 0; i < mesh.positions.length; i++)
    mesh.positions[i] = (mesh.positions[i]! + offset[i % 3]!) * scale;
}

/** Cap every open boundary loop (the scan's base holes) with a fan around its centroid. */
function closeHoles(mesh: Mesh): number {
  const directed = new Map<string, [number, number]>();
  for (let f = 0; f < mesh.indices.length; f += 3)
    for (let k = 0; k < 3; k++) {
      const a = mesh.indices[f + k]!,
        b = mesh.indices[f + ((k + 1) % 3)]!;
      directed.set(`${a},${b}`, [a, b]);
    }
  // A boundary edge has no twin running the other way; walk them into loops.
  const next = new Map<number, number>();
  for (const [a, b] of directed.values()) if (!directed.has(`${b},${a}`)) next.set(b, a);
  const positions = Array.from(mesh.positions);
  const indices = Array.from(mesh.indices);
  let loops = 0;
  while (next.size > 0) {
    const start = next.keys().next().value!;
    const loop: number[] = [];
    let v = start;
    while (next.has(v)) {
      loop.push(v);
      const n = next.get(v)!;
      next.delete(v);
      v = n;
    }
    if (loop.length < 3) continue;
    const centre = [0, 0, 0];
    for (const i of loop)
      for (let k = 0; k < 3; k++) centre[k]! += positions[i * 3 + k]! / loop.length;
    const c = positions.length / 3;
    positions.push(...centre);
    for (let i = 0; i < loop.length; i++) indices.push(loop[i]!, loop[(i + 1) % loop.length]!, c);
    loops++;
  }
  mesh.positions = Float32Array.from(positions);
  mesh.indices = Uint32Array.from(indices);
  return loops;
}

/** Vertex-clustering simplification: good enough for a collision field. */
function cluster(mesh: Mesh, cell: number): Mesh {
  const ids = new Map<string, number>();
  const sums: number[][] = [];
  const remap = new Uint32Array(mesh.positions.length / 3);
  for (let v = 0; v < remap.length; v++) {
    const p = [0, 1, 2].map((k) => mesh.positions[v * 3 + k]!);
    const key = p.map((x) => Math.floor(x / cell)).join(',');
    let id = ids.get(key);
    if (id === undefined) {
      id = sums.length;
      ids.set(key, id);
      sums.push([0, 0, 0, 0]);
    }
    const sum = sums[id]!;
    for (let k = 0; k < 3; k++) sum[k]! += p[k]!;
    sum[3]! += 1;
    remap[v] = id;
  }
  const positions = new Float32Array(sums.length * 3);
  sums.forEach((s, i) => {
    for (let k = 0; k < 3; k++) positions[i * 3 + k] = s[k]! / s[3]!;
  });
  const faces: number[] = [];
  const seen = new Set<string>();
  for (let f = 0; f < mesh.indices.length; f += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => remap[mesh.indices[f + k]!]!) as [
      number,
      number,
      number,
    ];
    if (a === b || b === c || a === c) continue;
    const key = [a, b, c].sort((x, y) => x - y).join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    faces.push(a, b, c);
  }
  return { positions, indices: Uint32Array.from(faces) };
}

function encodeMesh(mesh: Mesh): ArrayBuffer {
  const vertices = mesh.positions.length / 3;
  if (vertices > 65535) throw new Error('Display mesh needs 32-bit indices.');
  const min = [Infinity, Infinity, Infinity],
    max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i++) {
    min[i % 3] = Math.min(min[i % 3]!, mesh.positions[i]!);
    max[i % 3] = Math.max(max[i % 3]!, mesh.positions[i]!);
  }
  const header = 36;
  const buffer = new ArrayBuffer(header + vertices * 6 + mesh.indices.length * 2 + 2);
  const view = new DataView(buffer);
  view.setUint32(0, MESH_MAGIC, true);
  view.setUint32(4, vertices, true);
  view.setUint32(8, mesh.indices.length, true);
  for (let k = 0; k < 3; k++) {
    view.setFloat32(12 + k * 4, min[k]!, true);
    view.setFloat32(24 + k * 4, max[k]!, true);
  }
  const quantized = new Uint16Array(buffer, header, vertices * 3);
  for (let i = 0; i < mesh.positions.length; i++)
    quantized[i] = Math.round(
      ((mesh.positions[i]! - min[i % 3]!) / (max[i % 3]! - min[i % 3]!)) * 65535,
    );
  new Uint16Array(buffer, header + vertices * 6, mesh.indices.length).set(mesh.indices);
  return buffer;
}

const root = new URL('../', import.meta.url);
const bunny = await decodeDraco(new URL('demo/assets/honey/bunny.drc', root).pathname);
normalize(bunny);
console.log(`closed ${closeHoles(bunny)} holes`);
const coarse = cluster(bunny, CLUSTER);
console.log(
  `bunny: ${bunny.indices.length / 3} display triangles, ${coarse.indices.length / 3} collision triangles`,
);
const sdf = bakeMeshToSdf({
  positions: coarse.positions,
  indices: coarse.indices,
  resolution: SDF_RESOLUTION,
  padding: 0.04,
});
const out = new URL('public/models/honey/', root);
await mkdir(out, { recursive: true });
await writeFile(new URL('bunny.mesh.bin', out), Buffer.from(encodeMesh(bunny)));
await writeFile(new URL('bunny.sdf.bin', out), Buffer.from(encodeSdfBinary(sdf)));
console.log('wrote public/models/honey/bunny.mesh.bin and bunny.sdf.bin');
