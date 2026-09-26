import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import type { ElasticAsset } from '../../../demo/presets/elastic.js';

for (const name of ['donut', 'croissant', 'banana', 'ginger-bread']) {
  const asset = JSON.parse(
    readFileSync(new URL(`../../../public/models/elastic/${name}.json`, import.meta.url), 'utf8'),
  ) as ElasticAsset;
  it(`${name} retains texture coordinates and smooth normals across UV seams`, () => {
    const vertices = asset.positions.length / 3;
    expect(asset.uvs.length).toBe(vertices * 2);
    expect(asset.normals.length).toBe(vertices * 3);
    expect(asset.uvs.every(Number.isFinite)).toBe(true);
    expect(Math.max(...asset.uvs) - Math.min(...asset.uvs)).toBeGreaterThan(0.01);
    const shared = new Map<string, number[]>();
    for (let i = 0; i < vertices; i++) {
      const normal = asset.normals.slice(i * 3, i * 3 + 3);
      expect(Math.hypot(...normal)).toBeCloseTo(1, 4);
      const key = asset.positions
        .slice(i * 3, i * 3 + 3)
        .map((n) => n.toFixed(6))
        .join(',');
      const previous = shared.get(key);
      if (previous)
        for (let axis = 0; axis < 3; axis++) expect(normal[axis]).toBeCloseTo(previous[axis]!, 5);
      else shared.set(key, normal);
    }
  });
  for (const budget of [50, 250, 500, 750, 1250]) {
    it(`${name} has a connected, non-overlapping ${budget}-particle template`, () => {
      const body = asset.templates[budget]!;
      const count = body.positions.length / 3;
      // Thin limbs may keep a few samples beyond tiny budgets rather than split.
      expect(count).toBeGreaterThanOrEqual(budget);
      expect(count).toBeLessThanOrEqual(Math.ceil(budget * 1.2));
      expect(body.surface.length).toBe(count);
      const neighbors = Array.from({ length: count }, () => [] as number[]);
      for (let i = 0; i < body.edges.length; i += 2) {
        const a = body.edges[i]!,
          b = body.edges[i + 1]!;
        expect(a).toBeGreaterThanOrEqual(0);
        expect(b).toBeLessThan(count);
        expect(a).toBeLessThan(b);
        neighbors[a]!.push(b);
        neighbors[b]!.push(a);
      }
      const visited = new Set<number>(),
        pending = [0];
      while (pending.length) {
        const node = pending.pop()!;
        if (visited.has(node)) continue;
        visited.add(node);
        pending.push(...neighbors[node]!.filter((n) => !visited.has(n)));
      }
      expect(visited.size).toBe(count);
      const boundaryCount = body.surface.reduce((sum: number, flag: number) => sum + flag, 0);
      expect(boundaryCount).toBeGreaterThan(0);
      // At the smallest budget every sample lies on the surface.
      if (budget >= 250) expect(boundaryCount).toBeLessThan(count);
      expect(body.surface.slice(0, boundaryCount).every((flag: number) => flag === 1)).toBe(true);
      expect(body.surface.slice(boundaryCount).every((flag: number) => flag === 0)).toBe(true);
      const diameter = 0.03 * Math.cbrt(200 / budget);
      let minimumDistance = Infinity;
      for (let a = 0; a < count; a++)
        for (let b = a + 1; b < count; b++) {
          minimumDistance = Math.min(
            minimumDistance,
            Math.hypot(
              ...[0, 1, 2].map((k) => body.positions[a * 3 + k]! - body.positions[b * 3 + k]!),
            ),
          );
        }
      expect(minimumDistance).toBeCloseTo(diameter, 5);
      // A render mesh hole must stay empty, including its particle representation.
      if (name === 'donut') {
        expect(
          Math.min(
            ...Array.from({ length: count }, (_, i) =>
              Math.hypot(body.positions[i * 3]!, body.positions[i * 3 + 2]!),
            ),
          ),
        ).toBeGreaterThan(diameter);
      }
    });
  }
}
