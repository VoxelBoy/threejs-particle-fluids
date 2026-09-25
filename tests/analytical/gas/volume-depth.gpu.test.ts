import { expect, it } from 'vitest';
import { BoxGeometry, Mesh, PerspectiveCamera, RenderTarget, Scene, Vector3 } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { instancedArray } from 'three/tsl';
import { createParticleRenderer } from '../../../src/core/index.js';
import { VolumetricGasRenderer, type GasSystem } from '../../../src/gas/index.js';

it('smoke blends in front of an intersecting solid and stays behind a foreground solid', async () => {
  const renderer = await createParticleRenderer();
  const target = new RenderTarget(128, 128);
  renderer.setSize(128, 128);
  renderer.setRenderTarget(target);
  const scene = new Scene();
  const camera = new PerspectiveCamera(35, 1, 0.1, 10);
  camera.position.z = 3;
  const geometry = new BoxGeometry(0.16, 0.5, 0.12);
  const material = new MeshBasicNodeMaterial({ color: 0xff0000 });
  const front = new Mesh(geometry, material),
    middle = new Mesh(geometry, material);
  front.position.set(-0.22, 0, 0.72);
  middle.position.set(0.22, 0, 0);
  scene.add(front, middle);
  const positions: number[] = [];
  for (let z = -0.4; z <= 0.4; z += 0.06)
    for (let y = -0.4; y <= 0.4; y += 0.06)
      for (let x = -0.4; x <= 0.4; x += 0.06) positions.push(x, y, z, 0);
  const count = positions.length / 4;
  const gas = {
    capacity: count,
    lifetime: 8,
    smokePositions: instancedArray(Float32Array.from(positions), 'vec4'),
    smokeAlive: instancedArray(new Uint32Array(count).fill(1), 'uint'),
    smokeAge: instancedArray(count, 'float'),
  } as unknown as GasSystem;
  const volume = new VolumetricGasRenderer({
    gas,
    min: new Vector3(-0.5, -0.5, -0.5),
    max: new Vector3(0.5, 0.5, 0.5),
    resolution: [32, 32, 32],
    steps: 48,
    density: 5,
  });
  const capture = async () => {
    await renderer.renderAsync(scene, camera);
    return renderer.readRenderTargetPixelsAsync(target, 0, 0, 128, 128);
  };
  try {
    const before = await capture();
    await volume.update(renderer);
    scene.add(volume.object);
    const after = await capture();
    const channelDelta = (point: Vector3, channel: number) => {
      const p = point.clone().project(camera);
      const x = Math.round((p.x * 0.5 + 0.5) * 127);
      const y = Math.round((0.5 - p.y * 0.5) * 127);
      let delta = 0;
      for (let dy = -2; dy <= 2; dy++)
        for (let dx = -2; dx <= 2; dx++) {
          const i = ((y + dy) * 128 + x + dx) * 4 + channel;
          delta += Math.abs(Number(after[i]) - Number(before[i]));
        }
      return delta / 25;
    };
    // The front object's depth is before the ray entry; its color must survive.
    expect(channelDelta(front.position, 0)).toBeLessThan(2);
    expect(channelDelta(front.position, 1)).toBeLessThan(2);
    // The middle object's depth is before the box exit, but smoke occupies the
    // segment in front of it. Testing the box faces used to erase that segment.
    expect(channelDelta(middle.position, 1)).toBeGreaterThan(8);
    expect(channelDelta(new Vector3(), 1)).toBeGreaterThan(8);
  } finally {
    volume.dispose();
    geometry.dispose();
    material.dispose();
    target.dispose();
    renderer.dispose();
  }
}, 30_000);
