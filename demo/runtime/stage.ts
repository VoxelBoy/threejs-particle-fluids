import {
  BoxGeometry,
  Color,
  CylinderGeometry,
  Group,
  Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
  type Object3D,
} from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';

export function material(color: number, roughness = 0.4, metalness = 0.15): MeshStandardMaterial {
  return new MeshStandardMaterial({ color, roughness, metalness });
}

export function block(
  size: [number, number, number],
  position: [number, number, number],
  color = 0xabb4b5,
  radius = 0.035,
): Mesh {
  const geometry =
    radius > 0 ? new RoundedBoxGeometry(...size, 3, radius) : new BoxGeometry(...size);
  const mesh = new Mesh(geometry, material(color));
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export function basin(width = 1.65, depth = 1.15): Group {
  const group = new Group();
  group.add(block([width + 0.12, 0.12, depth + 0.12], [0, -0.075, 0], 0x9daeb6));
  const rim = 0.035;
  const color = 0x657882;
  group.add(block([width + 0.12, 0.035, rim], [0, 0.003, -depth / 2 - 0.03], color, 0.008));
  group.add(block([width + 0.12, 0.035, rim], [0, 0.003, depth / 2 + 0.03], color, 0.008));
  group.add(block([rim, 0.035, depth + 0.12], [-width / 2 - 0.03, 0.003, 0], color, 0.008));
  group.add(block([rim, 0.035, depth + 0.12], [width / 2 + 0.03, 0.003, 0], color, 0.008));
  return group;
}

export function pedestal(radius = 0.95): Mesh {
  const mesh = new Mesh(
    new CylinderGeometry(radius, radius, 0.11, 96),
    material(0x75828c, 0.38, 0.35),
  );
  mesh.position.y = -0.065;
  mesh.receiveShadow = true;
  mesh.castShadow = true;
  return mesh;
}

export function floor(): Mesh {
  const mesh = new Mesh(new PlaneGeometry(200, 200), material(0x03060a, 0.9, 0));
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = -0.15;
  mesh.receiveShadow = true;
  return mesh;
}

export function disposeObjects(objects: readonly Object3D[]): void {
  const geometries = new Set();
  const materials = new Set();
  for (const object of objects) {
    object.removeFromParent();
    object.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      if (!geometries.has(child.geometry)) {
        child.geometry.dispose();
        geometries.add(child.geometry);
      }
      for (const value of Array.isArray(child.material) ? child.material : [child.material]) {
        if (!materials.has(value)) {
          value.dispose();
          materials.add(value);
        }
      }
    });
  }
}

export const backdrop = new Color(0x10161e);
