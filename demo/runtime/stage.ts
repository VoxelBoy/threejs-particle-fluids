import {
  BoxGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  ExtrudeGeometry,
  Group,
  LatheGeometry,
  Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
  Shape,
  Vector2,
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

function roundedOutline(width: number, height: number, radius: number): Shape {
  const x = width / 2,
    y = height / 2;
  const shape = new Shape();
  shape.moveTo(-x + radius, -y);
  shape.lineTo(x - radius, -y);
  shape.quadraticCurveTo(x, -y, x, -y + radius);
  shape.lineTo(x, y - radius);
  shape.quadraticCurveTo(x, y, x - radius, y);
  shape.lineTo(-x + radius, y);
  shape.quadraticCurveTo(-x, y, -x, y - radius);
  shape.lineTo(-x, -y + radius);
  shape.quadraticCurveTo(-x, -y, -x + radius, -y);
  return shape;
}

function extruded(
  shape: Shape,
  depth: number,
  bevel: number,
  color: number,
  roughness = 0.4,
  metalness = 0.35,
): Mesh {
  const mesh = new Mesh(
    new ExtrudeGeometry(shape, {
      depth,
      bevelEnabled: true,
      bevelSize: bevel,
      bevelThickness: bevel,
      bevelSegments: 3,
      curveSegments: 12,
      steps: 1,
    }),
    material(color, roughness, metalness),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** Continuous frame, including rounded corners; there are no intersecting rails. */
export function panelFrame(width: number, height: number, rail = 0.025, depth = 0.03): Mesh {
  const outline = roundedOutline(width, height, 0.035);
  outline.holes.push(roundedOutline(width - rail * 2, height - rail * 2, 0.015));
  const frame = extruded(outline, depth, 0.0025, 0x546c79, 0.3, 0.65);
  frame.geometry.translate(0, 0, -depth / 2);
  return frame;
}

/** Beveled graphite plinth, inset deck, and recessed support feet. Top is y=0. */
export function platform(width: number, depth: number): Group {
  const group = new Group();
  const body = extruded(roundedOutline(width, depth, 0.1), 0.065, 0.012, 0x26343f, 0.32, 0.5);
  body.rotation.x = -Math.PI / 2;
  body.position.y = -0.095;
  const deck = extruded(
    roundedOutline(width - 0.055, depth - 0.055, 0.085),
    0.011,
    0.006,
    0x536873,
    0.52,
    0.12,
  );
  deck.name = 'PlatformDeck';
  deck.rotation.x = -Math.PI / 2;
  deck.position.y = -0.017;
  group.add(body, deck);
  const feet = new CylinderGeometry(0.045, 0.05, 0.043, 32);
  const feetMaterial = material(0x101a22, 0.8, 0);
  for (const x of [-1, 1])
    for (const z of [-1, 1]) {
      const foot = new Mesh(feet, feetMaterial);
      foot.position.set(x * (width / 2 - 0.15), -0.128, z * (depth / 2 - 0.15));
      group.add(foot);
    }
  // A small inlaid index mark gives the front edge a finished, manufactured detail.
  group.add(
    block([0.095, 0.006, 0.002], [-width / 2 + 0.19, -0.06, depth / 2 + 0.012], 0xb1bdc3, 0),
  );
  return group;
}

export function basin(width = 1.65, depth = 1.15, glassHeight = 0): Group {
  const group = platform(width + 0.13, depth + 0.13);
  const outline = roundedOutline(width + 0.1, depth + 0.1, 0.078);
  outline.holes.push(roundedOutline(width + 0.035, depth + 0.035, 0.043));
  const rim = extruded(outline, 0.024, 0.004, 0x7f949d, 0.28, 0.7);
  rim.rotation.x = -Math.PI / 2;
  rim.position.y = 0.003;
  group.add(rim);
  if (glassHeight > 0) {
    const shell = roundedOutline(width + 0.065, depth + 0.065, 0.055);
    shell.holes.push(roundedOutline(width + 0.045, depth + 0.045, 0.045));
    const glass = extruded(shell, glassHeight, 0.001, 0xb9dce5, 0.1, 0);
    const glassMaterial = glass.material as MeshStandardMaterial;
    glassMaterial.transparent = true;
    glassMaterial.opacity = 0.1;
    glassMaterial.depthWrite = false;
    glassMaterial.side = DoubleSide;
    glass.castShadow = false;
    glass.rotation.x = -Math.PI / 2;
    glass.position.y = 0.012;
    group.add(glass);
    const cap = extruded(shell, 0.003, 0.001, 0xa4bec8, 0.22, 0.5);
    cap.rotation.x = -Math.PI / 2;
    cap.position.y = glassHeight + 0.013;
    group.add(cap);
  }
  return group;
}

/** One-piece arch with a broad top rail supporting the fabric's entire hem. */
export function clothStand(): Group {
  const group = platform(1.78, 0.76);
  const arch = new Shape();
  arch.moveTo(-0.78, 0);
  arch.lineTo(-0.78, 1.55);
  arch.quadraticCurveTo(-0.78, 1.68, -0.65, 1.68);
  arch.lineTo(0.65, 1.68);
  arch.quadraticCurveTo(0.78, 1.68, 0.78, 1.55);
  arch.lineTo(0.78, 0);
  arch.lineTo(0.725, 0);
  arch.lineTo(0.725, 1.51);
  arch.quadraticCurveTo(0.725, 1.605, 0.63, 1.605);
  arch.lineTo(-0.63, 1.605);
  arch.quadraticCurveTo(-0.725, 1.605, -0.725, 1.51);
  arch.lineTo(-0.725, 0);
  arch.closePath();
  const frame = extruded(arch, 0.048, 0.004, 0x526b79, 0.28, 0.65);
  frame.position.z = -0.024;
  group.add(frame);
  // Flush anchor fasteners sit on the front face, without crossing the frame.
  for (const x of [-0.752, 0.752]) {
    const fastener = new Mesh(
      new CylinderGeometry(0.008, 0.008, 0.002, 24),
      material(0xb1bdc3, 0.3, 0.8),
    );
    fastener.rotation.x = Math.PI / 2;
    fastener.position.set(x, 0.05, 0.029);
    group.add(fastener);
  }
  return group;
}

export function pedestal(radius = 0.95): Mesh {
  const profile = [
    [0, -0.14],
    [radius - 0.07, -0.14],
    [radius - 0.05, -0.12],
    [radius - 0.05, -0.1],
    [radius - 0.01, -0.09],
    [radius, -0.075],
    [radius, -0.028],
    [radius - 0.012, -0.01],
    [radius - 0.035, 0],
    [0, 0],
  ].map(([x, y]) => new Vector2(x, y));
  const mesh = new Mesh(new LatheGeometry(profile, 96), material(0x405762, 0.32, 0.5));
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
