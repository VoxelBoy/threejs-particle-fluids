import type { Object3D, PerspectiveCamera, Scene, Vector2 } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import type { ParticleSystem, SimLoop } from '../src/core/index.js';

export type Values = Record<string, number>;
export interface Control {
  key: string;
  label: string;
  description: string;
  min: number;
  max: number;
  step: number;
  value: number;
  unit?: string;
  restart?: boolean;
}
export interface Preset {
  id: string;
  name: string;
  category: string;
  description: string;
  accent: string;
  number: string;
  camera: readonly [number, number, number];
  target: readonly [number, number, number];
  duration: number;
  controls: readonly Control[];
  build(context: BuildContext, values: Values): Promise<Experiment> | Experiment;
}
/** Particle budgets offered in the viewport. Presets size their particles to match. */
export const PARTICLE_LEVELS = [
  { id: 'low', label: 'Low', count: 1000 },
  { id: 'medium', label: 'Medium', count: 5000 },
  { id: 'high', label: 'High', count: 10000 },
  { id: 'ultra', label: 'Ultra', count: 15000 },
  { id: 'max', label: 'Max', count: 25000 },
] as const;
export type ParticleLevel = (typeof PARTICLE_LEVELS)[number]['id'];

export interface BuildContext {
  renderer: WebGPURenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  /** Target particle count for the preset's main material. */
  particles: number;
}
export interface Experiment {
  particles: ParticleSystem;
  loop: SimLoop;
  objects: Object3D[];
  particleCount: number;
  substeps: number;
  iterations: number;
  update?(dt: number, time: number): void | Promise<void>;
  prepareRender?(): void | Promise<void>;
  setParameter(key: string, value: number): void;
  setParticleView?(enabled: boolean): void;
  interact?(uv: Vector2): Promise<boolean>;
  dispose(): void;
}
