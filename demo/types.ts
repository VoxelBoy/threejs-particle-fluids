import type { Object3D, PerspectiveCamera, Scene, Vector2 } from 'three';
import type { WebGPURenderer } from 'three/webgpu';
import type { ParticleSystem, SimLoop } from '../src/index.js';

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
  /** Sidebar section the preset is listed under. */
  group: 'Liquids' | 'Soft Body' | 'Cloth' | 'Gases';
  description: string;
  accent: string;
  number: string;
  camera: readonly [number, number, number];
  target: readonly [number, number, number];
  duration: number;
  controls: readonly Control[];
  /** Particle count per level, when the defaults don't suit the material. */
  particleCounts?: Readonly<Record<ParticleLevel, number>>;
  build(context: BuildContext, values: Values): Promise<Experiment> | Experiment;
}
/** Particle budgets offered in the viewport. Presets size their particles to match. */
export const PARTICLE_LEVELS = [
  { id: 'low', label: 'Low', count: 5000 },
  { id: 'medium', label: 'Medium', count: 10000 },
  { id: 'high', label: 'High', count: 15000 },
  { id: 'ultra', label: 'Ultra', count: 25000 },
  { id: 'max', label: 'Max', count: 50000 },
] as const;
export type ParticleLevel = (typeof PARTICLE_LEVELS)[number]['id'];
export function particleCount(preset: Preset, level: ParticleLevel): number {
  return (
    preset.particleCounts?.[level] ?? PARTICLE_LEVELS.find((entry) => entry.id === level)!.count
  );
}
/** Elastic studies splits each level across 20 bodies; one baked template per level. */
export const ELASTIC_BODY_BUDGETS = PARTICLE_LEVELS.map((level) => level.count / 20);

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
  /** Screen-space reflections on liquid surfaces. */
  setReflections?(enabled: boolean): void;
  interact?(uv: Vector2): Promise<boolean>;
  dispose(): void;
}
