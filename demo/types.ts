import type { Object3D, PerspectiveCamera, Scene } from 'three';
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
export interface BuildContext {
  renderer: WebGPURenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  quality: 'balanced' | 'high';
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
  disturb?(): void;
  dispose(): void;
}
