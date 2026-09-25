import {
  ACESFilmicToneMapping,
  DirectionalLight,
  Fog,
  HemisphereLight,
  PerspectiveCamera,
  Scene,
  Vector2,
  Vector3,
} from 'three';
import { PMREMGenerator, type WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { FrameStepper, createParticleRenderer } from '../../src/core/index.js';
import type { Experiment, Preset, Values } from '../types.js';
import { backdrop, disposeObjects, floor } from './stage.js';

export interface Diagnostics {
  fps: number;
  frameMs: number;
  particles: number;
  time: number;
  substeps: number;
  iterations: number;
  limited: boolean;
}
export interface CameraState {
  position: Vector3;
  target: Vector3;
}

export class World {
  readonly canvas = document.createElement('canvas');
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(38, 1, 0.02, 100);
  renderer!: WebGPURenderer;
  experiment!: Experiment;
  controls!: OrbitControls;
  playing = true;
  looping = true;
  time = 0;
  private stopped = false;
  private frameId = 0;
  private pending: Promise<void> = Promise.resolve();
  private stepper = new FrameStepper({ fixedDt: 1 / 60, maxSubstepsPerFrame: 2 });
  private cleanup: (() => void)[] = [];
  private lastFrame = 0;
  private frameMs = 16.67;
  private lastReport = 0;
  private width = 0;
  private height = 0;
  private captureRequest: { resolve(blob: Blob): void; reject(error: Error): void } | undefined;

  constructor(
    readonly preset: Preset,
    private readonly onStats: (stats: Diagnostics) => void,
    private readonly onLoop: () => void,
    private readonly onError: (error: unknown) => void,
  ) {
    this.canvas.setAttribute(
      'aria-label',
      `${preset.name} interactive 3D simulation. Drag to orbit, scroll to zoom.`,
    );
    this.canvas.tabIndex = 0;
  }

  async init(
    host: HTMLElement,
    values: Values,
    quality: 'balanced' | 'high',
    cameraState?: CameraState,
  ): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
    this.renderer = await createParticleRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, quality === 'high' ? 1.75 : 1.25),
    );
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.9;
    this.renderer.shadowMap.enabled = true;
    this.scene.background = backdrop;
    this.scene.fog = new Fog(backdrop, 4, 10);
    const room = new RoomEnvironment();
    const pmrem = new PMREMGenerator(this.renderer);
    const environment = pmrem.fromScene(room, 0.04);
    this.scene.environment = environment.texture;
    this.scene.environmentIntensity = 0.45;
    room.dispose();
    pmrem.dispose();
    this.cleanup.push(() => environment.dispose());
    this.scene.add(new HemisphereLight(0xd7e6ff, 0x111721, 0.65));
    const key = new DirectionalLight(0xf8f1e5, 2.5);
    key.position.set(1.5, 4, 2);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = key.shadow.camera.bottom = -2;
    key.shadow.camera.right = key.shadow.camera.top = 2;
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 9;
    key.shadow.bias = -0.0005;
    key.shadow.normalBias = 0.012;
    this.cleanup.push(() => key.dispose());
    const rim = new DirectionalLight(0xa4c4ff, 1.5);
    rim.position.set(-2, 2.5, -2);
    this.scene.add(key, rim, floor());
    this.camera.position.fromArray(this.preset.camera);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.fromArray(this.preset.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.1;
    this.controls.maxDistance = 7;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    if (cameraState) {
      this.camera.position.copy(cameraState.position);
      this.controls.target.copy(cameraState.target);
    }
    this.controls.update();
    this.resize(host.clientWidth, host.clientHeight);
    this.experiment = await this.preset.build(
      { renderer: this.renderer, scene: this.scene, camera: this.camera, quality },
      values,
    );
    this.scene.add(...this.experiment.objects);
    const observer = new ResizeObserver(() => {
      this.width = host.clientWidth;
      this.height = host.clientHeight;
    });
    observer.observe(host);
    this.cleanup.push(() => observer.disconnect());
    await this.render();
    const visibility = () => {
      this.stepper.reset();
      this.lastFrame = 0;
    };
    document.addEventListener('visibilitychange', visibility);
    this.cleanup.push(() => document.removeEventListener('visibilitychange', visibility));
    // Device loss should surface as a recoverable UI state, never a frozen canvas.
    const backend = this.renderer.backend as unknown as { device?: GPUDevice };
    void backend.device?.lost.then((info) => {
      if (!this.stopped && info.reason !== 'destroyed')
        this.onError(
          new Error('The graphics device was interrupted. Reload the preset to continue.'),
        );
    });
  }

  private resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.renderer.setSize(this.width, this.height, false);
    this.camera.aspect = this.width / this.height;
    // Preserve enough horizontal space for the whole experiment in portrait layouts.
    this.camera.fov =
      (2 *
        Math.atan(Math.tan((19 * Math.PI) / 180) * Math.max(1, 1.15 / this.camera.aspect)) *
        180) /
      Math.PI;
    this.camera.updateProjectionMatrix();
  }

  private async render(): Promise<void> {
    const size = this.renderer.getSize(new Vector2());
    if (size.x !== this.width || size.y !== this.height) this.resize(this.width, this.height);
    this.controls.update();
    await this.experiment.prepareRender?.();
    this.renderer.render(this.scene, this.camera);
    if (this.captureRequest) {
      const request = this.captureRequest;
      this.captureRequest = undefined;
      await new Promise<void>((done) =>
        this.canvas.toBlob((blob) => {
          if (blob) request.resolve(blob);
          else request.reject(new Error('Image capture failed.'));
          done();
        }, 'image/png'),
      );
    }
    const backend = this.renderer.backend as unknown as { device?: GPUDevice };
    await backend.device?.queue.onSubmittedWorkDone();
  }

  start(): void {
    this.frameId = requestAnimationFrame((now) => {
      this.pending = this.frame(now).catch((error: unknown) => {
        this.stopped = true;
        this.onError(error);
      });
    });
  }

  private async frame(now: number): Promise<void> {
    if (this.stopped) return;
    let limited = false;
    if (document.hidden || !this.playing) this.stepper.reset();
    else {
      const result = await this.stepper.pump(now, async (dt) => {
        await this.experiment.update?.(dt, this.time);
        await this.experiment.loop.step(dt);
        this.time += dt;
      });
      limited = result.truncated;
    }
    if (!document.hidden) await this.render();
    if (this.lastFrame > 0)
      this.frameMs += (Math.min(1000, now - this.lastFrame) - this.frameMs) * 0.12;
    this.lastFrame = now;
    if (now - this.lastReport > 200) {
      this.lastReport = now;
      this.onStats({
        fps: 1000 / this.frameMs,
        frameMs: this.frameMs,
        particles: this.experiment.particleCount,
        time: this.time,
        substeps: this.experiment.substeps,
        iterations: this.experiment.iterations,
        limited,
      });
    }
    if (this.looping && this.time >= this.preset.duration) {
      this.onLoop();
      return;
    }
    if (!this.stopped) this.start();
  }

  cameraState(): CameraState {
    return { position: this.camera.position.clone(), target: this.controls.target.clone() };
  }
  resetCamera(): void {
    this.camera.position.fromArray(this.preset.camera);
    this.controls.target.fromArray(this.preset.target);
    this.controls.update();
  }

  screenshot(): Promise<Blob> {
    if (this.stopped) return Promise.reject(new Error('The scene is no longer active.'));
    this.captureRequest?.reject(new Error('Another image capture was requested.'));
    return new Promise((resolve, reject) => {
      this.captureRequest = { resolve, reject };
    });
  }

  async dispose(): Promise<void> {
    this.stopped = true;
    cancelAnimationFrame(this.frameId);
    await this.pending;
    this.captureRequest?.reject(new Error('The scene changed before capture completed.'));
    this.captureRequest = undefined;
    this.controls?.dispose();
    this.experiment?.dispose();
    disposeObjects([...this.scene.children]);
    this.cleanup.forEach((fn) => fn());
    this.renderer?.dispose();
    this.canvas.remove();
  }
}
