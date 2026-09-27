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
import {
  PMREMGenerator,
  RenderPipeline,
  type RTTNode,
  type Node,
  type TextureNode,
  type WebGPURenderer,
} from 'three/webgpu';
import { pass, renderOutput, rtt, vec4 } from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { denoise } from 'three/addons/tsl/display/DenoiseNode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
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
  ambientOcclusion = true;
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
  private pointerStart: Vector2 | undefined;
  private interaction: Vector2 | undefined;
  private renderPipeline!: RenderPipeline;
  private aoTarget: RTTNode | undefined;

  constructor(
    readonly preset: Preset,
    private readonly onStats: (stats: Diagnostics) => void,
    private readonly onLoop: () => void,
    private readonly onError: (error: unknown) => void,
    private readonly onInteraction: () => void,
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
    particles: number,
    cameraState?: CameraState,
  ): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
    this.renderer = await createParticleRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
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
      { renderer: this.renderer, scene: this.scene, camera: this.camera, particles },
      values,
    );
    this.scene.add(...this.experiment.objects);
    // Reconstruct normals from depth so custom volume shaders need no extra MRT
    // output. Half-resolution GTAO and edge-aware filtering keep the cost modest.
    // Normal reconstruction requires a single-sample depth texture. FXAA runs
    // after tone mapping to smooth silhouettes without multisampled depth reads.
    const scenePass = pass(this.scene, this.camera, { samples: 1 });
    const color = scenePass.getTextureNode('output');
    const depth = scenePass.getTextureNode('depth');
    // @ts-expect-error Three.js supports null to reconstruct normals; r184 typings omit it.
    const occlusion = ao(depth, null, this.camera);
    occlusion.resolutionScale = 0.5;
    occlusion.radius.value = 0.15;
    occlusion.thickness.value = 0.12;
    occlusion.samples.value = 12;
    occlusion.scale.value = 1.1;
    // @ts-expect-error Same optional-normal support as GTAONode.
    const filtered = denoise(occlusion.getTextureNode(), depth, null, this.camera);
    filtered.radius.value = 3;
    filtered.depthPhi.value = 0.05;
    // Denoise into its own half-resolution target too; the composite upsamples it
    // with bilinear filtering. Only the colour pass runs at full resolution.
    this.aoTarget = rtt(filtered as unknown as Node<'vec4'>, 1, 1);
    this.sizeAoTarget();
    this.renderPipeline = new RenderPipeline(this.renderer);
    const composite = vec4(color.rgb.mul(this.aoTarget.r.mul(0.7).add(0.3)), color.a);
    const display = rtt(
      renderOutput(composite, this.renderer.toneMapping, this.renderer.outputColorSpace),
    );
    this.renderPipeline.outputColorTransform = false;
    this.renderPipeline.outputNode = fxaa(display);
    this.cleanup.push(() => {
      this.renderPipeline.dispose();
      display.renderTarget?.dispose();
      display.dispose();
      occlusion.dispose();
      this.aoTarget?.renderTarget?.dispose();
      this.aoTarget?.dispose();
      // Denoise owns a small noise texture; it has no render target of its own.
      (filtered.noiseNode as TextureNode).value.dispose();
      scenePass.dispose();
    });
    const down = (event: PointerEvent) => {
      if (event.button === 0) this.pointerStart = new Vector2(event.clientX, event.clientY);
    };
    const up = (event: PointerEvent) => {
      if (
        this.pointerStart &&
        this.pointerStart.distanceTo(new Vector2(event.clientX, event.clientY)) < 5
      ) {
        const rect = this.canvas.getBoundingClientRect();
        this.interaction = new Vector2(
          (event.clientX - rect.left) / rect.width,
          (event.clientY - rect.top) / rect.height,
        );
      }
      this.pointerStart = undefined;
    };
    const cancel = () => {
      this.pointerStart = undefined;
    };
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Enter' && this.experiment.interact)
        this.interaction = new Vector2(0.5, 0.5);
    };
    this.canvas.addEventListener('pointerdown', down);
    this.canvas.addEventListener('pointerup', up);
    this.canvas.addEventListener('pointercancel', cancel);
    this.canvas.addEventListener('keydown', keydown);
    this.cleanup.push(() => {
      this.canvas.removeEventListener('pointerdown', down);
      this.canvas.removeEventListener('pointerup', up);
      this.canvas.removeEventListener('pointercancel', cancel);
      this.canvas.removeEventListener('keydown', keydown);
    });
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
    this.sizeAoTarget();
  }

  private sizeAoTarget(): void {
    if (!this.aoTarget) return;
    const size = this.renderer.getDrawingBufferSize(new Vector2());
    this.aoTarget.setSize(Math.max(1, Math.floor(size.x / 2)), Math.max(1, Math.floor(size.y / 2)));
  }

  private async render(): Promise<void> {
    const size = this.renderer.getSize(new Vector2());
    if (size.x !== this.width || size.y !== this.height) this.resize(this.width, this.height);
    this.controls.update();
    await this.experiment.prepareRender?.();
    if (this.interaction) {
      const uv = this.interaction;
      this.interaction = undefined;
      if (await this.experiment.interact?.(uv)) this.onInteraction();
    }
    if (this.ambientOcclusion) this.renderPipeline.render();
    else this.renderer.render(this.scene, this.camera);
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
        substeps: this.experiment.loop.substeps,
        iterations: this.experiment.loop.iterations,
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
