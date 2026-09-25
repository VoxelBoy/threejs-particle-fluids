import { WebGPURenderer } from 'three/webgpu';
import type { WebGPURendererParameters } from 'three/src/renderers/webgpu/WebGPURenderer.js';

/** Device limits required by the parallel scan and contact kernels. */
const REQUIRED_LIMITS: Record<string, number> = {
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxStorageBuffersPerShaderStage: 10,
};

/** Initialize a renderer with the solver's required limits and reject WebGL fallback. */
export async function createParticleRenderer(
  options: Partial<WebGPURendererParameters> = {},
): Promise<WebGPURenderer> {
  const renderer = new WebGPURenderer({
    ...options,
    requiredLimits: {
      ...(options.requiredLimits ?? {}),
      ...REQUIRED_LIMITS,
    },
  });
  await renderer.init();
  const backend = renderer.backend as { isWebGPUBackend?: boolean };
  if (backend.isWebGPUBackend !== true) {
    renderer.dispose();
    throw new Error(
      `createParticleRenderer: WebGPU adapter unavailable; three.js fell back to ${renderer.backend.constructor.name}. ` +
        `Three.js Particle Fluids requires WebGPU. ` +
        `Check that the browser supports WebGPU and, on Linux Chromium, that --enable-unsafe-webgpu is set.`,
    );
  }
  return renderer;
}
