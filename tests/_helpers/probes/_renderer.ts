// Shared probe helper: create a WebGPURenderer and assert the backend is
// actually WebGPU (three silently falls back to WebGL2 when unavailable).

import { WebGPURenderer } from 'three/webgpu';

// Chrome's default device limit is 256 invocations per workgroup — below the
// 1024 needed for single-pass Blelloch at W=1024. Requesting the higher limit
// explicitly; modern desktop adapters (Metal, Vulkan, D3D12) expose 1024.
export async function createWebGPURenderer(): Promise<WebGPURenderer> {
  const renderer = new WebGPURenderer({
    forceWebGL: false,
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: 1024,
      maxComputeWorkgroupSizeX: 1024,
    },
  });
  await renderer.init();
  const backend = renderer.backend as { isWebGPUBackend?: boolean };
  if (backend.isWebGPUBackend !== true) {
    renderer.dispose();
    throw new Error(
      `Phase 01 probe requires a WebGPU adapter; got ${renderer.backend.constructor.name}`,
    );
  }
  return renderer;
}
