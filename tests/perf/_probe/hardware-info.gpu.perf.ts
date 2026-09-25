// Phase Perf paper-gap diagnosis — H4 hardware reconciliation.
//
// Dumps the GPU adapter info, all WebGPU `limits` keys, and the set of
// supported features. Output prints a single-line JSON between
// `__PARTICLE_FLUIDS_PAPER_GAP_H4_BEGIN__` / `__PARTICLE_FLUIDS_PAPER_GAP_H4_END__`
// sentinels for the `tests/perf/_helpers/run-paper-gap.ts` driver to
// extract and persist to `tests/perf/results/paper-gap-h4-hardware-<stamp>.json`.
//
// Spec-sheet TFLOPS / memory-bandwidth lookup is done manually at
// diagnosis-report time and lives in the report. The probe captures
// only the data the API exposes — vendor, architecture, device,
// description, plus all numeric limits and feature flags.
//

import { describe, expect, it } from 'vitest';
import { createParticleRenderer } from '../../../src/core/index.js';

const SENTINEL_BEGIN = '__PARTICLE_FLUIDS_PAPER_GAP_H4_BEGIN__';
const SENTINEL_END = '__PARTICLE_FLUIDS_PAPER_GAP_H4_END__';

interface AdapterInfoSnapshot {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
}

interface HardwareInfoReport {
  readonly probeId: 'paper-gap-h4-hardware';
  readonly capturedAtIso: string;
  readonly adapter: AdapterInfoSnapshot;
  readonly features: readonly string[];
  readonly limits: Readonly<Record<string, number>>;
  readonly userAgent: string;
}

async function captureHardwareInfo(): Promise<HardwareInfoReport> {
  const renderer = await createParticleRenderer();
  const backend = renderer.backend as {
    readonly device?: GPUDevice;
    readonly adapter?: GPUAdapter;
  };
  const device = backend.device;

  if (!device) {
    renderer.dispose();
    throw new Error('paper-gap-h4: renderer.backend.device unavailable after init');
  }

  // GPUDevice exposes `adapterInfo` on modern browsers (spec-aligned with
  // GPUAdapter.info). Three.js's `backend.adapter` is also typically
  // present but is an undocumented surface (U-43); prefer the spec path.
  const deviceAdapterInfo = (device as unknown as { adapterInfo?: GPUAdapterInfo }).adapterInfo;
  const backendAdapterInfo = backend.adapter?.info;
  const info = deviceAdapterInfo ?? backendAdapterInfo;

  const adapterSnapshot: AdapterInfoSnapshot = {
    vendor: info?.vendor ?? '',
    architecture: info?.architecture ?? '',
    device: info?.device ?? '',
    description: info?.description ?? '',
  };

  const features: string[] = [];
  device.features.forEach((f) => features.push(f));
  features.sort();

  // GPUSupportedLimits exposes its keys as IDL prototype getters, not as
  // own enumerable properties — Object.keys(device.limits) returns []
  // even when the limits are populated. Enumerate against the known
  // spec key list. Source: WebGPU spec §3.6.2 (GPUSupportedLimits).
  const LIMIT_KEYS: readonly string[] = [
    'maxBindGroups',
    'maxBindGroupsPlusVertexBuffers',
    'maxBindingsPerBindGroup',
    'maxBufferSize',
    'maxColorAttachmentBytesPerSample',
    'maxColorAttachments',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupSizeX',
    'maxComputeWorkgroupSizeY',
    'maxComputeWorkgroupSizeZ',
    'maxComputeWorkgroupStorageSize',
    'maxComputeWorkgroupsPerDimension',
    'maxDynamicStorageBuffersPerPipelineLayout',
    'maxDynamicUniformBuffersPerPipelineLayout',
    'maxInterStageShaderComponents',
    'maxInterStageShaderVariables',
    'maxSampledTexturesPerShaderStage',
    'maxSamplersPerShaderStage',
    'maxStorageBufferBindingSize',
    'maxStorageBuffersInFragmentStage',
    'maxStorageBuffersInVertexStage',
    'maxStorageBuffersPerShaderStage',
    'maxStorageTexturesInFragmentStage',
    'maxStorageTexturesInVertexStage',
    'maxStorageTexturesPerShaderStage',
    'maxTextureArrayLayers',
    'maxTextureDimension1D',
    'maxTextureDimension2D',
    'maxTextureDimension3D',
    'maxUniformBufferBindingSize',
    'maxUniformBuffersPerShaderStage',
    'maxVertexAttributes',
    'maxVertexBufferArrayStride',
    'maxVertexBuffers',
    'minStorageBufferOffsetAlignment',
    'minUniformBufferOffsetAlignment',
  ];
  const limits: Record<string, number> = {};
  const limitsObj = device.limits as unknown as Record<string, number>;
  for (const key of [...LIMIT_KEYS].sort()) {
    const v = limitsObj[key];
    if (typeof v === 'number') limits[key] = v;
  }

  renderer.dispose();

  return {
    probeId: 'paper-gap-h4-hardware',
    capturedAtIso: new Date().toISOString(),
    adapter: adapterSnapshot,
    features,
    limits,
    userAgent:
      typeof navigator !== 'undefined' && typeof navigator.userAgent === 'string'
        ? navigator.userAgent
        : '',
  };
}

describe('Phase Perf paper-gap H4 — hardware info', () => {
  it('captures adapter info, features, and limits', async () => {
    const report = await captureHardwareInfo();

    // Single-line JSON between sentinels for the driver to extract.
    const line = JSON.stringify(report);
    // eslint-disable-next-line no-console
    console.log(SENTINEL_BEGIN);
    // eslint-disable-next-line no-console
    console.log(line);
    // eslint-disable-next-line no-console
    console.log(SENTINEL_END);

    // Soft asserts: the probe records the data, it does not assert any
    // specific hardware. We only fail if the API returned nothing
    // useful at all — that means either the adapter init is broken or
    // the browser is non-WebGPU-compliant.
    expect(report.adapter).toBeTruthy();
    expect(Object.keys(report.limits).length).toBeGreaterThan(0);
  }, 60_000);
});
