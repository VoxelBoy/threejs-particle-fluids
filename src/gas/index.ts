export { GasSystem, type GasSystemOptions } from './sim/GasSystem.js';
export { buildSmokeAdvectKernel, type BuildSmokeAdvectKernelArgs } from './sim/smokeAdvect.js';
export type { GasRenderer, SmokeTracers } from './render/types.js';
export {
  PointSpritesGasRenderer,
  type PointSpritesGasRendererOptions,
} from './render/PointSpritesGasRenderer.js';
export {
  VolumetricGasRenderer,
  type VolumetricGasRendererOptions,
} from './render/VolumetricGasRenderer.js';
