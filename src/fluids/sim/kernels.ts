/**
 *
 *
 * This file remains so existing fluid kernel imports (`./kernels.js`) keep
 * compiling unchanged — there is no public-API change for `src/fluids`.
 */
export {
  createSphKernelUniforms,
  emitPoly6,
  emitPoly6FromRSq,
  emitSpikyGrad,
  type SphKernelUniforms,
} from '../../core/index.js';
