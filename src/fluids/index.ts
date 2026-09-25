export { FluidSystem, type FluidSystemOptions } from './FluidSystem.js';
export {
  createSphKernelUniforms,
  emitPoly6,
  emitPoly6FromRSq,
  emitSpikyGrad,
  type SphKernelUniforms,
} from './sim/kernels.js';
export { buildLambdaKernel, type BuildLambdaKernelArgs } from './sim/lambda.js';
export {
  buildApplyDeltaKernel,
  buildPositionDeltaKernel,
  type BuildApplyDeltaKernelArgs,
  type BuildPositionDeltaKernelArgs,
} from './sim/positionDelta.js';
export { buildCoupledDeltaKernel, type BuildCoupledDeltaKernelArgs } from './sim/coupledDelta.js';
export {
  buildBoundaryVolumeKernel,
  type BuildBoundaryVolumeKernelArgs,
} from './sim/boundaryVolume.js';
export {
  buildVorticityPass1Kernel,
  buildVorticityPass2Kernel,
  buildVorticityPass3Kernel,
  type BuildVorticityPass1KernelArgs,
  type BuildVorticityPass2KernelArgs,
  type BuildVorticityPass3KernelArgs,
} from './sim/vorticity.js';
export {
  buildXsphApplyKernel,
  buildXsphComputeKernel,
  type BuildXsphApplyKernelArgs,
  type BuildXsphComputeKernelArgs,
} from './sim/xsph.js';
export {
  buildFusedVorticityXsphWalkKernel,
  type BuildFusedVorticityXsphWalkKernelArgs,
} from './sim/fusedVorticityXsphWalk.js';
export {
  buildFusedVorticityXsphApplyKernel,
  type BuildFusedVorticityXsphApplyKernelArgs,
} from './sim/fusedVorticityXsphApply.js';
export {
  buildApplyVelocityImpulseKernel,
  buildColorFieldNormalKernel,
  buildSurfaceTensionScatterKernel,
  createCohesionUniforms,
  type BuildApplyVelocityImpulseKernelArgs,
  type BuildColorFieldNormalKernelArgs,
  type BuildSurfaceTensionScatterKernelArgs,
  type CohesionUniforms,
} from './sim/cohesion.js';
export {
  buildAdhesionScatterKernel,
  createAdhesionUniforms,
  type AdhesionUniforms,
  type BuildAdhesionScatterKernelArgs,
} from './sim/adhesion.js';
export {
  buildSolidReactionScatterKernel,
  type BuildSolidReactionScatterKernelArgs,
} from './sim/solidReaction.js';
export { buildDragKernel, type BuildDragKernelArgs } from './sim/drag.js';

// --- Renderer (Phase 14) ---
export {
  FluidSurfaceRenderer,
  buildDefaultParams,
  type BooleanHandle,
  type ColorHandle,
  type DebugParams,
  type DebugView,
  type DepthParams,
  type FluidSurfaceParamDefaults,
  type FluidSurfaceParams,
  type FluidSurfaceRendererOptions,
  type NumberHandle,
  type SelectHandle,
  type SmoothingParams,
  type SmoothingResolution,
  type SurfaceParams,
  type ThicknessParams,
} from './render/index.js';
export { ViscositySolver, type ViscositySolverOptions } from './ViscositySolver.js';
