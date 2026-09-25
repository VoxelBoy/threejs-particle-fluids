export {
  ContactBuffer,
  ContactRecord,
  DEFAULT_LAMBDA_SCALE,
  type ContactBufferOptions,
} from './ContactBuffer.js';
export {
  DEFAULT_FRICTION_TABLE_GROUPS,
  FrictionTable,
  type FrictionTableOptions,
} from './FrictionTable.js';
export {
  buildContactGenerateKernel,
  buildContactGenerateRangedKernel,
  type BuildContactGenerateArgs,
  type BuildContactGenerateRangedArgs,
} from './generate.js';
export {
  ACCUMULATOR_HEADROOM,
  ContactAccumulator,
  buildApplyAccumulatorToBothKernel,
  buildApplyAccumulatorToPredictedKernel,
  buildResetAccumulatorKernel,
  buildResetOverflowFlagKernel,
  deriveAccumulatorScale,
  emitAccumulateDelta,
} from './accumulator.js';
export { buildContactSolveKernel } from './solve.js';
export { buildContactStabilizeKernel } from './stabilize.js';
export { buildCopyContactInvMassKernel } from './contactInvMass.js';
export {
  emitGeometrySelection,
  type ContactGeometryExtension,
  type EmitGeometrySelectionArgs,
} from './extension.js';
export { emitContactSolveCorrection, type EmitContactSolveCorrectionArgs } from './correction.js';
export {
  DEFAULT_MAX_VELOCITY,
  VELOCITY_ACCUMULATOR_HEADROOM,
  VelocityAccumulator,
  buildApplyVelocityAccumulatorKernel,
  buildResetVelocityAccumulatorKernel,
  buildResetVelocityOverflowFlagKernel,
  deriveVelocityAccumulatorScale,
  emitAccumulateVelocityDelta,
} from './velocityAccumulator.js';
export { buildContactFrictionVelocityKernel } from './frictionVelocity.js';
