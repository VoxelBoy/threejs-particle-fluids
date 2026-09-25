import { Mesh, PlaneGeometry, Scene, type Texture } from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import {
  Break,
  Discard,
  Fn,
  If,
  Loop,
  cameraProjectionMatrix,
  float,
  int,
  ivec2,
  positionLocal,
  screenCoordinate,
  screenSize,
  select,
  texture,
  textureLoad,
  textureSize,
  uniform,
  vec4,
} from 'three/tsl';
import { NRF_FAR_SENTINEL_THRESHOLD } from '../passes/smoothing.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Pass-2 debug materials (views #6–#12). Each shader samples the NRF
 * input/output texture(s) and emits a per-pixel diagnostic.
 *
 * The NRF intermediates (clampActivation, rangeMask, kernelSize,
 * biasDiscardMask) reproduce the production-pipeline math from
 * `passes/smoothing.ts` (Truong & Yuksel 2018 eqs. 1–9 + the
 * convention adapter for our positive-distance encoding). The math is
 * duplicated inline; if `passes/smoothing.ts` ever changes, this file
 * MUST track the change — any divergence makes views #8–#11 lie about
 * what production is computing.
 *
 * `depthGray` and `deltaPerIter` are algorithm-agnostic visualisers.
 */
export type Pass2DebugMode =
  | 'depthGray'
  | 'clampActivation'
  | 'rangeMask'
  | 'kernelSize'
  | 'biasDiscardMask'
  | 'deltaPerIter';

export interface BuildPass2DebugMaterialArgs {
  readonly mode: Pass2DebugMode;
  readonly inputTexture: Texture;
  /** For `deltaPerIter` only — texture holding iter K−1. */
  readonly prevTexture?: Texture;
  readonly depthScaleUniform: ReturnType<typeof uniform<'float', number>>;
  readonly derivativeScaleUniform: ReturnType<typeof uniform<'float', number>>;
  readonly filterSigmaUniform: ReturnType<typeof uniform<'float', number>>;
  readonly filterDeltaUniform: ReturnType<typeof uniform<'float', number>>;
  readonly filterMuUniform: ReturnType<typeof uniform<'float', number>>;
}

const MAX_KERNEL_HALF_WIDTH = 64;

export function buildPass2DebugMaterial(args: BuildPass2DebugMaterialArgs): MeshBasicNodeMaterial {
  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthTest = false;
  material.depthWrite = false;
  material.vertexNode = vec4(positionLocal.xy, 0.0, 1.0);

  const tex: Any = args.inputTexture;
  const prev: Any = args.prevTexture;
  const dScale: Any = args.derivativeScaleUniform;
  const sigmaUni: Any = args.filterSigmaUniform;
  const deltaUni: Any = args.filterDeltaUniform;

  material.outputNode = Fn(() => {
    // The source RT (`tex`) is at smoothing resolution, which can be
    // smaller than the canvas (smoothing.resolution = half / quarter).
    // Index texels in TEXTURE space, not canvas space — otherwise
    // canvas fragments outside the texture extent textureLoad-OOB to 0
    // and only the lower-left corner shows valid data. The kernel-size
    // math (sigma_i_f) and the 1D walk (xPos = px ± k) also need to be
    // in texture pixels to match the production smoothing pass.
    // textureSize requires a TextureNode; raw Texture (which textureLoad
    // auto-wraps) silently returns wrong dims and the fragment math
    // collapses to texel (0,0) for every pixel.
    const texSizeXY: Any = textureSize(texture(tex), int(0));
    const W: Any = texSizeXY.x.toUint();
    const Hres: Any = texSizeXY.y.toUint();
    const sx: Any = screenCoordinate.x;
    const sy: Any = screenCoordinate.y;
    const sw: Any = screenSize.x;
    const sh: Any = screenSize.y;
    const px: Any = sx.div(sw.toFloat()).mul(W.toFloat()).toUint();
    const py: Any = sy.div(sh.toFloat()).mul(Hres.toFloat()).toUint();

    const z_i: Any = textureLoad(tex, ivec2(px, py)).r.toVar();
    const isBackground: Any = z_i.greaterThanEqual(float(NRF_FAR_SENTINEL_THRESHOLD));
    Discard(isBackground);

    if (args.mode === 'depthGray') {
      const t: Any = z_i.div(args.depthScaleUniform).clamp(0.0, 1.0);
      return vec4(t, t, t, 1.0);
    }

    if (args.mode === 'deltaPerIter') {
      const zPrev: Any = textureLoad(prev, ivec2(px, py)).r;
      const prevIsBg: Any = zPrev.greaterThanEqual(float(NRF_FAR_SENTINEL_THRESHOLD));
      const delta: Any = select(prevIsBg, float(0.0), z_i.sub(zPrev));
      const pos: Any = delta.mul(dScale).clamp(0.0, 1.0);
      const neg: Any = delta.negate().mul(dScale).clamp(0.0, 1.0);
      return vec4(pos, 0.0, neg, 1.0);
    }

    const Fy: Any = (cameraProjectionMatrix as Any).element(1).y;
    const sigma_i_f: Any = Hres.toFloat()
      .mul(sigmaUni)
      .mul(Fy)
      .div(z_i.mul(2.0).max(float(1e-4)))
      .max(float(1.0));

    if (args.mode === 'kernelSize') {
      const t: Any = sigma_i_f.div(32.0).clamp(0.0, 1.0);
      return vec4(t, t, t, 1.0);
    }

    const halfWidth: Any = sigma_i_f.mul(3.0).ceil().toUint();
    const clampHits: Any = float(0.0).toVar();
    const rangeMaskHits: Any = float(0.0).toVar();
    const biasDropHits: Any = float(0.0).toVar();
    const totalPairs: Any = float(0.0).toVar();
    const deltaFar: Any = deltaUni.toVar();
    const deltaNear: Any = deltaUni.toVar();

    Loop(
      { start: int(1), end: int(MAX_KERNEL_HALF_WIDTH), type: 'int', condition: '<=' },
      ({ i: k }: { i: Any }) => {
        If(k.toUint().greaterThan(halfWidth), () => {
          Break();
        });

        const xPos: Any = px
          .toInt()
          .add(k)
          .max(int(0))
          .min(W.toInt().sub(int(1)));
        const xNeg: Any = px
          .toInt()
          .sub(k)
          .max(int(0))
          .min(W.toInt().sub(int(1)));
        const z_j_pos: Any = textureLoad(tex, ivec2(xPos, py)).r;
        const z_j_neg: Any = textureLoad(tex, ivec2(xNeg, py)).r;

        const tooClose_pos: Any = z_j_pos.lessThan(z_i.sub(deltaNear));
        const tooClose_neg: Any = z_j_neg.lessThan(z_i.sub(deltaNear));
        const dropPair: Any = tooClose_pos.or(tooClose_neg);

        totalPairs.assign(totalPairs.add(1.0));

        If(dropPair, () => {
          biasDropHits.assign(biasDropHits.add(1.0));
          If(tooClose_pos, () => {
            rangeMaskHits.assign(rangeMaskHits.add(1.0));
          });
          If(tooClose_neg, () => {
            rangeMaskHits.assign(rangeMaskHits.add(1.0));
          });
        }).Else(() => {
          const clamp_pos: Any = z_j_pos.greaterThan(z_i.add(deltaFar));
          const clamp_neg: Any = z_j_neg.greaterThan(z_i.add(deltaFar));
          If(clamp_pos, () => {
            clampHits.assign(clampHits.add(1.0));
          });
          If(clamp_neg, () => {
            clampHits.assign(clampHits.add(1.0));
          });

          const inRange_pos: Any = z_j_pos.lessThanEqual(z_i.add(deltaFar));
          const inRange_neg: Any = z_j_neg.lessThanEqual(z_i.add(deltaFar));
          If(inRange_pos, () => {
            const newFar: Any = z_j_pos.sub(z_i).add(deltaUni);
            const newNear: Any = z_i.sub(z_j_pos).add(deltaUni);
            deltaFar.assign(deltaFar.max(newFar));
            deltaNear.assign(deltaNear.max(newNear));
          });
          If(inRange_neg, () => {
            const newFar: Any = z_j_neg.sub(z_i).add(deltaUni);
            const newNear: Any = z_i.sub(z_j_neg).add(deltaUni);
            deltaFar.assign(deltaFar.max(newFar));
            deltaNear.assign(deltaNear.max(newNear));
          });
        });
      },
    );

    const totalSafe: Any = totalPairs.max(float(1.0));

    if (args.mode === 'clampActivation') {
      const v: Any = clampHits.div(totalSafe.mul(2.0)).clamp(0.0, 1.0);
      return vec4(v, v, v, 1.0);
    }
    if (args.mode === 'rangeMask') {
      const v: Any = rangeMaskHits.div(totalSafe.mul(2.0)).clamp(0.0, 1.0);
      return vec4(v, v, v, 1.0);
    }
    const v: Any = biasDropHits.div(totalSafe).clamp(0.0, 1.0);
    return vec4(v, v, v, 1.0);
  })();

  return material;
}

export function buildPass2DebugScene(material: MeshBasicNodeMaterial): {
  readonly mesh: Mesh;
  readonly scene: Scene;
} {
  const mesh = new Mesh(new PlaneGeometry(2, 2), material);
  mesh.frustumCulled = false;
  const scene = new Scene();
  scene.add(mesh);
  return { mesh, scene };
}
