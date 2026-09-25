import {
  Fn,
  atomicLoad,
  atomicStore,
  instanceIndex,
  instancedArray,
  int,
  struct,
  uint,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type { WebGPURenderer } from 'three/webgpu';

// TSL @types surface atomic/GPGPU nodes as bare `Node`, stripping the
// proxy-provided method chains. Same loose-alias pattern used in `integrate.ts`,
// `hashGrid/*.ts`, `constraints/*.ts`, and `_probe/*`.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Default fixed-point scale for the per-contact accumulated Lagrange
 * multiplier buffers (λ_n, λ_t) used by Macklin 2020 §3.5 static friction
 * gate and §3.6 velocity-friction clamp.
 *
 * Units: ticks per kg·m (the XPBD λ unit for positional constraints).
 * Magnitude budget:
 *   - Max storable signed value: 2^31 / 2^20 = 2048 kg·m.
 * At static equilibrium under gravity, λ_n per substep is
 * `m · g · h² ≈ 1·10·(1/240)² ≈ 1.7e-4` kg·m for MVP-scale particles, so
 * 2048 kg·m is many orders of magnitude of headroom.
 *   - Quantization: 2^-20 ≈ 9.5e-7 kg·m per tick. Smaller than the f32 ULP
 *     envelope of the downstream velocity-friction clamp at MVP scales, so
 *     quantization is not a physical-accuracy concern.
 */
export const DEFAULT_LAMBDA_SCALE = 1 << 20; // 2^20 ≈ 1.05e6 ticks per kg·m

/**
 * Per-contact record laid out as a TSL struct so all per-pair state lives in
 * a single storage buffer binding. Phase 21a refactor collapsed the prior
 * `pairs` (u32×2) + `lambdaNT` (atomic i32×2) + `normal` (vec4) into one
 * struct array, dropping `solve.ts`'s storage-buffer count from 10 → 8 with
 * softbody active. Phase 21 added the per-pair Coulomb-friction fields
 * `muS`/`muK` populated by `generate.ts` from a {@link FrictionTable} LUT
 * (per-phase-group authoring) so the solve and friction-velocity kernels
 * read scalar μ per pair instead of a global uniform.
 *
 * Field offsets (WGSL std430):
 *   - i        u32         offset 0  (4 B)
 *   - j        u32         offset 4  (4 B)
 *   - muS      f32         offset 8  (4 B) — per-pair static  μ_s (Macklin 2020 §3.5)
 *   - muK      f32         offset 12 (4 B) — per-pair kinetic μ_k (Macklin 2020 §3.6)
 *   - lambdaN  atomic<i32> offset 16 (4 B)
 *   - lambdaT  atomic<i32> offset 20 (4 B)
 *   - (padding) offset 24..31 (8 B) — vec3 alignment requires next field at 16-byte boundary
 *   - normal   vec3<f32>   offset 32 (12 B + 4 B trailing pad to maintain 16 B struct alignment)
 *   - total stride: 48 B per record.
 *
 * Pre-21a stride was 32 B (no μ fields, normal at offset 16). The +16 B comes
 * from vec3's mandatory 16-byte alignment forcing a padding gap once the
 * scalar prefix grows past 16 bytes — adding the two μ fields without
 * crossing 16 bytes was not possible. At MVP-typical 32k contacts the cost is
 * ~512 KB additional, immaterial against the 256 MB Apple-Silicon
 * `maxBufferSize`. Future consolidation candidate: pack `(μ_s, μ_k)` into a
 * single u32 via `pack2x16float` and drop one of the slots — saves no stride
 * (still 16 B-bound) but tightens the read path.
 */
export const ContactRecord = struct(
  {
    i: 'uint',
    j: 'uint',
    muS: 'float',
    muK: 'float',
    lambdaN: { type: 'int', atomic: true },
    lambdaT: { type: 'int', atomic: true },
    normal: 'vec3',
  },
  'ContactRecord',
);

/** Float32-words per record (12 — see field-offset table on {@link ContactRecord}). */
const RECORD_STRIDE_WORDS = 12;

export interface ContactBufferOptions {
  /**
   * Hard cap on the number of contact pairs emitted per substep. When the
   * contact generator would exceed this cap the extra pairs are silently
   * dropped; the caller is expected to watch for overflow via
   * {@link ContactBuffer.readbackCount} or a post-kernel CPU check and log a
   * diagnostic (plan §Contact generation — "log a diagnostic … we prefer
   * loud failure to silent dropped contacts").
   *
   * A reasonable default for uniform-radius scenes is `8 × capacity`: a
   * particle in dense contact touches at most its six axis-aligned neighbors
   * plus a face/edge/corner handful, so 8 pairs per particle / 2 (each pair
   * counted once with i < j ordering) ≈ 4 pairs contributed — times capacity
   * gives headroom for worst-case packing.
   */
  readonly maxContacts: number;

  /**
   * Fixed-point scale (ticks per kg·m) for the accumulated Lagrange
   * multiplier fields λ_n and λ_t. Defaults to {@link DEFAULT_LAMBDA_SCALE}.
   * Callers generally should not override this.
   */
  readonly lambdaScale?: number;
}

/**
 * Contact pair storage for the unified Macklin-2014 §6 contact pipeline.
 *
 * **Phase 21a layout** — one storage buffer of {@link ContactRecord} structs,
 * indexed by contact slot `c ∈ [0, maxContacts)`. Pre-21a layout used three
 * separate storage buffers (`pairs`, `lambdaNT`, `normal`); see git history
 * for the migration. Field-access pattern in kernels:
 *
 * ```ts
 * const rec = contacts.records.element(c);
 * const i = rec.get('i').toVar();
 * atomicAdd(rec.get('lambdaN'), ticks);
 * rec.get('normal').assign(n);
 * ```
 *
 * `counter` remains a separate single-slot atomic u32 buffer; struct buffers
 * with one element silently no-op atomic writes on three.js r184 + Apple
 * Silicon Chrome. The counter does not need to live in the struct.
 *
 *
 * Paper reference: Macklin 2014 §6.1 non-penetration constraint eq. (22) —
 * `C(x_i, x_j) = |x_{ij}| - r ≥ 0`. This buffer owns the per-pair `(i, j)`
 * identities, the Macklin 2020 §3.5 / §3.6 accumulated Lagrange multipliers,
 * and the post-Phase-15 per-pair contact normal (so the velocity-friction
 * pass reads the same `n` the position solve used).
 */
export class ContactBuffer {
  readonly renderer: WebGPURenderer;
  readonly maxContacts: number;

  /**
   * Per-contact record buffer. Length `maxContacts`. See {@link ContactRecord}
   * for field layout.
   */
  readonly records: StorageBufferNode<'struct'>;
  /** Single-slot atomic counter. */
  readonly counter: StorageBufferNode<'uint'>;

  /** Fixed-point ticks per kg·m for λ_n and λ_t. CPU-side scalar. */
  readonly lambdaScale: number;

  /** Compute kernel — dispatch once per substep before `generate`. */
  readonly resetKernel: ComputeNode;

  /**
   * Zero every record's λ_n / λ_t slots. Dispatched once per substep
   * alongside {@link ContactBuffer.resetKernel}, before the iter loop
   * begins. One thread per record.
   */
  readonly resetLambdaKernel: ComputeNode;

  private disposed = false;

  constructor(renderer: WebGPURenderer, options: ContactBufferOptions) {
    if (!Number.isInteger(options.maxContacts) || options.maxContacts <= 0) {
      throw new Error(
        `ContactBuffer: maxContacts must be a positive integer, got ${options.maxContacts}`,
      );
    }
    const lambdaScale = options.lambdaScale ?? DEFAULT_LAMBDA_SCALE;
    if (!Number.isFinite(lambdaScale) || lambdaScale <= 0) {
      throw new Error(
        `ContactBuffer: lambdaScale must be a positive finite number, got ${lambdaScale}`,
      );
    }
    this.renderer = renderer;
    this.maxContacts = options.maxContacts;
    this.lambdaScale = lambdaScale;

    this.records = instancedArray(options.maxContacts, ContactRecord as Any) as Any;
    this.counter = instancedArray(1, 'uint').toAtomic();

    const counter = this.counter;
    this.resetKernel = Fn(() => {
      // instanceIndex is always 0 here — dispatch is size 1 — but we still
      // use atomicStore rather than `.assign()` because the backing binding
      // is `array<atomic<u32>>`. See `hashGrid/cellIndex.ts` resetCounts.
      atomicStore(counter.element(uint(0)), uint(0));
    })().compute(1);

    const records: Any = this.records;
    this.resetLambdaKernel = Fn(() => {
      // One thread per record zeroes both λ_n and λ_t atomic fields.
      const c: Any = instanceIndex;
      const rec: Any = records.element(c);
      atomicStore(rec.get('lambdaN'), int(0));
      atomicStore(rec.get('lambdaT'), int(0));
    })().compute(options.maxContacts);
  }

  /**
   * Read the current valid contact count back to the CPU. Debug/test only —
   * a single-element u32 round-trip stalls the render loop.
   */
  async readbackCount(): Promise<number> {
    this.assertAlive();
    const buf = await this.renderer.getArrayBufferAsync(this.counter.value);
    return new Uint32Array(buf)[0]!;
  }

  /**
   * Read the records buffer back as `{i, j}` ordered flat `Uint32Array` of
   * length `2 · nContacts` (only the leading `2 · nContacts` slots are
   * valid; trailing slots are stale from previous substeps). Pre-21a's
   * `pairs` was `u32[2N]`; this method preserves the same return shape so
   * existing readback callers don't change.
   */
  async readbackPairs(): Promise<{
    readonly nContacts: number;
    readonly pairs: Uint32Array;
  }> {
    this.assertAlive();
    const [countBuf, recordsBuf] = await Promise.all([
      this.renderer.getArrayBufferAsync(this.counter.value),
      this.renderer.getArrayBufferAsync(this.records.value),
    ]);
    const nRaw = new Uint32Array(countBuf)[0]!;
    const nContacts = Math.min(nRaw, this.maxContacts);
    const u32 = new Uint32Array(recordsBuf);
    const out = new Uint32Array(2 * nContacts);
    for (let c = 0; c < nContacts; c++) {
      const base = c * RECORD_STRIDE_WORDS;
      out[2 * c] = u32[base]!; // i
      out[2 * c + 1] = u32[base + 1]!; // j
    }
    return { nContacts, pairs: out };
  }

  /**
   * TSL helper: expose the atomic counter as a plain u32 for kernels that
   * need to load it (coloring, per-particle inverted-list build). Hides the
   * `atomicLoad` / `Any`-cast dance at the call site.
   */
  loadCount(): Any {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const c: any = atomicLoad(this.counter.element(uint(0)));
    return c;
  }

  destroy(): void {
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('ContactBuffer has been destroyed');
  }
}
