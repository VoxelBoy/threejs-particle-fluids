import { Vector2 } from 'three';
import { uniformArray } from 'three/tsl';

// TSL @types surface uniformArray as a bare Node, stripping the
// proxy-provided method chains. Same loose-alias pattern used elsewhere
// in `src/core`.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/**
 * Default capacity for the per-friction-group lookup table. The friction
 * group is the LOW 16 bits of `ParticleSystem.phase` — decoupled from the
 * self-collision group (HIGH 16 bits, used by the contact-emit phase mask)
 * so a scene can have intra-pile contacts emit (self-collision = 0) AND
 * still differentiate friction across particles via the low bits. In
 * practice MVP scenes use small integers, so a 16-slot LUT comfortably
 * covers every documented use-case. Raise via
 * {@link FrictionTableOptions.maxGroups} if a scene needs more — uniform-
 * array memory cost is `8 B · maxGroups`.
 */
export const DEFAULT_FRICTION_TABLE_GROUPS = 16;

export interface FrictionTableOptions {
  /**
   * Number of group slots in the LUT. Indexed by the LOW 16 bits of
   * `ParticleSystem.phase` (the friction group; decoupled from the high-
   * 16-bit self-collision group). Defaults to
   * {@link DEFAULT_FRICTION_TABLE_GROUPS} (= 16).
   *
   * Group `0` is a valid index; scenes that don't differentiate by
   * friction group (everything with low-bits = 0) leave only slot 0 in
   * use, with the table's defaults still applied. Raising this past the
   * largest group index any scene uses is harmless — extra slots cost
   * 8 B each in the uniform buffer.
   */
  readonly maxGroups?: number;
  /** Default static-friction coefficient applied to every group at construction. */
  readonly defaultMuS: number;
  /** Default kinetic-friction coefficient applied to every group at construction. */
  readonly defaultMuK: number;
}

/**
 *
 *
 * **Phase encoding convention.** The friction group lives in the LOW 16
 * bits of `ParticleSystem.phase`. The HIGH 16 bits hold the independent
 * self-collision group consumed by the phase-mask in `generate.ts`'s
 * contact-emit walk. Decoupled because the two concepts are orthogonal:
 *   - Self-collision suppression: "should pairs within this group emit at
 *     all?" Used for fluid (PBF handles density), softbody (shape match
 *     handles rigidity), etc. Pile scenes set high-bits = 0 so intra-pile
 *     contacts emit normally.
 *   - Friction lookup: "what (μ_s, μ_k) does this particle contribute to
 *     a pair?" Used for any scene that wants surface-property
 *     differentiation independent of self-collision suppression.
 *
 * Scenes that need both — e.g. softbody-swimming where each body needs
 * intra-body suppression AND its own friction — encode the same group ID
 * into both halves: `phase = (group << 16) | group`.
 *
 *
 * Authoring API:
 *   - {@link setGroupFriction} writes a single group's `(μ_s, μ_k)` —
 *     the production path for per-material scenes (e.g. fluid + body
 *     where each body group needs distinct friction).
 *   - {@link setMuS} / {@link setMuK} apply a single coefficient to
 *     every group — back-compat for scenes that author through a single
 *     scalar slider (legacy `SimLoop.friction.muS.value = v` shape;
 *     three demo specs migrated from that pre-21).
 *
 */
export class FrictionTable {
  readonly maxGroups: number;
  /**
   * `Vector2[]` backing the uniform-array. Each entry: `.x = μ_s, .y = μ_k`.
   * Mutated in place by the authoring methods; `uniformArray` reads the
   * latest values per frame.
   */
  private readonly entries: Vector2[];
  /**
   * Uniform-array node bound by the contact-emission kernel. Type
   * deliberately broad — TSL's @types surface uniformArray as a loose
   * `Node` and the consumer site uses `.element(idx).x/.y`.
   */
  readonly lut: Any;

  constructor(options: FrictionTableOptions) {
    const maxGroups = options.maxGroups ?? DEFAULT_FRICTION_TABLE_GROUPS;
    if (!Number.isInteger(maxGroups) || maxGroups <= 0) {
      throw new Error(`FrictionTable: maxGroups must be a positive integer, got ${maxGroups}`);
    }
    if (
      !Number.isFinite(options.defaultMuS) ||
      options.defaultMuS < 0 ||
      !Number.isFinite(options.defaultMuK) ||
      options.defaultMuK < 0
    ) {
      throw new Error(
        `FrictionTable: defaultMuS and defaultMuK must be non-negative finite numbers, got muS=${options.defaultMuS} muK=${options.defaultMuK}`,
      );
    }
    this.maxGroups = maxGroups;
    this.entries = new Array(maxGroups);
    for (let g = 0; g < maxGroups; g++) {
      this.entries[g] = new Vector2(options.defaultMuS, options.defaultMuK);
    }
    this.lut = uniformArray(this.entries, 'vec2');
  }

  /**
   * Set `(μ_s, μ_k)` for a specific phase group. The `group` argument is
   * the value extracted from `phase >> 16` at contact emit time — i.e. the
   * 16-bit self-collision group. Throws if `group` is outside
   * `[0, maxGroups)`. Both coefficients must be non-negative finite
   * numbers.
   */
  setGroupFriction(group: number, muS: number, muK: number): void {
    if (!Number.isInteger(group) || group < 0 || group >= this.maxGroups) {
      throw new Error(
        `FrictionTable.setGroupFriction: group must be in [0, ${this.maxGroups}), got ${group}`,
      );
    }
    if (!Number.isFinite(muS) || muS < 0 || !Number.isFinite(muK) || muK < 0) {
      throw new Error(
        `FrictionTable.setGroupFriction: muS and muK must be non-negative finite numbers, got muS=${muS} muK=${muK}`,
      );
    }
    this.entries[group]!.set(muS, muK);
  }

  /** Apply `μ_s` to every group. Use for scenes authored through a single scalar slider. */
  setMuS(muS: number): void {
    if (!Number.isFinite(muS) || muS < 0) {
      throw new Error(`FrictionTable.setMuS: muS must be non-negative finite, got ${muS}`);
    }
    for (const e of this.entries) e.x = muS;
  }

  /** Apply `μ_k` to every group. Use for scenes authored through a single scalar slider. */
  setMuK(muK: number): void {
    if (!Number.isFinite(muK) || muK < 0) {
      throw new Error(`FrictionTable.setMuK: muK must be non-negative finite, got ${muK}`);
    }
    for (const e of this.entries) e.y = muK;
  }

  /** Read `(μ_s, μ_k)` for a group — primarily for tests / debug. */
  getGroupFriction(group: number): { readonly muS: number; readonly muK: number } {
    if (!Number.isInteger(group) || group < 0 || group >= this.maxGroups) {
      throw new Error(
        `FrictionTable.getGroupFriction: group must be in [0, ${this.maxGroups}), got ${group}`,
      );
    }
    const e = this.entries[group]!;
    return { muS: e.x, muK: e.y };
  }
}
