/**
 * Fixed-timestep accumulator for wall-clock frame pacing.
 *
 * Demo animate loops face two independent problems that look similar
 * at first:
 *
 *   1. **Physics stability.** The solver needs a small enough Δt to
 *      avoid stiff-force blow-up. `src/core`'s `SimLoop` handles
 *      this internally via its fixed `substeps × iterations` schedule
 *      (Macklin 2019 Small Steps: many small substeps beat fewer large
 *      ones). One `loop.step(Δt)` advances sim time by exactly `Δt`.
 *
 *   2. **Wall-clock pacing.** `requestAnimationFrame` fires at the
 *      display's refresh rate, but that rate is not guaranteed — a GC
 *      pause, a shader recompile, or a background tab swap can stall
 *      the tick to 30 ms (or longer). If we pass `dt = 1/60` to
 *      `loop.step()` regardless of real elapsed time, the simulation
 *      visibly slows down during stalls: sim-time-per-wall-second
 *      drops. If we pass real `dt` to `loop.step()`, physics stability
 *      is lost (each substep becomes `realDt / substeps` and can
 *      exceed the CFL envelope the scene was tuned for).
 *
 * The classical fix is the **fixed-timestep accumulator** (Gaffer on
 * Games, "Fix Your Timestep!", 2004). We always pass a *fixed* Δt to
 * `loop.step()` — the Δt the scene was tuned for — and consume
 * wall-clock time by running more or fewer `loop.step()` calls per
 * animate frame. Specifically:
 *
 *     accumulator += realDt
 *     while accumulator >= fixedDt AND substeps < cap:
 *         loop.step(fixedDt)
 *         accumulator -= fixedDt
 *
 * Under steady state (realDt ≈ fixedDt), this runs exactly one
 * `loop.step()` per animate frame. During a stall (realDt > fixedDt),
 * it catches up by running multiple steps in one animate tick. During
 * an overrun (realDt < fixedDt — should not happen with
 * requestAnimationFrame but can with a tight game loop), it skips a
 * step to avoid outpacing wall clock.
 *
 * **Spiral-of-death guard.** If the sim is persistently slower than
 * real-time (GPU saturated, too many particles for the budget), the
 * naive accumulator grows unbounded: each frame adds more time than
 * the sim can consume, next frame has even more backlog, and frame
 * time increases until the whole page freezes. We cap the accumulator
 * at `maxSubstepsPerFrame · fixedDt`. When the cap is hit, excess
 * sim-time is *discarded* — the simulation visibly slows down
 * relative to wall clock rather than spiraling. Diagnostic via the
 * `truncated` field of the pump() result.
 *
 *
 * Newton-3 / G4 properties carry through: each `loop.step(fixedDt)`
 * call is the same deterministic computation it was before, so the
 * tier-1 bit-exact per-particle kernels stay tier-1 bit-exact.
 * Running a different *count* of identical steps per frame does not
 * break that.
 */
export interface FrameStepperOptions {
  /**
   * Physics simulation timestep per `loop.step()` call, in seconds.
   * Typically `1/60`. Scene-tuned — the value the scene's substep
   * count / iteration count / XPBD compliance were chosen against.
   */
  readonly fixedDt: number;
  /**
   * Cap on substeps executed per animate frame. Defaults to 4 — a
   * conservative value that allows catching up from a ~4×-overrun
   * stall (67 ms browser hitch at 60 Hz target) without risking a
   * cascade where each frame accumulates more sim-time than it can
   * process. Raise for scenes with a generous per-step budget
   * (simple particle scenes: 8+), lower for heavy scenes where catch-up
   * itself is expensive (dam-break: 2–3).
   */
  readonly maxSubstepsPerFrame?: number;
}

export interface FrameStepperResult {
  /** Substeps executed this frame. `0 ≤ n ≤ maxSubstepsPerFrame`. */
  readonly substepsThisFrame: number;
  /**
   * True if the accumulator was truncated to prevent spiral of death
   * this frame. When true, the simulation's sim-time-per-wall-second
   * dropped below 1.0 — sim is falling behind wall clock.
   */
  readonly truncated: boolean;
  /**
   * Leftover wall-clock time that didn't add up to a full `fixedDt`,
   * in seconds. In `[0, fixedDt)`. Useful for render-side frame
   * interpolation (motion between the last-executed substep and the
   * next one), though MVP demos just render the post-substep state
   * directly.
   */
  readonly remainderSeconds: number;
}

export class FrameStepper {
  readonly fixedDt: number;
  readonly maxSubstepsPerFrame: number;
  private accumulator = 0;
  private lastMs: number | null = null;

  constructor(options: FrameStepperOptions) {
    if (!Number.isFinite(options.fixedDt) || options.fixedDt <= 0) {
      throw new Error(
        `FrameStepper: fixedDt must be a positive finite number, got ${options.fixedDt}`,
      );
    }
    const cap = options.maxSubstepsPerFrame ?? 4;
    if (!Number.isInteger(cap) || cap <= 0) {
      throw new Error(`FrameStepper: maxSubstepsPerFrame must be a positive integer, got ${cap}`);
    }
    this.fixedDt = options.fixedDt;
    this.maxSubstepsPerFrame = cap;
  }

  /**
   * Consume wall-clock time since the last `pump()` by running zero
   * or more `step(fixedDt)` calls. The callback MUST advance the sim
   * by exactly `fixedDt` per invocation — typically a direct
   * `loop.step(fixedDt)` wrap.
   *
   * First call establishes the zero point for `nowMs`; it runs no
   * substeps and returns `{ substepsThisFrame: 0, ... }`. Subsequent
   * calls consume the elapsed wall-clock time.
   *
   * @param nowMs Current wall-clock time in milliseconds. Typically
   *   `performance.now()`; the absolute zero does not matter, only
   *   deltas between successive calls.
   * @param step Advance-by-one-substep callback. Awaited per substep
   *   so GPU backpressure (e.g. the 2-frames-in-flight fence pattern
   *   from `examples/dam-break`) flows naturally through the stepper.
   */
  async pump(nowMs: number, step: (dt: number) => Promise<void>): Promise<FrameStepperResult> {
    if (this.lastMs === null) {
      this.lastMs = nowMs;
      return { substepsThisFrame: 0, truncated: false, remainderSeconds: 0 };
    }
    const realDt = Math.max(0, (nowMs - this.lastMs) / 1000);
    this.lastMs = nowMs;
    this.accumulator += realDt;

    // Spiral-of-death guard. If the accumulator has grown past the cap
    // (sim is falling behind wall-clock faster than we can catch up),
    // discard the excess so next frame doesn't keep piling up.
    const maxAccum = this.fixedDt * this.maxSubstepsPerFrame;
    let truncated = false;
    if (this.accumulator > maxAccum) {
      this.accumulator = maxAccum;
      truncated = true;
    }

    let substepsThisFrame = 0;
    while (this.accumulator >= this.fixedDt && substepsThisFrame < this.maxSubstepsPerFrame) {
      await step(this.fixedDt);
      this.accumulator -= this.fixedDt;
      substepsThisFrame += 1;
    }
    return {
      substepsThisFrame,
      truncated,
      remainderSeconds: this.accumulator,
    };
  }

  /**
   * Reset internal state. Call after a long pause (tab hidden, scene
   * reset, paused simulation) to prevent the next `pump()` call from
   * trying to "catch up" on the entire pause duration.
   */
  reset(): void {
    this.accumulator = 0;
    this.lastMs = null;
  }
}
