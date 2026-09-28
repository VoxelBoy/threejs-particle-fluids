// Per-kernel GPU timing. `SimLoop.step` submits a whole frame as one compute
// pass, so timestamps only give the frame's total. While a profiler is
// installed, every batched submission is split into one pass per kernel, and
// three.js records a timestamp pair for each, keyed by a uid that holds the
// kernel's node id. Splitting adds a little per-pass overhead and removes any
// overlap between neighboring dispatches, so the per-kernel sum runs a bit
// above the batched frame time; use it to rank kernels, and the batched time
// for totals.

import { TimestampQuery } from 'three/src/constants.js';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type { WebGPURenderer } from 'three/webgpu';

/** Time one kernel took per frame, summed over its dispatches. */
export interface KernelTiming {
  readonly name: string;
  /** Mean GPU milliseconds per frame. */
  readonly ms: number;
  /** Dispatches per frame. */
  readonly calls: number;
}

interface QueryPool {
  readonly timestamps: Map<string, number>;
  currentQueryIndex: number;
}

// The pool holds 2048 queries (1024 passes); resolve well before it fills.
const RESOLVE_EVERY = 400;

export class KernelProfiler {
  private readonly names = new Map<number, string>();
  private readonly totals = new Map<string, { ms: number; calls: number }>();
  private frames = 0;
  private restore: (() => void) | undefined;

  constructor(private readonly renderer: WebGPURenderer) {}

  /** Split batched submissions into timed single-kernel passes until {@link uninstall}. */
  install(): void {
    if (this.restore) return;
    const renderer = this.renderer as unknown as {
      computeAsync(nodes: ComputeNode | ComputeNode[], size?: unknown): Promise<void>;
      compute(nodes: ComputeNode | ComputeNode[], size?: unknown): void;
    };
    const original = renderer.computeAsync;
    let pending = 0;
    renderer.computeAsync = async (nodes, size) => {
      const list = Array.isArray(nodes) ? nodes : [nodes];
      for (const node of list) {
        this.names.set(node.id, node.name || `unnamed#${node.id}`);
        renderer.compute(node, size);
        if (++pending >= RESOLVE_EVERY) {
          pending = 0;
          await this.harvest();
        }
      }
    };
    this.restore = () => {
      renderer.computeAsync = original;
    };
  }

  uninstall(): void {
    this.restore?.();
    this.restore = undefined;
  }

  /** Start counting from zero, dropping any queries still pending. */
  async reset(): Promise<void> {
    await this.harvest();
    this.totals.clear();
    this.frames = 0;
  }

  /** Collect the queries of one finished frame. */
  async endFrame(): Promise<void> {
    await this.harvest();
    this.frames++;
  }

  /** Kernels sorted by time per frame, slowest first. */
  results(): KernelTiming[] {
    const frames = Math.max(this.frames, 1);
    return [...this.totals]
      .map(([name, t]) => ({ name, ms: t.ms / frames, calls: t.calls / frames }))
      .sort((a, b) => b.ms - a.ms);
  }

  private get pool(): QueryPool {
    const backend = this.renderer.backend as unknown as {
      timestampQueryPool: Record<string, QueryPool | undefined>;
    };
    const pool = backend.timestampQueryPool[TimestampQuery.COMPUTE];
    if (!pool) throw new Error('KernelProfiler: the renderer is not tracking timestamps');
    return pool;
  }

  private async harvest(): Promise<void> {
    await this.renderer.resolveTimestampsAsync(TimestampQuery.COMPUTE);
    const pool = this.pool;
    // uid is `<prefix>:<call>:<node id>:f<frame>`.
    for (const [uid, ms] of pool.timestamps) {
      const id = Number(uid.split(':')[2]);
      const name = this.names.get(id) ?? `unknown#${id}`;
      const entry = this.totals.get(name) ?? { ms: 0, calls: 0 };
      entry.ms += ms;
      entry.calls += 1;
      this.totals.set(name, entry);
    }
    pool.timestamps.clear();
  }
}
