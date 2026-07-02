/**
 * KernelWorkerPool — manages a fixed set of Workers initialised with the
 * simd-threads WASM binary over a shared WebAssembly.Memory (ADR-006 / P5.1).
 *
 * Invariants:
 *  - Workers NEVER call alloc/free/realloc (the arena is main-thread-only).
 *  - Each worker holds its own WASM instance but they ALL share the same
 *    linear memory, so pointers from the main-thread arena are valid in workers.
 *  - Work items are dispatched round-robin; one pending item per worker at a time.
 *
 * Crash / timeout policy:
 *  - If a worker doesn't respond within `timeoutMs` (default 30 s) we replace
 *    it with a fresh instance and reject the pending promise.  This ensures the
 *    pool never deadlocks after a worker crash.
 */

import { KERNEL_WORKER_SCRIPT } from './kernel-worker-script.js';

const isNode =
  typeof process !== 'undefined' &&
  typeof process.versions === 'object' &&
  typeof process.versions.node === 'string';

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

/** An in-flight request waiting for a worker reply. */
interface Pending {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  timer: ReturnType<typeof setTimeout> | null;
}

type WorkerLike = {
  postMessage(msg: unknown): void;
  on?(event: 'message', cb: (msg: unknown) => void): void;
  onmessage?: ((e: MessageEvent) => void) | null;
  terminate(): void;
};

/* ------------------------------------------------------------------ */
/* Worker creation                                                       */
/* ------------------------------------------------------------------ */

async function createRawWorker(): Promise<WorkerLike> {
  if (isNode) {
    const { Worker } = await import('node:worker_threads');
    return new Worker(KERNEL_WORKER_SCRIPT, { eval: true }) as unknown as WorkerLike;
  }
  /* Browser: wrap script in a Blob URL */
  const blob = new Blob([KERNEL_WORKER_SCRIPT], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  const w = new Worker(url);
  URL.revokeObjectURL(url);
  return w as unknown as WorkerLike;
}

function wireWorkerMessages(
  raw: WorkerLike,
  onMsg: (msg: unknown) => void,
): void {
  if (isNode && typeof raw.on === 'function') {
    raw.on('message', onMsg);
  } else {
    (raw as unknown as Worker).onmessage = (e: MessageEvent) => onMsg(e.data);
  }
}

/* ------------------------------------------------------------------ */
/* Pool                                                                  */
/* ------------------------------------------------------------------ */

/** Single slot in the worker pool. */
interface PoolSlot {
  raw: WorkerLike;
  pending: Map<number, Pending>;
}

export class KernelWorkerPool {
  private readonly memory: WebAssembly.Memory;
  private readonly wasmBytes: Uint8Array;
  private readonly timeoutMs: number;
  private readonly slots: PoolSlot[] = [];
  private nextId = 1;
  private nextSlot = 0;
  private terminated = false;

  /** Only {@link KernelWorkerPool.create} should be used (async init). */
  private constructor(
    memory: WebAssembly.Memory,
    wasmBytes: Uint8Array,
    timeoutMs: number,
  ) {
    this.memory = memory;
    this.wasmBytes = wasmBytes;
    this.timeoutMs = timeoutMs;
  }

  /**
   * Create and fully initialise a pool of `numWorkers` workers.
   * Each worker receives the threads WASM binary and the shared memory.
   */
  static async create(
    wasmBytes: Uint8Array,
    memory: WebAssembly.Memory,
    numWorkers: number,
    timeoutMs = 30_000,
  ): Promise<KernelWorkerPool> {
    const pool = new KernelWorkerPool(memory, wasmBytes, timeoutMs);
    const readyPromises: Promise<void>[] = [];
    for (let i = 0; i < numWorkers; i++) {
      readyPromises.push(pool.addWorker());
    }
    await Promise.all(readyPromises);
    return pool;
  }

  /** Add one worker slot and wait for it to signal 'ready'. */
  private async addWorker(): Promise<void> {
    const raw = await createRawWorker();
    const pending = new Map<number, Pending>();
    const slot: PoolSlot = { raw, pending };
    this.slots.push(slot);

    wireWorkerMessages(raw, (msg) => this.handleMessage(slot, msg));

    return new Promise<void>((resolve, reject) => {
      const id = this.nextId++;
      pending.set(id, {
        resolve: () => resolve(),
        reject,
        timer: null,
      });
      raw.postMessage({
        type: 'init',
        requestId: id,
        bytes: this.wasmBytes,
        memory: this.memory,
      });
    });
  }

  private handleMessage(slot: PoolSlot, msg: unknown): void {
    if (typeof msg !== 'object' || msg === null) return;
    const m = msg as Record<string, unknown>;
    const id = m['requestId'] as number;
    const entry = slot.pending.get(id);
    if (!entry) return;
    slot.pending.delete(id);
    if (entry.timer !== null) clearTimeout(entry.timer);

    if (m['type'] === 'error') {
      entry.reject(new Error(String(m['error'])));
    } else {
      entry.resolve(msg);
    }
  }

  /**
   * Send a 'kernel' message to the next available worker (round-robin) and
   * return the numeric result. Times out after `this.timeoutMs`.
   */
  async sendKernel(fn: string, args: number[]): Promise<number> {
    const slot = this.slots[this.nextSlot % this.slots.length]!;
    this.nextSlot++;
    const id = this.nextId++;
    const result = await this.sendToSlot<{ value: number }>(slot, id, {
      type: 'kernel',
      requestId: id,
      fn,
      args,
    });
    return result.value;
  }

  /**
   * Send a 'meta' (batched) message to the next worker (round-robin) and
   * return an array of numeric results (one per op in `ops`).
   */
  async sendMeta(ops: Array<{ fn: string; args: number[] }>): Promise<number[]> {
    const slot = this.slots[this.nextSlot % this.slots.length]!;
    this.nextSlot++;
    const id = this.nextId++;
    const result = await this.sendToSlot<{ values: number[] }>(slot, id, {
      type: 'meta',
      requestId: id,
      ops,
    });
    return result.values;
  }

  /**
   * Dispatch work to ALL slots (one message per slot) and await all results.
   * Used for parallel chunk dispatch.
   */
  async broadcastKernel(
    perSlot: Array<{ fn: string; args: number[] }>,
  ): Promise<number[]> {
    const count = Math.min(perSlot.length, this.slots.length);
    const promises: Promise<number>[] = [];
    for (let i = 0; i < count; i++) {
      const { fn, args } = perSlot[i]!;
      const slot = this.slots[i]!;
      const id = this.nextId++;
      promises.push(
        this.sendToSlot<{ value: number }>(slot, id, {
          type: 'kernel',
          requestId: id,
          fn,
          args,
        }).then((r) => r.value),
      );
    }
    return Promise.all(promises);
  }

  /**
   * Dispatch meta (batched) calls to ALL slots, one message per slot.
   * Returns an array of arrays: `result[i][j]` = value j from slot i.
   */
  async broadcastMeta(
    perSlot: Array<Array<{ fn: string; args: number[] }>>,
  ): Promise<number[][]> {
    const count = Math.min(perSlot.length, this.slots.length);
    const promises: Promise<number[]>[] = [];
    for (let i = 0; i < count; i++) {
      const ops = perSlot[i]!;
      const slot = this.slots[i]!;
      const id = this.nextId++;
      promises.push(
        this.sendToSlot<{ values: number[] }>(slot, id, {
          type: 'meta',
          requestId: id,
          ops,
        }).then((r) => r.values),
      );
    }
    return Promise.all(promises);
  }

  private sendToSlot<T>(
    slot: PoolSlot,
    id: number,
    msg: unknown,
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | null = null;
      if (this.timeoutMs > 0) {
        timer = setTimeout(() => {
          slot.pending.delete(id);
          /* Replace crashed/hung worker */
          this.replaceSlot(slot).catch(() => {/* ignore */});
          reject(new Error(`kernel worker timed out (${this.timeoutMs} ms)`));
        }, this.timeoutMs);
      }
      slot.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        timer,
      });
      slot.raw.postMessage(msg);
    });
  }

  /** Replace a crashed slot with a fresh worker. */
  private async replaceSlot(slot: PoolSlot): Promise<void> {
    try { slot.raw.terminate(); } catch { /* ignore */ }
    const idx = this.slots.indexOf(slot);
    if (idx === -1) return;
    /* Reject any remaining pending requests for this slot */
    for (const [, p] of slot.pending) {
      if (p.timer !== null) clearTimeout(p.timer);
      p.reject(new Error('worker replaced after crash/timeout'));
    }
    slot.pending.clear();
    const raw = await createRawWorker();
    const newSlot: PoolSlot = { raw, pending: new Map() };
    wireWorkerMessages(raw, (msg) => this.handleMessage(newSlot, msg));
    this.slots[idx] = newSlot;
    /* Re-initialise */
    const id = this.nextId++;
    await new Promise<void>((resolve, reject) => {
      newSlot.pending.set(id, { resolve: () => resolve(), reject, timer: null });
      raw.postMessage({
        type: 'init',
        requestId: id,
        bytes: this.wasmBytes,
        memory: this.memory,
      });
    });
  }

  /** Number of active worker slots. */
  get size(): number {
    return this.slots.length;
  }

  /** Terminate all workers. Pool must not be used after this. */
  terminate(): void {
    if (this.terminated) return;
    this.terminated = true;
    for (const slot of this.slots) {
      for (const [, p] of slot.pending) {
        if (p.timer !== null) clearTimeout(p.timer);
        p.reject(new Error('pool terminated'));
      }
      slot.pending.clear();
      try { slot.raw.terminate(); } catch { /* ignore */ }
    }
    this.slots.length = 0;
  }
}
