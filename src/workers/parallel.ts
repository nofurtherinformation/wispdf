/**
 * enableThreads — opt-in shared-memory parallel mode (ADR-006 / P5.1).
 *
 * Detection:
 *  - Browser: requires `crossOriginIsolated === true` (COOP+COEP headers).
 *  - Node.js: `SharedArrayBuffer` is always available (no isolation needed).
 *  If isolation is absent → console.warn + returns `false`.
 *
 * When enabled:
 *  - Loads `simd-threads.wasm` with a SAB-backed `WebAssembly.Memory`.
 *  - Calls `__wasm_init_memory()` (if exported) on the main instance to apply
 *    any passive data segments.  Workers instantiate the same binary with the
 *    same shared memory but skip `__wasm_init_memory` — the arena statics are
 *    already zero-initialised by wasm's memory-zero guarantee, so re-applying
 *    the data section from workers is neither needed nor safe.
 *  - Spawns a pool of `workers` (default 4) kernel workers.
 *  - Provides parallel chunked dispatch for elementwise and reduction kernels.
 *
 * Allocator invariant (ADR-006):
 *  Workers NEVER call alloc/free/realloc.  All memory allocation is done by
 *  the main thread before dispatching work.  This means:
 *   • Workers only read from data buffers (reductions) or write to pre-allocated
 *     output buffers (elementwise).
 *   • There are no data races on the arena state (HEAP_TOP / FREE_HEAD / GENERATION).
 *
 * Bit-exactness note:
 *  f64 sum/mean with parallel dispatch is NOT bit-identical to single-thread.
 *  Single-thread uses 2-stripe accumulation over all elements; parallel dispatch
 *  computes per-chunk partial sums (also 2-striped within each chunk) and then
 *  combines them left-to-right.  The combination order differs, producing
 *  different floating-point rounding.  Integer sums are always exact (no FP).
 *  This is an ADR-006-scope deviation; results are still IEEE-754 correct and
 *  deterministic for a fixed worker count.  See docs/threads.md for details.
 *
 * Chunk boundary alignment:
 *  All chunk boundaries are multiples of 8 elements so that validity-bitmap byte
 *  offsets are integer-aligned.  The sub-bitmap pointer for chunk starting at
 *  element `s` (a multiple of 8) is `vpPtr + s/8`, which correctly addresses
 *  the first byte of that chunk's portion of the Arrow-LSB bitmap.
 */

import { KernelWorkerPool } from './pool.js';

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

/** Configuration for {@link enableThreads}. */
export interface ThreadsConfig {
  /** Number of kernel workers (default 4). */
  workers?: number;
  /**
   * Directory containing `simd-threads.wasm`.
   * Node: filesystem path.  Browser: base URL.
   * Default: resolved relative to the loader module (same as scalar/simd.wasm).
   */
  wasmDir?: string | URL;
  /** Worker request timeout in ms (default 30 000).  0 = no timeout. */
  timeoutMs?: number;
  /** Initial shared memory pages (64 KiB each; default 16 = 1 MiB). */
  initialPages?: number;
  /** Maximum shared memory pages (default 16384 = 1 GiB). Must match --max-memory in build.sh. */
  maxPages?: number;
}

/** Handle returned by a successful {@link enableThreads} call. */
export interface ThreadsHandle {
  /** Always `true` — threads are active. */
  readonly enabled: true;

  /** Number of active worker slots. */
  readonly workers: number;

  /**
   * The SAB-backed shared WebAssembly.Memory used by all workers.
   * Column data for parallel operations must reside in this memory.
   * Use {@link alloc} to allocate buffers here.
   */
  readonly memory: WebAssembly.Memory;

  /**
   * Allocate `bytes` in the shared memory arena (main-thread-only).
   * Returns a 16-byte-aligned pointer, or throws on OOM.
   * Workers NEVER call alloc — only the main thread does (ADR-006 invariant).
   */
  alloc(bytes: number): number;

  /** Free a pointer previously returned by {@link alloc}. */
  free(ptr: number): void;

  /**
   * Call a kernel function synchronously on the main thread (single-thread path).
   * Useful for correctness comparison: `callKernel('sum_f64_null', ptr, 0, len)`.
   */
  callKernel(fn: string, ...args: number[]): number;

  /**
   * Parallel f64 sum over elements [0, len).  Null lanes (vpPtr ≠ 0) are skipped.
   * Returns the sum of non-null values, or 0 if all null.
   * NOTE: result may differ from single-thread due to FP accumulation order; see
   * docs/threads.md for the formal deviation note.
   */
  sumF64(dataPtr: number, vpPtr: number, len: number): Promise<number>;

  /**
   * Parallel f64 mean.  Returns NaN if len == 0 or all null.
   * Computed as totalSum / totalCount (exact weights via count_null per chunk).
   */
  meanF64(dataPtr: number, vpPtr: number, len: number): Promise<number>;

  /**
   * Parallel f64 min (NaN-aware, null-skipping).
   * Returns NaN if no valid non-NaN elements exist.
   */
  minF64(dataPtr: number, vpPtr: number, len: number): Promise<number>;

  /**
   * Parallel f64 max (NaN-aware, null-skipping).
   * Returns NaN if no valid non-NaN elements exist.
   */
  maxF64(dataPtr: number, vpPtr: number, len: number): Promise<number>;

  /**
   * Parallel elementwise add_f64 over pre-allocated buffers in shared memory.
   * Workers write directly to the shared output buffer (zero-copy).
   */
  addF64(aPtr: number, bPtr: number, outPtr: number, len: number): Promise<void>;

  /** Parallel elementwise sub_f64 (shared memory). */
  subF64(aPtr: number, bPtr: number, outPtr: number, len: number): Promise<void>;

  /** Parallel elementwise mul_f64 (shared memory). */
  mulF64(aPtr: number, bPtr: number, outPtr: number, len: number): Promise<void>;

  /**
   * Run a generic parallel elementwise binary kernel by name.
   * The kernel must have signature (i32 a, i32 b, i32 out, i32 len) -> ().
   */
  parallelElementwiseBinary(
    fn: string,
    aPtr: number,
    bPtr: number,
    outPtr: number,
    len: number,
    elemBytes: number,
  ): Promise<void>;

  /**
   * Run a generic parallel reduction kernel by name.
   * The kernel must have signature (i32 data, i32 vp, i32 len) -> <scalar>.
   * Returns an array of per-chunk partial results (caller combines).
   */
  parallelReduce(
    fn: string,
    dataPtr: number,
    vpPtr: number,
    len: number,
    elemBytes: number,
  ): Promise<number[]>;

  /** Terminate all workers. The handle must not be used after this. */
  terminate(): void;
}

/* ------------------------------------------------------------------ */
/* Detection helpers                                                    */
/* ------------------------------------------------------------------ */

const isNode_ =
  typeof process !== 'undefined' &&
  typeof process.versions === 'object' &&
  typeof process.versions.node === 'string';

/**
 * Returns true if SharedArrayBuffer is available and cross-origin isolation is
 * satisfied (required for browser; not required for Node.js).
 */
function isIsolated(): boolean {
  if (typeof SharedArrayBuffer === 'undefined') return false;
  if (isNode_) return true; /* Node.js never needs COOP/COEP */
  /* Browser: crossOriginIsolated must be true */
  return typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;
}

/* ------------------------------------------------------------------ */
/* Wasm loader for the threads variant                                  */
/* ------------------------------------------------------------------ */

/** Load a wasm file and return a Uint8Array backed by a plain ArrayBuffer. */
async function loadBytes(
  fileName: string,
  wasmDir: string | URL | undefined,
): Promise<Uint8Array> {
  if (isNode_) {
    const [{ readFile }, { fileURLToPath }] = await Promise.all([
      import('node:fs/promises'),
      import('node:url'),
    ]);
    let filePath: string;
    if (wasmDir === undefined) {
      filePath = fileURLToPath(new URL(fileName, import.meta.url));
    } else if (wasmDir instanceof URL) {
      filePath = fileURLToPath(new URL(fileName, wasmDir));
    } else if (wasmDir.startsWith('file:')) {
      filePath = fileURLToPath(new URL(fileName, wasmDir));
    } else {
      const { join } = await import('node:path');
      filePath = join(wasmDir, fileName);
    }
    const raw = await readFile(filePath);
    /* Node.js Buffer may use a pooled underlying ArrayBuffer; copy to own ArrayBuffer */
    const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
    return new Uint8Array(ab);
  }
  const base = wasmDir ?? new URL('.', import.meta.url);
  const url = new URL(fileName, base);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
}

/* ------------------------------------------------------------------ */
/* Chunk splitting                                                       */
/* ------------------------------------------------------------------ */

/**
 * Split [0, len) into at most `numWorkers` chunks, each starting at a
 * multiple of 8 elements (for Arrow-LSB validity-bitmap byte alignment).
 * Returns an array of [start, end) pairs.
 */
export function splitChunks(
  len: number,
  numWorkers: number,
): Array<[number, number]> {
  if (len === 0 || numWorkers <= 0) return [];
  /* Raw chunk size, then round UP to next multiple of 8 */
  const raw = Math.ceil(len / numWorkers);
  const cs = Math.ceil(raw / 8) * 8;
  const chunks: Array<[number, number]> = [];
  for (let s = 0; s < len; s += cs) {
    chunks.push([s, Math.min(s + cs, len)]);
  }
  return chunks;
}

/* ------------------------------------------------------------------ */
/* ThreadsHandle implementation                                          */
/* ------------------------------------------------------------------ */

/**
 * Raw wasm exports from the main thread's simd-threads.wasm instance.
 * Note: simd-threads.wasm uses --import-memory, so memory is NOT in exports.
 */
type ThreadsExports = {
  alloc: unknown;
  free: unknown;
  [fn: string]: unknown;
};

class ThreadsHandleImpl implements ThreadsHandle {
  readonly enabled = true as const;
  /** The SAB-backed shared memory — stored separately because simd-threads.wasm
   * IMPORTS its memory (--import-memory), so it does NOT appear in exports. */
  readonly memory: WebAssembly.Memory;
  private readonly exports: ThreadsExports;

  constructor(
    private readonly pool: KernelWorkerPool,
    mainInstance: WebAssembly.Instance,
    sharedMemory: WebAssembly.Memory,
  ) {
    this.exports = mainInstance.exports as unknown as ThreadsExports;
    this.memory = sharedMemory;
  }

  get workers(): number {
    return this.pool.size;
  }

  alloc(bytes: number): number {
    const ptr = (this.exports.alloc as (b: number) => number)(bytes);
    if (ptr === 0) throw new Error(`threads wasm OOM allocating ${bytes} bytes`);
    return ptr;
  }

  free(ptr: number): void {
    (this.exports.free as (p: number) => void)(ptr);
  }

  callKernel(fn: string, ...args: number[]): number {
    const f = this.exports[fn];
    if (typeof f !== 'function') throw new Error(`kernel not found: ${fn}`);
    return (f as (...a: number[]) => number)(...args);
  }

  /* ---------------------------------------------------------------- */
  /* f64 reductions                                                     */
  /* ---------------------------------------------------------------- */

  async sumF64(dataPtr: number, vpPtr: number, len: number): Promise<number> {
    const chunks = splitChunks(len, this.pool.size);
    if (chunks.length === 0) return 0;
    if (chunks.length === 1) {
      /* Single chunk — skip worker overhead */
      return this.pool.sendKernel('sum_f64_null', [
        dataPtr, vpPtr, len,
      ]);
    }
    const perSlot = chunks.map(([s, e]) => ({
      fn: 'sum_f64_null',
      args: [
        dataPtr + s * 8,
        vpPtr === 0 ? 0 : vpPtr + (s >> 3),
        e - s,
      ],
    }));
    const partials = await this.pool.broadcastKernel(perSlot);
    /* Combine left-to-right (deterministic for a given worker count).
     * This is NOT bit-identical to single-thread 2-stripe; see docs/threads.md. */
    let total = 0;
    for (const p of partials) total += p;
    return total;
  }

  async meanF64(dataPtr: number, vpPtr: number, len: number): Promise<number> {
    const chunks = splitChunks(len, this.pool.size);
    if (chunks.length === 0) return NaN;
    if (chunks.length === 1) {
      return this.pool.sendKernel('mean_f64_null', [dataPtr, vpPtr, len]);
    }
    /* Each worker returns [partial_sum, partial_count] */
    const perSlot = chunks.map(([s, e]) => {
      const adjData = dataPtr + s * 8;
      const adjVp = vpPtr === 0 ? 0 : vpPtr + (s >> 3);
      const clen = e - s;
      return [
        { fn: 'sum_f64_null',  args: [adjData, adjVp, clen] },
        { fn: 'count_null',    args: [adjVp,        clen] },
      ];
    });
    const results = await this.pool.broadcastMeta(perSlot);
    let totalSum = 0;
    let totalCount = 0;
    for (const [sum, count] of results) {
      totalSum += sum!;
      totalCount += count!;
    }
    return totalCount === 0 ? NaN : totalSum / totalCount;
  }

  async minF64(dataPtr: number, vpPtr: number, len: number): Promise<number> {
    const chunks = splitChunks(len, this.pool.size);
    if (chunks.length === 0) return NaN;
    if (chunks.length === 1) {
      return this.pool.sendKernel('min_f64_null', [dataPtr, vpPtr, len]);
    }
    const perSlot = chunks.map(([s, e]) => ({
      fn: 'min_f64_null',
      args: [dataPtr + s * 8, vpPtr === 0 ? 0 : vpPtr + (s >> 3), e - s],
    }));
    const partials = await this.pool.broadcastKernel(perSlot);
    return combineMinF64(partials);
  }

  async maxF64(dataPtr: number, vpPtr: number, len: number): Promise<number> {
    const chunks = splitChunks(len, this.pool.size);
    if (chunks.length === 0) return NaN;
    if (chunks.length === 1) {
      return this.pool.sendKernel('max_f64_null', [dataPtr, vpPtr, len]);
    }
    const perSlot = chunks.map(([s, e]) => ({
      fn: 'max_f64_null',
      args: [dataPtr + s * 8, vpPtr === 0 ? 0 : vpPtr + (s >> 3), e - s],
    }));
    const partials = await this.pool.broadcastKernel(perSlot);
    return combineMaxF64(partials);
  }

  /* ---------------------------------------------------------------- */
  /* elementwise binary ops                                             */
  /* ---------------------------------------------------------------- */

  addF64(aPtr: number, bPtr: number, outPtr: number, len: number): Promise<void> {
    return this.parallelElementwiseBinary('add_f64', aPtr, bPtr, outPtr, len, 8);
  }

  subF64(aPtr: number, bPtr: number, outPtr: number, len: number): Promise<void> {
    return this.parallelElementwiseBinary('sub_f64', aPtr, bPtr, outPtr, len, 8);
  }

  mulF64(aPtr: number, bPtr: number, outPtr: number, len: number): Promise<void> {
    return this.parallelElementwiseBinary('mul_f64', aPtr, bPtr, outPtr, len, 8);
  }

  async parallelElementwiseBinary(
    fn: string,
    aPtr: number,
    bPtr: number,
    outPtr: number,
    len: number,
    elemBytes: number,
  ): Promise<void> {
    const chunks = splitChunks(len, this.pool.size);
    if (chunks.length === 0) return;
    if (chunks.length === 1) {
      await this.pool.sendKernel(fn, [aPtr, bPtr, outPtr, len]);
      return;
    }
    const perSlot = chunks.map(([s, e]) => {
      const off = s * elemBytes;
      return { fn, args: [aPtr + off, bPtr + off, outPtr + off, e - s] };
    });
    /* Workers write directly to shared output buffer — no combination needed */
    await this.pool.broadcastKernel(perSlot);
  }

  async parallelReduce(
    fn: string,
    dataPtr: number,
    vpPtr: number,
    len: number,
    elemBytes: number,
  ): Promise<number[]> {
    const chunks = splitChunks(len, this.pool.size);
    if (chunks.length === 0) return [];
    const perSlot = chunks.map(([s, e]) => ({
      fn,
      args: [
        dataPtr + s * elemBytes,
        vpPtr === 0 ? 0 : vpPtr + (s >> 3),
        e - s,
      ],
    }));
    return this.pool.broadcastKernel(perSlot);
  }

  terminate(): void {
    this.pool.terminate();
  }
}

/* ------------------------------------------------------------------ */
/* Combination helpers                                                   */
/* ------------------------------------------------------------------ */

/**
 * Combine partial f64 mins.  Kernels return NaN for all-null/all-NaN chunks;
 * those are skipped.  If all partials are NaN, returns NaN.
 */
function combineMinF64(partials: number[]): number {
  let result = NaN;
  for (const p of partials) {
    if (!Number.isNaN(p)) {
      result = Number.isNaN(result) ? p : Math.min(result, p);
    }
  }
  return result;
}

/** Combine partial f64 maxes (symmetric to combineMinF64). */
function combineMaxF64(partials: number[]): number {
  let result = NaN;
  for (const p of partials) {
    if (!Number.isNaN(p)) {
      result = Number.isNaN(result) ? p : Math.max(result, p);
    }
  }
  return result;
}

/* ------------------------------------------------------------------ */
/* enableThreads                                                         */
/* ------------------------------------------------------------------ */

/**
 * Opt-in parallel mode (ADR-006).
 *
 * Returns a {@link ThreadsHandle} when threads are available and enabled,
 * or `false` (with a console.warn) when cross-origin isolation is absent.
 *
 * @example
 * ```ts
 * const th = await enableThreads({ workers: 4 });
 * if (th) {
 *   const sum = await th.sumF64(ptr, vp, len);
 *   // ...
 *   th.terminate();
 * }
 * ```
 */
export async function enableThreads(
  config: ThreadsConfig = {},
): Promise<ThreadsHandle | false> {
  /* ---- isolation detection (ADR-006) ---- */
  if (!isIsolated()) {
    console.warn(
      '[databonk] enableThreads: SharedArrayBuffer is not available or ' +
      'cross-origin isolation is missing (COOP/COEP headers required in browsers). ' +
      'Parallel mode not activated. See docs/threads.md for setup instructions.',
    );
    return false;
  }

  const numWorkers = config.workers ?? 4;
  const timeoutMs = config.timeoutMs ?? 30_000;
  /* simd-threads.wasm requires ≥17 pages (17 × 64 KiB) to cover the shadow stack
   * and BSS region laid out by wasm-ld.  Default to 32 pages (2 MiB) for headroom. */
  const initialPages = config.initialPages ?? 32;
  const maxPages = config.maxPages ?? 16_384;           /* 1 GiB (matches --max-memory in build.sh) */

  /* ---- load the threads wasm binary ---- */
  const wasmBytes = await loadBytes('simd-threads.wasm', config.wasmDir);

  /* ---- create shared memory ---- */
  const memory = new WebAssembly.Memory({
    initial: initialPages,
    maximum: maxPages,
    shared: true,
  });

  /* ---- instantiate main thread's wasm instance ---- */
  /* Use compile + instantiate(Module, imports) to avoid TypeScript overload
   * ambiguity (instantiate(BufferSource) vs instantiate(Module)). wasmBytes
   * is backed by a plain ArrayBuffer (guaranteed by loadBytes above). */
  const module = await WebAssembly.compile(wasmBytes.buffer as ArrayBuffer);
  const instance = await WebAssembly.instantiate(module, {
    env: { memory },
  });
  const exports = instance.exports as Record<string, unknown>;

  /* ---- apply passive data segments (if any) ---- */
  /* wasm-ld with --shared-memory may generate __wasm_init_memory to apply
   * passive data segments exactly once.  Our static-mut arena vars are all
   * zero-initialised (BSS) so there are typically no data segments to apply;
   * calling the function is a safe no-op in that case.
   * Workers intentionally skip this call — they see whatever the main thread
   * wrote, which is the correct initial state (all zeros). */
  if (typeof exports['__wasm_init_memory'] === 'function') {
    (exports['__wasm_init_memory'] as () => void)();
  }

  /* ---- build worker pool (workers share the same memory) ---- */
  const pool = await KernelWorkerPool.create(wasmBytes, memory, numWorkers, timeoutMs);

  return new ThreadsHandleImpl(pool, instance, memory);
}
