# Parallel mode — `enableThreads()` (ADR-006 / Phase 5)

The library's main path is single-threaded (see ADR-006).  `enableThreads()` is
an opt-in feature that uses a shared-memory WASM build and a worker pool to
parallelize elementwise and reduction kernels.

---

## Requirements

| Environment | Requirement |
|---|---|
| **Node.js ≥ 18** | `SharedArrayBuffer` is always enabled — no extra config. |
| **Browser** | Requires **cross-origin isolation**: serve the page with `Cross-Origin-Opener-Policy: same-origin` AND `Cross-Origin-Embedder-Policy: require-corp`. Then `crossOriginIsolated === true` and `SharedArrayBuffer` is available. |

If the required isolation is absent, `enableThreads()` emits a `console.warn`
and returns `false`.  The library continues to run single-threaded and all
existing behaviour is unaffected.

---

## Quick start

```typescript
import { init, DataFrame, enableThreads } from 'databonk';

await init(); // load the standard single-thread wasm

// Opt in to parallel mode:
const th = await enableThreads({ workers: 4 });
if (th) {
  // th.sumF64 / th.meanF64 / th.minF64 / th.maxF64 dispatch work to 4 workers
  const sum = await th.sumF64(dataPtr, vpPtr, len);
  th.terminate(); // clean up when done
}
```

### Node.js script

```typescript
import { enableThreads } from 'databonk';

const th = await enableThreads({ workers: 4 });
if (!th) throw new Error('threads unavailable');

// parallel sum
const result = await th.sumF64(ptr, 0, 10_000_000);
th.terminate();
```

---

## Browser: COOP / COEP setup

### Vite (`vite.config.ts`)

```typescript
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    {
      name: 'cross-origin-isolation',
      configureServer(server) {
        server.middlewares.use((_, res, next) => {
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          next();
        });
      },
    },
  ],
});
```

### webpack (`webpack.config.js`)

```javascript
const express = require('express');

module.exports = {
  devServer: {
    setupMiddlewares(middlewares, devServer) {
      devServer.app.use((_, res, next) => {
        res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
        next();
      });
      return middlewares;
    },
  },
};
```

### nginx

```nginx
location / {
    add_header Cross-Origin-Opener-Policy same-origin;
    add_header Cross-Origin-Embedder-Policy require-corp;
}
```

### Cloudflare Pages / Workers

Add response headers in your `_headers` file (Cloudflare Pages):

```
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
```

---

## How it works

### The simd-threads.wasm binary

A third wasm build (`simd-threads.wasm`) is produced alongside `scalar.wasm`
and `simd.wasm`.  It uses:

- `RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128"`
- Linker flags: `--import-memory --shared-memory --max-memory=1073741824`
- Built with nightly Rust + `-Zbuild-std=core` (required for `+atomics` on
  `wasm32-unknown-unknown` — nightly needed until atomics stabilise).

Unlike `scalar.wasm` / `simd.wasm` which export their own `WebAssembly.Memory`,
`simd-threads.wasm` **imports** `env.memory`.  The host (JS) creates a
`SharedArrayBuffer`-backed `WebAssembly.Memory` and passes it to every wasm
instance (main + all workers).

The `scalar.wasm` and `simd.wasm` main-path binaries are **not affected** — they
are built identically to before Phase 5 (verified by checksum).  Zero behaviour
change when `enableThreads()` is not called.

### Allocator invariant

The arena (bump+freelist) lives in the shared linear memory.  Its state
(`HEAP_TOP`, `FREE_HEAD`, `GENERATION`) is stored as zero-initialised static
variables.  Since wasm linear memory is zero-initialised by default, there are no
data segments that would be re-applied by worker instantiation — workers see the
exact state left by the main thread.

**Workers NEVER call `alloc`/`free`/`realloc`.**  All allocation is done by the
main thread before work is dispatched.  Workers are stateless kernel executors:
they receive pre-computed pointer arguments and call the appropriate kernel
export.  This is the parallel extension of the ABI §5.4 "kernels never alloc"
rule.

### Chunk dispatch

```
[0, len)  →  split into N ≤ numWorkers chunks, each boundary a multiple of 8
                for Arrow-LSB validity-bitmap byte alignment.

worker i receives:
  dataPtr + chunkStart × elemSize
  vpPtr === 0 ? 0 : vpPtr + chunkStart / 8   (bitmap byte-aligned pointer)
  chunkLen

workers write results directly into shared memory (elementwise)
  or return partial scalars via postMessage (reductions).
```

Reduction combination order is fixed left-to-right (chunk 0, chunk 1, …, chunk N-1)
for a given worker count, ensuring deterministic results.

### Floating-point deviation

**f64 `sum` and `mean` results from parallel dispatch are NOT bit-identical to
single-thread.**

Single-thread `sum_f64_null` uses a prescribed 2-stripe accumulation over all N
elements (element `i` → accumulator `i & 1`; result = acc0 + acc1).  Parallel
dispatch splits into chunks, each chunk is 2-striped independently, and the
per-chunk partial sums are combined left-to-right.  The different accumulation
order produces different IEEE-754 rounding for f64.

This deviation is intentional and justified (ADR-006 scope):

- Results are still IEEE-754 correct (no ULP blowup; relative error < 1e-9
  in practice for typical data).
- Results are **deterministic** for a fixed worker count — the same worker count
  always produces the same bit-pattern.
- Integer sums (`sum_i32_null`, `sum_u32_null`) are **exact** (integer arithmetic
  is order-insensitive modulo overflow, which is expected and documented).
- `min_f64_null` / `max_f64_null` are **exactly equal** to single-thread
  (min/max are order-insensitive).

If your application requires bit-exact f64 sums, run in single-thread mode
(do not call `enableThreads()`).

---

## API reference

```typescript
async function enableThreads(config?: ThreadsConfig): Promise<ThreadsHandle | false>

interface ThreadsConfig {
  workers?: number;     // default 4
  wasmDir?: string | URL;
  timeoutMs?: number;   // default 30 000 ms; 0 = no timeout
  initialPages?: number; // shared memory initial pages (64 KiB each); default 32 (minimum 17)
  maxPages?: number;     // shared memory maximum pages; default 16384 (1 GiB)
}

interface ThreadsHandle {
  readonly enabled: true;
  readonly workers: number;

  // Parallel reductions (f64)
  sumF64(dataPtr, vpPtr, len): Promise<number>;
  meanF64(dataPtr, vpPtr, len): Promise<number>;
  minF64(dataPtr, vpPtr, len): Promise<number>;
  maxF64(dataPtr, vpPtr, len): Promise<number>;

  // Parallel elementwise (f64)
  addF64(aPtr, bPtr, outPtr, len): Promise<void>;
  subF64(aPtr, bPtr, outPtr, len): Promise<void>;
  mulF64(aPtr, bPtr, outPtr, len): Promise<void>;

  // Generic parallel dispatch
  parallelReduce(fn, dataPtr, vpPtr, len, elemBytes): Promise<number[]>;
  parallelElementwiseBinary(fn, aPtr, bPtr, outPtr, len, elemBytes): Promise<void>;

  terminate(): void;
}
```

---

## Robustness

Worker crashes and timeouts do not hang the pool.  If a worker fails to respond
within `timeoutMs`, the pending request is rejected with an error, the worker is
terminated, and a fresh replacement is spawned and re-initialised automatically.

---

## Build notes for maintainers

The `simd-threads.wasm` build requires nightly Rust.  Exact commands documented
in `wasm/rust/build.sh`.  Install once in the container:

```bash
rustup toolchain install nightly
rustup target add wasm32-unknown-unknown --toolchain nightly
rustup component add rust-src --toolchain nightly
```

Then build all three variants:

```bash
docker run --rm -v "$PWD":/work -w /work dataframe-dev \
  bash -lc 'npm ci && bash wasm/rust/build.sh'
```

The `build.sh` script installs nightly automatically if not present.

---

## Performance expectations

Benchmark: 10M f64 elements, sum/mean/min, Node.js 22, Apple M-series (2026-07-02).

| Op   | 1 thread (ms) | 1 worker (ms) | 4 workers (ms) | Speedup (1t→4w) | Gate ≥1.8× |
|------|---------------|---------------|----------------|-----------------|------------|
| sum  | 2.74          | 2.84          | 0.82           | 3.34×           | PASS       |
| mean | 2.74          | 2.79          | 0.82           | 3.36×           | PASS       |
| min  | 4.38          | 4.46          | 1.24           | 3.54×           | PASS       |

Gate: ≥1.8× on 4 workers for all three ops.  Actual results are 3.3–3.5×
(well above the gate).  The 10M-element dataset is 80 MB; at ~100 GB/s
bandwidth memory-bus saturation would cap speedup, but in practice the
Apple M-series unified memory architecture scales here.
