/**
 * Parallel mode parity tests (ADR-006 / P5.1).
 *
 * Architecture note: the threads wasm uses a SEPARATE shared WebAssembly.Memory
 * from the reference (scalar/simd) wasm.  Data for parallel operations must live
 * in the threads wasm's shared memory (allocated via `th.alloc()`).  The tests
 * therefore:
 *   1. Allocate data in the THREADS shared memory via `th.alloc()` + `th.memory`.
 *   2. Compare parallel dispatch results to `th.callKernel()` (same wasm, same memory,
 *      single-thread path) — this is the correct parity baseline.
 *
 * Tests:
 *  1. enableThreads returns false when SAB unavailable (no-isolation no-op).
 *  2. enableThreads succeeds in Node.js (SAB always available).
 *  3. Parity: parallel reductions match single-thread within floating-point tolerance.
 *  4. Parity with nulls: sparse validity bitmaps handled correctly.
 *  5. fast-check property: random data, random null patterns.
 *  6. Elementwise parity: addF64 bit-identical to single-thread.
 *  7. splitChunks utility: boundary alignment and coverage.
 *  8. Zero-change flag-off: scalar/simd wasm files are valid; threads wasm imports memory.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';
import * as fc from 'fast-check';

const __dir = dirname(fileURLToPath(import.meta.url));
const WASM_DIR = join(__dir, '../../wasm/dist');

/* ------------------------------------------------------------------ */
/* enableThreads import                                                  */
/* ------------------------------------------------------------------ */

import { enableThreads, splitChunks } from '../../src/workers/index.js';
import type { ThreadsHandle } from '../../src/workers/index.js';

/* ------------------------------------------------------------------ */
/* Helpers: write data into the THREADS shared memory                   */
/* ------------------------------------------------------------------ */

function writeF64(handle: ThreadsHandle, data: number[]): number {
  const ptr = handle.alloc(data.length * 8);
  const view = new Float64Array(handle.memory.buffer, ptr, data.length);
  for (let i = 0; i < data.length; i++) view[i] = data[i]!;
  return ptr;
}

function packValidity(bits: number[], len: number): Uint8Array {
  const bytes = new Uint8Array(Math.ceil(len / 8));
  for (let i = 0; i < len; i++) {
    if (bits[i]) bytes[i >> 3] |= 1 << (i & 7);
  }
  return bytes;
}

function writeValidity(handle: ThreadsHandle, bits: number[], len: number): number {
  const bm = packValidity(bits, len);
  const ptr = handle.alloc(bm.length);
  const view = new Uint8Array(handle.memory.buffer, ptr, bm.length);
  view.set(bm);
  return ptr;
}

/* ------------------------------------------------------------------ */
/* Test state                                                            */
/* ------------------------------------------------------------------ */

let th: ThreadsHandle | false;

beforeAll(async () => {
  th = await enableThreads({
    workers: 4,
    wasmDir: WASM_DIR,
    timeoutMs: 15_000,
  });
}, 60_000);

afterAll(() => {
  if (th && th.enabled) th.terminate();
});

/* ------------------------------------------------------------------ */
/* Test 1: threads enabled in Node.js                                   */
/* ------------------------------------------------------------------ */

describe('enableThreads detection', () => {
  it('returns a ThreadsHandle in Node.js (SAB always available)', () => {
    expect(th).toBeTruthy();
    expect(th).not.toBe(false);
    if (th) expect(th.enabled).toBe(true);
  });

  it('reports 4 workers', () => {
    if (!th) return;
    expect(th.workers).toBe(4);
  });

  it('exposes a shared WebAssembly.Memory', () => {
    if (!th) return;
    expect(th.memory).toBeInstanceOf(WebAssembly.Memory);
    /* Shared memory is backed by SharedArrayBuffer */
    expect(th.memory.buffer).toBeInstanceOf(SharedArrayBuffer);
  });

  it('alloc/free work in shared memory', () => {
    if (!th) return;
    const ptr = th.alloc(64);
    expect(ptr).toBeGreaterThan(0);
    const view = new Uint8Array(th.memory.buffer, ptr, 64);
    view.fill(42);
    expect(view[0]).toBe(42);
    th.free(ptr);
  });
});

/* ------------------------------------------------------------------ */
/* Test 2: no-isolation no-op                                           */
/* ------------------------------------------------------------------ */

describe('no-isolation no-op', () => {
  it('returns false when SharedArrayBuffer is unavailable', async () => {
    const saved = (globalThis as Record<string, unknown>)['SharedArrayBuffer'];
    (globalThis as Record<string, unknown>)['SharedArrayBuffer'] = undefined;
    let warned = false;
    const origWarn = console.warn;
    console.warn = () => { warned = true; };
    try {
      const result = await enableThreads({ workers: 2, wasmDir: WASM_DIR });
      expect(result).toBe(false);
      expect(warned).toBe(true);
    } finally {
      (globalThis as Record<string, unknown>)['SharedArrayBuffer'] = saved;
      console.warn = origWarn;
    }
  }, 10_000);
});

/* ------------------------------------------------------------------ */
/* Test 3: parity — basic numeric data, no nulls                        */
/* ------------------------------------------------------------------ */

describe('parity — no nulls', () => {
  const LEN = 10_000;
  const data = Array.from({ length: LEN }, (_, i) => (i + 1) * 0.1);

  it('sumF64 within 1e-9 relative error', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const expected = th.callKernel('sum_f64_null', ptr, 0, LEN);
    const actual = await th.sumF64(ptr, 0, LEN);
    th.free(ptr);
    const relErr = Math.abs(actual - expected) / (Math.abs(expected) + 1e-15);
    expect(relErr).toBeLessThan(1e-9);
  }, 30_000);

  it('meanF64 within 1e-9 relative error', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const expected = th.callKernel('mean_f64_null', ptr, 0, LEN);
    const actual = await th.meanF64(ptr, 0, LEN);
    th.free(ptr);
    const relErr = Math.abs(actual - expected) / (Math.abs(expected) + 1e-15);
    expect(relErr).toBeLessThan(1e-9);
  }, 30_000);

  it('minF64 exact', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const expected = th.callKernel('min_f64_null', ptr, 0, LEN);
    const actual = await th.minF64(ptr, 0, LEN);
    th.free(ptr);
    expect(actual).toBe(expected);
  }, 30_000);

  it('maxF64 exact', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const expected = th.callKernel('max_f64_null', ptr, 0, LEN);
    const actual = await th.maxF64(ptr, 0, LEN);
    th.free(ptr);
    expect(actual).toBe(expected);
  }, 30_000);
});

/* ------------------------------------------------------------------ */
/* Test 4: parity with nulls                                             */
/* ------------------------------------------------------------------ */

describe('parity — with nulls', () => {
  const LEN = 1024;
  const data = Array.from({ length: LEN }, (_, i) => (i + 1) * 1.5);
  /* Every 3rd element is null */
  const bits = Array.from({ length: LEN }, (_, i) => (i % 3 === 0 ? 0 : 1));

  it('sumF64 with nulls within 1e-9 relative error', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const vp = writeValidity(th, bits, LEN);
    const expected = th.callKernel('sum_f64_null', ptr, vp, LEN);
    const actual = await th.sumF64(ptr, vp, LEN);
    th.free(ptr); th.free(vp);
    const relErr = Math.abs(actual - expected) / (Math.abs(expected) + 1e-15);
    expect(relErr).toBeLessThan(1e-9);
  }, 30_000);

  it('meanF64 with nulls within 1e-9 relative error', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const vp = writeValidity(th, bits, LEN);
    const expected = th.callKernel('mean_f64_null', ptr, vp, LEN);
    const actual = await th.meanF64(ptr, vp, LEN);
    th.free(ptr); th.free(vp);
    const relErr = Math.abs(actual - expected) / (Math.abs(expected) + 1e-15);
    expect(relErr).toBeLessThan(1e-9);
  }, 30_000);

  it('minF64 with nulls exact', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const vp = writeValidity(th, bits, LEN);
    const expected = th.callKernel('min_f64_null', ptr, vp, LEN);
    const actual = await th.minF64(ptr, vp, LEN);
    th.free(ptr); th.free(vp);
    expect(actual).toBe(expected);
  }, 30_000);

  it('maxF64 with nulls exact', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const vp = writeValidity(th, bits, LEN);
    const expected = th.callKernel('max_f64_null', ptr, vp, LEN);
    const actual = await th.maxF64(ptr, vp, LEN);
    th.free(ptr); th.free(vp);
    expect(actual).toBe(expected);
  }, 30_000);

  it('all-null input: sumF64 = 0', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const nullBits = Array.from({ length: LEN }, () => 0);
    const nullVp = writeValidity(th, nullBits, LEN);
    const actual = await th.sumF64(ptr, nullVp, LEN);
    th.free(ptr); th.free(nullVp);
    expect(actual).toBe(0);
  }, 30_000);

  it('all-null input: meanF64 = NaN', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const nullBits = Array.from({ length: LEN }, () => 0);
    const nullVp = writeValidity(th, nullBits, LEN);
    const actual = await th.meanF64(ptr, nullVp, LEN);
    th.free(ptr); th.free(nullVp);
    expect(Number.isNaN(actual)).toBe(true);
  }, 30_000);

  it('all-null input: minF64 = NaN', async () => {
    if (!th) return;
    const ptr = writeF64(th, data);
    const nullBits = Array.from({ length: LEN }, () => 0);
    const nullVp = writeValidity(th, nullBits, LEN);
    const actual = await th.minF64(ptr, nullVp, LEN);
    th.free(ptr); th.free(nullVp);
    expect(Number.isNaN(actual)).toBe(true);
  }, 30_000);
});

/* ------------------------------------------------------------------ */
/* Test 5: fast-check property — random data + null patterns            */
/* ------------------------------------------------------------------ */

/**
 * Catastrophic-cancellation-safe tolerance for parallel f64 sum.
 *
 * Parallel dispatch splits work into chunks and combines partial sums in a
 * different order from single-thread; the rounding error is bounded by
 *   |actSum - expSum| ≤ tol × Σ|validXi|
 * rather than by relative error of the sum itself, which can be near zero when
 * large positive and negative values cancel.  tol = 1e-10 gives a comfortable
 * margin over worst-case float64 rounding with up to 256 elements × 4 chunks.
 *
 * Special cases handled:
 *  - ±Infinity: exact equality fast-path (Inf === Inf, -Inf === -Inf).
 *    Without it, Inf - Inf = NaN and Math.abs(NaN) = NaN, which compares
 *    false to anything — a spurious failure.
 *  - NaN sum (Inf + (-Inf) = NaN even with noNaN values): both-NaN fast-path.
 *
 * This is the documented ADR-006 deviation; see docs/threads.md.
 */
function sumWithinTol(actSum: number, expSum: number, values: number[], bits: number[]): boolean {
  /* Fast-path: exact equality covers ±Infinity === ±Infinity and 0 === -0. */
  if (actSum === expSum) return true;
  /* Both NaN (e.g. Infinity + (-Infinity) in the data). */
  if (Number.isNaN(actSum) && Number.isNaN(expSum)) return true;
  /* Finite difference — use catastrophic-cancellation-safe bound. */
  const absErr = Math.abs(actSum - expSum);
  const scale = values.reduce((s, x, i) => s + ((bits[i] ?? 0) ? Math.abs(x) : 0), 0);
  return absErr <= 1e-10 * scale + 1e-20;
}

describe('fast-check: parallel vs single-thread parity', () => {
  /**
   * Shared async property: parallel sum (catastrophic-cancellation-safe tolerance)
   * and parallel min/max (bit-exact — order-insensitive reductions).
   *
   * min/max MUST be bit-exact: if they ever differ, the chunk combiner has a bug
   * (see combineMinF64/combineMaxF64 in src/workers/parallel.ts).
   */
  function makeParity(thHandle: NonNullable<typeof th>) {
    return fc.asyncProperty(
      fc.array(fc.float({ noNaN: true }), { minLength: 8, maxLength: 256 }),
      fc.array(fc.boolean(), { minLength: 1 }),
      async (values, nullMask) => {
        const len = values.length;
        const bits = Array.from({ length: len }, (_, i) =>
          nullMask[i % nullMask.length] ? 1 : 0,
        );
        const ptr = writeF64(thHandle, values);
        const hasNull = bits.some((b) => b === 0);
        const vp = hasNull ? writeValidity(thHandle, bits, len) : 0;

        try {
          /* Single-thread reference using same wasm/memory */
          const expSum = thHandle.callKernel('sum_f64_null', ptr, vp, len);
          const expMin = thHandle.callKernel('min_f64_null', ptr, vp, len);
          const expMax = thHandle.callKernel('max_f64_null', ptr, vp, len);

          /* Parallel dispatch */
          const actSum = await thHandle.sumF64(ptr, vp, len);
          const actMin = await thHandle.minF64(ptr, vp, len);
          const actMax = await thHandle.maxF64(ptr, vp, len);

          /* sum: catastrophic-cancellation-safe tolerance (extreme float32 values
           * like ±3.4e38 can cancel to ~0, making simple relative error blow up).
           * Tolerance is scaled by Σ|valid xi| — always non-zero when values exist.
           * This is the documented ADR-006 FP deviation; see docs/threads.md. */
          if (!sumWithinTol(actSum, expSum, values, bits)) return false;

          /* min/max: bit-exact (order-insensitive); any difference is a combiner bug */
          const eqMin =
            (Number.isNaN(expMin) && Number.isNaN(actMin)) || expMin === actMin;
          const eqMax =
            (Number.isNaN(expMax) && Number.isNaN(actMax)) || expMax === actMax;
          return eqMin && eqMax;
        } finally {
          thHandle.free(ptr);
          if (vp !== 0) thHandle.free(vp);
        }
      },
    );
  }

  it('sum/min/max agree on random data+nulls (100 runs)', async () => {
    if (!th) return;
    await fc.assert(makeParity(th), { numRuns: 100 });
  }, 120_000);

  /* Pinned seed: previously failed because the old relative-error check used
   * (expSum + 1e-15) as the denominator, which rounds to ~1e-15 when large
   * positive and negative floats cancel.  Confirmed: the failure was in SUM
   * (catastrophic cancellation), NOT in min/max — the chunk combiner is correct.
   * After switching to the sum(|x|)-scaled tolerance above, this seed passes. */
  it('pinned seed 1711909037 — sum catastrophic cancellation regression', async () => {
    if (!th) return;
    await fc.assert(makeParity(th), { seed: 1711909037, numRuns: 50 });
  }, 120_000);
});

/* ------------------------------------------------------------------ */
/* Test 6: elementwise parity — addF64                                   */
/* ------------------------------------------------------------------ */

describe('elementwise parity — addF64', () => {
  const LEN = 4096;
  const a = Array.from({ length: LEN }, (_, i) => i + 0.5);
  const b = Array.from({ length: LEN }, (_, i) => i * 2.0);

  it('parallel addF64 bit-identical to single-thread', async () => {
    if (!th) return;

    const aPtr = writeF64(th, a);
    const bPtr = writeF64(th, b);
    const outPtrRef = th.alloc(LEN * 8);
    const outPtrPar = th.alloc(LEN * 8);

    /* Single-thread reference (on shared memory, same wasm instance) */
    th.callKernel('add_f64', aPtr, bPtr, outPtrRef, LEN);

    /* Parallel dispatch — workers write directly to shared output buffer */
    await th.addF64(aPtr, bPtr, outPtrPar, LEN);

    const vRef = new Float64Array(th.memory.buffer, outPtrRef, LEN);
    const vPar = new Float64Array(th.memory.buffer, outPtrPar, LEN);
    for (let i = 0; i < LEN; i++) {
      expect(vPar[i]).toBe(vRef[i]);
    }

    th.free(aPtr);
    th.free(bPtr);
    th.free(outPtrRef);
    th.free(outPtrPar);
  }, 30_000);
});

/* ------------------------------------------------------------------ */
/* Test 7: splitChunks utility                                           */
/* ------------------------------------------------------------------ */

describe('splitChunks', () => {
  it('produces byte-aligned boundaries for various lengths', () => {
    for (const len of [0, 1, 7, 8, 9, 100, 1000, 10_000_000]) {
      const chunks = splitChunks(len, 4);
      for (const [start] of chunks) {
        expect(start % 8).toBe(0); /* Arrow-LSB bitmap byte-alignment */
      }
      if (chunks.length > 0) {
        expect(chunks[0]![0]).toBe(0);
        expect(chunks[chunks.length - 1]![1]).toBe(len);
      }
    }
  });

  it('total coverage equals len', () => {
    for (const len of [0, 10_007, 10_000_000]) {
      const chunks = splitChunks(len, 4);
      let covered = 0;
      for (const [s, e] of chunks) covered += e - s;
      expect(covered).toBe(len);
    }
  });

  it('no more than numWorkers chunks', () => {
    for (const nw of [1, 2, 4, 8]) {
      const chunks = splitChunks(10_000, nw);
      expect(chunks.length).toBeLessThanOrEqual(nw);
    }
  });

  it('returns [] for len=0', () => {
    expect(splitChunks(0, 4)).toEqual([]);
  });
});

/* ------------------------------------------------------------------ */
/* Test 8: zero-change flag-off — scalar/simd binaries unchanged         */
/* ------------------------------------------------------------------ */

describe('flag-off: existing binaries unchanged', () => {
  it('scalar.wasm is a valid wasm binary', async () => {
    const buf = await readFile(join(WASM_DIR, 'scalar.wasm'));
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    expect(dv.getUint32(0, false)).toBe(0x0061736d); /* wasm magic */
    expect(dv.getUint32(4, true)).toBe(0x00000001);  /* wasm version 1 */
  });

  it('simd.wasm is a valid wasm binary', async () => {
    const buf = await readFile(join(WASM_DIR, 'simd.wasm'));
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    expect(dv.getUint32(0, false)).toBe(0x0061736d);
    expect(dv.getUint32(4, true)).toBe(0x00000001);
  });

  it('scalar.wasm instantiates without imports (self-contained memory)', async () => {
    const buf = await readFile(join(WASM_DIR, 'scalar.wasm'));
    const { instance } = await WebAssembly.instantiate(buf, {});
    expect(typeof (instance.exports as Record<string, unknown>)['alloc']).toBe('function');
  });

  it('simd-threads.wasm requires env.memory import (shared memory)', async () => {
    const buf = await readFile(join(WASM_DIR, 'simd-threads.wasm'));
    /* Should FAIL without the import */
    await expect(WebAssembly.instantiate(buf, {})).rejects.toThrow();
    /* Should SUCCEED with a shared memory of at least 17 pages */
    const shared = new WebAssembly.Memory({ initial: 17, maximum: 16384, shared: true });
    const { instance } = await WebAssembly.instantiate(buf, { env: { memory: shared } });
    expect(typeof (instance.exports as Record<string, unknown>)['alloc']).toBe('function');
    expect(typeof (instance.exports as Record<string, unknown>)['sum_f64_null']).toBe('function');
  });
});
