/**
 * tests/temporal/temporal_frame.test.ts
 *
 * Conformance runner for temporal.json groups handled at the frame/expr layer:
 *
 *   - date32_compare_sort       kernel_reuse: min/max/argsort/eq/lt via i32 physical kernels
 *   - timestamp_compare_sort    kernel_reuse: min/max/argsort/eq via i64 physical kernels
 *   - restricted_temporal_arithmetic  kernel_reuse: sub/add i32/i64 kernels
 *   - temporal_arithmetic_errors      frame_error: dtype checker throws
 *   - temporal_casts            frame: reinterpret + scale casts
 *   - temporal_groupby_join     kernel_reuse: hash_i32/hash_i64 equal-hash property
 *
 * All cases in temporal.json EXCEPT the `frame_accessor` group (covered in conformance.test.ts).
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { readFileSync } from 'node:fs';

import { loadEnv, TestFrame, takeColumn, type TestEnv } from '../expr/helper.js';
import { createColumn, freeColumn, columnToArray } from '../../src/memory/column.js';
import { alignValidity, freeAligned } from '../../src/frame/util.js';
import { compile } from '../../src/expr/compile.js';
import { col, lit } from '../../src/expr/ast.js';
import { resolve } from '../../src/expr/dtypes.js';
import type { DType } from '../../src/memory/dtype.js';
import type { Cell } from '../../src/memory/column.js';

// ── Load fixture ──────────────────────────────────────────────────────────────

const __dir = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(__dir, '../conformance/fixtures/temporal.json');

interface FixtureCase {
  group: string;
  name: string;
  layer: string;
  export?: string;
  op?: string;
  dtype_a?: string;
  dtype_b?: string;
  dtype_logical?: string;
  dtype_physical?: string;
  cast?: string;
  note?: string;
  error_pattern?: string;
  property?: string;
  inputs?: unknown;
  expected?: unknown;
}

const fixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8')) as { cases: FixtureCase[] };
const allCases = fixture.cases;

// ── Test environment ─────────────────────────────────────────────────────────

let env: TestEnv;

beforeAll(async () => {
  env = await loadEnv();
});

afterAll(() => {
  // env has no cleanup (memory is GC'd)
});

// ── Helper: build a frame with one or two temporal columns ──────────────────

type Inputs_I32 = { data: number[]; vp?: number[] };
type Inputs_I64_Str = { data: string[]; vp?: number[] };
type Inputs_Arith = {
  a?: unknown[]; b?: unknown[];
  a_vp?: number[]; b_vp?: number[];
  s?: unknown;
  inout_perm?: number[]; desc?: number;
};

function buildValidity(vp: number[] | undefined, len: number): number[] | undefined {
  if (!vp) return undefined;
  // vp is an array of 0/1 per element; convert to a JS array for createColumn
  // (but createColumn reads nulls from the values array, not a vp array).
  // We encode nulls by using null in the values array.
  return vp;
}

/** Build i32 data array with nulls embedded from vp (1=valid, 0=null). */
function applyVpI32(data: number[], vp?: number[]): Array<number | null> {
  if (!vp) return data;
  return data.map((v, i) => (vp[i] === 0 ? null : v));
}

/** Build i64 (bigint) data array from decimal strings with nulls from vp. */
function applyVpI64(data: string[], vp?: number[]): Array<bigint | null> {
  return data.map((v, i) => (vp && vp[i] === 0 ? null : BigInt(v)));
}

// ── Helpers to call wasm kernels directly (kernel_reuse layer) ───────────────

function rawKernel(wasm: object, name: string): (...args: number[]) => number {
  const fn = (wasm as Record<string, unknown>)[name];
  if (typeof fn !== 'function') throw new Error(`kernel not found: ${name}`);
  return fn as (...args: number[]) => number;
}

function rawKernelBigInt(wasm: object, name: string): (...args: (number | bigint)[]) => bigint {
  const fn = (wasm as Record<string, unknown>)[name];
  if (typeof fn !== 'function') throw new Error(`kernel not found: ${name}`);
  return fn as (...args: (number | bigint)[]) => bigint;
}

// ── date32_compare_sort ──────────────────────────────────────────────────────

describe('temporal_frame — date32_compare_sort', () => {
  const cases = allCases.filter((c) => c.group === 'date32_compare_sort');

  for (const tc of cases) {
    it(tc.name, () => {
      const inp = tc.inputs as Inputs_I32 & { a?: number[]; b?: number[]; inout_perm?: number[]; desc?: number };
      const exp = tc.expected as { result?: number; out_mask?: number[]; out_perm?: number[] };
      const { ctx, wasm } = env;

      if (tc.export === 'min_i32_null' || tc.export === 'max_i32_null') {
        // Build a date32 column, call the physical kernel directly.
        const values = applyVpI32(inp.data, inp.vp);
        const col = createColumn(ctx, 'date32', values);
        try {
          const len = col.length;
          const validityBytes = Math.ceil(len / 8);
          const vbytes = Math.max(validityBytes, 1);
          const vp = col.validityPtr !== 0 ? col.validityPtr : 0;
          const fn = rawKernel(wasm, tc.export!);
          const result = fn(col.dataPtr, vp, len);
          expect(result).toBe(exp.result);
        } finally {
          freeColumn(ctx, col);
        }
      } else if (tc.export === 'argsort_i32') {
        const values = applyVpI32(inp.data, inp.vp);
        const col = createColumn(ctx, 'date32', values);
        try {
          const len = col.length;
          const permPtr = ctx.mod.alloc(Math.max(len * 4, 1));
          const permView = ctx.viewOf({ ptr: permPtr, length: len, dtype: 'i32' }) as Int32Array;
          const inPerm = inp.inout_perm!;
          for (let i = 0; i < len; i++) permView[i] = inPerm[i]!;
          const vp = col.validityPtr !== 0 ? col.validityPtr : 0;
          const fn = rawKernel(wasm, 'argsort_i32');
          const desc = inp.desc ?? 0;
          // Try 5-arg first, then 6-arg (scratch_ptr amendment)
          try {
            fn(col.dataPtr, vp, permPtr, len, desc);
          } catch {
            const scratchPtr = ctx.mod.alloc(Math.max(len * 4, 1));
            fn(col.dataPtr, vp, permPtr, len, desc, scratchPtr);
            ctx.mod.free(scratchPtr);
          }
          const result = Array.from(permView);
          ctx.viewOf.forget({ ptr: permPtr, length: len, dtype: 'i32' });
          ctx.mod.free(permPtr);
          expect(result).toEqual(exp.out_perm);
        } finally {
          freeColumn(ctx, col);
        }
      } else if (tc.export === 'eq_i32_mask' || tc.export === 'lt_i32_mask') {
        const a = inp.a!;
        const b = inp.b!;
        const colA = createColumn(ctx, 'date32', a);
        const colB = createColumn(ctx, 'date32', b);
        try {
          const len = a.length;
          const vbytes = Math.max(Math.ceil(len / 8), 1);
          const maskPtr = ctx.mod.alloc(vbytes);
          const fn = rawKernel(wasm, tc.export!);
          fn(colA.dataPtr, colB.dataPtr, maskPtr, len);
          const maskView = ctx.viewOf({ ptr: maskPtr, length: vbytes, dtype: 'u8' }) as Uint8Array;
          const result: number[] = [];
          for (let i = 0; i < len; i++) result.push((maskView[i >> 3]! >> (i & 7)) & 1);
          ctx.viewOf.forget({ ptr: maskPtr, length: vbytes, dtype: 'u8' });
          ctx.mod.free(maskPtr);
          expect(result).toEqual(exp.out_mask);
        } finally {
          freeColumn(ctx, colA);
          freeColumn(ctx, colB);
        }
      }
    });
  }
});

// ── timestamp_compare_sort ───────────────────────────────────────────────────

describe('temporal_frame — timestamp_compare_sort', () => {
  const cases = allCases.filter((c) => c.group === 'timestamp_compare_sort');

  for (const tc of cases) {
    it(tc.name, () => {
      const inp = tc.inputs as Inputs_I64_Str & { a?: string[]; b?: string[]; inout_perm?: number[]; desc?: number };
      const exp = tc.expected as { result?: string; out_mask?: number[]; out_perm?: number[] };
      const { ctx, wasm } = env;

      if (tc.export === 'min_i64_null' || tc.export === 'max_i64_null') {
        const values = applyVpI64(inp.data, inp.vp);
        const col = createColumn(ctx, 'timestamp', values);
        try {
          const len = col.length;
          const vp = col.validityPtr !== 0 ? col.validityPtr : 0;
          const fn = rawKernelBigInt(wasm, tc.export!);
          const result = fn(col.dataPtr, vp, len);
          expect(String(result)).toBe(exp.result);
        } finally {
          freeColumn(ctx, col);
        }
      } else if (tc.export === 'argsort_i64') {
        const values = applyVpI64(inp.data, inp.vp);
        const col = createColumn(ctx, 'timestamp', values);
        try {
          const len = col.length;
          const permPtr = ctx.mod.alloc(Math.max(len * 4, 1));
          const permView = ctx.viewOf({ ptr: permPtr, length: len, dtype: 'i32' }) as Int32Array;
          const inPerm = inp.inout_perm!;
          for (let i = 0; i < len; i++) permView[i] = inPerm[i]!;
          const vp = col.validityPtr !== 0 ? col.validityPtr : 0;
          const fn = rawKernel(wasm, 'argsort_i64');
          const desc = inp.desc ?? 0;
          try {
            fn(col.dataPtr, vp, permPtr, len, desc);
          } catch {
            const scratchPtr = ctx.mod.alloc(Math.max(len * 4, 1));
            fn(col.dataPtr, vp, permPtr, len, desc, scratchPtr);
            ctx.mod.free(scratchPtr);
          }
          const result = Array.from(permView);
          ctx.viewOf.forget({ ptr: permPtr, length: len, dtype: 'i32' });
          ctx.mod.free(permPtr);
          expect(result).toEqual(exp.out_perm);
        } finally {
          freeColumn(ctx, col);
        }
      } else if (tc.export === 'eq_i64_mask') {
        const a = inp.a!;
        const b = inp.b!;
        const colA = createColumn(ctx, 'timestamp', a.map(BigInt));
        const colB = createColumn(ctx, 'timestamp', b.map(BigInt));
        try {
          const len = a.length;
          const vbytes = Math.max(Math.ceil(len / 8), 1);
          const maskPtr = ctx.mod.alloc(vbytes);
          const fn = rawKernel(wasm, 'eq_i64_mask');
          fn(colA.dataPtr, colB.dataPtr, maskPtr, len);
          const maskView = ctx.viewOf({ ptr: maskPtr, length: vbytes, dtype: 'u8' }) as Uint8Array;
          const result: number[] = [];
          for (let i = 0; i < len; i++) result.push((maskView[i >> 3]! >> (i & 7)) & 1);
          ctx.viewOf.forget({ ptr: maskPtr, length: vbytes, dtype: 'u8' });
          ctx.mod.free(maskPtr);
          expect(result).toEqual(exp.out_mask);
        } finally {
          freeColumn(ctx, colA);
          freeColumn(ctx, colB);
        }
      }
    });
  }
});

// ── restricted_temporal_arithmetic ──────────────────────────────────────────

describe('temporal_frame — restricted_temporal_arithmetic', () => {
  const cases = allCases.filter((c) => c.group === 'restricted_temporal_arithmetic');

  for (const tc of cases) {
    it(tc.name, () => {
      const inp = tc.inputs as Inputs_Arith;
      const exp = tc.expected as { out?: unknown[]; out_vp?: number[] };
      const { ctx, wasm } = env;
      const expOut = exp.out!;
      const expVp = exp.out_vp;

      // Dispatch by export name
      if (tc.export === 'sub_i64') {
        // timestamp - timestamp -> i64
        const a = applyVpI64(inp.a as string[], inp.a_vp);
        const b = applyVpI64(inp.b as string[], inp.b_vp);
        const frame = new TestFrame(env, {
          a: { dtype: 'timestamp', values: a },
          b: { dtype: 'timestamp', values: b },
        });
        try {
          const plan = compile(col('a').sub(col('b')), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          for (let i = 0; i < expOut.length; i++) {
            if (expOut[i] === null) expect(result[i]).toBeNull();
            else expect(String(result[i] as bigint)).toBe(String(expOut[i]));
          }
        } finally {
          frame.free();
        }
      } else if (tc.export === 'add_i64_scalar' || tc.export === 'sub_i64_scalar') {
        // timestamp ± scalar ms -> timestamp
        const op = tc.export === 'add_i64_scalar' ? 'add' : 'sub';
        const a = applyVpI64(inp.a as string[], inp.a_vp);
        const s = Number(inp.s as string);
        const frame = new TestFrame(env, { a: { dtype: 'timestamp', values: a } });
        try {
          const expr = op === 'add' ? col('a').add(lit(s)) : col('a').sub(lit(s));
          const plan = compile(expr, frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          for (let i = 0; i < expOut.length; i++) {
            expect(String(result[i] as bigint)).toBe(String(expOut[i]));
          }
        } finally {
          frame.free();
        }
      } else if (tc.export === 'add_i64') {
        // timestamp + i64 column -> timestamp
        const a = applyVpI64(inp.a as string[], inp.a_vp);
        const b = applyVpI64(inp.b as string[], inp.b_vp);
        const frame = new TestFrame(env, {
          a: { dtype: 'timestamp', values: a },
          b: { dtype: 'i64', values: b },
        });
        try {
          const plan = compile(col('a').add(col('b')), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          for (let i = 0; i < expOut.length; i++) {
            if (expOut[i] === null) expect(result[i]).toBeNull();
            else expect(String(result[i] as bigint)).toBe(String(expOut[i]));
          }
        } finally {
          frame.free();
        }
      } else if (tc.export === 'sub_i32') {
        // date32 - date32 -> i32
        const a = applyVpI32(inp.a as number[], inp.a_vp);
        const b = applyVpI32(inp.b as number[], inp.b_vp);
        const frame = new TestFrame(env, {
          a: { dtype: 'date32', values: a },
          b: { dtype: 'date32', values: b },
        });
        try {
          const plan = compile(col('a').sub(col('b')), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          for (let i = 0; i < expOut.length; i++) {
            if (expOut[i] === null) expect(result[i]).toBeNull();
            else expect(result[i]).toBe(expOut[i]);
          }
        } finally {
          frame.free();
        }
      } else if (tc.export === 'add_i32_scalar' || tc.export === 'sub_i32_scalar') {
        // date32 ± scalar days -> date32
        const op = tc.export === 'add_i32_scalar' ? 'add' : 'sub';
        const a = applyVpI32(inp.a as number[], inp.a_vp);
        const s = inp.s as number;
        const frame = new TestFrame(env, { a: { dtype: 'date32', values: a } });
        try {
          const expr = op === 'add' ? col('a').add(lit(s)) : col('a').sub(lit(s));
          const plan = compile(expr, frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          for (let i = 0; i < expOut.length; i++) {
            expect(result[i]).toBe(expOut[i]);
          }
        } finally {
          frame.free();
        }
      }
    });
  }
});

// ── temporal_arithmetic_errors ───────────────────────────────────────────────

describe('temporal_frame — temporal_arithmetic_errors', () => {
  const cases = allCases.filter((c) => c.group === 'temporal_arithmetic_errors');

  for (const tc of cases) {
    it(tc.name, () => {
      const dtypeA = tc.dtype_a as DType;
      const dtypeB = tc.dtype_b as DType;
      const op = tc.op!;
      const errorPattern = tc.error_pattern!;

      // Build dummy values for the given dtypes.
      const valA: Cell[] = dtypeA === 'timestamp' ? [0n] : dtypeA === 'date32' ? [0] : [0n];
      const valB: Cell[] = dtypeB === 'timestamp' ? [0n] : dtypeB === 'date32' ? [0] : dtypeB === 'f64' ? [1.0] : [0n];

      const frame = new TestFrame(env, {
        a: { dtype: dtypeA, values: valA as unknown as number[] },
        b: { dtype: dtypeB, values: valB as unknown as number[] },
      });

      try {
        let expr;
        switch (op) {
          case 'add': expr = col('a').add(col('b')); break;
          case 'sub': expr = col('a').sub(col('b')); break;
          case 'mul': expr = col('a').mul(col('b')); break;
          case 'div': expr = col('a').div(col('b')); break;
          default: expr = col('a').add(col('b'));
        }
        expect(() => resolve(expr, frame)).toThrow(errorPattern);
      } finally {
        frame.free();
      }
    });
  }
});

// ── temporal_casts ───────────────────────────────────────────────────────────

describe('temporal_frame — temporal_casts', () => {
  const cases = allCases.filter((c) => c.group === 'temporal_casts');

  for (const tc of cases) {
    it(tc.name, () => {
      const { ctx } = env;
      const cast = tc.cast!;

      if (cast === 'date32_to_i32') {
        const inp = tc.inputs as { data_date32: number[] };
        const exp = tc.expected as { out_i32: number[] };
        const srcCol = createColumn(ctx, 'date32', inp.data_date32);
        try {
          const frame = new TestFrame(env, { d: { dtype: 'date32', values: inp.data_date32 } });
          const plan = compile(col('d').cast('i32'), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          expect(result).toEqual(exp.out_i32);
          frame.free();
        } finally {
          freeColumn(ctx, srcCol);
        }
      } else if (cast === 'i32_to_date32') {
        const inp = tc.inputs as { data_i32: number[] };
        const exp = tc.expected as { out_date32_days: number[] };
        const frame = new TestFrame(env, { v: { dtype: 'i32', values: inp.data_i32 } });
        try {
          const plan = compile(col('v').cast('date32'), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          expect(result).toEqual(exp.out_date32_days);
        } finally {
          frame.free();
        }
      } else if (cast === 'timestamp_to_i64') {
        const inp = tc.inputs as { data_ts_ms: string[] };
        const exp = tc.expected as { out_i64: string[] };
        const vals = inp.data_ts_ms.map(BigInt);
        const frame = new TestFrame(env, { ts: { dtype: 'timestamp', values: vals } });
        try {
          const plan = compile(col('ts').cast('i64'), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          expect(result.map(String)).toEqual(exp.out_i64);
        } finally {
          frame.free();
        }
      } else if (cast === 'i64_to_timestamp') {
        const inp = tc.inputs as { data_i64: string[] };
        const exp = tc.expected as { out_ts_ms: string[] };
        const vals = inp.data_i64.map(BigInt);
        const frame = new TestFrame(env, { v: { dtype: 'i64', values: vals } });
        try {
          const plan = compile(col('v').cast('timestamp'), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          expect(result.map(String)).toEqual(exp.out_ts_ms);
        } finally {
          frame.free();
        }
      } else if (cast === 'date32_to_timestamp') {
        const inp = tc.inputs as { data_date32: number[]; vp?: number[] };
        const exp = tc.expected as { out_ts_ms: (string | null)[]; out_vp?: number[] };
        const vals = applyVpI32(inp.data_date32, inp.vp);
        const frame = new TestFrame(env, { d: { dtype: 'date32', values: vals } });
        try {
          const plan = compile(col('d').cast('timestamp'), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          for (let i = 0; i < exp.out_ts_ms.length; i++) {
            if (exp.out_ts_ms[i] === null) expect(result[i]).toBeNull();
            else expect(String(result[i] as bigint)).toBe(exp.out_ts_ms[i]);
          }
        } finally {
          frame.free();
        }
      } else if (cast === 'timestamp_to_date32') {
        const inp = tc.inputs as { data_ts_ms: string[]; vp?: number[] };
        const exp = tc.expected as { out_date32: (number | null)[]; out_vp?: number[] };
        const vals = applyVpI64(inp.data_ts_ms, inp.vp);
        const frame = new TestFrame(env, { ts: { dtype: 'timestamp', values: vals } });
        try {
          const plan = compile(col('ts').cast('date32'), frame);
          const { column } = plan.execute();
          const result = takeColumn(ctx, column!);
          for (let i = 0; i < exp.out_date32.length; i++) {
            if (exp.out_date32[i] === null) expect(result[i]).toBeNull();
            else expect(result[i]).toBe(exp.out_date32[i]);
          }
        } finally {
          frame.free();
        }
      }
    });
  }
});

// ── temporal_groupby_join ─────────────────────────────────────────────────────

describe('temporal_frame — temporal_groupby_join', () => {
  const cases = allCases.filter((c) => c.group === 'temporal_groupby_join');

  for (const tc of cases) {
    it(tc.name, () => {
      const { ctx, wasm } = env;
      const inp = tc.inputs as { data: unknown[]; vp?: number[] };
      const isTimestamp = tc.dtype_logical === 'timestamp';

      if (tc.property === 'equal_inputs_equal_hashes') {
        // hash kernels output u64 per element (8 bytes); groupby.ts allocates len*8.
        if (isTimestamp) {
          const vals = applyVpI64(inp.data as string[], inp.vp);
          const col = createColumn(ctx, 'timestamp', vals);
          try {
            const len = col.length;
            const hashPtr = ctx.mod.alloc(Math.max(len * 8, 1));
            const av = alignValidity(ctx, col);
            try {
              rawKernel(wasm, 'hash_i64')(col.dataPtr, av.ptr, hashPtr, len);
              const hashView = ctx.viewOf({ ptr: hashPtr, length: len, dtype: 'i64' }) as BigInt64Array;
              const data = inp.data as string[];
              for (let i = 0; i < len; i++) {
                for (let j = i + 1; j < len; j++) {
                  const iValid = !inp.vp || inp.vp[i] !== 0;
                  const jValid = !inp.vp || inp.vp[j] !== 0;
                  if (iValid === jValid && (data[i] === data[j] || (!iValid && !jValid))) {
                    expect(hashView[i]).toBe(hashView[j]);
                  }
                }
              }
              ctx.viewOf.forget({ ptr: hashPtr, length: len, dtype: 'i64' });
            } finally {
              freeAligned(ctx, av);
              ctx.mod.free(hashPtr);
            }
          } finally {
            freeColumn(ctx, col);
          }
        } else {
          const vals = applyVpI32(inp.data as number[], inp.vp);
          const col = createColumn(ctx, 'date32', vals);
          try {
            const len = col.length;
            const hashPtr = ctx.mod.alloc(Math.max(len * 8, 1));
            const av = alignValidity(ctx, col);
            try {
              rawKernel(wasm, 'hash_i32')(col.dataPtr, av.ptr, hashPtr, len);
              const hashView = ctx.viewOf({ ptr: hashPtr, length: len, dtype: 'i64' }) as BigInt64Array;
              const data = inp.data as number[];
              for (let i = 0; i < len; i++) {
                for (let j = i + 1; j < len; j++) {
                  const iValid = !inp.vp || inp.vp[i] !== 0;
                  const jValid = !inp.vp || inp.vp[j] !== 0;
                  if (iValid === jValid && (data[i] === data[j] || (!iValid && !jValid))) {
                    expect(hashView[i]).toBe(hashView[j]);
                  }
                }
              }
              ctx.viewOf.forget({ ptr: hashPtr, length: len, dtype: 'i64' });
            } finally {
              freeAligned(ctx, av);
              ctx.mod.free(hashPtr);
            }
          } finally {
            freeColumn(ctx, col);
          }
        }
      }
    });
  }
});
