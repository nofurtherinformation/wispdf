/**
 * JS-side dispatch stubs for the reduce kernel family (v2.3 i64).
 *
 * Pointer args are i32 byte-offsets into wasm linear memory.
 * sum/min/max/first/last return wasm i64 → JS BigInt.
 * mean/std/var return wasm f64 → JS number.
 * nunique returns wasm i32 → JS number.
 */

export interface ReduceWasm {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: (...args: any[]) => any;
  memory: WebAssembly.Memory;
}

// i64 reductions — data ptr, vp ptr, len (i32)
export const sum_i64_null     = (w: ReduceWasm, data: number, vp: number, len: number): bigint => w.sum_i64_null(data, vp, len) as bigint;
export const mean_i64_null    = (w: ReduceWasm, data: number, vp: number, len: number): number => w.mean_i64_null(data, vp, len) as number;
export const min_i64_null     = (w: ReduceWasm, data: number, vp: number, len: number): bigint => w.min_i64_null(data, vp, len) as bigint;
export const max_i64_null     = (w: ReduceWasm, data: number, vp: number, len: number): bigint => w.max_i64_null(data, vp, len) as bigint;
export const std_i64_null     = (w: ReduceWasm, data: number, vp: number, len: number): number => w.std_i64_null(data, vp, len) as number;
export const var_i64_null     = (w: ReduceWasm, data: number, vp: number, len: number): number => w.var_i64_null(data, vp, len) as number;
export const nunique_i64_null = (w: ReduceWasm, data: number, vp: number, len: number): number => w.nunique_i64_null(data, vp, len) as number;
/** first_i64: returns bigint; writes 0|1 validity into out_valid (i32 ptr). */
export const first_i64_null   = (w: ReduceWasm, data: number, vp: number, len: number, outValid: number): bigint => w.first_i64_null(data, vp, len, outValid) as bigint;
/** last_i64: returns bigint; writes 0|1 validity into out_valid (i32 ptr). */
export const last_i64_null    = (w: ReduceWasm, data: number, vp: number, len: number, outValid: number): bigint => w.last_i64_null(data, vp, len, outValid) as bigint;
