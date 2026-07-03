/**
 * JS-side typed dispatch stubs for the elementwise kernel family (Phase 2).
 *
 * These are thin wrappers that take typed JS arguments and call the flat C ABI
 * exports in the loaded WASM module (contracts/wasm-abi.md §9).  Null
 * propagation, bitmap packing, and buffer lifetime are the caller's
 * responsibility — these stubs perform no allocation.
 *
 * All pointer arguments are i32 byte-offsets into wasm linear memory.
 */

/** Minimum subset of the Phase-1 memory-core exports required by these stubs. */
export interface ElementwiseWasm {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: (...args: any[]) => any;
  memory: WebAssembly.Memory;
}

// ── f64 arithmetic ──────────────────────────────────────────────────────────

export const add_f64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.add_f64(a, b, out, len);
export const sub_f64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.sub_f64(a, b, out, len);
export const mul_f64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mul_f64(a, b, out, len);
export const div_f64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.div_f64(a, b, out, len);
export const mod_f64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mod_f64(a, b, out, len);

export const add_f64_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.add_f64_scalar(a, s, out, len);
export const sub_f64_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.sub_f64_scalar(a, s, out, len);
export const mul_f64_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.mul_f64_scalar(a, s, out, len);
export const div_f64_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.div_f64_scalar(a, s, out, len);
export const mod_f64_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.mod_f64_scalar(a, s, out, len);

export const neg_f64 = (w: ElementwiseWasm, a: number, out: number, len: number) => w.neg_f64(a, out, len);

// ── f32 arithmetic ──────────────────────────────────────────────────────────

export const add_f32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.add_f32(a, b, out, len);
export const sub_f32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.sub_f32(a, b, out, len);
export const mul_f32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mul_f32(a, b, out, len);
export const div_f32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.div_f32(a, b, out, len);
export const mod_f32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mod_f32(a, b, out, len);

export const add_f32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.add_f32_scalar(a, s, out, len);
export const sub_f32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.sub_f32_scalar(a, s, out, len);
export const mul_f32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.mul_f32_scalar(a, s, out, len);
export const div_f32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.div_f32_scalar(a, s, out, len);
export const mod_f32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.mod_f32_scalar(a, s, out, len);

export const neg_f32 = (w: ElementwiseWasm, a: number, out: number, len: number) => w.neg_f32(a, out, len);

// ── i32 arithmetic ──────────────────────────────────────────────────────────

export const add_i32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.add_i32(a, b, out, len);
export const sub_i32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.sub_i32(a, b, out, len);
export const mul_i32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mul_i32(a, b, out, len);
export const div_i32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.div_i32(a, b, out, len);
export const mod_i32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mod_i32(a, b, out, len);

export const add_i32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.add_i32_scalar(a, s, out, len);
export const sub_i32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.sub_i32_scalar(a, s, out, len);
export const mul_i32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.mul_i32_scalar(a, s, out, len);
export const div_i32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.div_i32_scalar(a, s, out, len);
export const mod_i32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.mod_i32_scalar(a, s, out, len);

export const neg_i32 = (w: ElementwiseWasm, a: number, out: number, len: number) => w.neg_i32(a, out, len);

// ── u32 arithmetic ──────────────────────────────────────────────────────────

export const add_u32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.add_u32(a, b, out, len);
export const sub_u32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.sub_u32(a, b, out, len);
export const mul_u32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mul_u32(a, b, out, len);
export const div_u32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.div_u32(a, b, out, len);
export const mod_u32    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mod_u32(a, b, out, len);

export const add_u32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.add_u32_scalar(a, s, out, len);
export const sub_u32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.sub_u32_scalar(a, s, out, len);
export const mul_u32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.mul_u32_scalar(a, s, out, len);
export const div_u32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.div_u32_scalar(a, s, out, len);
export const mod_u32_scalar = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.mod_u32_scalar(a, s, out, len);

export const neg_u32 = (w: ElementwiseWasm, a: number, out: number, len: number) => w.neg_u32(a, out, len);

// ── Comparisons → Arrow-LSB bitmask ─────────────────────────────────────────

// f64
export const gt_f64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.gt_f64_mask(a, b, out, len);
export const ge_f64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ge_f64_mask(a, b, out, len);
export const lt_f64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.lt_f64_mask(a, b, out, len);
export const le_f64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.le_f64_mask(a, b, out, len);
export const eq_f64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.eq_f64_mask(a, b, out, len);
export const ne_f64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ne_f64_mask(a, b, out, len);

export const gt_f64_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.gt_f64_scalar_mask(a, s, out, len);
export const ge_f64_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.ge_f64_scalar_mask(a, s, out, len);
export const lt_f64_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.lt_f64_scalar_mask(a, s, out, len);
export const le_f64_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.le_f64_scalar_mask(a, s, out, len);
export const eq_f64_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.eq_f64_scalar_mask(a, s, out, len);
export const ne_f64_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.ne_f64_scalar_mask(a, s, out, len);

// f32
export const gt_f32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.gt_f32_mask(a, b, out, len);
export const ge_f32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ge_f32_mask(a, b, out, len);
export const lt_f32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.lt_f32_mask(a, b, out, len);
export const le_f32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.le_f32_mask(a, b, out, len);
export const eq_f32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.eq_f32_mask(a, b, out, len);
export const ne_f32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ne_f32_mask(a, b, out, len);

export const gt_f32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.gt_f32_scalar_mask(a, s, out, len);
export const ge_f32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.ge_f32_scalar_mask(a, s, out, len);
export const lt_f32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.lt_f32_scalar_mask(a, s, out, len);
export const le_f32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.le_f32_scalar_mask(a, s, out, len);
export const eq_f32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.eq_f32_scalar_mask(a, s, out, len);
export const ne_f32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.ne_f32_scalar_mask(a, s, out, len);

// i32 (signed)
export const gt_i32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.gt_i32_mask(a, b, out, len);
export const ge_i32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ge_i32_mask(a, b, out, len);
export const lt_i32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.lt_i32_mask(a, b, out, len);
export const le_i32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.le_i32_mask(a, b, out, len);
export const eq_i32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.eq_i32_mask(a, b, out, len);
export const ne_i32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ne_i32_mask(a, b, out, len);

export const gt_i32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.gt_i32_scalar_mask(a, s, out, len);
export const ge_i32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.ge_i32_scalar_mask(a, s, out, len);
export const lt_i32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.lt_i32_scalar_mask(a, s, out, len);
export const le_i32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.le_i32_scalar_mask(a, s, out, len);
export const eq_i32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.eq_i32_scalar_mask(a, s, out, len);
export const ne_i32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.ne_i32_scalar_mask(a, s, out, len);

// u32 (unsigned — no SIMD intrinsics, scalar loop)
export const gt_u32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.gt_u32_mask(a, b, out, len);
export const ge_u32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ge_u32_mask(a, b, out, len);
export const lt_u32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.lt_u32_mask(a, b, out, len);
export const le_u32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.le_u32_mask(a, b, out, len);
export const eq_u32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.eq_u32_mask(a, b, out, len);
export const ne_u32_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ne_u32_mask(a, b, out, len);

export const gt_u32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.gt_u32_scalar_mask(a, s, out, len);
export const ge_u32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.ge_u32_scalar_mask(a, s, out, len);
export const lt_u32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.lt_u32_scalar_mask(a, s, out, len);
export const le_u32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.le_u32_scalar_mask(a, s, out, len);
export const eq_u32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.eq_u32_scalar_mask(a, s, out, len);
export const ne_u32_scalar_mask = (w: ElementwiseWasm, a: number, s: number, out: number, len: number) => w.ne_u32_scalar_mask(a, s, out, len);

// ── Kleene three-valued logic ────────────────────────────────────────────────

/**
 * Kleene AND: and_kleene(a, a_vp, b, b_vp, out, out_vp, len)
 * a/b: per-element u8 bool (0|1); a_vp/b_vp: Arrow-LSB validity bitmap (0=null ptr → all-valid)
 */
export const and_kleene = (w: ElementwiseWasm, a: number, avp: number, b: number, bvp: number, out: number, outvp: number, len: number) =>
  w.and_kleene(a, avp, b, bvp, out, outvp, len);

/** Kleene OR (T dominates: T|N=T, F|N=N). */
export const or_kleene = (w: ElementwiseWasm, a: number, avp: number, b: number, bvp: number, out: number, outvp: number, len: number) =>
  w.or_kleene(a, avp, b, bvp, out, outvp, len);

/** Logical NOT of a bool column (not T=F, not F=T, not N=N). */
export const not_bool = (w: ElementwiseWasm, a: number, avp: number, out: number, outvp: number, len: number) =>
  w.not_bool(a, avp, out, outvp, len);

// ── Validity bitmap ops ──────────────────────────────────────────────────────

/** Pointwise AND of two Arrow-LSB validity bitmaps (a_vp, b_vp → out_vp). */
export const validity_and = (w: ElementwiseWasm, avp: number, bvp: number, outvp: number, len: number) =>
  w.validity_and(avp, bvp, outvp, len);

/** Pointwise OR of two Arrow-LSB validity bitmaps (a_vp, b_vp → out_vp). */
export const validity_or  = (w: ElementwiseWasm, avp: number, bvp: number, outvp: number, len: number) =>
  w.validity_or(avp, bvp, outvp, len);

// ── Casts ────────────────────────────────────────────────────────────────────

export const cast_f64_f32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f64_f32(i, ivp, o, ovp, len);
export const cast_f64_i32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f64_i32(i, ivp, o, ovp, len);
export const cast_f64_u32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f64_u32(i, ivp, o, ovp, len);
export const cast_f64_bool = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f64_bool(i, ivp, o, ovp, len);

export const cast_f32_f64  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f32_f64(i, ivp, o, ovp, len);
export const cast_f32_i32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f32_i32(i, ivp, o, ovp, len);
export const cast_f32_u32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f32_u32(i, ivp, o, ovp, len);
export const cast_f32_bool = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f32_bool(i, ivp, o, ovp, len);

export const cast_i32_f64  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i32_f64(i, ivp, o, ovp, len);
export const cast_i32_f32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i32_f32(i, ivp, o, ovp, len);
export const cast_i32_u32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i32_u32(i, ivp, o, ovp, len);
export const cast_i32_bool = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i32_bool(i, ivp, o, ovp, len);

export const cast_u32_f64  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_u32_f64(i, ivp, o, ovp, len);
export const cast_u32_f32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_u32_f32(i, ivp, o, ovp, len);
export const cast_u32_i32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_u32_i32(i, ivp, o, ovp, len);
export const cast_u32_bool = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_u32_bool(i, ivp, o, ovp, len);

export const cast_bool_f64 = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_bool_f64(i, ivp, o, ovp, len);
export const cast_bool_f32 = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_bool_f32(i, ivp, o, ovp, len);
export const cast_bool_i32 = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_bool_i32(i, ivp, o, ovp, len);
export const cast_bool_u32 = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_bool_u32(i, ivp, o, ovp, len);

// ── Null-aware utilities ─────────────────────────────────────────────────────

/** Replace nulls with a fill value. in_vp=0 → all-valid fast copy. */
export const fill_null_f64 = (w: ElementwiseWasm, inp: number, ivp: number, fill: number, out: number, len: number) => w.fill_null_f64(inp, ivp, fill, out, len);
export const fill_null_f32 = (w: ElementwiseWasm, inp: number, ivp: number, fill: number, out: number, len: number) => w.fill_null_f32(inp, ivp, fill, out, len);
export const fill_null_i32 = (w: ElementwiseWasm, inp: number, ivp: number, fill: number, out: number, len: number) => w.fill_null_i32(inp, ivp, fill, out, len);
export const fill_null_u32 = (w: ElementwiseWasm, inp: number, ivp: number, fill: number, out: number, len: number) => w.fill_null_u32(inp, ivp, fill, out, len);

/**
 * is_null: vp=0 → all outputs 0 (none are null).
 * Writes 1 per null element, 0 per valid element.
 */
export const is_null = (w: ElementwiseWasm, vp: number, out: number, len: number) => w.is_null(vp, out, len);

/** Expand an Arrow-LSB bitmask (1 bit/elem) to a u8 bool column (1 byte/elem). */
export const expand_mask_bool = (w: ElementwiseWasm, mask: number, out: number, len: number) => w.expand_mask_bool(mask, out, len);

// ── i64 arithmetic (v2.3) — scalars cross as wasm i64 → JS bigint ───────────

export const add_i64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.add_i64(a, b, out, len);
export const sub_i64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.sub_i64(a, b, out, len);
export const mul_i64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mul_i64(a, b, out, len);
export const div_i64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.div_i64(a, b, out, len);
export const mod_i64    = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.mod_i64(a, b, out, len);

export const add_i64_scalar = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.add_i64_scalar(a, s, out, len);
export const sub_i64_scalar = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.sub_i64_scalar(a, s, out, len);
export const mul_i64_scalar = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.mul_i64_scalar(a, s, out, len);
export const div_i64_scalar = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.div_i64_scalar(a, s, out, len);
export const mod_i64_scalar = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.mod_i64_scalar(a, s, out, len);

export const neg_i64 = (w: ElementwiseWasm, a: number, out: number, len: number) => w.neg_i64(a, out, len);

// i64 comparison masks (s is bigint for scalar variants)
export const gt_i64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.gt_i64_mask(a, b, out, len);
export const ge_i64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ge_i64_mask(a, b, out, len);
export const lt_i64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.lt_i64_mask(a, b, out, len);
export const le_i64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.le_i64_mask(a, b, out, len);
export const eq_i64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.eq_i64_mask(a, b, out, len);
export const ne_i64_mask = (w: ElementwiseWasm, a: number, b: number, out: number, len: number) => w.ne_i64_mask(a, b, out, len);

export const gt_i64_scalar_mask = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.gt_i64_scalar_mask(a, s, out, len);
export const ge_i64_scalar_mask = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.ge_i64_scalar_mask(a, s, out, len);
export const lt_i64_scalar_mask = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.lt_i64_scalar_mask(a, s, out, len);
export const le_i64_scalar_mask = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.le_i64_scalar_mask(a, s, out, len);
export const eq_i64_scalar_mask = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.eq_i64_scalar_mask(a, s, out, len);
export const ne_i64_scalar_mask = (w: ElementwiseWasm, a: number, s: bigint, out: number, len: number) => w.ne_i64_scalar_mask(a, s, out, len);

// i64 casts
export const cast_f64_i64  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f64_i64(i, ivp, o, ovp, len);
export const cast_f32_i64  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_f32_i64(i, ivp, o, ovp, len);
export const cast_i32_i64  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i32_i64(i, ivp, o, ovp, len);
export const cast_u32_i64  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_u32_i64(i, ivp, o, ovp, len);
export const cast_bool_i64 = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_bool_i64(i, ivp, o, ovp, len);
export const cast_i64_f64  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i64_f64(i, ivp, o, ovp, len);
export const cast_i64_f32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i64_f32(i, ivp, o, ovp, len);
export const cast_i64_i32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i64_i32(i, ivp, o, ovp, len);
export const cast_i64_u32  = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i64_u32(i, ivp, o, ovp, len);
export const cast_i64_bool = (w: ElementwiseWasm, i: number, ivp: number, o: number, ovp: number, len: number) => w.cast_i64_bool(i, ivp, o, ovp, len);

/** Replace nulls with a bigint fill value. in_vp=0 → all-valid fast copy. */
export const fill_null_i64 = (w: ElementwiseWasm, inp: number, ivp: number, fill: bigint, out: number, len: number) => w.fill_null_i64(inp, ivp, fill, out, len);
