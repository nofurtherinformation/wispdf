/**
 * Expression compiler + executor (Phase 3, P3.1 deliverables §3/§4).
 *
 * `compile(expr, frame)` type-checks (`./dtypes.ts`) then lowers the typed IR to an
 * ordered sequence of Phase-2 kernel calls over the frame's {@link Column}s, and runs
 * it. `compileFilter(predicate, frame)` produces a reusable {@link Selection} for the
 * `df.filter` path.
 *
 * ## Validity handling (dtypes.md §5)
 * Elementwise arithmetic/comparison kernels are data-only; null propagation is a
 * separate `validity_and` / unary copy. Integer `div`/`mod` additionally null out
 * zero-divisor rows (dtypes.md §3.2) via a `divisor != 0` mask AND'd into validity —
 * the data-only kernels cannot express that themselves.
 *
 * ## Fusion (spec P3.1)
 *  - **compare → filter**: a comparison lowers straight to a 1-bit mask that `filter`
 *    consumes; no bool column is materialised (see {@link compileFilter} /
 *    {@link Selection.compact}). One mask, one compaction per column.
 *  - **elementwise chains**: an op whose operand is an *owned* temp writes its output
 *    **in place** into that buffer, so a linear chain reuses ONE data buffer instead
 *    of allocating one per node (see {@link Runtime}).
 *
 * ## Buffer lifecycle
 * Every temporary is owned by the {@link Runtime}; the result's buffers are transferred
 * out (fully owned by the returned {@link Column}) and everything else is freed — no
 * leaks (verified against arena stats in the leak tests).
 */

import { DTYPES, type DType } from '../memory/dtype.js';
import type { MemoryContext } from '../memory/context.js';
import type { Column } from '../memory/column.js';
import type { DtComponent } from './ast.js';
import { getBit, setBit, validityBytes } from '../memory/bitmap.js';
import {
  writeDictionary,
  decodeDictionary,
  decodeSlot,
  type Dictionary,
} from '../memory/dictionary.js';
import type { Cell } from '../memory/column.js';

import type { Expr, ArithOp, CompareOp, AggOp } from './ast.js';
import { resolve, type TExpr } from './dtypes.js';
import type { FrameView } from './frame.js';
import { ExprError } from './errors.js';
import {
  Runtime,
  statsOf,
  materializeValidity,
  popcountMask,
  freeValidity,
  ALL_VALID,
  type Validity,
  type ExecStats,
} from './runtime.js';
import { extractComponents } from '../temporal/vectorize.js';

export type { ExecStats } from './runtime.js';

// ── Public result types ───────────────────────────────────────────────────────

/** Whether a compiled expression yields a column or a single scalar (aggregation). */
export type ResultKind = 'column' | 'scalar';

/** A scalar (aggregation) result. */
export interface ScalarResult {
  readonly value: Cell;
  readonly dtype: DType;
}

/** The outcome of {@link CompiledPlan.execute}. */
export interface PlanResult {
  readonly kind: ResultKind;
  /** Present iff `kind === 'column'`. Caller owns it — free via `freeColumn`. */
  readonly column?: Column;
  /** Present iff `kind === 'scalar'`. */
  readonly scalar?: ScalarResult;
  /** Kernel-call + allocation counters for the run (plan inspection). */
  readonly stats: ExecStats;
}

/** A type-checked, ready-to-run expression. */
export interface CompiledPlan {
  /** Result dtype (`'bool'` for predicates; the scalar dtype for aggregations). */
  readonly dtype: DType;
  /** Whether {@link execute} yields a column or a scalar. */
  readonly resultKind: ResultKind;
  /** Run the plan over the bound frame, returning the result + stats. */
  execute(): PlanResult;
}

/**
 * Type-check `expr` against `frame` and bind it for execution. Throws
 * {@link ExprError} on any dtype/column error (before running anything).
 */
export function compile(expr: Expr, frame: FrameView): CompiledPlan {
  const t = resolve(expr, frame);
  const resultKind: ResultKind = t.kind === 'agg' ? 'scalar' : 'column';
  return {
    dtype: t.dtype,
    resultKind,
    execute(): PlanResult {
      const rt = new Runtime(frame);
      try {
        if (t.kind === 'agg') {
          const s = evalAgg(rt, frame, t);
          return { kind: 'scalar', scalar: s, stats: statsOf(rt.trace) };
        }
        const v = evalExpr(rt, frame, t);
        const column = finalizeColumn(rt, v);
        return { kind: 'column', column, stats: statsOf(rt.trace) };
      } finally {
        rt.freeAll();
      }
    },
  };
}

// ── Selection / filter ────────────────────────────────────────────────────────

/**
 * A compiled selection over a frame: the effective row mask plus a `compact` that
 * materialises the kept rows of any column. Fusion (a): a bare comparison predicate
 * lowers to ONE mask; each `compact` is ONE `filter` kernel — no bool column.
 */
export interface Selection {
  /** Number of selected rows. */
  readonly count: number;
  /** Compact `col` to the selected rows (caller owns the result — `freeColumn`). */
  compact(col: Column): Column;
  /** Release the mask + scratch. Call once after all `compact`s. */
  free(): void;
  /** Live kernel/alloc counters (accumulates across `compact` calls). */
  readonly stats: ExecStats;
}

/** A type-checked predicate ready to produce a {@link Selection}. */
export interface CompiledFilter {
  execute(): Selection;
}

/**
 * Type-check `predicate` (must be boolean) and bind it as a filter. The `df.filter`
 * path calls `execute()` once, then `compact`s each surviving column.
 */
export function compileFilter(predicate: Expr, frame: FrameView): CompiledFilter {
  const t = resolve(predicate, frame);
  if (t.dtype !== 'bool') {
    throw new ExprError(
      `filter predicate must be boolean, got ${t.dtype}. Use a comparison / boolean expression.`,
    );
  }
  return {
    execute(): Selection {
      const rt = new Runtime(frame);
      const mask = toMask(rt, evalExpr(rt, frame, t));
      // Effective selection = predicate-true AND predicate-valid (null → dropped, §4.5).
      let effPtr: number;
      if (mask.validity.ptr === 0) {
        effPtr = mask.maskPtr; // already owned by rt
      } else if (mask.ownsMask) {
        rt.call('validity_and', mask.maskPtr, mask.validity.ptr, mask.maskPtr, rt.len);
        freeValidity(rt, mask.validity);
        effPtr = mask.maskPtr;
      } else {
        const out = rt.alloc(rt.validityBytes, 'mask');
        rt.call('validity_and', mask.maskPtr, mask.validity.ptr, out, rt.len);
        freeValidity(rt, mask.validity);
        effPtr = out;
      }
      const count = popcountMask(rt, effPtr, rt.len);
      return makeSelection(rt, frame, effPtr, count);
    },
  };
}

function makeSelection(rt: Runtime, frame: FrameView, effPtr: number, count: number): Selection {
  let selIdx: Int32Array | null = null; // lazily built for null-column validity compaction
  const selectedIndices = (): Int32Array => {
    if (selIdx) return selIdx;
    const out = new Int32Array(count);
    const mask = rt.view(effPtr, validityBytes(rt.len), 'u8') as Uint8Array;
    let k = 0;
    for (let i = 0; i < rt.len; i++) if ((mask[i >> 3]! >> (i & 7)) & 1) out[k++] = i;
    selIdx = out;
    return out;
  };

  return {
    count,
    compact(col: Column): Column {
      const info = DTYPES[col.dtype];
      const outData = frame.ctx.mod.alloc(Math.max(count * info.size, 1));
      // One compaction per column: filter the storage buffer by the effective mask.
      const got = rt.call(`filter_${filterToken(col.dtype)}`, col.dataPtr, effPtr, outData, rt.len);
      if (got !== count) {
        throw new Error(`filter count mismatch: kernel ${got}, expected ${count}`);
      }
      let validityPtr = 0;
      if (col.validityPtr !== 0) {
        const idx = selectedIndices();
        const src = rt.view(
          col.validityPtr,
          validityBytes(col.validityBitOffset + rt.len),
          'u8',
        ) as Uint8Array;
        let nulls = 0;
        const vbytes = validityBytes(count);
        const outVp = frame.ctx.mod.alloc(Math.max(vbytes, 1));
        const dst = rt.view(outVp, vbytes, 'u8') as Uint8Array;
        dst.fill(0);
        for (let k = 0; k < count; k++) {
          if (getBit(src, col.validityBitOffset + idx[k]!)) setBit(dst, k);
          else nulls++;
        }
        if (nulls === 0) {
          frame.ctx.viewOf.forget({ ptr: outVp, length: vbytes, dtype: 'u8' });
          frame.ctx.mod.free(outVp);
        } else {
          validityPtr = outVp;
        }
      }
      const dict = col.dict ? copyDictionary(frame.ctx, col.dict) : null;
      return {
        dtype: col.dtype,
        length: count,
        dataPtr: outData,
        validityPtr,
        validityBitOffset: 0,
        dict,
        owned: true,
      };
    },
    free(): void {
      rt.freeAll();
    },
    get stats(): ExecStats {
      return statsOf(rt.trace);
    },
  };
}

// ── Intermediate value representations ────────────────────────────────────────

/** A materialised numeric / utf8 column value. */
interface ColVal {
  readonly rep: 'column';
  readonly dtype: DType;
  dataPtr: number;
  ownsData: boolean;
  validity: Validity;
  dict: Dictionary | null;
  ownsDict: boolean;
}
/** A boolean value as a 1-bit mask + null-propagation validity (compare output). */
interface MaskVal {
  readonly rep: 'mask';
  maskPtr: number;
  ownsMask: boolean;
  validity: Validity;
}
/** A boolean value as a u8 bool column + validity (Kleene form). */
interface BoolColVal {
  readonly rep: 'boolcol';
  dataPtr: number;
  ownsData: boolean;
  validity: Validity;
}
type Val = ColVal | MaskVal | BoolColVal;

const MIRROR: Record<CompareOp, CompareOp> = {
  gt: 'lt',
  lt: 'gt',
  ge: 'le',
  le: 'ge',
  eq: 'eq',
  ne: 'ne',
};
const COMMUTATIVE: ReadonlySet<ArithOp> = new Set<ArithOp>(['add', 'mul']);
/** Casts that can introduce range/NaN nulls (dtypes.md §2 ⚠ cells). */
const LOSSY_CAST: ReadonlySet<string> = new Set([
  'f64_i32', 'f64_u32', 'f64_bool',
  'f32_i32', 'f32_u32', 'f32_bool',
  'i32_u32', 'u32_i32',
  // v2: float→i64 is lossy (range-null when |x|≥2^63 or NaN); i64→* is lossless (ADR-009)
  'f64_i64', 'f32_i64',
]);

const MS_PER_DAY_N = 86_400_000n;

// ── Core evaluator ────────────────────────────────────────────────────────────

function evalExpr(rt: Runtime, frame: FrameView, t: TExpr): Val {
  switch (t.kind) {
    case 'col':
      return evalCol(rt, frame, t.name, t.dtype);
    case 'lit':
      return broadcastScalar(rt, t.value, t.dtype);
    case 'arith':
      return evalArith(rt, frame, t);
    case 'neg':
      return evalNeg(rt, frame, t.dtype, t.operand);
    case 'compare':
      return evalCompare(rt, frame, t);
    case 'bool':
      return evalKleene(rt, frame, t.op, t.left, t.right);
    case 'not':
      return evalNot(rt, frame, t.operand);
    case 'isNull':
      return evalIsNull(rt, frame, t.operand);
    case 'fillNull':
      return evalFillNull(rt, frame, t.dtype, t.operand, t.value);
    case 'cast':
      return evalCast(rt, frame, t.from, t.dtype, t.operand);
    case 'agg': {
      const s = evalAgg(rt, frame, t);
      return broadcastScalar(rt, s.value, t.dtype);
    }
    case 'dt':
      return evalDt(rt, frame, t.component, t.operand);
  }
}

function evalCol(rt: Runtime, frame: FrameView, name: string, dtype: DType): Val {
  const col = frame.getColumn(name);
  if (!col) throw new ExprError(`column '${name}' vanished during execution`);
  const validity = materializeValidity(rt, col.validityPtr, col.validityBitOffset);
  if (dtype === 'bool') {
    return { rep: 'boolcol', dataPtr: col.dataPtr, ownsData: false, validity };
  }
  return {
    rep: 'column',
    dtype,
    dataPtr: col.dataPtr,
    ownsData: false,
    validity,
    dict: col.dict,
    ownsDict: false,
  };
}

/** Evaluate a non-scalar operand expected to be a numeric column. */
function evalNumericColumn(rt: Runtime, frame: FrameView, t: TExpr): ColVal {
  const v = evalExpr(rt, frame, t);
  if (v.rep !== 'column') throw new Error(`internal: expected numeric column, got ${v.rep}`);
  return v;
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

function isScalarOperand(t: TExpr): boolean {
  return (
    t.kind === 'lit' ||
    t.kind === 'agg' ||
    (t.kind === 'cast' && isScalarOperand(t.operand))
  );
}

function evalArith(
  rt: Runtime,
  frame: FrameView,
  t: Extract<TExpr, { kind: 'arith' }>,
): ColVal {
  const dtype = t.dtype;
  const ls = isScalarOperand(t.left);
  const rs = isScalarOperand(t.right);
  if (ls && rs) {
    const lcol = broadcastScalar(rt, evalScalarCell(rt, frame, t.left), dtype) as ColVal;
    return arithScalar(rt, frame, t.op, lcol, evalScalarCell(rt, frame, t.right), dtype);
  }
  if (rs) {
    const col = evalNumericColumn(rt, frame, t.left);
    return arithScalar(rt, frame, t.op, col, evalScalarCell(rt, frame, t.right), dtype);
  }
  if (ls) {
    const col = evalNumericColumn(rt, frame, t.right);
    return arithScalarLeft(rt, frame, t.op, evalScalarCell(rt, frame, t.left), col, dtype);
  }
  const l = evalNumericColumn(rt, frame, t.left);
  const r = evalNumericColumn(rt, frame, t.right);
  return arithVec(rt, frame, t.op, l, r, dtype);
}

function arithScalar(
  rt: Runtime,
  frame: FrameView,
  op: ArithOp,
  col: ColVal,
  s: Cell,
  dtype: DType,
): ColVal {
  const size = DTYPES[dtype].size;
  const out = col.ownsData ? col.dataPtr : rt.alloc(rt.len * size, 'data');
  const integerDivZero =
    (op === 'div' || op === 'mod') &&
    ((dtype === 'i32' || dtype === 'u32') && s === 0 || dtype === 'i64' && s === 0n);

  if (s === null || integerDivZero) {
    // null scalar, or integer divide/mod by literal 0 → entire column is null.
    freeValidity(rt, col.validity);
    fillZeros(rt, out, rt.len * size);
    return colOf(dtype, out, allNull(rt), null, false);
  }
  // Physical kernel token (temporal types route to i32/i64 kernels, dtypes.md §7/ADR-010).
  const kd = DTYPES[dtype].wasm;
  if (kd === 'i64') {
    const sI64 = typeof s === 'bigint' ? s : BigInt(s as number);
    rt.callBigInt(`${op}_i64_scalar`, col.dataPtr, sI64, out, rt.len);
  } else {
    rt.call(`${op}_${kd}_scalar`, col.dataPtr, s as number, out, rt.len);
  }
  return colOf(dtype, out, col.validity, null, false);
}

function arithScalarLeft(
  rt: Runtime,
  frame: FrameView,
  op: ArithOp,
  s: Cell,
  col: ColVal,
  dtype: DType,
): ColVal {
  if (COMMUTATIVE.has(op)) return arithScalar(rt, frame, op, col, s, dtype);
  // Non-commutative with the scalar on the left: materialise it and go vector.
  const lcol = broadcastScalar(rt, s, dtype) as ColVal;
  return arithVec(rt, frame, op, lcol, col, dtype);
}

function arithVec(
  rt: Runtime,
  frame: FrameView,
  op: ArithOp,
  l: ColVal,
  r: ColVal,
  dtype: DType,
): ColVal {
  const size = DTYPES[dtype].size;
  const needZeroMask =
    (op === 'div' || op === 'mod') && (dtype === 'i32' || dtype === 'u32' || dtype === 'i64');

  // Compute the divisor≠0 mask BEFORE the data kernel may overwrite r's buffer.
  let zeroMask: Validity = ALL_VALID;
  if (needZeroMask) {
    const m = rt.alloc(rt.validityBytes, 'mask');
    if (dtype === 'i64') {
      rt.callBigInt('ne_i64_scalar_mask', r.dataPtr, 0n, m, rt.len);
    } else {
      rt.call(`ne_${dtype}_scalar_mask`, r.dataPtr, 0, m, rt.len);
    }
    zeroMask = { ptr: m, owns: true };
  }

  const out = l.ownsData ? l.dataPtr : r.ownsData ? r.dataPtr : rt.alloc(rt.len * size, 'data');
  const kd = DTYPES[dtype].wasm; // physical kernel token
  rt.call(`${op}_${kd}`, l.dataPtr, r.dataPtr, out, rt.len);
  if (out !== l.dataPtr && l.ownsData) rt.free(l.dataPtr);
  if (out !== r.dataPtr && r.ownsData) rt.free(r.dataPtr);

  let validity = combineValidity(rt, l.validity, r.validity);
  if (needZeroMask) validity = combineValidity(rt, validity, zeroMask);
  return colOf(dtype, out, validity, null, false);
}

function evalNeg(rt: Runtime, frame: FrameView, dtype: DType, operand: TExpr): ColVal {
  const v = evalNumericColumn(rt, frame, operand);
  const size = DTYPES[dtype].size;
  const out = v.ownsData ? v.dataPtr : rt.alloc(rt.len * size, 'data');
  rt.call(`neg_${dtype}`, v.dataPtr, out, rt.len);
  return colOf(dtype, out, v.validity, null, false);
}

// ── Comparisons ───────────────────────────────────────────────────────────────

function evalCompare(
  rt: Runtime,
  frame: FrameView,
  t: Extract<TExpr, { kind: 'compare' }>,
): Val {
  const d = t.operandDtype;
  const ls = isScalarOperand(t.left);
  const rs = isScalarOperand(t.right);

  // Both operands are scalars → constant predicate.
  if (ls && rs) {
    const a = evalScalarCell(rt, frame, t.left);
    const b = evalScalarCell(rt, frame, t.right);
    return broadcastPredicate(rt, applyCompareScalar(t.op, a, b));
  }

  if (d === 'bool') return evalBoolCompare(rt, frame, t, ls);
  if (d === 'utf8') return evalUtf8Compare(rt, frame, t, ls);

  // Numeric/temporal comparison → mask.
  // Physical kernel token: temporal types route to i32/i64 kernels (ADR-010).
  const kd = DTYPES[d].wasm;
  const maskPtr = rt.alloc(rt.validityBytes, 'mask');
  if (ls || rs) {
    const col = evalNumericColumn(rt, frame, ls ? t.right : t.left);
    const s = evalScalarCell(rt, frame, ls ? t.left : t.right);
    if (s === null) {
      freeValidity(rt, col.validity);
      rt.free(col.ownsData ? col.dataPtr : 0);
      const m = rt.view(maskPtr, rt.validityBytes, 'u8') as Uint8Array;
      m.fill(0);
      return { rep: 'mask', maskPtr, ownsMask: true, validity: allNull(rt) };
    }
    const op = ls ? MIRROR[t.op] : t.op; // mirror when the scalar is on the left
    if (kd === 'i64') {
      const sI64 = typeof s === 'bigint' ? s : BigInt(s as number);
      rt.callBigInt(`${op}_i64_scalar_mask`, col.dataPtr, sI64, maskPtr, rt.len);
    } else {
      rt.call(`${op}_${kd}_scalar_mask`, col.dataPtr, s as number, maskPtr, rt.len);
    }
    if (col.ownsData) rt.free(col.dataPtr);
    return { rep: 'mask', maskPtr, ownsMask: true, validity: col.validity };
  }
  const l = evalNumericColumn(rt, frame, t.left);
  const r = evalNumericColumn(rt, frame, t.right);
  rt.call(`${t.op}_${kd}_mask`, l.dataPtr, r.dataPtr, maskPtr, rt.len);
  if (l.ownsData) rt.free(l.dataPtr);
  if (r.ownsData) rt.free(r.dataPtr);
  return { rep: 'mask', maskPtr, ownsMask: true, validity: combineValidity(rt, l.validity, r.validity) };
}

function evalUtf8Compare(
  rt: Runtime,
  frame: FrameView,
  t: Extract<TExpr, { kind: 'compare' }>,
  scalarLeft: boolean,
): MaskVal {
  const col = evalNumericColumn(rt, frame, scalarLeft ? t.right : t.left);
  const s = evalScalarCell(rt, frame, scalarLeft ? t.left : t.right);
  const maskPtr = rt.alloc(rt.validityBytes, 'mask');
  const m = rt.view(maskPtr, rt.validityBytes, 'u8') as Uint8Array;

  if (s === null) {
    m.fill(0);
    if (col.ownsData) rt.free(col.dataPtr);
    freeValidity(rt, col.validity);
    if (col.ownsDict) freeDict(frame, col.dict);
    return { rep: 'mask', maskPtr, ownsMask: true, validity: allNull(rt) };
  }
  const strings = col.dict ? decodeDictionary(frame.ctx, col.dict) : [];
  const slot = strings.indexOf(s as string);
  if (slot >= 0) {
    rt.call(`${t.op}_i32_scalar_mask`, col.dataPtr, slot, maskPtr, rt.len);
  } else {
    // literal absent from the dictionary: eq → all-false, ne → all-true.
    m.fill(t.op === 'ne' ? 0xff : 0x00);
  }
  if (col.ownsData) rt.free(col.dataPtr);
  if (col.ownsDict) freeDict(frame, col.dict);
  return { rep: 'mask', maskPtr, ownsMask: true, validity: col.validity };
}

function evalBoolCompare(
  rt: Runtime,
  frame: FrameView,
  t: Extract<TExpr, { kind: 'compare' }>,
  scalarLeft: boolean,
): Val {
  const litVal = evalScalarCell(rt, frame, scalarLeft ? t.left : t.right) as boolean;
  const b = toBoolCol(rt, evalExpr(rt, frame, scalarLeft ? t.right : t.left));
  // eq(true)/ne(false) → identity; eq(false)/ne(true) → logical not.
  const invert = t.op === 'eq' ? litVal === false : litVal === true;
  return invert ? notBool(rt, b) : b;
}

// ── Boolean (Kleene) / not ────────────────────────────────────────────────────

function evalKleene(
  rt: Runtime,
  frame: FrameView,
  op: 'and' | 'or',
  leftT: TExpr,
  rightT: TExpr,
): BoolColVal {
  const a = toBoolCol(rt, evalExpr(rt, frame, leftT));
  const b = toBoolCol(rt, evalExpr(rt, frame, rightT));
  const kernel = op === 'and' ? 'and_kleene' : 'or_kleene';
  const out = a.ownsData ? a.dataPtr : b.ownsData ? b.dataPtr : rt.alloc(rt.len, 'data');
  const bothValid = a.validity.ptr === 0 && b.validity.ptr === 0;
  let validity: Validity;
  if (bothValid) {
    const scratch = rt.alloc(rt.validityBytes, 'scratch');
    rt.call(kernel, a.dataPtr, 0, b.dataPtr, 0, out, scratch, rt.len);
    rt.free(scratch);
    validity = ALL_VALID;
  } else {
    const outVp = rt.alloc(rt.validityBytes, 'validity');
    rt.call(kernel, a.dataPtr, a.validity.ptr, b.dataPtr, b.validity.ptr, out, outVp, rt.len);
    freeValidity(rt, a.validity);
    freeValidity(rt, b.validity);
    validity = { ptr: outVp, owns: true };
  }
  if (out !== a.dataPtr && a.ownsData) rt.free(a.dataPtr);
  if (out !== b.dataPtr && b.ownsData) rt.free(b.dataPtr);
  return { rep: 'boolcol', dataPtr: out, ownsData: true, validity };
}

function evalNot(rt: Runtime, frame: FrameView, operand: TExpr): BoolColVal {
  return notBool(rt, toBoolCol(rt, evalExpr(rt, frame, operand)));
}

function notBool(rt: Runtime, a: BoolColVal): BoolColVal {
  const out = a.ownsData ? a.dataPtr : rt.alloc(rt.len, 'data');
  let validity: Validity;
  if (a.validity.ptr === 0) {
    const scratch = rt.alloc(rt.validityBytes, 'scratch');
    rt.call('not_bool', a.dataPtr, 0, out, scratch, rt.len);
    rt.free(scratch);
    validity = ALL_VALID;
  } else {
    const outVp = rt.alloc(rt.validityBytes, 'validity');
    rt.call('not_bool', a.dataPtr, a.validity.ptr, out, outVp, rt.len);
    freeValidity(rt, a.validity);
    validity = { ptr: outVp, owns: true };
  }
  if (out !== a.dataPtr && a.ownsData) rt.free(a.dataPtr);
  return { rep: 'boolcol', dataPtr: out, ownsData: true, validity };
}

// ── isNull ────────────────────────────────────────────────────────────────────

function evalIsNull(rt: Runtime, frame: FrameView, operand: TExpr): BoolColVal {
  const v = evalExpr(rt, frame, operand);
  const validity = valueValidity(v);
  const out = rt.alloc(rt.len, 'data');
  rt.call('is_null', validity.ptr, out, rt.len);
  freeVal(rt, frame, v); // is_null's result has no nulls; the operand is done.
  return { rep: 'boolcol', dataPtr: out, ownsData: true, validity: ALL_VALID };
}

// ── fillNull ──────────────────────────────────────────────────────────────────

function evalFillNull(
  rt: Runtime,
  frame: FrameView,
  dtype: DType,
  operandT: TExpr,
  value: Cell,
): Val {
  if (dtype === 'bool') return fillNullBool(rt, frame, operandT, value as boolean);
  if (dtype === 'utf8') return fillNullUtf8(rt, frame, operandT, value as string);
  const col = evalNumericColumn(rt, frame, operandT);
  if (col.validity.ptr === 0) return col; // no nulls → identity
  const size = DTYPES[dtype].size;
  const out = col.ownsData ? col.dataPtr : rt.alloc(rt.len * size, 'data');
  const kd = DTYPES[dtype].wasm; // physical kernel token for temporal routing
  if (kd === 'i64') {
    const bigVal = typeof value === 'bigint' ? value : BigInt(value as number);
    rt.callBigInt('fill_null_i64', col.dataPtr, col.validity.ptr, bigVal, out, rt.len);
  } else {
    rt.call(`fill_null_${kd}`, col.dataPtr, col.validity.ptr, value as number, out, rt.len);
  }
  freeValidity(rt, col.validity);
  return colOf(dtype, out, ALL_VALID, null, false);
}

function fillNullBool(
  rt: Runtime,
  frame: FrameView,
  operandT: TExpr,
  value: boolean,
): BoolColVal {
  const b = toBoolCol(rt, evalExpr(rt, frame, operandT));
  if (b.validity.ptr === 0) return b;
  const out = b.ownsData ? b.dataPtr : rt.alloc(rt.len, 'data');
  const src = rt.view(b.dataPtr, rt.len, 'u8') as Uint8Array;
  const vp = rt.view(b.validity.ptr, rt.validityBytes, 'u8') as Uint8Array;
  const dst = rt.view(out, rt.len, 'u8') as Uint8Array;
  const fill = value ? 1 : 0;
  for (let i = 0; i < rt.len; i++) dst[i] = getBit(vp, i) ? src[i]! : fill;
  freeValidity(rt, b.validity);
  return { rep: 'boolcol', dataPtr: out, ownsData: true, validity: ALL_VALID };
}

function fillNullUtf8(
  rt: Runtime,
  frame: FrameView,
  operandT: TExpr,
  value: string,
): ColVal {
  const col = evalNumericColumn(rt, frame, operandT);
  if (col.validity.ptr === 0) return col;
  const strings = col.dict ? decodeDictionary(frame.ctx, col.dict) : [];
  let slot = strings.indexOf(value);
  let dict = col.dict;
  let ownsDict = col.ownsDict;
  if (slot < 0) {
    slot = strings.length;
    dict = writeDictionary(frame.ctx, [...strings, value]);
    rt.track(dict.offsetsPtr, (dict.count + 1) * 4, 'dict');
    rt.track(dict.bytesPtr, Math.max(dict.bytesLen, 1), 'dict');
    if (col.ownsDict) freeDict(frame, col.dict);
    ownsDict = true;
  }
  const out = col.ownsData ? col.dataPtr : rt.alloc(rt.len * 4, 'data');
  const srcIdx = rt.view(col.dataPtr, rt.len, 'i32') as Int32Array;
  const vp = rt.view(col.validity.ptr, rt.validityBytes, 'u8') as Uint8Array;
  const dstIdx = rt.view(out, rt.len, 'i32') as Int32Array;
  for (let i = 0; i < rt.len; i++) dstIdx[i] = getBit(vp, i) ? srcIdx[i]! : slot;
  freeValidity(rt, col.validity);
  return colOf('utf8', out, ALL_VALID, dict, ownsDict);
}

// ── cast ──────────────────────────────────────────────────────────────────────

function evalCast(
  rt: Runtime,
  frame: FrameView,
  from: DType,
  to: DType,
  operandT: TExpr,
): Val {
  const v = evalExpr(rt, frame, operandT);
  if (from === to) return v; // identity elided (dtypes.md §2)

  const inData = v.rep === 'boolcol' ? v.dataPtr : (v as ColVal).dataPtr;
  const inOwns = v.rep === 'boolcol' ? v.ownsData : (v as ColVal).ownsData;
  const inValidity = valueValidity(v);

  // Reinterpret casts (date32↔i32, timestamp↔i64): no kernel, pure dtype relabel (dtypes.md §7.2).
  if ((from === 'date32' && to === 'i32') || (from === 'i32' && to === 'date32') ||
      (from === 'timestamp' && to === 'i64') || (from === 'i64' && to === 'timestamp')) {
    return { ...(v as ColVal), dtype: to };
  }

  // Scale cast: date32 → timestamp (×86_400_000 ms, dtypes.md §7.2).
  if (from === 'date32' && to === 'timestamp') {
    // 1. cast_i32_i64 (lossless widening from i32 to i64)
    const tmpPtr = rt.alloc(rt.len * 8, 'data');
    const scratch = rt.alloc(rt.validityBytes, 'scratch');
    rt.call('cast_i32_i64', inData, inValidity.ptr, tmpPtr, scratch, rt.len);
    rt.free(scratch);
    // 2. mul_i64_scalar(86_400_000) — in-place on tmpPtr
    rt.callBigInt('mul_i64_scalar', tmpPtr, MS_PER_DAY_N, tmpPtr, rt.len);
    if (inOwns) rt.free(inData);
    return colOf('timestamp', tmpPtr, inValidity, null, false);
  }

  // Floor-div cast: timestamp → date32 (JS-side BigInt floor-div, dtypes.md §7.2).
  if (from === 'timestamp' && to === 'date32') {
    const msView = rt.view(inData, rt.len, 'i64') as BigInt64Array;
    const outPtr = rt.alloc(rt.len * 4, 'data');
    const outView = rt.view(outPtr, rt.len, 'i32') as Int32Array;
    for (let i = 0; i < rt.len; i++) {
      const ms = msView[i]!;
      let d = ms / MS_PER_DAY_N;
      if (ms % MS_PER_DAY_N !== 0n && ms < 0n) d -= 1n;
      outView[i] = Number(d);
    }
    if (inOwns) rt.free(inData);
    return colOf('date32', outPtr, inValidity, null, false);
  }

  const outSize = DTYPES[to].size;
  const canReuse = inOwns && DTYPES[from].size === outSize;
  const out = canReuse ? inData : rt.alloc(rt.len * outSize, 'data');
  const lossy = LOSSY_CAST.has(`${from}_${to}`);

  let validity: Validity;
  if (lossy) {
    const outVp = rt.alloc(rt.validityBytes, 'validity');
    rt.call(`cast_${from}_${to}`, inData, inValidity.ptr, out, outVp, rt.len);
    freeValidity(rt, inValidity);
    validity = { ptr: outVp, owns: true };
  } else {
    // Lossless: output validity equals input validity; run the kernel with a scratch.
    const scratch = rt.alloc(rt.validityBytes, 'scratch');
    rt.call(`cast_${from}_${to}`, inData, inValidity.ptr, out, scratch, rt.len);
    rt.free(scratch);
    validity = inValidity;
  }
  if (out !== inData && inOwns) rt.free(inData);
  if (to === 'bool') return { rep: 'boolcol', dataPtr: out, ownsData: true, validity };
  return colOf(to, out, validity, null, false);
}

// ── dt accessors ─────────────────────────────────────────────────────────────

/**
 * Evaluate a dt accessor node: extract a single calendar component from a
 * date32 or timestamp column.
 *
 * Implementation (ADR-010, ADR-012):
 *   - date32: Int32Array view of days-since-epoch → civil.date32ToFields via extractComponents.
 *   - timestamp (UTC): BigInt64Array view of epoch-ms → civil.timestampUtcToFields.
 *   - timestamp (tz): same view, but routed through tz.getTzComponents per valid row.
 *
 * tz metadata is read from the source column when the operand is a direct
 * column reference (the common path: `col('ts').dt.year()`). Cast expressions
 * do not carry tz metadata.
 *
 * Result dtype is always i32 (dtypes.md §10).
 */
function evalDt(rt: Runtime, frame: FrameView, c: DtComponent, t: TExpr): ColVal {
  const v = evalNumericColumn(rt, frame, t);
  const len = rt.len;

  // Resolve tz metadata: only available when the operand is a direct col ref.
  let tz: string | undefined;
  if (t.kind === 'col') {
    const srcCol = frame.getColumn(t.name);
    if (srcCol?.tz) tz = srcCol.tz;
  }

  // Get the validity bitmap (null = all-valid, matches extractComponents contract).
  let validityBitmap: Uint8Array | null = null;
  if (v.validity.ptr !== 0) {
    validityBitmap = rt.view(v.validity.ptr, rt.validityBytes, 'u8') as Uint8Array;
  }

  // View the source data as the appropriate typed array.
  let extractResult: { data: Int32Array };
  if (v.dtype === 'date32') {
    const dataView = rt.view(v.dataPtr, len, 'i32') as Int32Array;
    extractResult = extractComponents(dataView, validityBitmap, c);
  } else {
    // timestamp: BigInt64Array (wasm='i64', 8 bytes per element)
    const dataView = rt.view(v.dataPtr, len, 'i64') as BigInt64Array;
    extractResult = extractComponents(dataView, validityBitmap, c, tz);
  }

  // Allocate a new wasm i32 buffer and copy the JS-side result into it.
  const outPtr = rt.alloc(len * 4, 'data');
  const outView = rt.view(outPtr, len, 'i32') as Int32Array;
  outView.set(extractResult.data);

  // Free the source data buffer if we own it (validity is forwarded as-is).
  if (v.ownsData) rt.free(v.dataPtr);
  if (v.ownsDict && v.dict) freeDict(frame, v.dict);

  return colOf('i32', outPtr, v.validity, null, false);
}

// ── Aggregations ──────────────────────────────────────────────────────────────

function evalAgg(
  rt: Runtime,
  frame: FrameView,
  t: Extract<TExpr, { kind: 'agg' }>,
): ScalarResult {
  const v = evalExpr(rt, frame, t.operand);
  const validity = valueValidity(v);
  const dataPtr = v.rep === 'mask' ? 0 : v.rep === 'boolcol' ? v.dataPtr : v.dataPtr;
  const dict = v.rep === 'column' ? v.dict : null;
  const opD = t.operandDtype;
  const storage: DType = opD === 'utf8' ? 'i32' : opD; // utf8 reduces over its i32 indices
  const len = rt.len;
  const nonNull = validity.ptr === 0 ? len : rt.call('count_null', validity.ptr, len);

  const value = computeAgg(rt, frame, t.op, opD, storage, dataPtr, validity.ptr, len, nonNull, dict);
  freeVal(rt, frame, v);
  return { value, dtype: t.dtype };
}

function computeAgg(
  rt: Runtime,
  frame: FrameView,
  op: AggOp,
  opD: DType,
  storage: DType,
  dataPtr: number,
  vp: number,
  len: number,
  nonNull: number,
  dict: Dictionary | null,
): Cell {
  /**
   * Physical kernel token: temporal types route to i32/i64; utf8 routes to i32 (dict indices).
   * For non-utf8/non-temporal dtypes, pkd === opD (no change).
   */
  const pkd = opD === 'utf8' ? 'i32' : DTYPES[opD].wasm;
  const conv = (x: number): number => (storage === 'u32' ? x >>> 0 : x);
  switch (op) {
    case 'count':
      return nonNull;
    case 'nunique':
      return rt.call(`nunique_${pkd}_null`, dataPtr, vp, len);
    case 'sum':
      if (pkd === 'i64') return rt.callBigInt('sum_i64_null', dataPtr, vp, len);
      return rt.call(`sum_${opD}_null`, dataPtr, vp, len);
    case 'mean':
      return nonNull >= 1 ? rt.call(`mean_${opD}_null`, dataPtr, vp, len) : null;
    case 'std':
      return nonNull >= 2 ? rt.call(`std_${opD}_null`, dataPtr, vp, len) : null;
    case 'var':
      return nonNull >= 2 ? rt.call(`var_${opD}_null`, dataPtr, vp, len) : null;
    case 'min':
    case 'max':
      if (!nonNull) return null;
      if (pkd === 'i64') return rt.callBigInt(`${op}_i64_null`, dataPtr, vp, len);
      return conv(rt.call(`${op}_${pkd}_null`, dataPtr, vp, len));
    case 'first':
    case 'last': {
      const ovPtr = rt.alloc(4, 'scratch');
      (rt.view(ovPtr, 1, 'i32') as Int32Array)[0] = 0;
      let rawCell: number | bigint;
      if (pkd === 'i64') {
        rawCell = rt.callBigInt(`${op}_i64_null`, dataPtr, vp, len, ovPtr);
      } else {
        rawCell = rt.call(`${op}_${pkd}_null`, dataPtr, vp, len, ovPtr);
      }
      const ok = (rt.view(ovPtr, 1, 'i32') as Int32Array)[0] !== 0;
      rt.free(ovPtr);
      if (!ok) return null;
      if (pkd === 'i64') return rawCell; // bigint (i64 or timestamp)
      if (opD === 'utf8') return decodeSlot(frame.ctx, dict!, rawCell as number);
      return conv(rawCell as number);
    }
  }
}

// ── Representation coercions ──────────────────────────────────────────────────

function toBoolCol(rt: Runtime, v: Val): BoolColVal {
  if (v.rep === 'boolcol') return v;
  if (v.rep === 'mask') {
    const out = rt.alloc(rt.len, 'data');
    rt.call('expand_mask_bool', v.maskPtr, out, rt.len);
    if (v.ownsMask) rt.free(v.maskPtr);
    return { rep: 'boolcol', dataPtr: out, ownsData: true, validity: v.validity };
  }
  throw new Error('internal: expected a boolean value');
}

function toMask(rt: Runtime, v: Val): MaskVal {
  if (v.rep === 'mask') return v;
  if (v.rep === 'boolcol') {
    const out = rt.alloc(rt.validityBytes, 'mask');
    const data = rt.view(v.dataPtr, rt.len, 'u8') as Uint8Array;
    const m = rt.view(out, rt.validityBytes, 'u8') as Uint8Array;
    m.fill(0);
    for (let i = 0; i < rt.len; i++) if (data[i]) setBit(m, i);
    if (v.ownsData) rt.free(v.dataPtr);
    return { rep: 'mask', maskPtr: out, ownsMask: true, validity: v.validity };
  }
  throw new Error('internal: expected a boolean value');
}

function valueValidity(v: Val): Validity {
  return v.validity;
}

/**
 * Kernel dtype token for filter/gather storage.
 * Temporal types map to their physical kernel token (DTYPES[].wasm).
 */
function filterToken(dtype: DType): string {
  if (dtype === 'bool') return 'u8';
  if (dtype === 'utf8') return 'i32';
  return DTYPES[dtype].wasm; // temporals: date32→'i32', timestamp→'i64'
}

// ── Scalars ───────────────────────────────────────────────────────────────────

function evalScalarCell(rt: Runtime, frame: FrameView, t: TExpr): Cell {
  if (t.kind === 'lit') {
    // A number literal adopted as i64 (from safe-int check) needs BigInt conversion.
    if (t.dtype === 'i64' && typeof t.value === 'number') return BigInt(t.value as number);
    return t.value;
  }
  if (t.kind === 'agg') return evalAgg(rt, frame, t).value;
  if (t.kind === 'cast') return castScalar(evalScalarCell(rt, frame, t.operand), t.from, t.dtype);
  throw new Error(`internal: ${t.kind} is not a scalar operand`);
}

// 2^63 = 9223372036854775808 (exact in float64, the first out-of-range value for i64)
const TWO_63 = 9223372036854775808;

/** Apply a single-value cast (dtypes.md §2 + §8) for a scalar operand. */
function castScalar(value: Cell, from: DType, to: DType): Cell {
  if (value === null) return null;
  if (from === to) return value;
  // i64 source → cast to float/int/bool
  if (from === 'i64') {
    const bv = value as bigint;
    if (to === 'f64') return Number(bv);
    if (to === 'f32') return Math.fround(Number(bv));
    if (to === 'i32') return Number(BigInt.asIntN(32, bv));
    if (to === 'u32') return Number(BigInt.asUintN(32, bv));
    if (to === 'bool') return bv !== 0n;
    return null; // unreachable for valid cast matrix paths
  }
  // float/int/bool source → i64
  if (to === 'i64') {
    if (typeof value === 'boolean') return value ? 1n : 0n;
    const n = value as number;
    if (!Number.isFinite(n)) return null; // NaN/±Infinity → null (lossy cast)
    const tr = Math.trunc(n);
    if (tr >= TWO_63 || tr < -TWO_63) return null; // out of i64 range → null
    return BigInt(tr);
  }
  // existing non-i64 casts
  if (to === 'bool') {
    if (typeof value === 'number' && Number.isNaN(value)) return null;
    return value !== 0 && value !== false;
  }
  const num = typeof value === 'boolean' ? (value ? 1 : 0) : (value as number);
  if (to === 'f64') return num;
  if (to === 'f32') return Math.fround(num);
  // integer targets: truncate toward zero, range-fail → null
  if (!Number.isFinite(num)) return null;
  const tr = Math.trunc(num);
  if (to === 'i32') return tr >= -2147483648 && tr <= 2147483647 ? tr : null;
  if (num < 0) return null; // u32: negative inputs → null (dtypes.md §2)
  return tr <= 4294967295 ? tr : null;
}

function applyCompareScalar(op: CompareOp, a: Cell, b: Cell): boolean | null {
  if (a === null || b === null) return null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const aa = a as any, bb = b as any;
  switch (op) {
    case 'gt': return aa > bb;
    case 'ge': return aa >= bb;
    case 'lt': return aa < bb;
    case 'le': return aa <= bb;
    case 'eq': return aa === bb;
    case 'ne': return aa !== bb;
  }
}

// ── Broadcasting ──────────────────────────────────────────────────────────────

function broadcastScalar(rt: Runtime, value: Cell, dtype: DType): ColVal | BoolColVal {
  if (dtype === 'utf8') {
    throw new ExprError('broadcasting a utf8 scalar to a column is not supported in P3.1.');
  }
  if (dtype === 'bool') {
    const out = rt.alloc(rt.len, 'data');
    const view = rt.view(out, rt.len, 'u8') as Uint8Array;
    view.fill(value === null ? 0 : value ? 1 : 0);
    return { rep: 'boolcol', dataPtr: out, ownsData: true, validity: value === null ? allNull(rt) : ALL_VALID };
  }
  const size = DTYPES[dtype].size;
  const out = rt.alloc(rt.len * size, 'data');
  // i64 and timestamp both use BigInt64Array (wasm='i64').
  if (dtype === 'i64' || dtype === 'timestamp') {
    const view = rt.view(out, rt.len, 'i64') as BigInt64Array;
    const bigVal = value === null ? 0n : typeof value === 'bigint' ? value : BigInt(value as number);
    for (let i = 0; i < rt.len; i++) view[i] = bigVal;
    return colOf(dtype, out, value === null ? allNull(rt) : ALL_VALID, null, false);
  }
  // date32 uses Int32Array (wasm='i32'): value is a number (day count).
  const view = rt.view(out, rt.len, DTYPES[dtype].view);
  const num = value === null ? 0 : (value as number);
  for (let i = 0; i < rt.len; i++) (view as unknown as { [i: number]: number })[i] = num;
  return colOf(dtype, out, value === null ? allNull(rt) : ALL_VALID, null, false);
}

function broadcastPredicate(rt: Runtime, v: boolean | null): MaskVal {
  const out = rt.alloc(rt.validityBytes, 'mask');
  const m = rt.view(out, rt.validityBytes, 'u8') as Uint8Array;
  m.fill(v === true ? 0xff : 0x00);
  return { rep: 'mask', maskPtr: out, ownsMask: true, validity: v === null ? allNull(rt) : ALL_VALID };
}

// ── Validity combinators ──────────────────────────────────────────────────────

function combineValidity(rt: Runtime, a: Validity, b: Validity): Validity {
  if (a.ptr === 0 && b.ptr === 0) return ALL_VALID;
  if (b.ptr === 0) return a;
  if (a.ptr === 0) return b;
  const out = rt.alloc(rt.validityBytes, 'validity');
  rt.call('validity_and', a.ptr, b.ptr, out, rt.len);
  freeValidity(rt, a);
  freeValidity(rt, b);
  return { ptr: out, owns: true };
}

function allNull(rt: Runtime): Validity {
  const ptr = rt.alloc(rt.validityBytes, 'validity');
  (rt.view(ptr, rt.validityBytes, 'u8') as Uint8Array).fill(0);
  return { ptr, owns: true };
}

function fillZeros(rt: Runtime, ptr: number, bytes: number): void {
  (rt.view(ptr, bytes, 'u8') as Uint8Array).fill(0);
}

// ── Value construction / teardown ─────────────────────────────────────────────

function colOf(
  dtype: DType,
  dataPtr: number,
  validity: Validity,
  dict: Dictionary | null,
  ownsDict: boolean,
): ColVal {
  return { rep: 'column', dtype, dataPtr, ownsData: true, validity, dict, ownsDict };
}

/** Free every runtime-owned buffer a value holds (used when a value is discarded). */
function freeVal(rt: Runtime, frame: FrameView, v: Val): void {
  if (v.rep === 'mask') {
    if (v.ownsMask) rt.free(v.maskPtr);
    freeValidity(rt, v.validity);
    return;
  }
  if (v.rep === 'boolcol') {
    if (v.ownsData) rt.free(v.dataPtr);
    freeValidity(rt, v.validity);
    return;
  }
  if (v.ownsData) rt.free(v.dataPtr);
  freeValidity(rt, v.validity);
  if (v.ownsDict) freeDict(frame, v.dict);
}

function freeDict(frame: FrameView, dict: Dictionary | null): void {
  if (!dict) return;
  frame.ctx.viewOf.forget({ ptr: dict.offsetsPtr, length: dict.count + 1, dtype: 'i32' });
  frame.ctx.viewOf.forget({ ptr: dict.bytesPtr, length: dict.bytesLen, dtype: 'u8' });
  frame.ctx.mod.free(dict.offsetsPtr);
  frame.ctx.mod.free(dict.bytesPtr);
}

// ── Result materialisation (full ownership transfer) ──────────────────────────

/**
 * Turn a computed {@link Val} into a self-owning {@link Column}. Any buffer still
 * borrowed from a source column is copied so the returned column can be freed
 * independently (no aliasing with the frame's columns).
 */
function finalizeColumn(rt: Runtime, v: Val): Column {
  if (v.rep === 'mask') return finalizeColumn(rt, toBoolCol(rt, v));

  const ctx = rt.ctx;
  const dtype: DType = v.rep === 'boolcol' ? 'bool' : v.dtype;
  const info = DTYPES[dtype];

  // Data buffer: transfer if owned, else copy into an independent buffer.
  let outData: number;
  if (v.ownsData) {
    outData = v.dataPtr;
    rt.transfer(v.dataPtr);
  } else {
    outData = ctx.mod.alloc(Math.max(rt.len * info.size, 1));
    copyBytes(rt, v.dataPtr, outData, rt.len * info.size);
  }

  // Validity: transfer if owned, else copy (already bit-0 aligned by materializeValidity).
  const validity = v.validity;
  let validityPtr = 0;
  if (validity.ptr !== 0) {
    if (validity.owns) {
      validityPtr = validity.ptr;
      rt.transfer(validity.ptr);
    } else {
      validityPtr = ctx.mod.alloc(Math.max(rt.validityBytes, 1));
      copyBytes(rt, validity.ptr, validityPtr, rt.validityBytes);
    }
  }

  // Dictionary (utf8): transfer tracked temp, else copy the source's slots verbatim.
  let dict: Dictionary | null = null;
  if (v.rep === 'column' && v.dict) {
    if (v.ownsDict) {
      dict = v.dict;
      rt.transfer(v.dict.offsetsPtr);
      rt.transfer(v.dict.bytesPtr);
    } else {
      dict = copyDictionary(ctx, v.dict);
    }
  }

  return { dtype, length: rt.len, dataPtr: outData, validityPtr, validityBitOffset: 0, dict, owned: true };
}

function copyBytes(rt: Runtime, src: number, dst: number, bytes: number): void {
  if (bytes <= 0) return;
  const s = rt.view(src, bytes, 'u8') as Uint8Array;
  const d = rt.view(dst, bytes, 'u8') as Uint8Array;
  d.set(s);
}

/** Copy a dictionary's buffers verbatim (slot order preserved → indices stay valid). */
function copyDictionary(ctx: MemoryContext, dict: Dictionary): Dictionary {
  const offBytes = (dict.count + 1) * 4;
  const offsetsPtr = ctx.mod.alloc(Math.max(offBytes, 1));
  const bytesPtr = ctx.mod.alloc(Math.max(dict.bytesLen, 1));
  (ctx.viewOf({ ptr: offsetsPtr, length: dict.count + 1, dtype: 'i32' }) as Int32Array).set(
    ctx.viewOf({ ptr: dict.offsetsPtr, length: dict.count + 1, dtype: 'i32' }) as Int32Array,
  );
  if (dict.bytesLen > 0) {
    (ctx.viewOf({ ptr: bytesPtr, length: dict.bytesLen, dtype: 'u8' }) as Uint8Array).set(
      ctx.viewOf({ ptr: dict.bytesPtr, length: dict.bytesLen, dtype: 'u8' }) as Uint8Array,
    );
  }
  return { count: dict.count, offsetsPtr, bytesPtr, bytesLen: dict.bytesLen };
}
