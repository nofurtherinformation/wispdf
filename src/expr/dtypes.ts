/**
 * Type & validity resolution (Phase 3, P3.1 deliverable §2; dtypes.md §3.1 + §5).
 *
 * `resolve(expr, schema)` type-checks an {@link Expr} and lowers it to a **typed IR**
 * ({@link TExpr}) the compiler executes directly. It encodes:
 *
 *   - result-dtype inference (arith lattice dtypes.md §3.1, comparison/boolean → `bool`,
 *     aggregation result matrix dtypes.md §4.3),
 *   - the single implicit conversion — **int→float widening in mixed arithmetic** —
 *     materialised as an explicit {@link TCast} node so the compiler just emits a cast
 *     kernel (dtypes.md §5 "cast-insertion points"),
 *   - literal typing: a bare numeric literal adopts the dtype of the operand it is
 *     combined with; a *fractional* literal against an integer column triggers the same
 *     widening (documented extension of the §3.1 rule to literal operands),
 *   - unsupported-mix errors naming **both** dtypes and the op (spec §4 ergonomics).
 *
 * Identity casts survive as `TCast{from===to}`; the compiler elides the kernel
 * (dtypes.md §2). Range/validity nulling is a runtime concern handled by the cast /
 * div / mod kernels, not here.
 */

import type { DType } from '../memory/dtype.js';
import {
  type Expr,
  type ExprNode,
  type ArithOp,
  type CompareOp,
  type BoolOp,
  type AggOp,
  type DtComponent,
  type ScalarValue,
} from './ast.js';
import {
  dtypeMismatch,
  unsupportedDtype,
  unsupportedCast,
  unknownColumn,
  badLiteral,
  ExprError,
} from './errors.js';

// ── Schema ────────────────────────────────────────────────────────────────────

/** The dtype view of a frame the type checker needs (a subset of `FrameView`). */
export interface Schema {
  /** Dtype of column `name`, or `undefined` if absent. */
  dtypeOf(name: string): DType | undefined;
  /** All column names (for nearest-match suggestions). */
  columnNames(): readonly string[];
}

/** Build a {@link Schema} from a plain `name → dtype` record (handy in tests). */
export function schemaOf(record: Readonly<Record<string, DType>>): Schema {
  const names = Object.keys(record);
  return {
    dtypeOf: (name) => record[name],
    columnNames: () => names,
  };
}

// ── Typed IR ──────────────────────────────────────────────────────────────────

/** A type-resolved expression node. `dtype` is the node's result dtype. */
export type TExpr =
  | Readonly<{ kind: 'col'; name: string; dtype: DType }>
  | Readonly<{ kind: 'lit'; value: ScalarValue; dtype: DType }>
  | Readonly<{ kind: 'arith'; op: ArithOp; dtype: DType; left: TExpr; right: TExpr }>
  | Readonly<{ kind: 'neg'; dtype: DType; operand: TExpr }>
  | Readonly<{
      kind: 'compare';
      op: CompareOp;
      dtype: 'bool';
      operandDtype: DType;
      left: TExpr;
      right: TExpr;
    }>
  | Readonly<{ kind: 'bool'; op: BoolOp; dtype: 'bool'; left: TExpr; right: TExpr }>
  | Readonly<{ kind: 'not'; dtype: 'bool'; operand: TExpr }>
  | Readonly<{ kind: 'isNull'; dtype: 'bool'; operand: TExpr }>
  | Readonly<{ kind: 'fillNull'; dtype: DType; operand: TExpr; value: ScalarValue }>
  | Readonly<{ kind: 'cast'; dtype: DType; from: DType; operand: TExpr }>
  | Readonly<{
      kind: 'agg';
      op: AggOp;
      dtype: DType;
      operandDtype: DType;
      operand: TExpr;
    }>
  /** dt accessor: extract a calendar field (result dtype = 'i32', dtypes.md §10). */
  | Readonly<{ kind: 'dt'; component: DtComponent; dtype: 'i32'; operand: TExpr }>;

/** `true` for a value dtype that participates in numeric arithmetic (dtypes.md §3.2). */
const NUMERIC: ReadonlySet<DType> = new Set<DType>(['f64', 'f32', 'i32', 'u32', 'i64']);
const INTEGER: ReadonlySet<DType> = new Set<DType>(['i32', 'u32', 'i64']);
const FLOAT: ReadonlySet<DType> = new Set<DType>(['f64', 'f32']);
/** Logical temporal dtypes (ADR-010). NOT in NUMERIC; use restricted algebra (dtypes.md §9). */
const TEMPORAL: ReadonlySet<DType> = new Set<DType>(['date32', 'timestamp']);

function isNumeric(d: DType): boolean {
  return NUMERIC.has(d);
}

/** Is `v` representable in the integer dtype `d` (for literal adoption)? */
function intInRange(v: number, d: 'i32' | 'u32'): boolean {
  if (!Number.isInteger(v)) return false;
  return d === 'i32' ? v >= -2147483648 && v <= 2147483647 : v >= 0 && v <= 4294967295;
}

// ── Literal classification ────────────────────────────────────────────────────

/** A bare numeric literal with no pinned dtype — the only *adoptable* operand. */
function adoptableNumLit(e: Expr): number | null {
  const n = e.node;
  return n.kind === 'lit' && n.dtype === null && typeof n.value === 'number' ? n.value : null;
}

function litTExpr(value: ScalarValue, dtype: DType): TExpr {
  return { kind: 'lit', value, dtype };
}

/** Wrap `t` in a cast to `to` iff its dtype differs (used for widening). */
function widen(t: TExpr, to: DType): TExpr {
  return t.dtype === to ? t : { kind: 'cast', dtype: to, from: t.dtype, operand: t };
}

// ── Public entry ──────────────────────────────────────────────────────────────

/** Resolve `expr` against `schema`, returning the typed IR. Throws {@link ExprError}. */
export function resolve(expr: Expr, schema: Schema): TExpr {
  return resolveNode(expr, schema);
}

/** Convenience: the top-level result dtype of `expr` (`'bool'` for predicates). */
export function inferType(expr: Expr, schema: Schema): DType {
  return resolve(expr, schema).dtype;
}

function resolveNode(e: Expr, schema: Schema): TExpr {
  const node: ExprNode = e.node;
  switch (node.kind) {
    case 'col': {
      const dt = schema.dtypeOf(node.name);
      if (dt === undefined) throw unknownColumn(node.name, schema.columnNames());
      return { kind: 'col', name: node.name, dtype: dt };
    }
    case 'lit': {
      // A bare literal with no sibling context defaults: bigint→i64, int→i32, frac→f64.
      if (node.dtype !== null) return litTExpr(node.value, node.dtype);
      if (typeof node.value === 'bigint') return litTExpr(node.value, 'i64');
      if (typeof node.value === 'number') {
        return litTExpr(node.value, Number.isInteger(node.value) ? 'i32' : 'f64');
      }
      return litTExpr(node.value, typeof node.value === 'string' ? 'utf8' : 'bool');
    }
    case 'arith':
      return resolveArith(node.op, node.left, node.right, schema);
    case 'neg':
      return resolveNeg(node.operand, schema);
    case 'compare':
      return resolveCompare(node.op, node.left, node.right, schema);
    case 'bool':
      return resolveBool(node.op, node.left, node.right, schema);
    case 'not':
      return resolveNot(node.operand, schema);
    case 'isNull':
      return { kind: 'isNull', dtype: 'bool', operand: resolveNode(node.operand, schema) };
    case 'fillNull':
      return resolveFillNull(node.operand, node.value, schema);
    case 'cast':
      return resolveCast(node.operand, node.to, schema);
    case 'agg':
      return resolveAgg(node.op, node.operand, schema);
    case 'dt':
      return resolveDt(node.component, node.operand, schema);
  }
}

// ── Arithmetic (dtypes.md §3.1/§3.2) ──────────────────────────────────────────

function resolveArith(op: ArithOp, leftE: Expr, rightE: Expr, schema: Schema): TExpr {
  const ln = adoptableNumLit(leftE);
  const rn = adoptableNumLit(rightE);

  // Both bare numeric literals → default each, then apply the lattice with no casts.
  if (ln !== null && rn !== null) {
    const ld: DType = Number.isInteger(ln) ? 'i32' : 'f64';
    const rd: DType = Number.isInteger(rn) ? 'i32' : 'f64';
    const opDtype = ld === rd ? ld : 'f64'; // i32⊕f64 → f64
    return {
      kind: 'arith',
      op,
      dtype: opDtype,
      left: litTExpr(ln, opDtype),
      right: litTExpr(rn, opDtype),
    };
  }

  // One bare numeric literal + one typed operand.
  if (ln !== null || rn !== null) {
    const litVal = (ln ?? rn) as number;
    const typed = resolveNode(ln !== null ? rightE : leftE, schema);
    const { opDtype, colTarget } = arithLitVsTyped(op, litVal, typed.dtype);
    const typedT = widen(typed, colTarget);
    const litT = litTExpr(litVal, opDtype);
    const left = ln !== null ? litT : typedT;
    const right = ln !== null ? typedT : litT;
    return { kind: 'arith', op, dtype: opDtype, left, right };
  }

  // Two typed operands.
  const L = resolveNode(leftE, schema);
  const R = resolveNode(rightE, schema);
  const opDtype = arithCommon(op, L.dtype, R.dtype);
  return { kind: 'arith', op, dtype: opDtype, left: widen(L, opDtype), right: widen(R, opDtype) };
}

/** Operation dtype for a literal vs a typed operand; `colTarget` = cast for the column. */
function arithLitVsTyped(
  op: ArithOp,
  value: number,
  d: DType,
): { opDtype: DType; colTarget: DType } {
  // Temporal restricted algebra (dtypes.md §9): literal offset; validation untested.
  if (d === 'timestamp') return { opDtype: 'timestamp', colTarget: 'timestamp' };
  if (d === 'date32') return { opDtype: 'date32', colTarget: 'date32' };
  if (!isNumeric(d)) {
    throw unsupportedDtype(op, d, 'arithmetic');
  }
  if (FLOAT.has(d)) return { opDtype: d, colTarget: d }; // literal adopts the float dtype
  if (d === 'i64') {
    // number literal vs i64 column: adopt as i64 if safe-int, else widen both to f64.
    if (Number.isSafeInteger(value)) return { opDtype: 'i64', colTarget: 'i64' };
    return { opDtype: 'f64', colTarget: 'f64' };
  }
  // i32/u32 column: adopt if the literal is an in-range integer, else widen to f64.
  if (intInRange(value, d as 'i32' | 'u32')) return { opDtype: d, colTarget: d };
  return { opDtype: 'f64', colTarget: 'f64' }; // fractional / out-of-range → int→float
}

/** Common arithmetic dtype for two typed operands (dtypes.md §3.1 + v2 §8 + §9 temporal). */
function arithCommon(op: ArithOp, a: DType, b: DType): DType {
  // Temporal restricted algebra (dtypes.md §9): check before numeric rules.
  if (TEMPORAL.has(a) || TEMPORAL.has(b)) {
    return temporalArithResult(op, a, b);
  }
  if (!isNumeric(a)) throw unsupportedDtype(op, a, 'arithmetic');
  if (!isNumeric(b)) throw unsupportedDtype(op, b, 'arithmetic');
  if (a === b) return a;
  // v2 i64 lattice: i32/u32 ⊕ i64 → i64; i64 ⊕ f64 → f64; i64 ⊕ f32 → f64
  if ((a === 'i64' && (b === 'i32' || b === 'u32')) || (b === 'i64' && (a === 'i32' || a === 'u32'))) return 'i64';
  if ((a === 'i64' && FLOAT.has(b)) || (b === 'i64' && FLOAT.has(a))) return 'f64';
  // int → float widening (v1 rule, non-i64)
  if ((a === 'i32' || a === 'u32') && b === 'f64') return 'f64';
  if ((b === 'i32' || b === 'u32') && a === 'f64') return 'f64';
  if ((a === 'i32' || a === 'u32') && b === 'f32') return 'f32';
  if ((b === 'i32' || b === 'u32') && a === 'f32') return 'f32';
  throw dtypeMismatch(op, a, b, 'insert an explicit .cast() (only int→float widening is implicit).');
}

/**
 * Temporal restricted arithmetic (dtypes.md §9): only the listed pairs are legal.
 * Everything else throws a dtype error naming op + both dtypes.
 */
function temporalArithResult(op: ArithOp, a: DType, b: DType): DType {
  // timestamp − timestamp → i64 (ms duration)
  if (op === 'sub' && a === 'timestamp' && b === 'timestamp') return 'i64';
  // timestamp ± integer-ms-column → timestamp
  if ((op === 'add' || op === 'sub') && a === 'timestamp' && INTEGER.has(b)) return 'timestamp';
  // commutative add: integer + timestamp
  if (op === 'add' && INTEGER.has(a) && b === 'timestamp') return 'timestamp';
  // date32 − date32 → i32 (days duration)
  if (op === 'sub' && a === 'date32' && b === 'date32') return 'i32';
  // date32 ± integer-days-column → date32  (i32/u32 only; not i64 per §9)
  if ((op === 'add' || op === 'sub') && a === 'date32' && (b === 'i32' || b === 'u32')) return 'date32';
  if (op === 'add' && (a === 'i32' || a === 'u32') && b === 'date32') return 'date32';
  throw new ExprError(`${op}(${a},${b})`);
}

function resolveNeg(operandE: Expr, schema: Schema): TExpr {
  const t = resolveNode(operandE, schema);
  if (!isNumeric(t.dtype)) {
    throw unsupportedDtype('neg', t.dtype);
  }
  return { kind: 'neg', dtype: t.dtype, operand: t };
}

// ── dt accessor (dtypes.md §10) ─────────────────────────────────────────────────

function resolveDt(component: DtComponent, operandE: Expr, schema: Schema): TExpr {
  const t = resolveNode(operandE, schema);
  if (!TEMPORAL.has(t.dtype)) throw unsupportedDtype('dt', t.dtype);
  // Result is always i32 (field value: year, month, day, etc.; dtypes.md §10).
  return { kind: 'dt', component, dtype: 'i32', operand: t };
}

// ── Comparisons (dtypes.md §4.1; exact-match except literal coercion) ──────────

function resolveCompare(op: CompareOp, leftE: Expr, rightE: Expr, schema: Schema): TExpr {
  const isEqNe = op === 'eq' || op === 'ne';

  // String / bool literal vs a typed operand (utf8 / bool eq-ne).
  const strLit = stringLit(leftE) ?? stringLit(rightE);
  const boolLit = boolLitOf(leftE) ?? boolLitOf(rightE);

  const ln = adoptableNumLit(leftE);
  const rn = adoptableNumLit(rightE);

  // Both bare numeric literals → default each and compare in the common dtype.
  if (ln !== null && rn !== null) {
    const opDtype: DType = Number.isInteger(ln) && Number.isInteger(rn) ? 'i32' : 'f64';
    return cmp(op, opDtype, litTExpr(ln, opDtype), litTExpr(rn, opDtype));
  }

  // One bare numeric literal + one typed operand.
  if (ln !== null || rn !== null) {
    const litVal = (ln ?? rn) as number;
    const typed = resolveNode(ln !== null ? rightE : leftE, schema);
    // Temporal vs number literal: literal = physical unit (days for date32, ms for timestamp).
    if (typed.dtype === 'date32') {
      const litT = litTExpr(litVal, 'date32');
      return cmp(op, 'date32', ln !== null ? litT : typed, ln !== null ? typed : litT);
    }
    if (typed.dtype === 'timestamp') {
      const litT = litTExpr(litVal, 'timestamp');
      return cmp(op, 'timestamp', ln !== null ? litT : typed, ln !== null ? typed : litT);
    }
    if (!isNumeric(typed.dtype)) {
      throw dtypeMismatch(op, ln !== null ? 'f64' : typed.dtype, ln !== null ? typed.dtype : 'f64',
        'cannot compare a number to a non-numeric column.');
    }
    const opDtype = cmpLitVsTyped(litVal, typed.dtype);
    const typedT = widen(typed, opDtype);
    const litT = litTExpr(litVal, opDtype);
    return cmp(op, opDtype, ln !== null ? litT : typedT, ln !== null ? typedT : litT);
  }

  // utf8: eq/ne against a string literal (dict-index compare); ordering unsupported.
  if (strLit !== null) {
    const typed = resolveNode(stringLit(leftE) !== null ? rightE : leftE, schema);
    if (typed.dtype !== 'utf8') {
      throw dtypeMismatch(op, 'utf8', typed.dtype, 'cannot compare a string to a non-utf8 column.');
    }
    if (!isEqNe) throw unsupportedDtype(op, 'utf8', 'string ordering');
    const litT = litTExpr(strLit, 'utf8');
    return cmp(op, 'utf8', stringLit(leftE) !== null ? litT : typed, stringLit(leftE) !== null ? typed : litT);
  }

  // bool: eq/ne against a boolean literal (lowered to identity / not).
  if (boolLit !== null) {
    const typed = resolveNode(boolLitOf(leftE) !== null ? rightE : leftE, schema);
    if (typed.dtype !== 'bool') {
      throw dtypeMismatch(op, 'bool', typed.dtype, 'cannot compare a boolean to a non-bool column.');
    }
    if (!isEqNe) throw unsupportedDtype(op, 'bool');
    const litT = litTExpr(boolLit, 'bool');
    return cmp(op, 'bool', boolLitOf(leftE) !== null ? litT : typed, boolLitOf(leftE) !== null ? typed : litT);
  }

  // Two typed operands: dtypes must match exactly (no widening in comparisons).
  const L = resolveNode(leftE, schema);
  const R = resolveNode(rightE, schema);
  if (L.dtype !== R.dtype) {
    throw dtypeMismatch(op, L.dtype, R.dtype, 'cast');
  }
  const d = L.dtype;
  if (d === 'utf8' || d === 'bool') {
    // column-vs-column utf8/bool comparison needs dict unification / xor — not in P3.1.
    throw unsupportedDtype(op, d, 'column-vs-column');
  }
  // Temporal dtypes are allowed in column-vs-column comparisons (route to physical kernel).
  return cmp(op, d, L, R);
}

function cmp(op: CompareOp, operandDtype: DType, left: TExpr, right: TExpr): TExpr {
  return { kind: 'compare', op, dtype: 'bool', operandDtype, left, right };
}

/** Operand dtype for a numeric literal vs a typed numeric column in a comparison. */
function cmpLitVsTyped(value: number, d: DType): DType {
  if (FLOAT.has(d)) return d;
  if (d === 'i64') return Number.isSafeInteger(value) ? 'i64' : 'f64';
  // i32/u32 column: exact match if in-range integer, else literal-driven widen to f64.
  return intInRange(value, d as 'i32' | 'u32') ? d : 'f64';
}

function stringLit(e: Expr): string | null {
  const n = e.node;
  return n.kind === 'lit' && typeof n.value === 'string' ? n.value : null;
}
function boolLitOf(e: Expr): boolean | null {
  const n = e.node;
  return n.kind === 'lit' && typeof n.value === 'boolean' ? n.value : null;
}

// ── Boolean / not (dtypes.md §4.2) ────────────────────────────────────────────

function resolveBool(op: BoolOp, leftE: Expr, rightE: Expr, schema: Schema): TExpr {
  const L = resolveNode(leftE, schema);
  const R = resolveNode(rightE, schema);
  if (L.dtype !== 'bool') throw unsupportedDtype(op, L.dtype, 'boolean');
  if (R.dtype !== 'bool') throw unsupportedDtype(op, R.dtype, 'boolean');
  return { kind: 'bool', op, dtype: 'bool', left: L, right: R };
}

function resolveNot(operandE: Expr, schema: Schema): TExpr {
  const t = resolveNode(operandE, schema);
  if (t.dtype !== 'bool') throw unsupportedDtype('not', t.dtype, 'boolean');
  return { kind: 'not', dtype: 'bool', operand: t };
}

// ── fillNull / cast (dtypes.md §4.5, §2) ──────────────────────────────────────

function resolveFillNull(operandE: Expr, value: ScalarValue, schema: Schema): TExpr {
  const t = resolveNode(operandE, schema);
  validateFill(value, t.dtype);
  return { kind: 'fillNull', dtype: t.dtype, operand: t, value };
}

function validateFill(value: ScalarValue, d: DType): void {
  if (d === 'utf8') {
    if (typeof value !== 'string') throw badLiteral(value, d, 'fillNull');
    return;
  }
  if (d === 'bool') {
    if (typeof value !== 'boolean') throw badLiteral(value, d, 'fillNull');
    return;
  }
  // i64 and timestamp both use bigint at the boundary (dtypes.md §6/§11)
  if (d === 'i64' || d === 'timestamp') {
    if (typeof value !== 'bigint' && typeof value !== 'number') throw badLiteral(value, d, 'fillNull');
    if (typeof value === 'number' && !Number.isSafeInteger(value)) throw badLiteral(value, d, 'fillNull');
    return;
  }
  if (typeof value !== 'number') throw badLiteral(value, d, 'fillNull');
  if ((d === 'i32' || d === 'u32') && !intInRange(value, d)) throw badLiteral(value, d, 'fillNull');
}

/**
 * The v2 cast matrix (dtypes.md §2 + §8 + §7.2 temporal). `true` = allowed; missing = ✗.
 * Temporal reinterpret casts: date32↔i32, timestamp↔i64 (free, no kernel).
 * Scale casts: date32→timestamp (×86_400_000), timestamp→date32 (floor-div).
 */
const CAST_OK: Readonly<Record<DType, ReadonlySet<DType>>> = {
  f64: new Set(['f64', 'f32', 'i32', 'u32', 'bool', 'i64']),
  f32: new Set(['f64', 'f32', 'i32', 'u32', 'bool', 'i64']),
  // i32 + date32 reinterpret (dtypes.md §7.2)
  i32: new Set(['f64', 'f32', 'i32', 'u32', 'bool', 'i64', 'date32']),
  u32: new Set(['f64', 'f32', 'i32', 'u32', 'bool', 'i64']),
  bool: new Set(['f64', 'f32', 'i32', 'u32', 'bool', 'i64']),
  utf8: new Set(['utf8']),
  // i64 ↔ utf8 is ✗; i64 + timestamp reinterpret (dtypes.md §7.2)
  i64: new Set(['f64', 'f32', 'i32', 'u32', 'bool', 'i64', 'timestamp']),
  // date32: reinterpret to i32, scale to timestamp; identity
  date32: new Set(['i32', 'timestamp', 'date32']),
  // timestamp: reinterpret to i64, floor-div to date32; identity
  timestamp: new Set(['i64', 'date32', 'timestamp']),
};

function resolveCast(operandE: Expr, to: DType, schema: Schema): TExpr {
  const t = resolveNode(operandE, schema);
  const from = t.dtype;
  if (!CAST_OK[from]?.has(to)) throw unsupportedCast(from, to);
  return { kind: 'cast', dtype: to, from, operand: t };
}

// ── Aggregations (dtypes.md §4.3) ─────────────────────────────────────────────

function resolveAgg(op: AggOp, operandE: Expr, schema: Schema): TExpr {
  const t = resolveNode(operandE, schema);
  const dtype = aggResult(op, t.dtype);
  return { kind: 'agg', op, dtype, operandDtype: t.dtype, operand: t };
}

/** Result dtype of aggregation `op` over operand dtype `d`; throws if unsupported. */
export function aggResult(op: AggOp, d: DType): DType {
  switch (op) {
    case 'count':
      return 'i32'; // non-null count, any dtype (dtypes.md §4.4)
    case 'nunique':
      if (isNumeric(d) || d === 'utf8' || TEMPORAL.has(d)) return 'i32';
      throw unsupportedDtype('nunique', d);
    case 'sum':
      if (FLOAT.has(d)) return d; // f64→f64, f32→f32
      if (d === 'i64') return 'i64'; // i64 sum → i64 (bigint wrapping, ADR-009)
      if (d === 'i32' || d === 'u32') return 'f64'; // int sums widen to f64 (ABI §9)
      throw unsupportedDtype('sum', d);
    case 'mean':
    case 'std':
    case 'var':
      if (isNumeric(d)) return 'f64';
      throw unsupportedDtype(op, d);
    case 'min':
    case 'max':
      if (isNumeric(d) || TEMPORAL.has(d)) return d;
      throw unsupportedDtype(op, d);
    case 'first':
    case 'last':
      if (isNumeric(d) || d === 'utf8' || TEMPORAL.has(d)) return d;
      throw unsupportedDtype(op, d);
  }
}

export { ExprError };
