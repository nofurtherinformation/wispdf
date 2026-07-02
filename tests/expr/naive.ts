/**
 * A naive JS evaluator over the **typed IR** (`TExpr`) — the oracle the compiled
 * plan is checked against. It is deliberately independent of the compiler's kernel
 * lowering: it interprets dtypes.md value/null/Kleene semantics directly in JS. Both
 * consume the same `resolve()` output, so this isolates the compiler's kernel +
 * validity + fusion logic (not its type inference, which unit tests cover).
 */

import type { TExpr } from '../../src/expr/dtypes.js';
import type { ArithOp, CompareOp, AggOp } from '../../src/expr/ast.js';
import type { DType } from '../../src/memory/dtype.js';

export type Cell = number | string | boolean | null;
export type JSFrame = Record<string, Cell[]>;

/** Evaluate a column-valued typed expression to a JS array. */
export function naiveColumn(t: TExpr, frame: JSFrame, n: number): Cell[] {
  switch (t.kind) {
    case 'col':
      return frame[t.name]!.slice();
    case 'lit':
      return new Array<Cell>(n).fill(t.value);
    case 'cast': {
      const a = naiveColumn(t.operand, frame, n);
      return a.map((v) => castCell(v, t.from, t.dtype));
    }
    case 'neg': {
      const a = naiveColumn(t.operand, frame, n);
      return a.map((v) => (v === null ? null : negCell(v as number, t.dtype)));
    }
    case 'arith': {
      const a = naiveColumn(t.left, frame, n);
      const b = naiveColumn(t.right, frame, n);
      return a.map((x, i) =>
        x === null || b[i] === null ? null : arithCell(t.op, x as number, b[i] as number, t.dtype),
      );
    }
    case 'compare': {
      const a = naiveColumn(t.left, frame, n);
      const b = naiveColumn(t.right, frame, n);
      return a.map((x, i) => (x === null || b[i] === null ? null : compareCell(t.op, x, b[i]!)));
    }
    case 'bool': {
      const a = naiveColumn(t.left, frame, n);
      const b = naiveColumn(t.right, frame, n);
      return a.map((x, i) => kleene(t.op, x as boolean | null, b[i] as boolean | null));
    }
    case 'not': {
      const a = naiveColumn(t.operand, frame, n);
      return a.map((v) => (v === null ? null : !(v as boolean)));
    }
    case 'isNull': {
      const a = naiveColumn(t.operand, frame, n);
      return a.map((v) => v === null);
    }
    case 'fillNull': {
      const a = naiveColumn(t.operand, frame, n);
      return a.map((v) => (v === null ? t.value : v));
    }
    case 'agg':
      return new Array<Cell>(n).fill(naiveScalar(t, frame, n));
  }
}

/** Evaluate an aggregation typed expression to a scalar cell. */
export function naiveScalar(t: Extract<TExpr, { kind: 'agg' }>, frame: JSFrame, n: number): Cell {
  const col = naiveColumn(t.operand, frame, n);
  const nonNull = col.filter((v) => v !== null) as (number | string)[];
  return aggCell(t.op, nonNull, t.operandDtype);
}

// ── per-element helpers ───────────────────────────────────────────────────────

function arithCell(op: ArithOp, a: number, b: number, dtype: DType): Cell {
  switch (dtype) {
    case 'f64':
      return op === 'add' ? a + b : op === 'sub' ? a - b : op === 'mul' ? a * b : op === 'div' ? a / b : a % b;
    case 'f32': {
      const r = op === 'add' ? a + b : op === 'sub' ? a - b : op === 'mul' ? a * b : op === 'div' ? a / b : a % b;
      return Math.fround(r);
    }
    case 'i32':
      switch (op) {
        case 'add': return (a + b) | 0;
        case 'sub': return (a - b) | 0;
        case 'mul': return Math.imul(a, b);
        case 'div': return b === 0 ? null : Math.trunc(a / b) | 0;
        case 'mod': return b === 0 ? null : (a % b) | 0;
      }
      break;
    case 'u32': {
      const ua = a >>> 0;
      const ub = b >>> 0;
      switch (op) {
        case 'add': return (ua + ub) >>> 0;
        case 'sub': return (ua - ub) >>> 0;
        case 'mul': return Math.imul(ua, ub) >>> 0;
        case 'div': return ub === 0 ? null : Math.trunc(ua / ub) >>> 0;
        case 'mod': return ub === 0 ? null : ua % ub >>> 0;
      }
      break;
    }
  }
  throw new Error(`arith on non-numeric dtype ${dtype}`);
}

function negCell(a: number, dtype: DType): number {
  switch (dtype) {
    case 'f64': return -a;
    case 'f32': return Math.fround(-a);
    case 'i32': return -a | 0;
    case 'u32': return -a >>> 0;
    default: throw new Error(`neg on ${dtype}`);
  }
}

function compareCell(op: CompareOp, a: Cell, b: Cell): boolean {
  switch (op) {
    case 'gt': return (a as number) > (b as number);
    case 'ge': return (a as number) >= (b as number);
    case 'lt': return (a as number) < (b as number);
    case 'le': return (a as number) <= (b as number);
    case 'eq': return a === b;
    case 'ne': return a !== b;
  }
}

function kleene(op: 'and' | 'or', a: boolean | null, b: boolean | null): boolean | null {
  if (op === 'and') {
    if (a === false || b === false) return false;
    if (a === null || b === null) return null;
    return true;
  }
  if (a === true || b === true) return true;
  if (a === null || b === null) return null;
  return false;
}

function castCell(v: Cell, from: DType, to: DType): Cell {
  if (v === null) return null;
  if (from === to) return v;
  if (to === 'bool') {
    if (typeof v === 'number' && Number.isNaN(v)) return null;
    return v !== 0 && v !== false;
  }
  const num = typeof v === 'boolean' ? (v ? 1 : 0) : (v as number);
  if (to === 'f64') return num;
  if (to === 'f32') return Math.fround(num);
  if (!Number.isFinite(num)) return null;
  const tr = Math.trunc(num);
  if (to === 'i32') return tr >= -2147483648 && tr <= 2147483647 ? tr : null;
  if (num < 0) return null; // u32: negative inputs → null (dtypes.md §2)
  return tr <= 4294967295 ? tr : null;
}

// ── aggregations (dtypes.md §4.3, skipna) ─────────────────────────────────────

function aggCell(op: AggOp, nonNull: (number | string)[], dtype: DType): Cell {
  const nums = nonNull as number[];
  switch (op) {
    case 'count':
      return nonNull.length;
    case 'nunique':
      return new Set(nonNull).size;
    case 'sum':
      return nums.reduce((s, x) => s + x, 0);
    case 'mean':
      return nums.length >= 1 ? nums.reduce((s, x) => s + x, 0) / nums.length : null;
    case 'min':
    case 'max': {
      if (nums.length < 1) return null;
      const finite = nums.filter((x) => !Number.isNaN(x));
      if (finite.length === 0) return NaN; // all-NaN
      return op === 'min' ? Math.min(...finite) : Math.max(...finite);
    }
    case 'std':
    case 'var': {
      if (nums.length < 2) return null;
      const mean = nums.reduce((s, x) => s + x, 0) / nums.length;
      const v = nums.reduce((s, x) => s + (x - mean) * (x - mean), 0) / (nums.length - 1);
      return op === 'var' ? v : Math.sqrt(v);
    }
    case 'first':
      return nonNull.length ? nonNull[0]! : null;
    case 'last':
      return nonNull.length ? nonNull[nonNull.length - 1]! : null;
  }
}

/** Value equality treating NaN===NaN and -0===0 (value, not bit, comparison). */
export function cellEq(a: Cell, b: Cell): boolean {
  if (typeof a === 'number' && typeof b === 'number') {
    if (Number.isNaN(a) && Number.isNaN(b)) return true;
    return a === b;
  }
  return a === b;
}

/** Approximate equality for float aggregations (accumulation order differs). */
export function cellClose(a: Cell, b: Cell, rel = 1e-9): boolean {
  if (a === null || b === null) return a === b;
  if (typeof a === 'number' && typeof b === 'number') {
    if (Number.isNaN(a) && Number.isNaN(b)) return true;
    if (a === b) return true;
    return Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b));
  }
  return a === b;
}
