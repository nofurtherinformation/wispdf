/**
 * Expression AST (Phase 3, P3.1 deliverable §1).
 *
 * The public expression surface from spec §4. `col('a')` / `lit(v)` are the two
 * leaf builders; every operator is a chainable method returning a **new**
 * {@link Expr}. Nodes are immutable (deep-frozen), so an `Expr` can be shared and
 * re-compiled without aliasing hazards.
 *
 *   col('a').gt(5).and(col('b').eq('x'))
 *   col('a').add(col('b')).mul(2)
 *   col('a').cast('f32').sum()
 *
 * Type resolution (result dtype, the single int→float widening rule, cast-insertion
 * points, unsupported-mix errors) lives in `./dtypes.ts`; lowering to kernel calls
 * lives in `./compile.ts`. This file is pure data + ergonomics — no wasm, no memory.
 */

import type { DType } from '../memory/dtype.js';

/** Binary arithmetic operators (dtypes.md §3.1/§3.2). */
export type ArithOp = 'add' | 'sub' | 'mul' | 'div' | 'mod';

/** dt accessor field names (dtypes.md §10, ADR-010). */
export type DtComponent =
  | 'year' | 'month' | 'day'
  | 'hour' | 'minute' | 'second' | 'millisecond'
  | 'weekday' | 'dayOfYear' | 'quarter';

/** str namespace op names (dtypes.md §13). v1 has only 'slice'. */
export type StrOp = 'slice';
/** Comparison operators → boolean/mask (dtypes.md §4.1). */
export type CompareOp = 'gt' | 'ge' | 'lt' | 'le' | 'eq' | 'ne';
/** Short-circuit-free three-valued boolean operators (dtypes.md §4.2). */
export type BoolOp = 'and' | 'or';
/** Reduction operators (dtypes.md §4.3). */
export type AggOp =
  | 'sum'
  | 'mean'
  | 'min'
  | 'max'
  | 'count'
  | 'nunique'
  | 'std'
  | 'var'
  | 'first'
  | 'last';

/** A raw JS scalar that a literal / fill value can hold. */
export type ScalarValue = number | bigint | string | boolean;

/** The immutable AST node inside every {@link Expr}. Discriminated on `kind`. */
export type ExprNode =
  | Readonly<{ kind: 'col'; name: string }>
  | Readonly<{ kind: 'lit'; value: ScalarValue; dtype: DType | null }>
  | Readonly<{ kind: 'arith'; op: ArithOp; left: Expr; right: Expr }>
  | Readonly<{ kind: 'neg'; operand: Expr }>
  | Readonly<{ kind: 'compare'; op: CompareOp; left: Expr; right: Expr }>
  | Readonly<{ kind: 'bool'; op: BoolOp; left: Expr; right: Expr }>
  | Readonly<{ kind: 'not'; operand: Expr }>
  | Readonly<{ kind: 'isNull'; operand: Expr }>
  | Readonly<{ kind: 'fillNull'; operand: Expr; value: ScalarValue }>
  | Readonly<{ kind: 'cast'; operand: Expr; to: DType }>
  | Readonly<{ kind: 'agg'; op: AggOp; operand: Expr }>
  /** dt accessor: extract a calendar field from a date32 or timestamp column. */
  | Readonly<{ kind: 'dt'; component: DtComponent; operand: Expr }>
  /**
   * str.slice: substring via JS String.prototype.slice semantics (dtypes.md §13).
   * Applied to dictionary values once (O(unique)), then indices remapped (O(rows)).
   */
  | Readonly<{ kind: 'strSlice'; operand: Expr; start: number; end: number | undefined }>;

/** Anything accepted where an expression operand is expected. Raw scalars wrap to `lit`. */
export type ExprLike = Expr | ScalarValue;

/** Wrap an {@link ExprLike} into an {@link Expr} (raw scalar → `lit`). */
export function toExpr(x: ExprLike): Expr {
  return x instanceof Expr ? x : lit(x);
}

/**
 * An immutable expression tree node with a chainable, pandas-familiar surface
 * (spec §4). Every method returns a new `Expr`; the receiver is never mutated.
 */
export class Expr {
  /** The frozen AST node this expression wraps. */
  readonly node: ExprNode;

  /** @internal — construct via {@link col}/{@link lit} or a chained method. */
  constructor(node: ExprNode) {
    this.node = Object.freeze(node);
    Object.freeze(this);
  }

  // ── arithmetic (dtypes.md §3.1/§3.2) ──────────────────────────────────────
  add(other: ExprLike): Expr {
    return arith('add', this, other);
  }
  sub(other: ExprLike): Expr {
    return arith('sub', this, other);
  }
  mul(other: ExprLike): Expr {
    return arith('mul', this, other);
  }
  div(other: ExprLike): Expr {
    return arith('div', this, other);
  }
  mod(other: ExprLike): Expr {
    return arith('mod', this, other);
  }
  neg(): Expr {
    return new Expr({ kind: 'neg', operand: this });
  }

  // ── comparison (dtypes.md §4.1) ───────────────────────────────────────────
  gt(other: ExprLike): Expr {
    return compare('gt', this, other);
  }
  ge(other: ExprLike): Expr {
    return compare('ge', this, other);
  }
  lt(other: ExprLike): Expr {
    return compare('lt', this, other);
  }
  le(other: ExprLike): Expr {
    return compare('le', this, other);
  }
  eq(other: ExprLike): Expr {
    return compare('eq', this, other);
  }
  ne(other: ExprLike): Expr {
    return compare('ne', this, other);
  }

  // ── boolean, three-valued Kleene (dtypes.md §4.2) ─────────────────────────
  and(other: ExprLike): Expr {
    return new Expr({ kind: 'bool', op: 'and', left: this, right: toExpr(other) });
  }
  or(other: ExprLike): Expr {
    return new Expr({ kind: 'bool', op: 'or', left: this, right: toExpr(other) });
  }
  not(): Expr {
    return new Expr({ kind: 'not', operand: this });
  }

  // ── null utilities (dtypes.md §4.5) ───────────────────────────────────────
  isNull(): Expr {
    return new Expr({ kind: 'isNull', operand: this });
  }
  /** `notNull` = `not(isNull)` (dtypes.md §4.5). */
  notNull(): Expr {
    return this.isNull().not();
  }
  fillNull(value: ScalarValue): Expr {
    return new Expr({ kind: 'fillNull', operand: this, value });
  }

  // ── cast (dtypes.md §2, explicit only) ────────────────────────────────────
  cast(to: DType): Expr {
    return new Expr({ kind: 'cast', operand: this, to });
  }

  // ── aggregations (dtypes.md §4.3) ─────────────────────────────────────────
  sum(): Expr {
    return agg('sum', this);
  }
  mean(): Expr {
    return agg('mean', this);
  }
  min(): Expr {
    return agg('min', this);
  }
  max(): Expr {
    return agg('max', this);
  }
  count(): Expr {
    return agg('count', this);
  }
  nunique(): Expr {
    return agg('nunique', this);
  }
  std(): Expr {
    return agg('std', this);
  }
  var(): Expr {
    return agg('var', this);
  }
  first(): Expr {
    return agg('first', this);
  }
  last(): Expr {
    return agg('last', this);
  }

  /**
   * dt accessor namespace for date32 / timestamp columns (dtypes.md §10, ADR-010).
   * Returns a {@link DtProxy} with `.year()`, `.month()`, `.day()`, `.hour()`,
   * `.minute()`, `.second()`, `.millisecond()`, `.weekday()`, `.dayOfYear()`, `.quarter()`.
   * Each method returns an `i32` Expr.
   */
  get dt(): DtProxy {
    return new DtProxy(this);
  }

  /**
   * str namespace for `utf8` columns (dtypes.md §13).
   * Returns a {@link StrProxy} with `.slice(start, end?)`.
   * Throws a dtype error at compile time if the column is not `utf8`.
   */
  get str(): StrProxy {
    return new StrProxy(this);
  }

  /** Readable, unambiguous rendering for `console.log` / error messages. */
  toString(): string {
    return render(this.node);
  }
}

/**
 * dt accessor proxy returned by `Expr.dt`. Every method produces an `i32` Expr
 * extracting the named calendar field from the parent date32 / timestamp column.
 */
export class DtProxy {
  constructor(private readonly operand: Expr) {}
  year(): Expr { return dt('year', this.operand); }
  month(): Expr { return dt('month', this.operand); }
  day(): Expr { return dt('day', this.operand); }
  hour(): Expr { return dt('hour', this.operand); }
  minute(): Expr { return dt('minute', this.operand); }
  second(): Expr { return dt('second', this.operand); }
  millisecond(): Expr { return dt('millisecond', this.operand); }
  weekday(): Expr { return dt('weekday', this.operand); }
  dayOfYear(): Expr { return dt('dayOfYear', this.operand); }
  quarter(): Expr { return dt('quarter', this.operand); }
}

/**
 * str accessor proxy returned by `Expr.str`. Provides string operations over
 * `utf8` columns; dtypes.md §13. A dtype error is raised at compile time (not
 * here) if the parent expression is not `utf8`.
 */
export class StrProxy {
  constructor(private readonly operand: Expr) {}

  /**
   * Substring via `JS String.prototype.slice` semantics (dtypes.md §13).
   *
   * - Negative `start`/`end`: count from the end of the string.
   * - `end` omitted: slice to the end of the string.
   * - Out-of-range indices clamp (same as JS).
   * - Null rows propagate null.
   * - UTF-16 code-unit indexing (same as every JS string API).
   *   **Surrogate-pair caveat:** a supplementary character (emoji, CJK extension, etc.)
   *   occupies two code units; `slice` may split a surrogate pair, producing an
   *   unpaired surrogate (well-defined but unusual in JS, not valid UTF-8).
   *   If you need grapheme-cluster or codepoint semantics, pre-process with JS before
   *   loading into the frame.
   */
  slice(start: number, end?: number): Expr {
    return new Expr({ kind: 'strSlice', operand: this.operand, start, end });
  }
}

// ── leaf builders ───────────────────────────────────────────────────────────

/** Reference the frame column named `name`. */
export function col(name: string): Expr {
  return new Expr({ kind: 'col', name });
}

/**
 * A scalar literal. Its dtype is normally inferred from the operand it is combined
 * with (an integer numeric literal adopts an integer column's dtype, a fractional
 * one triggers int→float widening — see `./dtypes.ts`). Pass `dtype` to pin it.
 */
export function lit(value: ScalarValue, dtype?: DType): Expr {
  return new Expr({ kind: 'lit', value, dtype: dtype ?? null });
}

// ── internal constructors ─────────────────────────────────────────────────────

function arith(op: ArithOp, left: Expr, right: ExprLike): Expr {
  return new Expr({ kind: 'arith', op, left, right: toExpr(right) });
}

function compare(op: CompareOp, left: Expr, right: ExprLike): Expr {
  return new Expr({ kind: 'compare', op, left, right: toExpr(right) });
}

function agg(op: AggOp, operand: Expr): Expr {
  return new Expr({ kind: 'agg', op, operand });
}

function dt(component: DtComponent, operand: Expr): Expr {
  return new Expr({ kind: 'dt', component, operand });
}

// ── rendering ─────────────────────────────────────────────────────────────────

const ARITH_SYM: Record<ArithOp, string> = {
  add: '+',
  sub: '-',
  mul: '*',
  div: '/',
  mod: '%',
};

function renderScalar(v: ScalarValue): string {
  if (typeof v === 'string') return JSON.stringify(v);
  if (typeof v === 'bigint') return String(v); // no 'n' suffix per spec
  return String(v);
}

function render(node: ExprNode): string {
  switch (node.kind) {
    case 'col':
      return `col(${JSON.stringify(node.name)})`;
    case 'lit':
      return node.dtype
        ? `lit(${renderScalar(node.value)}, ${node.dtype})`
        : `lit(${renderScalar(node.value)})`;
    case 'arith':
      return `(${render(node.left.node)} ${ARITH_SYM[node.op]} ${render(node.right.node)})`;
    case 'neg':
      return `(-${render(node.operand.node)})`;
    case 'compare':
      return `${render(node.left.node)}.${node.op}(${render(node.right.node)})`;
    case 'bool':
      return `${render(node.left.node)}.${node.op}(${render(node.right.node)})`;
    case 'not':
      return `${render(node.operand.node)}.not()`;
    case 'isNull':
      return `${render(node.operand.node)}.isNull()`;
    case 'fillNull':
      return `${render(node.operand.node)}.fillNull(${renderScalar(node.value)})`;
    case 'cast':
      return `${render(node.operand.node)}.cast(${node.to})`;
    case 'agg':
      return `${render(node.operand.node)}.${node.op}()`;
    case 'dt':
      return `${render(node.operand.node)}.dt.${node.component}()`;
    case 'strSlice':
      return node.end === undefined
        ? `${render(node.operand.node)}.str.slice(${node.start})`
        : `${render(node.operand.node)}.str.slice(${node.start}, ${node.end})`;
  }
}
