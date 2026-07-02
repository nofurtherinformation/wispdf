/**
 * Expression layer (Phase 3, P3.1) — public surface.
 *
 * The DataFrame layer (P3.2) builds directly on this. The two entry points:
 *
 *   - `compile(expr, frame)`  → a {@link CompiledPlan}; `.execute()` runs it and
 *     returns either a `Column` (element-wise / cast / fillNull / boolean result) or
 *     a scalar (aggregation), plus kernel/alloc {@link ExecStats}.
 *   - `compileFilter(pred, frame)` → `.execute()` gives a {@link Selection} whose
 *     `.compact(col)` materialises the kept rows of a column (compare→filter fusion).
 *
 * `inferType(expr, schema)` type-checks without running (result dtype / errors), for
 * `withColumn` dtype wiring and early validation. Frames implement {@link FrameView}
 * (which extends {@link Schema}).
 */

// AST + builders
export {
  Expr,
  col,
  lit,
  toExpr,
  type ExprLike,
  type ExprNode,
  type ScalarValue,
  type ArithOp,
  type CompareOp,
  type BoolOp,
  type AggOp,
} from './ast.js';

// Type resolution
export {
  resolve,
  inferType,
  aggResult,
  schemaOf,
  type Schema,
  type TExpr,
} from './dtypes.js';

// Errors
export {
  ExprError,
  dtypeMismatch,
  unsupportedDtype,
  unsupportedCast,
  unknownColumn,
  badLiteral,
  nearest,
} from './errors.js';

// Frame contract + kernel dispatch
export { callKernel, type FrameView, type KernelWasm, type KernelFn } from './frame.js';

// Compiler + executor
export {
  compile,
  compileFilter,
  type CompiledPlan,
  type CompiledFilter,
  type PlanResult,
  type ResultKind,
  type ScalarResult,
  type Selection,
  type ExecStats,
} from './compile.js';
