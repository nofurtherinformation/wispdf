/**
 * Expression-layer errors (Phase 3, P3.1). Spec §4 ergonomics: dtype mismatches
 * name **both** dtypes and the operation; unknown column names suggest the nearest
 * match. All expression-time failures are {@link ExprError} so the frame layer can
 * catch and re-surface them uniformly.
 */

import type { DType } from '../memory/dtype.js';

/** Base class for every expression compile/eval error. */
export class ExprError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ExprError';
  }
}

/**
 * A binary op was handed two dtypes it cannot combine (dtypes.md §3.1: only
 * int→float widening is implicit). Message names both dtypes and the op.
 */
export function dtypeMismatch(op: string, left: DType, right: DType, hint?: string): ExprError {
  const tail = hint ? ` ${hint}` : '';
  return new ExprError(
    `dtype mismatch in '${op}': cannot combine ${left} and ${right}.${tail}`,
  );
}

/** An op was applied to a dtype it does not support at all (e.g. arithmetic on utf8). */
export function unsupportedDtype(op: string, dtype: DType, hint?: string): ExprError {
  const tail = hint ? ` ${hint}` : '';
  return new ExprError(`operation '${op}' is not supported for dtype ${dtype}.${tail}`);
}

/** A cast that is not in the v1 matrix (dtypes.md §2, the ✗ cells). */
export function unsupportedCast(from: DType, to: DType): ExprError {
  return new ExprError(
    `cast from ${from} to ${to} is not supported in v1` +
      (from === 'utf8' || to === 'utf8'
        ? ' (numeric↔utf8 conversion is not a kernel cast; use CSV inference / toString).'
        : '.'),
  );
}

/** Column `name` was not found; suggest the closest known column if any. */
export function unknownColumn(name: string, known: readonly string[]): ExprError {
  const near = nearest(name, known);
  const suffix = near ? ` Did you mean '${near}'?` : '';
  const list = known.length ? ` Known columns: ${known.map((k) => `'${k}'`).join(', ')}.` : '';
  return new ExprError(`unknown column '${name}'.${suffix}${list}`);
}

/** A literal value's JS type does not match the dtype it must adopt. */
export function badLiteral(value: unknown, dtype: DType, op: string): ExprError {
  return new ExprError(
    `literal ${JSON.stringify(value)} is not a valid ${dtype} value for '${op}'.`,
  );
}

// ── nearest-name suggestion (Levenshtein, capped) ─────────────────────────────

/** Closest known name to `name` within a small edit distance, else `null`. */
export function nearest(name: string, known: readonly string[]): string | null {
  let best: string | null = null;
  let bestD = Infinity;
  for (const k of known) {
    const d = levenshtein(name, k);
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  // Only suggest if the edit distance is small relative to the name length.
  const threshold = Math.max(2, Math.ceil(name.length / 3));
  return best !== null && bestD <= threshold ? best : null;
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = new Array<number>(n + 1);
  let curr = new Array<number>(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    const ai = a.charCodeAt(i - 1);
    for (let j = 1; j <= n; j++) {
      const cost = ai === b.charCodeAt(j - 1) ? 0 : 1;
      curr[j] = Math.min(prev[j]! + 1, curr[j - 1]! + 1, prev[j - 1]! + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n]!;
}
