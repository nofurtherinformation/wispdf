/**
 * Naive JS oracle for the public DataFrame API.
 * Used by fuzz tests to verify library output against a pure-JS reference.
 * Implements the same null / NaN / infinity semantics as the library spec.
 */

export type Cell = number | string | boolean | null;
export type ORow = Record<string, Cell>;
export type OFrame = ORow[];

// ── Frame creation ────────────────────────────────────────────────────────────

export function makeOracleFrame(
  a: (number | null)[],
  b: (number | null)[],
  g: (string | null)[],
  flag: (boolean | null)[],
): OFrame {
  const n = a.length;
  return Array.from({ length: n }, (_, i) => ({
    a: a[i] ?? null,
    b: b[i] ?? null,
    g: g[i] ?? null,
    flag: flag[i] ?? null,
  }));
}

// ── Schema-preserving ops ─────────────────────────────────────────────────────

/** Filter: keep rows where col > val (null or NaN → false, filtered out). */
export function oracleFilterGt(frame: OFrame, col: string, val: number): OFrame {
  return frame.filter((r) => {
    const v = r[col];
    return typeof v === 'number' && !isNaN(v) && v > val;
  });
}

/** Filter: keep rows where col < val. */
export function oracleFilterLt(frame: OFrame, col: string, val: number): OFrame {
  return frame.filter((r) => {
    const v = r[col];
    return typeof v === 'number' && !isNaN(v) && v < val;
  });
}

/** Filter: keep rows where utf8 col equals val. */
export function oracleFilterEqStr(frame: OFrame, col: string, val: string): OFrame {
  return frame.filter((r) => r[col] === val);
}

/** withColumn 'a' = col('a') * lit(val) — f64 arithmetic (NaN propagates). */
export function oracleWithColumnMul(frame: OFrame, name: string, val: number): OFrame {
  return frame.map((r) => ({
    ...r,
    [name]: r[name] === null ? null : (r[name] as number) * val,
  }));
}

/** withColumn 'a' = col('a') + lit(val) — f64 arithmetic. */
export function oracleWithColumnAdd(frame: OFrame, name: string, val: number): OFrame {
  return frame.map((r) => ({
    ...r,
    [name]: r[name] === null ? null : (r[name] as number) + val,
  }));
}

/** withColumn 'b' = col('b') * lit(val) — i32 with wrapping. */
export function oracleWithColumnI32Mul(frame: OFrame, name: string, val: number): OFrame {
  return frame.map((r) => ({
    ...r,
    [name]: r[name] === null ? null : i32wrap((r[name] as number) * val),
  }));
}

/** withColumn 'b' = col('b') + lit(val) — i32 with wrapping. */
export function oracleWithColumnI32Add(frame: OFrame, name: string, val: number): OFrame {
  return frame.map((r) => ({
    ...r,
    [name]: r[name] === null ? null : i32wrap((r[name] as number) + val),
  }));
}

/** sortValues by `col` ascending, nulls last; stable. */
export function oracleSortAsc(frame: OFrame, col: string): OFrame {
  return frame.slice().sort((ra, rb) => {
    const a = ra[col];
    const b = rb[col];
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (typeof a === 'number' && isNaN(a) && typeof b === 'number' && isNaN(b)) return 0;
    if (typeof a === 'number' && isNaN(a)) return 1; // NaN sorts last like null
    if (typeof b === 'number' && isNaN(b)) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  });
}

/** sortValues by `col` descending, nulls last; stable. */
export function oracleSortDesc(frame: OFrame, col: string): OFrame {
  return frame.slice().sort((ra, rb) => {
    const a = ra[col];
    const b = rb[col];
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (typeof a === 'number' && isNaN(a) && typeof b === 'number' && isNaN(b)) return 0;
    if (typeof a === 'number' && isNaN(a)) return 1;
    if (typeof b === 'number' && isNaN(b)) return -1;
    if (a < b) return 1;
    if (a > b) return -1;
    return 0;
  });
}

/** head(n). */
export function oracleHead(frame: OFrame, n: number): OFrame {
  return frame.slice(0, n);
}

/** slice(i, j). */
export function oracleSlice(frame: OFrame, i: number, j: number): OFrame {
  return frame.slice(i, j);
}

/** select — keep only the named columns. */
export function oracleSelect(frame: OFrame, cols: string[]): OFrame {
  return frame.map((r) => {
    const out: ORow = {};
    for (const c of cols) out[c] = r[c] ?? null;
    return out;
  });
}

// ── Terminal agg ops ──────────────────────────────────────────────────────────

/** groupby('g').agg({a: fn}) — output rows sorted by g for deterministic comparison. */
export function oracleGroupbyAgg(
  frame: OFrame,
  key: string,
  aggCol: string,
  fn: 'sum' | 'mean' | 'count' | 'min' | 'max',
): ORow[] {
  // Collect all non-null values per group.  NaN is a VALID value (validity bitmap = 1),
  // NOT null — do not conflate NaN with null here.  See contracts/dtypes.md §4.
  const groups = new Map<Cell, number[]>();
  for (const r of frame) {
    const k = r[key] ?? null;
    const v = r[aggCol];
    if (!groups.has(k)) groups.set(k, []);
    // Only null / undefined means "null" in the bitmap sense; NaN is a valid, non-null value.
    if (v != null) {
      groups.get(k)!.push(v as number);
    }
  }
  const rows: ORow[] = [];
  for (const [k, vals] of groups) {
    let agg: number | null;
    switch (fn) {
      case 'sum':
        // NaN poisons sum (spec §4.3: "a valid NaN in the data → result NaN").
        agg = vals.reduce((s, x) => s + x, 0);
        break;
      case 'mean':
        // NaN poisons mean.
        agg = vals.length > 0 ? vals.reduce((s, x) => s + x, 0) / vals.length : null;
        break;
      case 'count':
        // count = # non-null entries; NaN is non-null (spec §4.4).
        agg = vals.length;
        break;
      case 'min': {
        // NaN is skipped (IEEE compare, spec §4.3); all-NaN → no non-NaN candidates → null.
        const nonNaN = vals.filter((x) => !Number.isNaN(x));
        agg = nonNaN.length > 0 ? Math.min(...nonNaN) : null;
        break;
      }
      case 'max': {
        const nonNaN = vals.filter((x) => !Number.isNaN(x));
        agg = nonNaN.length > 0 ? Math.max(...nonNaN) : null;
        break;
      }
    }
    rows.push({ [key]: k, [aggCol]: agg });
  }
  // Sort by key for deterministic comparison (null first)
  rows.sort((a, b) => {
    const ka = a[key];
    const kb = b[key];
    if (ka === null && kb === null) return 0;
    if (ka === null) return -1;
    if (kb === null) return 1;
    if (ka < kb) return -1;
    if (ka > kb) return 1;
    return 0;
  });
  return rows;
}

// ── Comparison helpers ────────────────────────────────────────────────────────

/** Compare two cells for equality, treating NaN as equal to NaN. */
export function cellEq(a: Cell, b: Cell): boolean {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  if (typeof a === 'number' && typeof b === 'number') {
    if (isNaN(a) && isNaN(b)) return true;
    return a === b;
  }
  return a === b;
}

/** Compare two frames column-by-column, row-by-row. Returns null on success, error string on failure. */
export function compareFrames(
  libCols: Record<string, Cell[]>,
  oracle: OFrame,
  cols: string[],
): string | null {
  const n = oracle.length;
  for (const c of cols) {
    const libCol = libCols[c];
    if (!libCol) return `column '${c}' missing from library output`;
    if (libCol.length !== n) return `column '${c}' length mismatch: lib=${libCol.length} oracle=${n}`;
    for (let i = 0; i < n; i++) {
      const lo = libCol[i] ?? null;
      const oo = oracle[i]?.[c] ?? null;
      if (!cellEq(lo, oo)) {
        return `row ${i} col '${c}': lib=${JSON.stringify(lo)} oracle=${JSON.stringify(oo)}`;
      }
    }
  }
  return null;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/** Truncate to signed 32-bit integer (WASM i32 wrapping). */
function i32wrap(n: number): number {
  return n | 0;
}
