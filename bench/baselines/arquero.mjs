/**
 * Baseline: Arquero.
 * Nulls: Arquero uses JS null natively in its column arrays.
 */
import * as aq from 'arquero';
const { op } = aq;

/**
 * Build an Arquero table from the raw numeric dataset.
 */
export function buildNumeric(raw) {
  const { n, a, b, c, aValid, bValid, cValid } = raw;

  // Build plain arrays for Arquero — it accepts null for missing values.
  const aArr = new Array(n);
  const bArr = new Array(n);
  const cArr = new Array(n);
  for (let i = 0; i < n; i++) {
    aArr[i] = aValid[i] ? a[i] : null;
    bArr[i] = bValid[i] ? b[i] : null;
    cArr[i] = cValid[i] ? c[i] : null;
  }
  return aq.table({ a: aArr, b: bArr, c: cArr });
}

/**
 * Build an Arquero table from the raw string dataset.
 */
export function buildString(raw) {
  const { n, a, b, c, aValid, bValid, cValid, gIndices, gDict } = raw;
  const aArr = new Array(n);
  const bArr = new Array(n);
  const cArr = new Array(n);
  const gArr = new Array(n);
  for (let i = 0; i < n; i++) {
    aArr[i] = aValid[i] ? a[i] : null;
    bArr[i] = bValid[i] ? b[i] : null;
    cArr[i] = cValid[i] ? c[i] : null;
    gArr[i] = gDict[gIndices[i]];
  }
  return aq.table({ a: aArr, b: bArr, c: cArr, g: gArr });
}

// ── ops ─────────────────────────────────────────────────────────────────────

/** Elementwise a+b → new table with column 'sum'. */
export function opAdd(table) {
  return table.derive({ sum: (d) => d.a + d.b });
}

/** filter a > 0.5. */
export function opFilter(table) {
  return table.filter((d) => d.a > 0.5);
}

/** Null-aware sum(a) — Arquero skips nulls by default. */
export function opSum(table) {
  return table.rollup({ s: op.sum('a') }).get('s', 0);
}

/** groupby(g).sum(a). */
export function opGroupbySum(table) {
  return table.groupby('g').rollup({ s: op.sum('a') });
}

/** End-to-end pipeline: filter(a>0.5) → groupby(g) → sum(a). */
export function opPipeline(table) {
  return table
    .filter((d) => d.a > 0.5)
    .groupby('g')
    .rollup({ s: op.sum('a') });
}
