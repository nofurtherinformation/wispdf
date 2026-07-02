/**
 * Baseline: array-of-objects (naive JS).
 * Nulls are represented as JS null in object properties.
 */

/**
 * Convert raw typed-array dataset to array-of-objects format.
 * Each row: { a: number|null, b: number|null, c: number|null }
 */
export function buildNumeric(raw) {
  const { n, a, b, c, aValid, bValid, cValid } = raw;
  const rows = new Array(n);
  for (let i = 0; i < n; i++) {
    rows[i] = {
      a: aValid[i] ? a[i] : null,
      b: bValid[i] ? b[i] : null,
      c: cValid[i] ? c[i] : null,
    };
  }
  return rows;
}

/**
 * Convert raw string dataset to array-of-objects format.
 * Each row: { a, b, c (nullable as above), g: string }
 */
export function buildString(raw) {
  const { n, a, b, c, aValid, bValid, cValid, gIndices, gDict } = raw;
  const rows = new Array(n);
  for (let i = 0; i < n; i++) {
    rows[i] = {
      a: aValid[i] ? a[i] : null,
      b: bValid[i] ? b[i] : null,
      c: cValid[i] ? c[i] : null,
      g: gDict[gIndices[i]],
    };
  }
  return rows;
}

// ── ops ─────────────────────────────────────────────────────────────────────

/** Elementwise a+b → new array, null if either is null. */
export function opAdd(rows) {
  const out = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    out[i] = r.a === null || r.b === null ? null : r.a + r.b;
  }
  return out;
}

/** filter a > 0.5 → new array of matching rows. */
export function opFilter(rows) {
  const out = [];
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].a !== null && rows[i].a > 0.5) out.push(rows[i]);
  }
  return out;
}

/** Null-aware sum(a). */
export function opSum(rows) {
  let s = 0;
  for (let i = 0; i < rows.length; i++) {
    const v = rows[i].a;
    if (v !== null) s += v;
  }
  return s;
}

/** groupby(g).sum(a) on string dataset rows. */
export function opGroupbySum(rows) {
  const map = new Map();
  for (let i = 0; i < rows.length; i++) {
    const { g, a } = rows[i];
    const cur = map.get(g);
    if (cur === undefined) {
      map.set(g, a === null ? 0 : a);
    } else {
      map.set(g, cur + (a === null ? 0 : a));
    }
  }
  return map;
}

/** End-to-end pipeline: filter(a>0.5) → groupby(g) → sum(a). */
export function opPipeline(rows) {
  const filtered = [];
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].a !== null && rows[i].a > 0.5) filtered.push(rows[i]);
  }
  return opGroupbySum(filtered);
}
