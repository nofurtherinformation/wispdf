/**
 * Baseline: Danfo.js (informational).
 * If danfojs fails to load, records a skip entry.
 */

let dfd;
let loadError;

try {
  // danfojs uses CJS exports; dynamic import handles both.
  const mod = await import('danfojs');
  dfd = mod.default ?? mod;
} catch (e) {
  loadError = e.message;
}

export const SKIP = loadError
  ? { status: 'skipped', reason: `import failed: ${loadError}` }
  : null;

/**
 * Build a Danfo DataFrame from the raw numeric dataset.
 * Danfo uses NaN for missing float values.
 */
export function buildNumeric(raw) {
  if (SKIP) return null;
  const { n, a, b, c, aValid, bValid, cValid } = raw;
  const aArr = new Array(n);
  const bArr = new Array(n);
  const cArr = new Array(n);
  for (let i = 0; i < n; i++) {
    aArr[i] = aValid[i] ? a[i] : NaN;
    bArr[i] = bValid[i] ? b[i] : NaN;
    cArr[i] = cValid[i] ? c[i] : NaN;
  }
  return new dfd.DataFrame({ a: aArr, b: bArr, c: cArr });
}

export function buildString(raw) {
  if (SKIP) return null;
  const { n, a, b, c, aValid, bValid, cValid, gIndices, gDict } = raw;
  const aArr = new Array(n);
  const bArr = new Array(n);
  const cArr = new Array(n);
  const gArr = new Array(n);
  for (let i = 0; i < n; i++) {
    aArr[i] = aValid[i] ? a[i] : NaN;
    bArr[i] = bValid[i] ? b[i] : NaN;
    cArr[i] = cValid[i] ? c[i] : NaN;
    gArr[i] = gDict[gIndices[i]];
  }
  return new dfd.DataFrame({ a: aArr, b: bArr, c: cArr, g: gArr });
}

// ── ops ─────────────────────────────────────────────────────────────────────

export function opAdd(df) {
  return df['a'].add(df['b']);
}

export function opFilter(df) {
  const mask = df['a'].gt(0.5);
  return df.loc({ rows: mask });
}

export function opSum(df) {
  return df['a'].sum();
}

export function opGroupbySum(df) {
  return df.groupby(['g']).agg({ a: 'sum' });
}

export function opPipeline(df) {
  const mask = df['a'].gt(0.5);
  const filtered = df.loc({ rows: mask });
  return filtered.groupby(['g']).agg({ a: 'sum' });
}
