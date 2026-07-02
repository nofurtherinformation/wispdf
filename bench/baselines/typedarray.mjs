/**
 * Baseline: hand-written loops over typed arrays.
 * This is THE reference the WASM kernels must beat 1.5×.
 *
 * Nulls: aValid[i] = 1 means valid; 0 means null.
 * For f64 columns we also set NaN in the data array on null slots,
 * but we check the validity mask (faster branch on Uint8Array).
 */

/**
 * Build the typed-array dataset from raw.
 * Returns the raw object directly — no conversion needed,
 * since makeNumericRaw / makeStringRaw already produce typed arrays.
 */
export function buildNumeric(raw) {
  return raw; // already typed arrays
}

export function buildString(raw) {
  return raw; // already typed arrays + gIndices + gDict
}

// ── ops ─────────────────────────────────────────────────────────────────────

/**
 * Elementwise a+b → Float64Array result.
 * Output is NaN where either input is null.
 */
export function opAdd(ds) {
  const { n, a, b, aValid, bValid } = ds;
  const out = new Float64Array(n);
  const outValid = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (aValid[i] & bValid[i]) {
      out[i] = a[i] + b[i];
      outValid[i] = 1;
    } else {
      out[i] = NaN;
      outValid[i] = 0;
    }
  }
  return out;
}

/** filter a > 0.5 → Int32Array of matching row indices. */
export function opFilter(ds) {
  const { n, a, aValid } = ds;
  // worst-case pre-allocate
  const indices = new Int32Array(n);
  let len = 0;
  for (let i = 0; i < n; i++) {
    if (aValid[i] && a[i] > 0.5) indices[len++] = i;
  }
  return indices.subarray(0, len);
}

/** Null-aware sum(a). Skips null slots via validity mask. */
export function opSum(ds) {
  const { n, a, aValid } = ds;
  let s = 0;
  for (let i = 0; i < n; i++) {
    if (aValid[i]) s += a[i];
  }
  return s;
}

/**
 * groupby(g).sum(a) on string dataset.
 * Uses gIndices (Int32Array) for group key — no string hashing.
 */
export function opGroupbySum(ds) {
  const { n, a, aValid, gIndices } = ds;
  // 10K unique groups
  const sums = new Float64Array(10_000);
  for (let i = 0; i < n; i++) {
    if (aValid[i]) sums[gIndices[i]] += a[i];
  }
  return sums;
}

/** End-to-end pipeline: filter(a>0.5) → groupby(g) → sum(a). */
export function opPipeline(ds) {
  const { n, a, aValid, gIndices } = ds;
  const sums = new Float64Array(10_000);
  for (let i = 0; i < n; i++) {
    if (aValid[i] && a[i] > 0.5) sums[gIndices[i]] += a[i];
  }
  return sums;
}
