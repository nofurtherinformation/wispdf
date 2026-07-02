/**
 * Fixed-seed dataset generators for the benchmark harness.
 * Uses mulberry32 PRNG for reproducibility.
 *
 * Returns the "raw" format: typed arrays + validity masks.
 * Each baseline converts to its own representation.
 */

/** mulberry32 PRNG — returns a function that yields floats in [0, 1). */
export function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
  };
}

const NULL_RATE = 0.05; // ~5% nulls per column

/**
 * Numeric dataset: f64 cols a, b; i32 col c; ~5% nulls each.
 *
 * Raw format returned:
 *   { n, a: Float64Array, b: Float64Array, c: Int32Array,
 *     aValid: Uint8Array, bValid: Uint8Array, cValid: Uint8Array }
 * where valid[i] = 1 means the value is non-null.
 * Null slots: a[i] = NaN, b[i] = NaN, c[i] = 0.
 */
export function makeNumericRaw(n, seed = 0xdeadbeef) {
  const rng = mulberry32(seed);

  const a = new Float64Array(n);
  const b = new Float64Array(n);
  const c = new Int32Array(n);
  const aValid = new Uint8Array(n);
  const bValid = new Uint8Array(n);
  const cValid = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    const av = rng() >= NULL_RATE;
    const bv = rng() >= NULL_RATE;
    const cv = rng() >= NULL_RATE;

    aValid[i] = av ? 1 : 0;
    bValid[i] = bv ? 1 : 0;
    cValid[i] = cv ? 1 : 0;

    a[i] = av ? rng() : NaN;
    b[i] = bv ? rng() : NaN;
    c[i] = cv ? ((rng() * 1000) | 0) : 0;

    // consume rng slots even when null to keep positions stable
    if (!av) rng();
    if (!bv) rng();
    if (!cv) rng();
  }

  return { n, a, b, c, aValid, bValid, cValid };
}

/**
 * String dataset: same numeric cols a, b, c + utf8 col g with 10K unique strings.
 * g has no nulls (to keep groupby deterministic).
 *
 * gIndices: Int32Array of indices into gDict (0..9999)
 * gDict: string[] of length 10000
 */
export function makeStringRaw(n, seed = 0xcafebabe) {
  const base = makeNumericRaw(n, seed);
  const rng = mulberry32(seed ^ 0x1234);

  const G_UNIQUE = 10_000;
  const gDict = Array.from({ length: G_UNIQUE }, (_, i) =>
    'g' + String(i).padStart(5, '0')
  );
  const gIndices = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    gIndices[i] = (rng() * G_UNIQUE) | 0;
  }

  return { ...base, gIndices, gDict };
}
