/**
 * Arrow validity-bitmap helpers (ABI §4.1): **LSB-first, `1 = valid`, `0 = null`**.
 *
 * A bitmap is `ceil(len / 8)` bytes; element `i` is valid iff
 * `bitmap[i >> 3] & (1 << (i & 7))` is nonzero. Bits past `len` in the final byte
 * are padding and are written `0` (kernels must not depend on them). These helpers
 * operate on a `Uint8Array` view obtained through `viewOf` — never a cached view.
 *
 * NOTE ON SLICES: a sliced column shares its parent's bitmap and reads it at a
 * *bit offset* (`Column.validityBitOffset`). Callers pass `bitOffset + i` here, so
 * these primitives stay offset-agnostic (`i` is an absolute bit index).
 */

/** Bytes needed for a `len`-bit validity bitmap: `ceil(len / 8)`. */
export function validityBytes(len: number): number {
  return (len + 7) >> 3;
}

/** True iff bit `i` is set (element valid) in an LSB-first bitmap. */
export function getBit(bitmap: Uint8Array, i: number): boolean {
  return (bitmap[i >> 3]! & (1 << (i & 7))) !== 0;
}

/** Set bit `i` (mark element valid). */
export function setBit(bitmap: Uint8Array, i: number): void {
  const byte = i >> 3;
  bitmap[byte] = (bitmap[byte] ?? 0) | (1 << (i & 7));
}

/** Clear bit `i` (mark element null). */
export function clearBit(bitmap: Uint8Array, i: number): void {
  const byte = i >> 3;
  bitmap[byte] = (bitmap[byte] ?? 0) & ~(1 << (i & 7));
}
