/**
 * AssemblyScript scalar kernels – ADR-007 language spike.
 *
 * Three kernels (ADR-008 ABI):
 *   add_f64        – element-wise f64 addition
 *   sum_f64_null   – null-aware f64 sum (Arrow LSB validity bitmap)
 *   cmp_gt_f64_mask – comparison to LSB bitmask
 *
 * Allocator:
 *   alloc(size) -> ptr  – bump allocator seeded from __heap_base
 */

// ---------------------------------------------------------------------------
// Bump allocator
// ---------------------------------------------------------------------------

let _bump: usize = 0;

@inline
function bump_init(): void {
  if (!_bump) _bump = __heap_base;
}

export function alloc(size: usize): usize {
  bump_init();
  const ptr = _bump;
  _bump += (size + 7) & ~(7 as usize);
  const needed_pages = ((_bump + 65535) >> 16) as i32;
  if (needed_pages > memory.size()) {
    memory.grow(needed_pages - memory.size());
  }
  return ptr;
}

// ---------------------------------------------------------------------------
// add_f64
// ---------------------------------------------------------------------------

export function add_f64(
  a_ptr: usize,
  b_ptr: usize,
  out_ptr: usize,
  len: usize,
): void {
  for (let i: usize = 0; i < len; i++) {
    const off = i << 3;
    store<f64>(out_ptr + off, load<f64>(a_ptr + off) + load<f64>(b_ptr + off));
  }
}

// ---------------------------------------------------------------------------
// sum_f64_null – Arrow LSB validity bitmap (1 = valid, 0 = null → skip)
// ---------------------------------------------------------------------------

export function sum_f64_null(
  ptr: usize,
  validity_ptr: usize,
  len: usize,
): f64 {
  let s0: f64 = 0.0;
  let s1: f64 = 0.0;
  let s2: f64 = 0.0;
  let s3: f64 = 0.0;

  let i: usize = 0;

  // 4-element unrolled loop; fast path when all 4 bits fit in one byte
  while (i + 3 < len) {
    const byte_idx = i >> 3;
    const bit = (i & 7) as u32; // shift amount must be u32

    if (bit < 5) {
      // All four validity bits in the same byte
      const bv = load<u8>(validity_ptr + byte_idx) as u32;
      if ((bv >> bit      ) & 1) s0 += load<f64>(ptr + (i      << 3));
      if ((bv >> (bit + 1)) & 1) s1 += load<f64>(ptr + ((i + 1) << 3));
      if ((bv >> (bit + 2)) & 1) s2 += load<f64>(ptr + ((i + 2) << 3));
      if ((bv >> (bit + 3)) & 1) s3 += load<f64>(ptr + ((i + 3) << 3));
      i += 4;
    } else {
      // Bits span two bytes – fall back to scalar for this group
      const bv0 = load<u8>(validity_ptr + byte_idx) as u32;
      if ((bv0 >> bit) & 1) s0 += load<f64>(ptr + (i << 3));
      i++;
      for (let k: usize = 0; k < 3; k++) {
        const ki = i + k;
        const bv = load<u8>(validity_ptr + (ki >> 3)) as u32;
        if ((bv >> ((ki & 7) as u32)) & 1) {
          s1 += load<f64>(ptr + (ki << 3));
        }
      }
      i += 3;
    }
  }

  // Remainder
  while (i < len) {
    const bv = load<u8>(validity_ptr + (i >> 3)) as u32;
    if ((bv >> ((i & 7) as u32)) & 1) s0 += load<f64>(ptr + (i << 3));
    i++;
  }

  return s0 + s1 + s2 + s3;
}

// ---------------------------------------------------------------------------
// cmp_gt_f64_mask – write LSB bitmask where a[i] > scalar
// ---------------------------------------------------------------------------

export function cmp_gt_f64_mask(
  ptr: usize,
  scalar: f64,
  out_mask_ptr: usize,
  len: usize,
): void {
  const out_bytes: usize = (len + 7) >> 3;

  // Zero output buffer
  for (let b: usize = 0; b < out_bytes; b++) {
    store<u8>(out_mask_ptr + b, 0);
  }

  // 8 elements per iteration → 1 output byte (no byte-boundary issues)
  let i: usize = 0;
  while (i + 7 < len) {
    const base = ptr + (i << 3);
    let bv: u8 = 0;
    if (load<f64>(base      ) > scalar) bv |= 0x01;
    if (load<f64>(base +  8 ) > scalar) bv |= 0x02;
    if (load<f64>(base + 16 ) > scalar) bv |= 0x04;
    if (load<f64>(base + 24 ) > scalar) bv |= 0x08;
    if (load<f64>(base + 32 ) > scalar) bv |= 0x10;
    if (load<f64>(base + 40 ) > scalar) bv |= 0x20;
    if (load<f64>(base + 48 ) > scalar) bv |= 0x40;
    if (load<f64>(base + 56 ) > scalar) bv |= 0x80;
    store<u8>(out_mask_ptr + (i >> 3), bv);
    i += 8;
  }

  // Remainder
  while (i < len) {
    if (load<f64>(ptr + (i << 3)) > scalar) {
      const bidx = i >> 3;
      const bpos = (i & 7) as u8;
      store<u8>(out_mask_ptr + bidx,
        load<u8>(out_mask_ptr + bidx) | ((1 as u8) << bpos));
    }
    i++;
  }
}
