/**
 * AssemblyScript SIMD kernels – ADR-007 language spike.
 *
 * Same ABI as scalar.ts; v128 intrinsics used where profitable.
 * Compile with: asc assembly/simd.ts --enable simd
 *
 * SIMD strategy:
 *   add_f64         – f64x2 addition, 2 elements/cycle
 *   sum_f64_null    – f64x2 accumulator; nulls zeroed via v128.and
 *   cmp_gt_f64_mask – 4× f64x2.gt per output byte (8 elements/iter)
 */

// ---------------------------------------------------------------------------
// Bump allocator (identical to scalar)
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
// add_f64 – f64x2, 2 elements/step
// ---------------------------------------------------------------------------

export function add_f64(
  a_ptr: usize,
  b_ptr: usize,
  out_ptr: usize,
  len: usize,
): void {
  let i: usize = 0;

  while (i + 1 < len) {
    const off = i << 3;
    v128.store(out_ptr + off,
      f64x2.add(v128.load(a_ptr + off), v128.load(b_ptr + off)));
    i += 2;
  }

  // Scalar tail (when len is odd)
  if (i < len) {
    const off = i << 3;
    store<f64>(out_ptr + off, load<f64>(a_ptr + off) + load<f64>(b_ptr + off));
  }
}

// ---------------------------------------------------------------------------
// sum_f64_null – two f64x2 accumulators; validity mask via v128.and
//
// Null handling: build a v128 mask (-1LL per valid lane, 0 per null lane),
// AND it with the data pair before adding.  Nulls contribute +0.0.
// ---------------------------------------------------------------------------

export function sum_f64_null(
  ptr: usize,
  validity_ptr: usize,
  len: usize,
): f64 {
  let acc0 = f64x2.splat(0.0);
  let acc1 = f64x2.splat(0.0);

  let i: usize = 0;

  // 4 elements per iteration (two v128 loads + two SIMD adds)
  while (i + 3 < len) {
    const byte_idx = i >> 3;
    const bit = (i & 7) as u32; // shift amount must be u32

    let v0: i64, v1: i64, v2: i64, v3: i64;

    if (bit < 5) {
      const bv = load<u8>(validity_ptr + byte_idx) as u32;
      v0 = (bv >> bit      ) & 1 ? -1 : 0;
      v1 = (bv >> (bit + 1)) & 1 ? -1 : 0;
      v2 = (bv >> (bit + 2)) & 1 ? -1 : 0;
      v3 = (bv >> (bit + 3)) & 1 ? -1 : 0;
    } else if (bit == 5) {
      const b0 = load<u8>(validity_ptr + byte_idx) as u32;
      const b1 = load<u8>(validity_ptr + byte_idx + 1) as u32;
      v0 = (b0 >> 5) & 1 ? -1 : 0;
      v1 = (b0 >> 6) & 1 ? -1 : 0;
      v2 = (b0 >> 7) & 1 ? -1 : 0;
      v3 = b1 & 1 ? -1 : 0;
    } else if (bit == 6) {
      const b0 = load<u8>(validity_ptr + byte_idx) as u32;
      const b1 = load<u8>(validity_ptr + byte_idx + 1) as u32;
      v0 = (b0 >> 6) & 1 ? -1 : 0;
      v1 = (b0 >> 7) & 1 ? -1 : 0;
      v2 = b1 & 1 ? -1 : 0;
      v3 = (b1 >> 1) & 1 ? -1 : 0;
    } else { // bit == 7
      const b0 = load<u8>(validity_ptr + byte_idx) as u32;
      const b1 = load<u8>(validity_ptr + byte_idx + 1) as u32;
      v0 = (b0 >> 7) & 1 ? -1 : 0;
      v1 = b1 & 1 ? -1 : 0;
      v2 = (b1 >> 1) & 1 ? -1 : 0;
      v3 = (b1 >> 2) & 1 ? -1 : 0;
    }

    const off0 = i << 3;
    const off2 = (i + 2) << 3;

    // Build validity masks: -1LL per valid lane, 0 per null
    let mask0 = i64x2.splat(v0);
    mask0 = i64x2.replace_lane(mask0, 1, v1);
    let mask1 = i64x2.splat(v2);
    mask1 = i64x2.replace_lane(mask1, 1, v3);

    acc0 = f64x2.add(acc0, v128.and(v128.load(ptr + off0), mask0));
    acc1 = f64x2.add(acc1, v128.and(v128.load(ptr + off2), mask1));

    i += 4;
  }

  // Reduce accumulators
  let sum =
    f64x2.extract_lane(acc0, 0) + f64x2.extract_lane(acc0, 1) +
    f64x2.extract_lane(acc1, 0) + f64x2.extract_lane(acc1, 1);

  // Scalar tail
  while (i < len) {
    const bv = load<u8>(validity_ptr + (i >> 3)) as u32;
    if ((bv >> ((i & 7) as u32)) & 1) sum += load<f64>(ptr + (i << 3));
    i++;
  }

  return sum;
}

// ---------------------------------------------------------------------------
// cmp_gt_f64_mask – 4× f64x2.gt per byte (8 elements / output byte)
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

  const sv = f64x2.splat(scalar);
  let i: usize = 0;

  // 8 elements per iteration → 1 output byte
  while (i + 7 < len) {
    const base = ptr + (i << 3);
    const c0 = f64x2.gt(v128.load(base     ), sv); // elements 0,1
    const c1 = f64x2.gt(v128.load(base + 16), sv); // elements 2,3
    const c2 = f64x2.gt(v128.load(base + 32), sv); // elements 4,5
    const c3 = f64x2.gt(v128.load(base + 48), sv); // elements 6,7

    // f64x2.gt: lane is all-1s for true (-1LL), 0 for false.
    // Bit 0 of each i64 lane encodes true/false.
    const b0 = i64x2.extract_lane(c0, 0) & 1;
    const b1 = i64x2.extract_lane(c0, 1) & 1;
    const b2 = i64x2.extract_lane(c1, 0) & 1;
    const b3 = i64x2.extract_lane(c1, 1) & 1;
    const b4 = i64x2.extract_lane(c2, 0) & 1;
    const b5 = i64x2.extract_lane(c2, 1) & 1;
    const b6 = i64x2.extract_lane(c3, 0) & 1;
    const b7 = i64x2.extract_lane(c3, 1) & 1;

    const bv: u8 = (
      b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) |
      (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7)
    ) as u8;
    store<u8>(out_mask_ptr + (i >> 3), bv);
    i += 8;
  }

  // Scalar tail
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
