//! Rust WASM spike for ADR-007 language evaluation.
//!
//! Three kernels matching the AssemblyScript spike:
//!   - add_f64:         element-wise f64 addition
//!   - sum_f64_null:    null-aware sum (Arrow LSB validity bitmap)
//!   - cmp_gt_f64_mask: comparison to LSB bitmask output
//!
//! Plus a static bump allocator (alloc export) for the bench harness.
//!
//! Build with RUSTFLAGS="-C target-feature=+simd128" for SIMD variant;
//! scalar variant omits that flag. Both builds use the same source:
//! #[cfg(target_feature = "simd128")] gates the SIMD paths.

#![no_std]

#[cfg(target_feature = "simd128")]
use core::arch::wasm32;

// ---------------------------------------------------------------------------
// Panic handler (required for no_std)
// ---------------------------------------------------------------------------

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In release wasm with panic=abort this is never reached, but we need
    // the symbol. Use the wasm unreachable trap instruction.
    #[cfg(target_arch = "wasm32")]
    core::arch::wasm32::unreachable();
    #[cfg(not(target_arch = "wasm32"))]
    loop {}
}

// ---------------------------------------------------------------------------
// Static bump allocator
// ---------------------------------------------------------------------------
//
// Starts at 2 MB to stay above the default 1 MB wasm-ld stack and any data
// section. Grows wasm linear memory on demand.

const HEAP_START: u32 = 2 * 1024 * 1024; // 2 MB

static mut BUMP_PTR: u32 = HEAP_START;

/// Allocate `size` bytes aligned to 8 bytes. Returns a pointer into wasm
/// linear memory. For the bench harness only — not a general allocator.
#[no_mangle]
pub unsafe extern "C" fn alloc(size: u32) -> *mut u8 {
    let ptr = BUMP_PTR;
    // 8-byte align the next allocation
    let end = (ptr + size + 7) & !7u32;
    // Grow linear memory if the new end exceeds current allocated pages
    let needed_pages = ((end as usize) + 65535) / 65536;
    let current_pages = core::arch::wasm32::memory_size(0);
    if needed_pages > current_pages {
        core::arch::wasm32::memory_grow(0, needed_pages - current_pages);
    }
    BUMP_PTR = end;
    ptr as *mut u8
}

// ---------------------------------------------------------------------------
// Kernel 1: add_f64
// Element-wise addition of two f64 arrays into an output array.
// SIMD path processes 2 f64s per instruction (f64x2).
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn add_f64(
    a_ptr: *const f64,
    b_ptr: *const f64,
    out_ptr: *mut f64,
    len: u32,
) {
    let len = len as usize;

    #[cfg(target_feature = "simd128")]
    {
        let mut i = 0usize;
        // Process 2 elements per SIMD instruction
        while i + 2 <= len {
            let a = wasm32::v128_load(a_ptr.add(i) as *const wasm32::v128);
            let b = wasm32::v128_load(b_ptr.add(i) as *const wasm32::v128);
            let r = wasm32::f64x2_add(a, b);
            wasm32::v128_store(out_ptr.add(i) as *mut wasm32::v128, r);
            i += 2;
        }
        // Scalar tail
        while i < len {
            *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
            i += 1;
        }
    }

    #[cfg(not(target_feature = "simd128"))]
    {
        for i in 0..len {
            *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: sum_f64_null
// Null-aware reduction. validity_ptr points to an Arrow LSB validity bitmap:
//   byte j, bit k (LSB-first) → element (j*8 + k) is valid iff bit is 1.
// Returns the sum of all valid elements; invalid elements are skipped.
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn sum_f64_null(
    ptr: *const f64,
    validity_ptr: *const u8,
    len: u32,
) -> f64 {
    let len = len as usize;

    #[cfg(target_feature = "simd128")]
    {
        // SIMD accumulator: two f64 lanes
        let mut acc = wasm32::f64x2_splat(0.0f64);
        // Separate scalar accumulator for partially-valid pairs. Adding a single
        // valid value via f64x2_splat would place it in BOTH lanes and thus
        // double-count it after the horizontal reduction, so single values are
        // summed here instead.
        let mut scalar_acc = 0.0f64;
        let mut i = 0usize;

        // Process pairs of elements. For each pair, check 2 validity bits.
        // If both valid: SIMD add. Otherwise: scalar fallback per element.
        while i + 2 <= len {
            let byte_idx = i / 8;
            let bit_off = (i % 8) as u32;
            let byte = *validity_ptr.add(byte_idx);
            let bits = (byte >> bit_off) & 0x3u8;

            if bits == 0x3 {
                // Both elements valid — use SIMD add
                let v = wasm32::v128_load(ptr.add(i) as *const wasm32::v128);
                acc = wasm32::f64x2_add(acc, v);
            } else {
                // At least one invalid — scalar fallback (add once, not splatted)
                if bits & 1 != 0 {
                    scalar_acc += *ptr.add(i);
                }
                if bits & 2 != 0 {
                    scalar_acc += *ptr.add(i + 1);
                }
            }
            i += 2;
        }

        // Horizontal reduction of the two accumulator lanes + scalar accumulator
        let lane0 = wasm32::f64x2_extract_lane::<0>(acc);
        let lane1 = wasm32::f64x2_extract_lane::<1>(acc);
        let mut total = lane0 + lane1 + scalar_acc;

        // Scalar tail (last element if len is odd)
        while i < len {
            let byte_idx = i / 8;
            let bit_off = (i % 8) as u32;
            let byte = *validity_ptr.add(byte_idx);
            if (byte >> bit_off) & 1 != 0 {
                total += *ptr.add(i);
            }
            i += 1;
        }

        total
    }

    #[cfg(not(target_feature = "simd128"))]
    {
        let mut sum = 0.0f64;
        for i in 0..len {
            let byte_idx = i / 8;
            let bit_off = (i % 8) as u32;
            let byte = *validity_ptr.add(byte_idx);
            if (byte >> bit_off) & 1 != 0 {
                sum += *ptr.add(i);
            }
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: cmp_gt_f64_mask
// For each element i: out_mask[i/8] bit (i%8) = 1 iff data[i] > scalar.
// Output bitmask is Arrow LSB format (same as validity bitmap).
// SIMD path: f64x2_gt → i64x2_bitmask gives 2 bits per SIMD op → pack
// 4 ops into 1 output byte (8 elements per byte).
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn cmp_gt_f64_mask(
    ptr: *const f64,
    scalar: f64,
    out_mask_ptr: *mut u8,
    len: u32,
) {
    let len = len as usize;

    #[cfg(target_feature = "simd128")]
    {
        let splat = wasm32::f64x2_splat(scalar);
        let mut i = 0usize;
        let mut byte_idx = 0usize;

        // Process 8 elements at a time → 4 SIMD ops → 1 output byte
        while i + 8 <= len {
            let mut byte: u8 = 0;
            // 4 groups of 2 elements each
            let a0 = wasm32::v128_load(ptr.add(i) as *const wasm32::v128);
            let a1 = wasm32::v128_load(ptr.add(i + 2) as *const wasm32::v128);
            let a2 = wasm32::v128_load(ptr.add(i + 4) as *const wasm32::v128);
            let a3 = wasm32::v128_load(ptr.add(i + 6) as *const wasm32::v128);

            let c0 = wasm32::f64x2_gt(a0, splat);
            let c1 = wasm32::f64x2_gt(a1, splat);
            let c2 = wasm32::f64x2_gt(a2, splat);
            let c3 = wasm32::f64x2_gt(a3, splat);

            // i64x2_bitmask: bit 0 = MSB of lane 0, bit 1 = MSB of lane 1
            // For f64x2_gt result: lane is all-1s (true) or all-0s (false)
            // so MSB is 1 iff the comparison was true.
            byte |= (wasm32::i64x2_bitmask(c0) as u8) & 0x03;
            byte |= ((wasm32::i64x2_bitmask(c1) as u8) & 0x03) << 2;
            byte |= ((wasm32::i64x2_bitmask(c2) as u8) & 0x03) << 4;
            byte |= ((wasm32::i64x2_bitmask(c3) as u8) & 0x03) << 6;

            *out_mask_ptr.add(byte_idx) = byte;
            i += 8;
            byte_idx += 1;
        }

        // Scalar tail
        let mut current_byte: u8 = 0;
        for j in i..len {
            if *ptr.add(j) > scalar {
                current_byte |= 1u8 << (j & 7);
            }
            if (j & 7) == 7 {
                *out_mask_ptr.add(j / 8) = current_byte;
                current_byte = 0;
            }
        }
        if len & 7 != 0 {
            *out_mask_ptr.add(len / 8) = current_byte;
        }
    }

    #[cfg(not(target_feature = "simd128"))]
    {
        // Zero mask output bytes
        let mask_bytes = (len + 7) / 8;
        for k in 0..mask_bytes {
            *out_mask_ptr.add(k) = 0;
        }
        for i in 0..len {
            if *ptr.add(i) > scalar {
                *out_mask_ptr.add(i / 8) |= 1u8 << (i & 7);
            }
        }
    }
}
