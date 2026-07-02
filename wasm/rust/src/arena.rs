//! Arena allocator: bump pointer + address-ordered coalescing free list.
//!
//! Implements the Phase-1 memory-core ABI (`contracts/wasm-abi.md` §3/§9):
//!
//! | export           | signature                     | semantics |
//! |------------------|-------------------------------|-----------|
//! | `alloc`          | `(i32 size) -> i32`           | 16-byte-aligned ptr, `0` on OOM; `alloc(0)` → valid ptr (do not deref). |
//! | `free`           | `(i32 ptr) -> ()`             | `free(0)` no-op; frees a block from `alloc`/`realloc`. |
//! | `realloc`        | `(i32 ptr,i32 new_size)->i32` | 16-byte-aligned; preserves `min(old,new)` bytes; `0` on OOM (old stays valid); `realloc(0,n)` ≡ `alloc(n)`. |
//! | `mem_generation` | `() -> i32`                   | increments on every successful `memory.grow`. |
//!
//! Invariants:
//!   * Every non-null allocation is **16-byte aligned** and has a 16-byte
//!     header immediately below the returned pointer, so `payload = block + 16`
//!     stays 16-aligned (legal aligned `v128` loads; natural alignment for all
//!     v1 dtypes). Block sizes are multiples of 16.
//!   * Only `alloc`/`realloc` ever call `memory.grow`, and each successful grow
//!     bumps the generation counter (§2). Kernels never grow memory (§5).
//!   * `free` inserts into an address-ordered list and coalesces with adjacent
//!     free neighbours; a freed block that reaches the top of the heap is
//!     returned to the bump pointer ("trim top"). Together these keep the heap
//!     high-water bounded under repeated alloc/free cycles (freelist reuse).
//!
//! Single-threaded wasm: `static mut` state is sound (no concurrency). We only
//! ever read/write the statics by value (never take `&`/`&mut`), so the
//! `static_mut_refs` lint does not fire.

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

/// Bytes of per-block bookkeeping, kept at 16 so the payload is 16-aligned.
const HEADER_SIZE: usize = 16;
/// Smallest payload handed out (also covers `alloc(0)`), keeps blocks usable.
const MIN_PAYLOAD: usize = 16;
/// Smallest whole block (header + min payload); used as the split threshold.
const MIN_BLOCK: usize = HEADER_SIZE + MIN_PAYLOAD; // 32
/// wasm page size.
const PAGE: usize = 65536;

// Header field byte offsets within a block (block start is 16-aligned):
//   [ 0.. 4) size  : u32  total block bytes (header + payload), multiple of 16
//   [ 4.. 8) next  : u32  next free block start (free list only; 0 = none)
//   [ 8..16) reserved (padding to keep the payload 16-aligned)

#[inline(always)]
fn align16(n: usize) -> usize {
    (n + 15) & !15
}

/// Total block size needed to satisfy a request of `size` payload bytes.
#[inline(always)]
fn needed_total(size: usize) -> usize {
    let payload = align16(size);
    let payload = if payload < MIN_PAYLOAD { MIN_PAYLOAD } else { payload };
    HEADER_SIZE + payload
}

// ---------------------------------------------------------------------------
// Global allocator state (single-threaded wasm)
// ---------------------------------------------------------------------------

/// Top of the bump region. `0` means "not yet initialised" (no real block ever
/// starts at offset 0 because the heap begins well above the data + stack).
static mut HEAP_TOP: usize = 0;
/// Head of the address-ordered free list (block start), `0` = empty.
static mut FREE_HEAD: usize = 0;
/// Generation counter (ABI §2); bumped on every successful `memory.grow`.
static mut GENERATION: i32 = 0;

/// Address of the linker-provided heap base (`__heap_base`): end of static data
/// and the shadow stack. Everything at or above this is ours to manage.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn heap_base() -> usize {
    extern "C" {
        static __heap_base: u8;
    }
    // The *address* of `__heap_base` is the heap base offset (no deref).
    core::ptr::addr_of!(__heap_base) as usize
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn heap_base() -> usize {
    // Non-wasm builds exist only so host tooling (`cargo check`, IDEs) works;
    // the allocator is never exercised off-wasm.
    0x0010_0000
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn wasm_memory_size() -> usize {
    core::arch::wasm32::memory_size(0)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn wasm_memory_grow(pages: usize) -> usize {
    core::arch::wasm32::memory_grow(0, pages)
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
unsafe fn wasm_memory_size() -> usize {
    0
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
unsafe fn wasm_memory_grow(_pages: usize) -> usize {
    usize::MAX // pretend OOM off-wasm
}

#[inline(always)]
unsafe fn init() {
    if HEAP_TOP == 0 {
        HEAP_TOP = align16(heap_base());
    }
}

// ---------------------------------------------------------------------------
// Header accessors
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn get_size(block: usize) -> usize {
    *(block as *const u32) as usize
}
#[inline(always)]
unsafe fn set_size(block: usize, size: usize) {
    *(block as *mut u32) = size as u32;
}
#[inline(always)]
unsafe fn get_next(block: usize) -> usize {
    *((block + 4) as *const u32) as usize
}
#[inline(always)]
unsafe fn set_next(block: usize, next: usize) {
    *((block + 4) as *mut u32) = next as u32;
}

/// Ensure linear memory covers byte offset `end` (exclusive). Grows if needed;
/// returns `false` on a failed `memory.grow` (OOM). Bumps the generation
/// counter on every successful grow (ABI §2/§3).
#[inline]
unsafe fn ensure_capacity(end: usize) -> bool {
    let current_bytes = (wasm_memory_size() as u64) * (PAGE as u64);
    if (end as u64) <= current_bytes {
        return true;
    }
    #[allow(clippy::manual_div_ceil)]
    let needed_pages = (((end as u64) - current_bytes + (PAGE as u64) - 1) / (PAGE as u64)) as usize;
    if wasm_memory_grow(needed_pages) == usize::MAX {
        return false; // OOM — do not advance any state
    }
    GENERATION = GENERATION.wrapping_add(1);
    true
}

// ---------------------------------------------------------------------------
// Public ABI exports
// ---------------------------------------------------------------------------

/// Allocate `size` payload bytes. Returns a 16-byte-aligned pointer, or `0` on
/// OOM. `alloc(0)` returns a valid aligned pointer that must not be dereferenced.
///
/// # Safety
/// ABI boundary; see module docs. Callers pass a non-negative `size`.
#[no_mangle]
pub unsafe extern "C" fn alloc(size: i32) -> i32 {
    if size < 0 {
        return 0;
    }
    init();
    let total = needed_total(size as usize);

    // 1. First-fit search of the free list (split large blocks).
    let mut prev = 0usize;
    let mut cur = FREE_HEAD;
    while cur != 0 {
        let bsize = get_size(cur);
        if bsize >= total {
            let next = get_next(cur);
            if bsize >= total + MIN_BLOCK {
                // Split: keep `total` in `cur`, return the remainder to the list
                // in `cur`'s address-ordered slot.
                let rem = cur + total;
                set_size(rem, bsize - total);
                set_next(rem, next);
                set_size(cur, total);
                if prev == 0 {
                    FREE_HEAD = rem;
                } else {
                    set_next(prev, rem);
                }
            } else {
                // Take the whole block.
                if prev == 0 {
                    FREE_HEAD = next;
                } else {
                    set_next(prev, next);
                }
            }
            return (cur + HEADER_SIZE) as i32;
        }
        prev = cur;
        cur = get_next(cur);
    }

    // 2. Bump-allocate a fresh block at the top of the heap.
    let block = HEAP_TOP;
    // wasm32 usize is 32-bit: a request that runs past the 4 GiB address space
    // wraps. Treat that as OOM instead of handing out a wrapped pointer.
    let new_top = match block.checked_add(total) {
        Some(v) => v,
        None => return 0,
    };
    if !ensure_capacity(new_top) {
        return 0; // OOM: leave HEAP_TOP unchanged
    }
    HEAP_TOP = new_top;
    set_size(block, total);
    (block + HEADER_SIZE) as i32
}

/// Free a block previously returned by `alloc`/`realloc`. `free(0)` is a no-op.
/// Double-free is undefined behaviour (guarded only by tests).
///
/// # Safety
/// ABI boundary; `ptr` must be `0` or a live allocation.
#[no_mangle]
pub unsafe extern "C" fn free(ptr: i32) {
    if ptr == 0 {
        return;
    }
    init();
    let block = (ptr as u32 as usize) - HEADER_SIZE;
    let size = get_size(block);
    insert_free(block, size);
}

/// Resize `ptr` to hold `new_size` payload bytes. Returns a 16-byte-aligned
/// pointer (possibly different) with the first `min(old, new)` bytes preserved,
/// or `0` on OOM (in which case the original block stays valid).
/// `realloc(0, n)` ≡ `alloc(n)`.
///
/// # Safety
/// ABI boundary; `ptr` must be `0` or a live allocation.
#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: i32, new_size: i32) -> i32 {
    if ptr == 0 {
        return alloc(new_size);
    }
    if new_size < 0 {
        return 0; // invalid; original stays valid
    }
    init();
    let block = (ptr as u32 as usize) - HEADER_SIZE;
    let old_total = get_size(block);
    let old_payload = old_total - HEADER_SIZE;
    let new_total = needed_total(new_size as usize);

    // Shrink (or same capacity): keep the block in place, split off any tail.
    if new_total <= old_total {
        if old_total - new_total >= MIN_BLOCK {
            set_size(block, new_total);
            let rem = block + new_total;
            set_size(rem, old_total - new_total);
            set_next(rem, 0);
            insert_free(rem, old_total - new_total);
        }
        return ptr;
    }

    // Grow, fast path: `block` is the top-of-heap block — extend it in place.
    if block + old_total == HEAP_TOP {
        let new_top = match block.checked_add(new_total) {
            Some(v) => v,
            None => return 0, // would run past the 4 GiB address space → OOM
        };
        if !ensure_capacity(new_top) {
            return 0; // OOM: original untouched, still valid
        }
        HEAP_TOP = new_top;
        set_size(block, new_total);
        return ptr;
    }

    // Grow, general path: allocate elsewhere, copy, free the original.
    let np = alloc(new_size);
    if np == 0 {
        return 0; // OOM: original untouched, still valid
    }
    let copy_bytes = if old_payload < new_size as usize {
        old_payload
    } else {
        new_size as usize
    };
    core::ptr::copy_nonoverlapping(
        (ptr as u32 as usize) as *const u8,
        (np as u32 as usize) as *mut u8,
        copy_bytes,
    );
    free(ptr);
    np
}

/// Current generation counter (ABI §2). Changes on every successful grow.
///
/// # Safety
/// Pure read of allocator state.
#[no_mangle]
pub unsafe extern "C" fn mem_generation() -> i32 {
    GENERATION
}

// ---------------------------------------------------------------------------
// Free-list insertion with coalescing + top trimming
// ---------------------------------------------------------------------------

/// Insert `[block, block+size)` into the address-ordered free list, coalescing
/// with an immediately-adjacent previous and/or next free block, then return
/// the block to the bump pointer if it now reaches the top of the heap.
unsafe fn insert_free(block: usize, size: usize) {
    let mut block = block;
    let mut size = size;

    // Locate the neighbours: `prev` is the last free block below `block`,
    // `next` the first free block above it.
    let mut prev = 0usize;
    let mut next = FREE_HEAD;
    while next != 0 && next < block {
        prev = next;
        next = get_next(next);
    }

    // Coalesce forward with `next` if physically adjacent.
    if next != 0 && block + size == next {
        size += get_size(next);
        next = get_next(next);
    }

    // Coalesce backward with `prev` if physically adjacent.
    if prev != 0 && prev + get_size(prev) == block {
        block = prev;
        size += get_size(prev);
        prev = pred_of(block); // predecessor of the merged (=prev) block
    }

    // Write the (possibly merged) block and splice it in.
    set_size(block, size);
    set_next(block, next);
    if prev == 0 {
        FREE_HEAD = block;
    } else {
        set_next(prev, block);
    }

    // Trim top: a free block flush with the heap top goes back to the bump
    // pointer so a later large request can reuse the space without growing.
    if block + size == HEAP_TOP {
        if prev == 0 {
            FREE_HEAD = get_next(block);
        } else {
            set_next(prev, get_next(block));
        }
        HEAP_TOP = block;
    }
}

/// Free-list predecessor of `block` (node whose `next == block`), or `0` if
/// `block` is the head. Assumes `block` is currently linked in the list.
#[inline]
unsafe fn pred_of(block: usize) -> usize {
    let mut p = 0usize;
    let mut c = FREE_HEAD;
    while c != 0 && c != block {
        p = c;
        c = get_next(c);
    }
    p
}
