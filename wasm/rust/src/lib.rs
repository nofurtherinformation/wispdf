//! dataframe-wasm — columnar dataframe WASM core (production crate).
//!
//! Phase 1 ships the **arena allocator** (see [`arena`]). Phase 2 adds kernel
//! families (elementwise / reduce / select / hash) as sibling modules, each a
//! flat C ABI per `contracts/wasm-abi.md` §5. Keep this crate:
//!   - `#![no_std]`, `panic = "abort"`, `crate-type = ["cdylib"]`,
//!   - target `wasm32-unknown-unknown`,
//!   - two binaries from one source (`scalar.wasm`, `simd.wasm`), SIMD paths
//!     gated by `#[cfg(target_feature = "simd128")]` inside each kernel family.
//!
//! Exports (ABI §9, Phase 1 "memory core"): `memory` (linker-provided),
//! `alloc`, `free`, `realloc`, `mem_generation`.

#![no_std]

mod arena;
mod elementwise;
mod hash;
mod reduce;
mod select;

// Re-export the allocator API for potential Rust-side use by future kernels.
// The wasm exports themselves come from `#[no_mangle]` in `arena` regardless.
pub use arena::{alloc, free, mem_generation, realloc};

/// Panic handler required by `#![no_std]`. With `panic = "abort"` a panic
/// lowers to a wasm `unreachable` trap (ABI §7); on valid inputs kernels and
/// the allocator never panic.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    #[cfg(target_arch = "wasm32")]
    core::arch::wasm32::unreachable();
    #[cfg(not(target_arch = "wasm32"))]
    loop {}
}
