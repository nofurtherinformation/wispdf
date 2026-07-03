/**
 * Compiler runtime (Phase 3, P3.1 deliverable §3 buffer lifecycle).
 *
 * A {@link Runtime} wraps a {@link FrameView}'s memory context + kernel exports and
 * is the single owner of every **temporary** the compiler allocates. It:
 *
 *   - allocates/free's through the arena, tracking live temp pointers so the plan
 *     leaves **zero** leaked buffers (verified against the arena high-water / free
 *     balance in the leak tests),
 *   - dispatches kernels by ABI name (ABI §6) and records an ordered {@link Trace}
 *     of kernel calls + allocations so fusion is verifiable by plan inspection,
 *   - hands out `viewOf` views (never cached — ADR-001).
 *
 * Ownership model: a pointer is in {@link Runtime.owned} iff the runtime allocated it
 * and has neither freed nor *transferred* it. Source-column buffers are never owned.
 * At the end of a plan the result's buffers are transferred out; everything else is
 * freed.
 */

import type { MemoryContext } from '../memory/context.js';
import type { FrameView, KernelWasm } from './frame.js';
import { callKernel, callKernelBigInt } from './frame.js';
import type { ViewDType, ColumnView, ColumnBuffer } from '../memory/views.js';
import { validityBytes, getBit, setBit } from '../memory/bitmap.js';

/** Why a temp buffer was allocated (plan-inspection classification). */
export type AllocPurpose = 'data' | 'validity' | 'mask' | 'scratch' | 'dict' | 'index';

/** Ordered record of what a plan did — the inspectable "kernel-call plan". */
export interface Trace {
  /** Kernel export names, in call order (ABI §6). */
  readonly kernels: string[];
  /** Allocations, in order, by purpose. */
  readonly allocs: Array<{ bytes: number; purpose: AllocPurpose }>;
  /** Free count (for balance checks). */
  frees: number;
}

/** Derived, test-friendly counters over a {@link Trace}. */
export interface ExecStats {
  /** Total kernel invocations. */
  readonly kernelCalls: number;
  /** Kernel names in call order. */
  readonly kernels: readonly string[];
  /** Total temp allocations. */
  readonly allocations: number;
  /** Data-buffer allocations only (the elementwise-fusion metric). */
  readonly dataAllocations: number;
  /** Mask-buffer allocations only (the compare→filter fusion metric). */
  readonly maskAllocations: number;
  /** Free calls issued. */
  readonly frees: number;
}

export function statsOf(trace: Trace): ExecStats {
  let data = 0;
  let mask = 0;
  for (const a of trace.allocs) {
    if (a.purpose === 'data') data++;
    else if (a.purpose === 'mask') mask++;
  }
  return {
    kernelCalls: trace.kernels.length,
    kernels: trace.kernels.slice(),
    allocations: trace.allocs.length,
    dataAllocations: data,
    maskAllocations: mask,
    frees: trace.frees,
  };
}

export class Runtime {
  readonly ctx: MemoryContext;
  readonly wasm: KernelWasm;
  /** Frame row count — every column and temp has this element length. */
  readonly len: number;
  /** Live temp pointers this runtime still owns. */
  readonly owned = new Set<number>();
  readonly trace: Trace = { kernels: [], allocs: [], frees: 0 };
  /** ColumnBuffers registered with `viewOf`, so they can be forgotten on free. */
  private readonly viewed: ColumnBuffer[] = [];

  constructor(frame: FrameView) {
    this.ctx = frame.ctx;
    this.wasm = frame.wasm;
    this.len = frame.length;
  }

  /** Bytes for a `len`-element validity bitmap. */
  get validityBytes(): number {
    return validityBytes(this.len);
  }

  /** Allocate `bytes` of temp scratch; tracks + records it. */
  alloc(bytes: number, purpose: AllocPurpose): number {
    const ptr = this.ctx.mod.alloc(bytes);
    if (ptr === 0) throw new Error(`out of wasm memory allocating ${bytes} bytes`);
    this.owned.add(ptr);
    this.trace.allocs.push({ bytes, purpose });
    return ptr;
  }

  /** Register an externally-allocated pointer (e.g. from `writeDictionary`) as a temp. */
  track(ptr: number, bytes: number, purpose: AllocPurpose): void {
    if (ptr === 0) return;
    this.owned.add(ptr);
    this.trace.allocs.push({ bytes, purpose });
  }

  /** Free a temp pointer if this runtime owns it (no-op otherwise). */
  free(ptr: number): void {
    if (ptr !== 0 && this.owned.delete(ptr)) {
      this.forgetViews(ptr);
      this.ctx.mod.free(ptr);
      this.trace.frees++;
    }
  }

  /** Transfer ownership of `ptr` out of the runtime (to the returned result). */
  transfer(ptr: number): void {
    this.owned.delete(ptr);
    this.forgetViews(ptr);
  }

  /** Free every temp still owned (call at the end of a plan). */
  freeAll(): void {
    for (const buf of this.viewed) this.ctx.viewOf.forget(buf);
    this.viewed.length = 0;
    for (const ptr of this.owned) {
      this.ctx.mod.free(ptr);
      this.trace.frees++;
    }
    this.owned.clear();
  }

  /** Dispatch kernel `name` (records the call). */
  call(name: string, ...args: number[]): number {
    this.trace.kernels.push(name);
    return callKernel(this.wasm, name, args);
  }

  /**
   * Like {@link call} but accepts `bigint` args and may return `bigint`.
   * Required for i64 scalar kernels and i64 reductions (wasm i64 ↔ JS BigInt).
   */
  callBigInt(name: string, ...args: (number | bigint)[]): number | bigint {
    this.trace.kernels.push(name);
    return callKernelBigInt(this.wasm, name, args);
  }

  /** A live `viewOf` view (never cache the result — ADR-001). */
  view(ptr: number, length: number, dtype: ViewDType): ColumnView {
    const buf: ColumnBuffer = { ptr, length, dtype };
    this.viewed.push(buf);
    return this.ctx.viewOf(buf);
  }

  /** Drop `viewOf` registrations for a freed/transferred pointer. */
  private forgetViews(ptr: number): void {
    for (let i = this.viewed.length - 1; i >= 0; i--) {
      const buf = this.viewed[i]!;
      if (buf.ptr === ptr) {
        this.ctx.viewOf.forget(buf);
        this.viewed.splice(i, 1);
      }
    }
  }
}

// ── validity / bit helpers ────────────────────────────────────────────────────

/** A validity reference: `ptr === 0` means all-valid; `owns` = runtime-allocated. */
export interface Validity {
  readonly ptr: number;
  readonly owns: boolean;
}

/** The shared all-valid sentinel (ABI §4.1 `validity_ptr == 0`). */
export const ALL_VALID: Validity = { ptr: 0, owns: false };

/** Free a validity buffer if it is runtime-owned. */
export function freeValidity(rt: Runtime, v: Validity): void {
  if (v.owns) rt.free(v.ptr);
}

/**
 * Return a bit-0-aligned validity pointer for a (possibly sliced) column
 * (`memory.d.ts` — a slice carries a bit offset). Root columns (offset 0) pass
 * through zero-copy; a mid-byte offset is realigned into an owned temp.
 */
export function materializeValidity(
  rt: Runtime,
  validityPtr: number,
  bitOffset: number,
): Validity {
  if (validityPtr === 0) return ALL_VALID;
  if ((bitOffset & 7) === 0) {
    return { ptr: validityPtr + (bitOffset >> 3), owns: false };
  }
  const bytes = rt.validityBytes;
  const out = rt.alloc(bytes, 'validity');
  const src = rt.view(validityPtr, validityBytes(bitOffset + rt.len), 'u8') as Uint8Array;
  const dst = rt.view(out, bytes, 'u8') as Uint8Array;
  dst.fill(0);
  for (let i = 0; i < rt.len; i++) if (getBit(src, bitOffset + i)) setBit(dst, i);
  return { ptr: out, owns: true };
}

/** Population count of the first `len` bits of an Arrow-LSB mask. */
export function popcountMask(rt: Runtime, maskPtr: number, len: number): number {
  if (len === 0) return 0;
  const u8 = rt.view(maskPtr, validityBytes(len), 'u8') as Uint8Array;
  let n = 0;
  for (let i = 0; i < len; i++) if ((u8[i >> 3]! >> (i & 7)) & 1) n++;
  return n;
}
