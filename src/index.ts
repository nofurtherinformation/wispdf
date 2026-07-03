/**
 * dataframe — columnar WASM dataframe library
 */

// Phase 1 — memory core: wasm loader, arena allocator, viewOf() layer.
export * from './memory/index.js';

// Phase 3 — expression AST + compiler (P3.1).
export * from './expr/index.js';

// Phase 3 — DataFrame / Series / GroupBy / join API (P3.2).
export * from './frame/index.js';

// Phase 5 — opt-in parallel mode (ADR-006) lives in the "databonk/workers" subpath
// export (separate bundle entry) to keep the main entry inside the §1 size budget.
// Type-only re-exports are free:
export type { ThreadsConfig, ThreadsHandle } from './workers/index.js';

// Phase 6 — I/O: CSV reader, JSON wrappers, Arrow IPC (P6.E).
export * from './io/index.js';

export const VERSION = '0.2.0';

/** Returns a greeting string. Placeholder retained for the scaffold smoke test. */
export function hello(name = 'world'): string {
  return `Hello, ${name}! dataframe v${VERSION}`;
}
