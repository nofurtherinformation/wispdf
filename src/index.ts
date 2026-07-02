/**
 * dataframe — columnar WASM dataframe library
 */

// Phase 1 — memory core: wasm loader, arena allocator, viewOf() layer.
export * from './memory/index.js';

// Phase 3 — expression AST + compiler (P3.1).
export * from './expr/index.js';

// Phase 3 — DataFrame / Series / GroupBy / join API (P3.2).
export * from './frame/index.js';

// Phase 5 — opt-in parallel mode (ADR-006).
export { enableThreads, splitChunks } from './workers/index.js';
export type { ThreadsConfig, ThreadsHandle } from './workers/index.js';

export const VERSION = '0.0.1';

/** Returns a greeting string. Placeholder retained for the scaffold smoke test. */
export function hello(name = 'world'): string {
  return `Hello, ${name}! dataframe v${VERSION}`;
}
