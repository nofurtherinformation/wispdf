/**
 * dataframe — columnar WASM dataframe library
 */

// Phase 1 — memory core: wasm loader, arena allocator, viewOf() layer.
export * from './memory/index.js';

export const VERSION = '0.0.1';

/**
 * Returns a greeting string. Placeholder until real API is wired up.
 */
export function hello(name = 'world'): string {
  return `Hello, ${name}! dataframe v${VERSION}`;
}
