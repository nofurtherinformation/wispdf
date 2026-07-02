/**
 * Parallel mode exports (ADR-006 / P5.1).
 *
 * `enableThreads()` is the single entry point for opt-in shared-memory
 * parallelism.  See docs/threads.md for full setup instructions.
 */

export { enableThreads, splitChunks } from './parallel.js';
export type { ThreadsConfig, ThreadsHandle } from './parallel.js';
export { KernelWorkerPool } from './pool.js';
