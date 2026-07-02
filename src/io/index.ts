/**
 * I/O module (Phase 6, Agent E): CSV reader, JSON wrappers, Arrow IPC.
 *
 * JSON path: `fromRecords` / `toRecords` on DataFrame are the canonical path;
 * `fromJSON` / `toJSON` are thin wrappers for ergonomics (see json.ts).
 */

export { fromCSV, ChunkParser, type FromCsvOptions } from './csv.js';
export { fromJSON, toJSON } from './json.js';
export { toArrow, fromArrow } from './arrow.js';
