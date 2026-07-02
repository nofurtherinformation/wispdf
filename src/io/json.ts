/**
 * JSON I/O thin wrappers.
 *
 * The spec (§4 + §7) asks for fromJSON/toJSON — but DataFrame.fromRecords and
 * df.toRecords() already ARE the JSON path:
 *
 *   df.toRecords()          → Array<Record<string, Cell>> — serialize with JSON.stringify
 *   DataFrame.fromRecords() → DataFrame                  — parse with JSON.parse first
 *
 * This module exports `fromJSON` and `toJSON` as thin wrappers so callers have a
 * discoverable one-liner, but the parallel implementation is intentionally avoided:
 * these delegates call through to the existing frame-layer implementation (no duplicate
 * column-building code). They add value only in ergonomics, not behavior.
 *
 * fromRecords / toRecords remain the canonical JSON path per the spec.
 */

import { DataFrame, type FrameOptions } from '../frame/dataframe.js';
import type { Cell } from '../memory/column.js';

/**
 * Parse a JSON array-of-records string into a DataFrame.
 * Equivalent to `DataFrame.fromRecords(JSON.parse(json), opts)`.
 *
 * @param json - A JSON string encoding an array of plain objects.
 * @param opts - Forwarded to DataFrame.fromRecords.
 * @throws SyntaxError if `json` is not valid JSON.
 * @throws TypeError if the parsed value is not an array.
 */
export function fromJSON(json: string, opts: FrameOptions = {}): DataFrame {
  const records = JSON.parse(json) as unknown;
  if (!Array.isArray(records)) {
    throw new TypeError(
      `fromJSON: expected a JSON array of records, got ${typeof records}`,
    );
  }
  return DataFrame.fromRecords(records as ReadonlyArray<Readonly<Record<string, Cell>>>, opts);
}

/**
 * Serialize a DataFrame to a compact JSON string (array of record objects).
 * Equivalent to `JSON.stringify(df.toRecords())`.
 *
 * @param df - The DataFrame to serialize.
 * @returns A JSON string. Null cells serialize as JSON `null`.
 */
export function toJSON(df: DataFrame): string {
  return JSON.stringify(df.toRecords());
}
