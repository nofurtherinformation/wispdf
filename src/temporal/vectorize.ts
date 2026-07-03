/**
 * src/temporal/vectorize.ts
 *
 * Vectorized helpers for dt accessor extraction. These are the functions that
 * the future frame layer will call (Part 1 deliverable — frame integration
 * happens in a later task).
 *
 * Shape: extractComponents(view, validity, component, tz?) → {data, validity}
 *   - view: BigInt64Array (timestamp) or Int32Array (date32)
 *   - validity: null (all-valid) or Uint8Array (Arrow LSB bitmap, 1=valid)
 *   - component: one of the dt accessor field names
 *   - tz: optional IANA timezone string (timestamp only; undefined = UTC path)
 *   - returns: Int32Array of extracted component values + a copied validity
 *
 * Performance notes:
 *   - UTC path: no Intl calls; pure integer math via civil.ts.
 *   - tz path: one cached Intl.DateTimeFormat per tz (no per-row Intl construct);
 *     one formatToParts call per valid row. See tz.ts for the formatter cache.
 *   - Null rows: output data is 0 (unspecified); validity bit is 0 (null).
 *
 * Conformance: dtypes.md §10–§11, ADR-010.
 */

import {
  date32ToFields,
  timestampUtcToFields,
  type CivilFields,
} from './civil.js';
import { getTzComponents } from './tz.js';

// ── Types ────────────────────────────────────────────────────────────────────

/** Valid dt accessor component names (dtypes.md §10). */
export type DtComponent =
  | 'year' | 'month' | 'day'
  | 'hour' | 'minute' | 'second' | 'millisecond'
  | 'weekday' | 'dayOfYear' | 'quarter';

/** Result of extractComponents: component data array + validity bitmap. */
export interface ExtractResult {
  /** Component value per element; 0 where null. */
  readonly data: Int32Array;
  /**
   * Copied validity bitmap (Arrow LSB, 1=valid), or null if the input was
   * all-valid and no rows are null. Matches the caller's validity input.
   */
  readonly validity: Uint8Array | null;
}

// ── Validity bitmap helpers ───────────────────────────────────────────────────

/** Read one validity bit from an Arrow LSB bitmap. */
function isValid(bitmap: Uint8Array, i: number): boolean {
  return ((bitmap[i >> 3] ?? 0) & (1 << (i & 7))) !== 0;
}

// ── Core function ─────────────────────────────────────────────────────────────

/**
 * extractComponents: vectorized dt accessor extraction over a typed-array view.
 *
 * @param view    - BigInt64Array for timestamp columns; Int32Array for date32.
 * @param validity - Arrow LSB bitmap (null = all-valid / no nulls in the column).
 * @param component - Which accessor field to extract.
 * @param tz      - Optional IANA tz string for timestamp columns.
 *                  Undefined / absent → UTC path (no Intl calls).
 * @returns ExtractResult with data Int32Array and copied validity.
 */
export function extractComponents(
  view: BigInt64Array | Int32Array,
  validity: Uint8Array | null,
  component: DtComponent,
  tz?: string,
): ExtractResult {
  const len = view.length;
  const data = new Int32Array(len);
  const isDate32 = view instanceof Int32Array;
  const hasTz = typeof tz === 'string' && tz.length > 0;

  for (let i = 0; i < len; i++) {
    // Null check: if validity bitmap present, skip null rows (leave data as 0)
    if (validity !== null && !isValid(validity, i)) {
      continue; // data[i] stays 0; validity bit stays 0 via copied bitmap
    }

    let fields: CivilFields;

    if (isDate32) {
      // Int32Array: date32 day counts
      fields = date32ToFields((view as Int32Array)[i] ?? 0);
    } else if (hasTz) {
      // BigInt64Array + tz: tz-aware path (one cached Intl.DateTimeFormat)
      const epochMs = Number((view as BigInt64Array)[i] ?? 0n);
      fields = getTzComponents(epochMs, tz as string);
    } else {
      // BigInt64Array + no tz: UTC integer-math path (no Intl)
      const ms = (view as BigInt64Array)[i] ?? 0n;
      fields = timestampUtcToFields(ms);
    }

    data[i] = fields[component];
  }

  return { data, validity };
}
