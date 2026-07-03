/**
 * dt-field extractor — delegates to src/temporal civil-math and tz modules.
 * No Date objects per-row; pure integer arithmetic for UTC path, cached
 * Intl.DateTimeFormat per zone for the tz path (ADR-010 §5, ADR-012).
 *
 * extractField: epoch-ms → one component (timestamp path, UTC or tz-aware).
 * extractDayField: day-count → one component (date32 path, UTC always).
 * @internal
 */

import type { DtComponent } from './ast.js';
export type { DtComponent };

import { timestampUtcToFields, date32ToFields } from '../temporal/civil.js';
import { getTzComponents } from '../temporal/tz.js';

/**
 * Extract one calendar component from epoch-ms.
 * weekday: ISO 8601 — Mon=1 … Sun=7.
 *
 * @param ms  - epoch milliseconds as number (safe for |ms| ≤ 2^53).
 *              Callers convert BigInt64Array elements with Number() before
 *              passing here — safe for the full timestamp range that fits in a
 *              JS safe integer.
 * @param c   - which component to extract.
 * @param tz  - optional IANA tz string; absent/undefined → UTC integer path.
 */
export function extractField(ms: number, c: DtComponent, tz?: string): number {
  const fields =
    typeof tz === 'string' && tz.length > 0
      ? getTzComponents(ms, tz)
      : timestampUtcToFields(BigInt(Math.trunc(ms)));
  // CivilFields uses 'millisecond' (not 'ms') as the public accessor name.
  return fields[c];
}

/**
 * date32 (days since epoch) → extract field.
 * Time-of-day fields are always 0 for date32 (dtypes.md §10).
 */
export function extractDayField(days: number, c: DtComponent): number {
  return date32ToFields(days)[c];
}
