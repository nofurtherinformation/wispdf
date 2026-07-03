/**
 * Minimal dt-field extractor — UTC only.
 * Uses native Date getUTC* (zero extra dependency).
 * tz-aware path moves behind a lazy subpath if size is threatened (ADR-010 §5).
 * @internal
 */

import type { DtComponent } from './ast.js';
export type { DtComponent };

const MS_DAY = 86_400_000;

/**
 * Extract one calendar component from epoch-ms (UTC).
 * weekday: ISO 8601 — Mon=1 … Sun=7.
 */
export function extractField(ms: number, c: DtComponent): number {
  const d = new Date(ms);
  switch (c) {
    case 'year':        return d.getUTCFullYear();
    case 'month':       return d.getUTCMonth() + 1;
    case 'day':         return d.getUTCDate();
    case 'hour':        return d.getUTCHours();
    case 'minute':      return d.getUTCMinutes();
    case 'second':      return d.getUTCSeconds();
    case 'millisecond': return d.getUTCMilliseconds();
    case 'weekday':     { const w = d.getUTCDay(); return w === 0 ? 7 : w; }
    case 'dayOfYear':   return Math.floor((ms - Date.UTC(d.getUTCFullYear(), 0, 1)) / MS_DAY) + 1;
    case 'quarter':     return Math.ceil((d.getUTCMonth() + 1) / 3);
  }
}

/** date32 (days since epoch) → extract field. */
export function extractDay(days: number, c: DtComponent): number {
  return extractField(days * MS_DAY, c);
}
