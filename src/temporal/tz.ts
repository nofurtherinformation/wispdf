/**
 * src/temporal/tz.ts
 *
 * Timezone-aware component extraction for timestamp columns.
 * Uses a cached Intl.DateTimeFormat per IANA zone string to convert UTC epoch-ms
 * to local civil fields.
 *
 * ICU / Intl dependency:
 *   Requires Node >= 18 with full-icu support (system ICU or --icu-data-dir).
 *   All evergreen browsers include the full IANA tz database via ICU.
 *   DST rules follow the platform ICU DB — the fixture values were computed with
 *   Node v22.23.1 (ICU 78.2). Tests skip tz-aware cases and warn when the ICU
 *   version differs from the fixture compute environment.
 *
 * Conformance: dtypes.md §10 ("tz metadata behavior"), ADR-010.
 */

import {
  civilToDays,
  isoWeekday,
  dayOfYear as doyFn,
  type CivilFields,
} from './civil.js';

// ── Formatter cache ───────────────────────────────────────────────────────────

/** One Intl.DateTimeFormat instance per IANA tz string. Thread-local in browser workers. */
const fmtCache = new Map<string, Intl.DateTimeFormat>();

/**
 * getFormatter: returns a cached Intl.DateTimeFormat for the given IANA tz string.
 * Throws a descriptive RangeError if the tz name is invalid.
 *
 * The formatter is configured with h23 hourCycle so that midnight is 00:xx and
 * there is no 12/24 ambiguity. formatToParts does not expose sub-second precision;
 * the millisecond component is derived from `epochMs % 1000` (tz-offset-independent).
 */
function getFormatter(tz: string): Intl.DateTimeFormat {
  const cached = fmtCache.get(tz);
  if (cached !== undefined) return cached;

  let fmt: Intl.DateTimeFormat;
  try {
    fmt = new Intl.DateTimeFormat('en', {
      timeZone: tz,
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
      second: 'numeric',
      hourCycle: 'h23',
    });
  } catch (e) {
    throw new RangeError(
      `Invalid IANA timezone: "${tz}". ${e instanceof Error ? e.message : String(e)}`,
    );
  }

  fmtCache.set(tz, fmt);
  return fmt;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * getIcuVersion: returns the ICU version string from process.versions.icu if
 * available (Node.js), or undefined in browsers. Used by the test runner to
 * conditionally skip tz-aware fixture cases when ICU differs from the fixture
 * compute environment (Node v22.23.1 / ICU 78.2).
 */
export function getIcuVersion(): string | undefined {
  if (
    typeof process !== 'undefined' &&
    process.versions != null &&
    typeof process.versions['icu'] === 'string'
  ) {
    return process.versions['icu'];
  }
  return undefined;
}

/**
 * validateIanaZone: throws a descriptive RangeError if the tz name is invalid.
 * Useful for eager validation before storing tz metadata on a column.
 */
export function validateIanaZone(tz: string): void {
  getFormatter(tz); // throws on invalid tz
}

/**
 * getTzComponents: extract civil date+time fields for `epochMs` in the given
 * IANA `tz`. Returns the same CivilFields shape as the UTC path.
 *
 * Millisecond is derived from `((epochMs % 1000) + 1000) % 1000`, which is
 * correct for both positive and negative epoch-ms values and is independent of
 * the timezone offset (sub-second precision is not affected by tz).
 *
 * Weekday and dayOfYear are derived from the LOCAL year/month/day (the date in
 * the target timezone, not UTC), using the same ISO 8601 weekday algorithm as
 * the UTC path (ADR-010).
 *
 * @param epochMs - UTC epoch milliseconds as a JS number. The caller must
 *   convert BigInt64Array values with Number() before passing here; this
 *   function is safe for |epochMs| ≤ 2^53 (Number.MAX_SAFE_INTEGER).
 * @param tz - IANA timezone string, e.g. "America/Chicago", "UTC", "+05:30".
 *   Throws RangeError on invalid zone names.
 */
export function getTzComponents(epochMs: number, tz: string): CivilFields {
  const fmt = getFormatter(tz);
  const parts = fmt.formatToParts(epochMs);

  let year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
  for (const part of parts) {
    switch (part.type) {
      case 'year':   year   = parseInt(part.value, 10); break;
      case 'month':  month  = parseInt(part.value, 10); break;
      case 'day':    day    = parseInt(part.value, 10); break;
      case 'hour':   hour   = parseInt(part.value, 10); break;
      case 'minute': minute = parseInt(part.value, 10); break;
      case 'second': second = parseInt(part.value, 10); break;
    }
  }

  // Sub-second component is tz-independent; handle negative epochMs correctly.
  const millisecond = ((epochMs % 1000) + 1000) % 1000;

  // weekday and dayOfYear are derived from the LOCAL date fields (year/month/day
  // as returned by Intl), not from the UTC date. The local day number is computed
  // via civilToDays so we can apply the same isoWeekday formula.
  const localDayNum = civilToDays(year, month, day);

  return {
    year,
    month,
    day,
    hour,
    minute,
    second,
    millisecond,
    weekday: isoWeekday(localDayNum),
    dayOfYear: doyFn(year, month, day),
    quarter: Math.ceil(month / 3),
  };
}
