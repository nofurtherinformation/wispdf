/**
 * src/temporal/civil.ts
 *
 * Pure integer civil-calendar math for date32 and timestamp types.
 * No Date objects, no allocations in loops, no locale dependencies.
 *
 * Howard Hinnant civil_from_days / days_from_civil algorithms:
 *   http://howardhinnant.github.io/date_algorithms.html
 *   (public domain; C++ ported to JS integer arithmetic)
 *
 * Conformance: dtypes.md §10, ADR-010.
 * Weekday: ISO 8601 Mon=1..Sun=7, locked in ADR-010.
 */

// ── Types ────────────────────────────────────────────────────────────────────

/** Proleptic Gregorian calendar date (year/month/day). */
export interface CivilDate {
  readonly year: number;
  readonly month: number; // [1, 12]
  readonly day: number;   // [1, 31]
}

/**
 * Full civil date+time broken down from a millisecond timestamp.
 * Note: the last field is named `ms` (millisecond component [0, 999]),
 * not `millisecond`, to distinguish it from the epoch-ms input.
 */
export interface CivilDateTime extends CivilDate {
  readonly hour: number;   // [0, 23]
  readonly minute: number; // [0, 59]
  readonly second: number; // [0, 59]
  readonly ms: number;     // [0, 999]
}

/**
 * All dt accessor fields (dtypes.md §10). Field `millisecond` matches the
 * fixture / public accessor name.
 */
export interface CivilFields {
  readonly year: number;
  readonly month: number;
  readonly day: number;
  readonly hour: number;
  readonly minute: number;
  readonly second: number;
  readonly millisecond: number; // [0, 999]
  readonly weekday: number;     // ISO 8601: Mon=1..Sun=7
  readonly dayOfYear: number;   // [1, 366]
  readonly quarter: number;     // [1, 4]
}

// ── Constants ────────────────────────────────────────────────────────────────

const MS_PER_DAY = 86_400_000;
const MS_PER_DAY_N = 86_400_000n;

// Cumulative days before the start of each month in a non-leap year.
// Index 0 = January: 0 days before Jan 1.
const DAYS_BEFORE_MONTH: readonly number[] = [
  0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334,
];

// ── Public pure functions ────────────────────────────────────────────────────

/**
 * daysToCivil: proleptic Gregorian day number → {year, month, day}.
 *
 * Algorithm: Howard Hinnant civil_from_days
 *   http://howardhinnant.github.io/date_algorithms.html
 *   The era origin is 1 Mar 0000 (shifted by 719468 days from 1970-01-01).
 *   Works for the full i32 date32 range and beyond.
 */
export function daysToCivil(days: number): CivilDate {
  const z = days + 719468;
  // Floor-divide z into 400-year eras (the C++ formula "`(z>=0 ? z : z-146096)/146097`"
  // with truncating C++ integer division is equivalent to Math.floor(z/146097)).
  const era = Math.floor(z / 146097);
  const doe = z - era * 146097;                                                 // day-of-era  [0, 146096]
  const yoe = Math.floor(
    (doe - Math.floor(doe / 1460) + Math.floor(doe / 36524) - Math.floor(doe / 146096)) / 365,
  );                                                                             // year-of-era [0, 399]
  const y = yoe + era * 400;
  const doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100)); // day-of-year [0, 365]
  const mp = Math.floor((5 * doy + 2) / 153);                                  // month pos   [0, 11]
  const d = doy - Math.floor((153 * mp + 2) / 5) + 1;                          // day         [1, 31]
  const m = mp < 10 ? mp + 3 : mp - 9;                                         // month       [1, 12]
  return { year: y + (m <= 2 ? 1 : 0), month: m, day: d };
}

/**
 * civilToDays: {year, month, day} → proleptic Gregorian day number since 1970-01-01.
 *
 * Algorithm: Howard Hinnant days_from_civil
 *   http://howardhinnant.github.io/date_algorithms.html
 *   Exact inverse of daysToCivil.
 */
export function civilToDays(y: number, m: number, d: number): number {
  const yAdj = y - (m <= 2 ? 1 : 0);
  const era = Math.floor(yAdj / 400);
  const yoe = yAdj - era * 400;                                                  // [0, 399]
  const doy = Math.floor((153 * (m > 2 ? m - 3 : m + 9) + 2) / 5) + d - 1;    // [0, 365]
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;   // [0, 146096]
  return era * 146097 + doe - 719468;
}

/**
 * isLeapYear: proleptic Gregorian leap-year rule.
 * Divisible by 4, except centuries, except 400-year multiples.
 */
export function isLeapYear(y: number): boolean {
  return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

/**
 * dayOfYear: 1-based day of year (1 = Jan 1, 365/366 = Dec 31).
 */
export function dayOfYear(y: number, m: number, d: number): number {
  const base = (DAYS_BEFORE_MONTH[m - 1] ?? 0) + d;
  return m > 2 && isLeapYear(y) ? base + 1 : base;
}

/**
 * isoWeekday: ISO 8601 weekday from a day number (days since 1970-01-01).
 * Returns 1 (Monday) through 7 (Sunday).
 *
 * 1970-01-01 (day 0) is Thursday (= 4).
 * Derivation (ADR-010): w = ((days + 3) % 7 + 7) % 7; weekday = w + 1.
 * The `+7)%7` ensures negative-safe modulo (JS `%` can be negative).
 *
 * Verification fixtures (ADR-010):
 *   day -4 (1969-12-28 Sun) → 7;  day 0 (Thu) → 4;  day 4 (Mon) → 1.
 */
export function isoWeekday(days: number): number {
  return (((days + 3) % 7 + 7) % 7) + 1;
}

/**
 * floorDivMs: floor-divide a BigInt epoch-ms value by 86_400_000.
 * Returns an integer day number in JS number range.
 *
 * BigInt `/` truncates toward zero; this corrects for negative non-exact
 * instants so that the result is always the mathematical floor.
 *
 * Worked examples (dtypes.md §7.2):
 *   -1n          → -1  (1969-12-31 23:59:59.999; trunc would give 0 — wrong)
 *   -86_400_001n → -2  (1969-12-30 23:59:59.999; trunc would give -1 — wrong)
 *   -86_400_000n → -1  (1969-12-31 00:00:00.000; exact multiple, trunc = floor)
 *   86_400_500n  →  1  (1970-01-02 00:00:00.500; trunc = floor for positive)
 */
export function floorDivMs(ms: bigint): number {
  let d = ms / MS_PER_DAY_N;
  if (ms % MS_PER_DAY_N !== 0n && ms < 0n) d -= 1n;
  return Number(d);
}

/**
 * msToCivil: epoch-milliseconds → {year, month, day, hour, minute, second, ms}.
 * Input may be bigint (BigInt64Array element) or a safe-integer number.
 * Uses negative-safe floor division; the `ms` field is the millisecond component [0, 999].
 *
 * Cross-check: -1 ms → {1969, 12, 31, 23, 59, 59, 999}
 *   floorDiv(-1, 86400000) = -1; msOfDay = -1 - (-1 * 86400000) = 86399999;
 *   hour = 23, min = 59, sec = 59, ms = 999. ✓
 */
export function msToCivil(ms: bigint | number): CivilDateTime {
  const msB: bigint =
    typeof ms === 'bigint' ? ms : BigInt(Math.trunc(ms));
  const dayNum = floorDivMs(msB);
  // msOfDay is always in [0, MS_PER_DAY - 1]
  const msOfDay = Number(msB - BigInt(dayNum) * MS_PER_DAY_N);

  const { year, month, day } = daysToCivil(dayNum);
  const hour = Math.floor(msOfDay / 3_600_000);
  const rem1 = msOfDay % 3_600_000;
  const minute = Math.floor(rem1 / 60_000);
  const rem2 = rem1 % 60_000;
  const second = Math.floor(rem2 / 1_000);
  const msComponent = rem2 % 1_000;

  return { year, month, day, hour, minute, second, ms: msComponent };
}

// ── High-level field extractors ───────────────────────────────────────────────

/**
 * date32ToFields: all dt accessor fields for a date32 day count.
 * Time-of-day fields are always 0 for date32 (dtypes.md §10).
 */
export function date32ToFields(days: number): CivilFields {
  const { year, month, day } = daysToCivil(days);
  return {
    year,
    month,
    day,
    hour: 0,
    minute: 0,
    second: 0,
    millisecond: 0,
    weekday: isoWeekday(days),
    dayOfYear: dayOfYear(year, month, day),
    quarter: Math.ceil(month / 3),
  };
}

/**
 * timestampUtcToFields: all dt accessor fields for a UTC timestamp.
 * Uses integer civil math only — no Date objects, no Intl.
 */
export function timestampUtcToFields(ms: bigint | number): CivilFields {
  const msB: bigint =
    typeof ms === 'bigint' ? ms : BigInt(Math.trunc(ms));
  const dayNum = floorDivMs(msB);
  const msOfDay = Number(msB - BigInt(dayNum) * MS_PER_DAY_N);

  const { year, month, day } = daysToCivil(dayNum);
  const hour = Math.floor(msOfDay / 3_600_000);
  const rem1 = msOfDay % 3_600_000;
  const minute = Math.floor(rem1 / 60_000);
  const rem2 = rem1 % 60_000;
  const second = Math.floor(rem2 / 1_000);
  const millisecond = rem2 % 1_000;

  return {
    year,
    month,
    day,
    hour,
    minute,
    second,
    millisecond,
    weekday: isoWeekday(dayNum),
    dayOfYear: dayOfYear(year, month, day),
    quarter: Math.ceil(month / 3),
  };
}
