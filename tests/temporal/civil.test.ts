/**
 * tests/temporal/civil.test.ts
 *
 * Property tests and table-driven tests for src/temporal/civil.ts.
 * No WASM needed — pure JS integer math.
 *
 * Coverage:
 *  - civilToDays(daysToCivil(d)) === d  round-trip over a wide range (property)
 *  - Cross-check a sample of day→date against Date.UTC (where safe-integer range)
 *  - Leap-year table: 1900 (not), 2000 (leap), 2100 (not), 2024 (leap)
 *  - floorDivMs negative cases from dtypes.md §7.2 worked example
 *  - isoWeekday full week fixture (ADR-010 §Decision weekday)
 *  - msToCivil: pre-epoch end-of-day = 1969-12-31T23:59:59.999
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  daysToCivil,
  civilToDays,
  isLeapYear,
  dayOfYear,
  isoWeekday,
  floorDivMs,
  msToCivil,
} from '../../src/temporal/civil.js';

// ── Round-trip property ──────────────────────────────────────────────────────

describe('civilToDays(daysToCivil(d)) round-trip', () => {
  it('holds for d in [-500000, 500000]', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -500_000, max: 500_000 }),
        (d) => {
          const { year, month, day } = daysToCivil(d);
          return civilToDays(year, month, day) === d;
        },
      ),
      { numRuns: 5_000 },
    );
  });

  it('holds for large positive day values (far future)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 500_001, max: 2_000_000 }),
        (d) => {
          const { year, month, day } = daysToCivil(d);
          return civilToDays(year, month, day) === d;
        },
      ),
      { numRuns: 1_000 },
    );
  });

  it('holds for large negative day values (far past)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -2_000_000, max: -500_001 }),
        (d) => {
          const { year, month, day } = daysToCivil(d);
          return civilToDays(year, month, day) === d;
        },
      ),
      { numRuns: 1_000 },
    );
  });
});

// ── Cross-check against Date.UTC ─────────────────────────────────────────────

describe('daysToCivil vs Date.UTC cross-check', () => {
  // For years in [1970, 9000] Date.UTC is reliable (no 2-digit year ambiguity).
  it('matches Date.UTC for days in [0, 365*1000] (years 1970 ~ 4706)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 365 * 1_000 }),
        (d) => {
          const { year, month, day } = daysToCivil(d);
          if (year < 100 || year > 9000) return true; // skip edge cases
          const utcMs = Date.UTC(year, month - 1, day);
          const expectedDay = utcMs / 86_400_000;
          return expectedDay === d;
        },
      ),
      { numRuns: 2_000 },
    );
  });

  it('matches Date.UTC for some pre-epoch dates (days in [-10000, -1])', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -10_000, max: -1 }),
        (d) => {
          const { year, month, day } = daysToCivil(d);
          if (year < 1 || year > 9000) return true;
          const utcMs = Date.UTC(year, month - 1, day);
          const expectedDay = utcMs / 86_400_000;
          return expectedDay === d;
        },
      ),
      { numRuns: 1_000 },
    );
  });

  it('known spot-checks', () => {
    // 2000-02-29 (leap day): day 11016
    expect(daysToCivil(11016)).toEqual({ year: 2000, month: 2, day: 29 });
    expect(civilToDays(2000, 2, 29)).toBe(11016);
    // 1969-12-31: day -1
    expect(daysToCivil(-1)).toEqual({ year: 1969, month: 12, day: 31 });
    expect(civilToDays(1969, 12, 31)).toBe(-1);
    // 2038-01-19 (Y2K38): day 24855
    expect(daysToCivil(24855)).toEqual({ year: 2038, month: 1, day: 19 });
    expect(civilToDays(2038, 1, 19)).toBe(24855);
    // Epoch: day 0
    expect(daysToCivil(0)).toEqual({ year: 1970, month: 1, day: 1 });
    expect(civilToDays(1970, 1, 1)).toBe(0);
  });
});

// ── Leap-year table ───────────────────────────────────────────────────────────

describe('isLeapYear', () => {
  it('1900: not leap (divisible by 100, not by 400)', () => {
    expect(isLeapYear(1900)).toBe(false);
  });

  it('2000: leap (divisible by 400)', () => {
    expect(isLeapYear(2000)).toBe(true);
  });

  it('2100: not leap (divisible by 100, not by 400)', () => {
    expect(isLeapYear(2100)).toBe(false);
  });

  it('2024: leap (divisible by 4, not by 100)', () => {
    expect(isLeapYear(2024)).toBe(true);
  });

  it('2023: not leap (not divisible by 4)', () => {
    expect(isLeapYear(2023)).toBe(false);
  });

  it('400: leap', () => {
    expect(isLeapYear(400)).toBe(true);
  });

  it('1: not leap', () => {
    expect(isLeapYear(1)).toBe(false);
  });
});

// ── dayOfYear ─────────────────────────────────────────────────────────────────

describe('dayOfYear', () => {
  it('Jan 1 is always day 1', () => {
    expect(dayOfYear(2000, 1, 1)).toBe(1);
    expect(dayOfYear(1970, 1, 1)).toBe(1);
    expect(dayOfYear(1969, 1, 1)).toBe(1);
  });

  it('2000-02-29 (leap): day 60', () => {
    expect(dayOfYear(2000, 2, 29)).toBe(60);
  });

  it('1969-12-31 (non-leap): day 365', () => {
    expect(dayOfYear(1969, 12, 31)).toBe(365);
  });

  it('2038-01-19: day 19', () => {
    expect(dayOfYear(2038, 1, 19)).toBe(19);
  });

  it('non-leap Dec 31 is day 365; leap Dec 31 is day 366', () => {
    expect(dayOfYear(2023, 12, 31)).toBe(365);
    expect(dayOfYear(2024, 12, 31)).toBe(366);
  });

  it('Mar 1 is day 60 in non-leap, day 61 in leap', () => {
    expect(dayOfYear(2023, 3, 1)).toBe(60);
    expect(dayOfYear(2024, 3, 1)).toBe(61);
  });
});

// ── isoWeekday ────────────────────────────────────────────────────────────────

describe('isoWeekday', () => {
  // Full-week fixture from ADR-010 and temporal.json date32_dt_weekday_coverage_full_week
  const WEEK: Array<[number, number, string]> = [
    [-4, 7, '1969-12-28 Sun'],
    [-3, 1, '1969-12-29 Mon'],
    [-2, 2, '1969-12-30 Tue'],
    [-1, 3, '1969-12-31 Wed'],
    [ 0, 4, '1970-01-01 Thu'],
    [ 1, 5, '1970-01-02 Fri'],
    [ 2, 6, '1970-01-03 Sat'],
    [ 3, 7, '1970-01-04 Sun'],
  ];

  for (const [days, expected, label] of WEEK) {
    it(`day ${days} (${label}) → weekday ${expected}`, () => {
      expect(isoWeekday(days)).toBe(expected);
    });
  }

  it('property: isoWeekday always returns 1..7', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -1_000_000, max: 1_000_000 }),
        (d) => {
          const w = isoWeekday(d);
          return w >= 1 && w <= 7;
        },
      ),
    );
  });

  it('property: consecutive days differ by 1 (mod 7), wrapping Sun→Mon', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -1_000_000, max: 999_999 }),
        (d) => {
          const w1 = isoWeekday(d);
          const w2 = isoWeekday(d + 1);
          // Sunday (7) wraps to Monday (1)
          const expected = w1 === 7 ? 1 : w1 + 1;
          return w2 === expected;
        },
      ),
    );
  });
});

// ── floorDivMs ───────────────────────────────────────────────────────────────

describe('floorDivMs', () => {
  // Worked examples from dtypes.md §7.2
  it('-1n → -1 (floor, not 0 which trunc gives)', () => {
    expect(floorDivMs(-1n)).toBe(-1);
  });

  it('-86_400_001n → -2 (floor, not -1 which trunc gives)', () => {
    expect(floorDivMs(-86_400_001n)).toBe(-2);
  });

  it('-86_400_000n → -1 (exact multiple, floor = trunc)', () => {
    expect(floorDivMs(-86_400_000n)).toBe(-1);
  });

  it('-172_800_001n → -3', () => {
    expect(floorDivMs(-172_800_001n)).toBe(-3);
  });

  it('-172_800_000n → -2 (exact multiple)', () => {
    expect(floorDivMs(-172_800_000n)).toBe(-2);
  });

  it('0n → 0', () => {
    expect(floorDivMs(0n)).toBe(0);
  });

  it('86_400_000n → 1 (positive exact)', () => {
    expect(floorDivMs(86_400_000n)).toBe(1);
  });

  it('86_400_500n → 1 (positive non-exact: floor = trunc)', () => {
    expect(floorDivMs(86_400_500n)).toBe(1);
  });

  it('951_782_400_000n → 11016 (2000-02-29)', () => {
    expect(floorDivMs(951_782_400_000n)).toBe(11016);
  });

  it('property: result * MS_PER_DAY <= ms < (result+1) * MS_PER_DAY', () => {
    fc.assert(
      fc.property(
        // Use safe-integer range for easy conversion
        fc.bigInt({ min: BigInt(-1e12), max: BigInt(1e12) }),
        (ms) => {
          const d = BigInt(floorDivMs(ms));
          const DAY = 86_400_000n;
          return d * DAY <= ms && ms < (d + 1n) * DAY;
        },
      ),
      { numRuns: 2_000 },
    );
  });
});

// ── msToCivil ────────────────────────────────────────────────────────────────

describe('msToCivil', () => {
  it('0 ms → 1970-01-01T00:00:00.000', () => {
    expect(msToCivil(0n)).toEqual({ year: 1970, month: 1, day: 1, hour: 0, minute: 0, second: 0, ms: 0 });
  });

  it('-1 ms → 1969-12-31T23:59:59.999 (critical negative-floor fixture)', () => {
    expect(msToCivil(-1n)).toEqual({ year: 1969, month: 12, day: 31, hour: 23, minute: 59, second: 59, ms: 999 });
  });

  it('-86_400_000 ms → 1969-12-31T00:00:00.000', () => {
    expect(msToCivil(-86_400_000n)).toEqual({ year: 1969, month: 12, day: 31, hour: 0, minute: 0, second: 0, ms: 0 });
  });

  it('3_723_000 ms → 1970-01-01T01:02:03.000', () => {
    expect(msToCivil(3_723_000n)).toEqual({ year: 1970, month: 1, day: 1, hour: 1, minute: 2, second: 3, ms: 0 });
  });

  it('951_782_400_000 ms → 2000-02-29T00:00:00.000', () => {
    expect(msToCivil(951_782_400_000n)).toEqual({ year: 2000, month: 2, day: 29, hour: 0, minute: 0, second: 0, ms: 0 });
  });

  it('2_147_483_647_000 ms → 2038-01-19T03:14:07.000', () => {
    // dayNum = floor(2147483647000 / 86400000) = 24855
    // msOfDay = 2147483647000 - 24855 * 86400000 = 11647000
    // hour = 3 (11647000 / 3600000 = 3.235)
    // min = 14 (11647000 % 3600000 = 847000; 847000/60000 = 14.1)
    // sec = 7 (847000 % 60000 = 7000; 7000/1000 = 7)
    expect(msToCivil(2_147_483_647_000n)).toEqual({ year: 2038, month: 1, day: 19, hour: 3, minute: 14, second: 7, ms: 0 });
  });

  it('accepts number input as well as bigint', () => {
    expect(msToCivil(0)).toEqual({ year: 1970, month: 1, day: 1, hour: 0, minute: 0, second: 0, ms: 0 });
    expect(msToCivil(3_723_000)).toEqual({ year: 1970, month: 1, day: 1, hour: 1, minute: 2, second: 3, ms: 0 });
  });
});
