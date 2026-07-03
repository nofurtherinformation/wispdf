/**
 * tests/temporal/conformance.test.ts
 *
 * Conformance runner for frame_accessor-layer cases in
 * tests/conformance/fixtures/temporal.json.
 *
 * Groups COVERED by this file (layer === "frame_accessor"):
 *   - dt_accessors_date32
 *   - dt_accessors_timestamp_utc
 *   - dt_accessors_timestamp_tz
 *
 * Groups SKIPPED (with explicit comment per task spec; handled in later tasks):
 *   - date32_compare_sort        → layer "kernel_reuse" (wasm i32 kernels — later task)
 *   - timestamp_compare_sort     → layer "kernel_reuse" (wasm i64 kernels — later task)
 *   - restricted_temporal_arithmetic → layer "kernel_reuse" (wasm kernels — later task)
 *   - temporal_arithmetic_errors → layer "frame_error"  (frame/expr layer — later task)
 *   - temporal_casts             → layer "frame"        (frame/expr layer — later task)
 *   - temporal_groupby_join      → layer "kernel_reuse" (wasm hash kernels — later task)
 *
 * ICU caveat (from fixture root): tz-aware cases were computed with Node v22.23.1
 * ICU 78.2. The test runner checks process.versions.icu and skips tz cases with a
 * warning if the ICU version differs from "78.2".
 */

import { describe, it, expect } from 'vitest';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { readFileSync } from 'node:fs';
import {
  date32ToFields,
  timestampUtcToFields,
  getIcuVersion,
  getTzComponents,
  type CivilFields,
} from '../../src/temporal/index.js';

// ── Load fixture ──────────────────────────────────────────────────────────────

const __dir = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(__dir, '../conformance/fixtures/temporal.json');

interface FixtureRoot {
  family: string;
  icu_caveat: string;
  cases: FixtureCase[];
}

interface SingleInput {
  date32?: number | null;
  ts_ms?: string | null;
}

interface WeekdayInputRow {
  date32: number;
  expected_weekday: number;
  date?: string;
}

interface FixtureExpected {
  year?: number | null;
  month?: number | null;
  day?: number | null;
  hour?: number | null;
  minute?: number | null;
  second?: number | null;
  millisecond?: number | null;
  weekday?: number | null;
  dayOfYear?: number | null;
  quarter?: number | null;
}

interface FixtureCase {
  group: string;
  name: string;
  layer: string;
  dtype?: string;
  tz?: string | null;
  note?: string;
  // Single-value cases
  input?: SingleInput;
  expected?: FixtureExpected;
  // Multi-row weekday coverage
  inputs?: WeekdayInputRow[] | unknown;
}

const fixture = JSON.parse(
  readFileSync(FIXTURE_PATH, 'utf-8'),
) as FixtureRoot;

// ── ICU version check ─────────────────────────────────────────────────────────

const FIXTURE_ICU = '78.2';
const runtimeIcu = getIcuVersion();
const icuMatches = runtimeIcu === FIXTURE_ICU;

if (!icuMatches) {
  console.warn(
    `[temporal conformance] ICU version mismatch: fixture computed with ${FIXTURE_ICU}, ` +
      `runtime has ${runtimeIcu ?? 'unknown'}. tz-aware accessor cases will be SKIPPED.`,
  );
} else {
  console.log(`[temporal conformance] ICU version: ${runtimeIcu} — tz-aware cases active.`);
}

// ── Helper: compare a field value to expected (null means field must be null) ─

function checkFields(
  result: CivilFields | null,
  expected: FixtureExpected,
  caseName: string,
): void {
  for (const [field, expVal] of Object.entries(expected) as Array<[keyof FixtureExpected, number | null | undefined]>) {
    if (expVal === undefined) continue; // fixture doesn't check this field

    if (expVal === null) {
      // Null input → accessor result should be null (represented here as result === null)
      expect(result, `${caseName}: result should be null for null input`).toBeNull();
      return; // one null check is enough; all fields would be null
    }

    // Non-null: result must have valid fields
    expect(result, `${caseName}: result should not be null for non-null input`).not.toBeNull();
    if (result === null) return;

    const got = result[field as keyof CivilFields];
    expect(got, `${caseName}.${field}`).toBe(expVal);
  }
}

// ── Filter to frame_accessor cases only ───────────────────────────────────────

const accessorCases = fixture.cases.filter((c) => c.layer === 'frame_accessor');

// ── Test runner ───────────────────────────────────────────────────────────────

describe('temporal conformance — frame_accessor layer', () => {

  // ── dt_accessors_date32 ─────────────────────────────────────────────────────

  describe('dt_accessors_date32', () => {
    const cases = accessorCases.filter((c) => c.group === 'dt_accessors_date32');

    for (const tc of cases) {
      it(tc.name, () => {
        // Special case: weekday coverage uses an `inputs` array rather than single `input`
        if (tc.name === 'date32_dt_weekday_coverage_full_week') {
          const rows = tc.inputs as WeekdayInputRow[];
          for (const row of rows) {
            const result = date32ToFields(row.date32);
            expect(result.weekday, `weekday for day ${row.date32} (${row.date ?? ''})`).toBe(
              row.expected_weekday,
            );
          }
          return;
        }

        const input = tc.input as SingleInput;
        const expected = tc.expected as FixtureExpected;

        if (input.date32 === null || input.date32 === undefined) {
          // Null input → null result (validity copied; all fields null)
          checkFields(null, expected, tc.name);
          return;
        }

        const result = date32ToFields(input.date32);
        checkFields(result, expected, tc.name);
      });
    }
  });

  // ── dt_accessors_timestamp_utc ───────────────────────────────────────────────

  describe('dt_accessors_timestamp_utc', () => {
    const cases = accessorCases.filter((c) => c.group === 'dt_accessors_timestamp_utc');

    for (const tc of cases) {
      it(tc.name, () => {
        const input = tc.input as SingleInput;
        const expected = tc.expected as FixtureExpected;

        if (input.ts_ms === null || input.ts_ms === undefined) {
          checkFields(null, expected, tc.name);
          return;
        }

        const ms = BigInt(input.ts_ms);
        const result = timestampUtcToFields(ms);
        checkFields(result, expected, tc.name);
      });
    }
  });

  // ── dt_accessors_timestamp_tz ────────────────────────────────────────────────

  describe('dt_accessors_timestamp_tz', () => {
    const cases = accessorCases.filter((c) => c.group === 'dt_accessors_timestamp_tz');

    for (const tc of cases) {
      const skipTz = !icuMatches && tc.tz !== null && tc.tz !== undefined && tc.tz !== 'UTC';

      it(
        skipTz ? `[SKIPPED — ICU mismatch] ${tc.name}` : tc.name,
        { skip: skipTz },
        () => {
          const input = tc.input as SingleInput;
          const expected = tc.expected as FixtureExpected;

          if (input.ts_ms === null || input.ts_ms === undefined) {
            checkFields(null, expected, tc.name);
            return;
          }

          const epochMs = Number(BigInt(input.ts_ms));
          const tz = tc.tz as string; // non-null for this group (validated by fixture)
          const result = getTzComponents(epochMs, tz);
          checkFields(result, expected, tc.name);
        },
      );
    }
  });

  // ── Completeness: assert we found accessor cases ──────────────────────────────

  it('fixture has dt_accessors_date32 cases', () => {
    const count = accessorCases.filter((c) => c.group === 'dt_accessors_date32').length;
    expect(count).toBeGreaterThan(0);
  });

  it('fixture has dt_accessors_timestamp_utc cases', () => {
    const count = accessorCases.filter((c) => c.group === 'dt_accessors_timestamp_utc').length;
    expect(count).toBeGreaterThan(0);
  });

  it('fixture has dt_accessors_timestamp_tz cases', () => {
    const count = accessorCases.filter((c) => c.group === 'dt_accessors_timestamp_tz').length;
    expect(count).toBeGreaterThan(0);
  });

  // ── Skipped groups (explicit, per task spec) ─────────────────────────────────
  //
  // The following groups are NOT tested in this file because they belong to the
  // wasm kernel layer or the frame/expr layer, which are implemented in later tasks:
  //
  //   date32_compare_sort        → kernel_reuse (min/max/argsort/eq/lt on i32 wasm)
  //   timestamp_compare_sort     → kernel_reuse (min/max/argsort/eq on i64 wasm)
  //   restricted_temporal_arithmetic → kernel_reuse (sub/add i32/i64 wasm)
  //   temporal_arithmetic_errors → frame_error  (frame/expr layer dtype checks)
  //   temporal_casts             → frame         (frame/expr layer cast ops)
  //   temporal_groupby_join      → kernel_reuse  (hash_i32/hash_i64 wasm)

});
