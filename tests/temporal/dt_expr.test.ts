/**
 * tests/temporal/dt_expr.test.ts
 *
 * v2.5b: dt accessor conformance through the expr/Series path (ADR-012).
 *
 * Coverage:
 *   1. Conformance via compile/execute (expr path) — dt_accessors_* fixture groups.
 *   2. Conformance via Series.dt — same fixture groups, same expected values.
 *   3. Property test — expr-path components == pure-module components over 500
 *      random timestamps (including pre-epoch) and 200 random date32 day counts.
 *   4. Disallowed accessors — hour/minute/second/millisecond on date32 throw TypeError.
 *   5. Wrong dtype — dt accessor on a non-temporal column throws.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { readFileSync } from 'node:fs';

import { loadEnv, TestFrame, takeColumn, type TestEnv } from '../expr/helper.js';
import { createColumn, freeColumn } from '../../src/memory/column.js';
import { Series } from '../../src/frame/series.js';
import { compile } from '../../src/expr/compile.js';
import { col } from '../../src/expr/ast.js';
import { getIcuVersion } from '../../src/temporal/tz.js';
import {
  date32ToFields,
  timestampUtcToFields,
  type CivilFields,
} from '../../src/temporal/civil.js';
import type { DtComponent } from '../../src/temporal/vectorize.js';

// ── Load fixture ──────────────────────────────────────────────────────────────

const __dir = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(__dir, '../conformance/fixtures/temporal.json');

interface SingleInput {
  date32?: number | null;
  ts_ms?: string | null;
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
  input?: SingleInput;
  expected?: FixtureExpected;
  inputs?: unknown;
}

const fixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8')) as { icu_caveat?: string; cases: FixtureCase[] };
const allCases = fixture.cases;

// ── ICU version gate (same logic as conformance.test.ts) ─────────────────────

const FIXTURE_ICU = '78.2';
const runtimeIcu = getIcuVersion();
const icuMatches = runtimeIcu === FIXTURE_ICU;

// ── Test environment ─────────────────────────────────────────────────────────

let env: TestEnv;
beforeAll(async () => {
  env = await loadEnv();
});

// ── All dt accessor components in declaration order ───────────────────────────

const ALL_COMPONENTS: DtComponent[] = [
  'year', 'month', 'day', 'hour', 'minute', 'second', 'millisecond',
  'weekday', 'dayOfYear', 'quarter',
];

// ── 1. Conformance: expr compile path ────────────────────────────────────────

describe('dt_accessors via expr compile path', () => {

  // ── date32 ──────────────────────────────────────────────────────────────────

  describe('dt_accessors_date32 — expr path', () => {
    const cases = allCases.filter(
      (c) => c.group === 'dt_accessors_date32' && c.input !== undefined,
    );

    for (const tc of cases) {
      it(tc.name, () => {
        const input = tc.input as SingleInput;
        const expected = tc.expected as FixtureExpected;

        // null input → all expected fields should be null
        if (input.date32 === null || input.date32 === undefined) {
          // Build a null date32 column
          const frame = new TestFrame(env, { x: { dtype: 'date32', values: [null] } });
          try {
            for (const [field, expVal] of Object.entries(expected) as [string, number | null][]) {
              if (expVal === undefined) continue;
              const compName = field as DtComponent;
              // date32 disallows time-of-day fields — skip for the null input test
              // (the dtype error is tested in §4 below)
              if (!['year','month','day','weekday','dayOfYear','quarter'].includes(compName)) continue;
              const plan = compile(col('x').dt[compName](), frame);
              const { column } = plan.execute();
              const result = takeColumn(env.ctx, column!);
              expect(result[0], `${tc.name}.${field}`).toBeNull();
            }
          } finally {
            frame.free();
          }
          return;
        }

        const frame = new TestFrame(env, { x: { dtype: 'date32', values: [input.date32] } });
        try {
          for (const [field, expVal] of Object.entries(expected) as [string, number | null][]) {
            if (expVal === undefined) continue;
            const compName = field as DtComponent;
            // date32 time-of-day fields should always yield 0 (from civil.ts),
            // but the type-checker doesn't restrict them at the expr layer.
            const plan = compile(col('x').dt[compName](), frame);
            const { column } = plan.execute();
            const result = takeColumn(env.ctx, column!);
            expect(result[0], `${tc.name}.${field}`).toBe(expVal);
          }
        } finally {
          frame.free();
        }
      });
    }
  });

  // ── timestamp UTC ─────────────────────────────────────────────────────────

  describe('dt_accessors_timestamp_utc — expr path', () => {
    const cases = allCases.filter((c) => c.group === 'dt_accessors_timestamp_utc');

    for (const tc of cases) {
      it(tc.name, () => {
        const input = tc.input as SingleInput;
        const expected = tc.expected as FixtureExpected;

        if (input.ts_ms === null || input.ts_ms === undefined) {
          const frame = new TestFrame(env, { x: { dtype: 'timestamp', values: [null] } });
          try {
            for (const [field, expVal] of Object.entries(expected) as [string, number | null][]) {
              if (expVal === undefined) continue;
              const compName = field as DtComponent;
              const plan = compile(col('x').dt[compName](), frame);
              const { column } = plan.execute();
              const result = takeColumn(env.ctx, column!);
              expect(result[0], `${tc.name}.${field}`).toBeNull();
            }
          } finally {
            frame.free();
          }
          return;
        }

        const ms = BigInt(input.ts_ms);
        const frame = new TestFrame(env, { x: { dtype: 'timestamp', values: [ms] } });
        try {
          for (const [field, expVal] of Object.entries(expected) as [string, number | null][]) {
            if (expVal === undefined) continue;
            const compName = field as DtComponent;
            const plan = compile(col('x').dt[compName](), frame);
            const { column } = plan.execute();
            const result = takeColumn(env.ctx, column!);
            expect(result[0], `${tc.name}.${field}`).toBe(expVal);
          }
        } finally {
          frame.free();
        }
      });
    }
  });

  // ── timestamp tz ──────────────────────────────────────────────────────────

  describe('dt_accessors_timestamp_tz — expr path', () => {
    const cases = allCases.filter((c) => c.group === 'dt_accessors_timestamp_tz');

    for (const tc of cases) {
      const skipTz = !icuMatches && tc.tz !== null && tc.tz !== undefined && tc.tz !== 'UTC';

      it(
        skipTz ? `[SKIPPED — ICU mismatch] ${tc.name}` : tc.name,
        { skip: skipTz },
        () => {
          const input = tc.input as SingleInput;
          const expected = tc.expected as FixtureExpected;
          const tz = tc.tz as string;

          const ms = input.ts_ms != null ? BigInt(input.ts_ms) : null;

          // Build a timestamp column with tz metadata via createColumn directly.
          const srcCol = createColumn(env.ctx, 'timestamp', [ms], tz);
          // Wrap in a mini-frame that exposes tz through getColumn.
          const frame: import('../../src/expr/frame.js').FrameView = {
            length: 1,
            ctx: env.ctx,
            wasm: env.wasm,
            dtypeOf: () => 'timestamp',
            columnNames: () => ['x'],
            getColumn: () => srcCol,
          };
          try {
            for (const [field, expVal] of Object.entries(expected) as [string, number | null][]) {
              if (expVal === undefined) continue;
              const compName = field as DtComponent;
              const plan = compile(col('x').dt[compName](), frame);
              const { column } = plan.execute();
              const result = takeColumn(env.ctx, column!);
              if (ms === null) {
                expect(result[0], `${tc.name}.${field}`).toBeNull();
              } else {
                expect(result[0], `${tc.name}.${field}`).toBe(expVal);
              }
            }
          } finally {
            freeColumn(env.ctx, srcCol);
          }
        },
      );
    }
  });
});

// ── 2. Conformance: Series.dt path ──────────────────────────────────────────

describe('dt_accessors via Series.dt path', () => {

  // ── date32 ──────────────────────────────────────────────────────────────────

  describe('dt_accessors_date32 — Series.dt path', () => {
    const cases = allCases.filter(
      (c) => c.group === 'dt_accessors_date32' && c.input !== undefined,
    );

    for (const tc of cases) {
      it(tc.name, () => {
        const input = tc.input as SingleInput;
        const expected = tc.expected as FixtureExpected;

        const day = input.date32 ?? null;
        const srcCol = createColumn(env.ctx, 'date32', [day]);
        const s = new Series(env.ctx, 'x', srcCol);

        try {
          for (const [field, expVal] of Object.entries(expected) as [string, number | null][]) {
            if (expVal === undefined) continue;
            const compName = field as DtComponent;
            // Skip time-of-day on date32 (tested for errors in §4)
            if (!['year','month','day','weekday','dayOfYear','quarter'].includes(compName)) continue;

            const result = s.dt[compName]().toArray();
            if (day === null) {
              expect(result[0], `${tc.name}.${field}`).toBeNull();
            } else {
              expect(result[0], `${tc.name}.${field}`).toBe(expVal);
            }
          }
        } finally {
          freeColumn(env.ctx, srcCol);
        }
      });
    }
  });

  // ── timestamp UTC ─────────────────────────────────────────────────────────

  describe('dt_accessors_timestamp_utc — Series.dt path', () => {
    const cases = allCases.filter((c) => c.group === 'dt_accessors_timestamp_utc');

    for (const tc of cases) {
      it(tc.name, () => {
        const input = tc.input as SingleInput;
        const expected = tc.expected as FixtureExpected;

        const ms = input.ts_ms != null ? BigInt(input.ts_ms) : null;
        const srcCol = createColumn(env.ctx, 'timestamp', [ms]);
        const s = new Series(env.ctx, 'x', srcCol);

        try {
          for (const [field, expVal] of Object.entries(expected) as [string, number | null][]) {
            if (expVal === undefined) continue;
            const compName = field as DtComponent;
            const result = s.dt[compName]().toArray();
            if (ms === null) {
              expect(result[0], `${tc.name}.${field}`).toBeNull();
            } else {
              expect(result[0], `${tc.name}.${field}`).toBe(expVal);
            }
          }
        } finally {
          freeColumn(env.ctx, srcCol);
        }
      });
    }
  });

  // ── timestamp tz ──────────────────────────────────────────────────────────

  describe('dt_accessors_timestamp_tz — Series.dt path', () => {
    const cases = allCases.filter((c) => c.group === 'dt_accessors_timestamp_tz');

    for (const tc of cases) {
      const skipTz = !icuMatches && tc.tz !== null && tc.tz !== undefined && tc.tz !== 'UTC';

      it(
        skipTz ? `[SKIPPED — ICU mismatch] ${tc.name}` : tc.name,
        { skip: skipTz },
        () => {
          const input = tc.input as SingleInput;
          const expected = tc.expected as FixtureExpected;
          const tz = tc.tz as string;

          const ms = input.ts_ms != null ? BigInt(input.ts_ms) : null;
          const srcCol = createColumn(env.ctx, 'timestamp', [ms], tz);
          const s = new Series(env.ctx, 'x', srcCol);

          try {
            for (const [field, expVal] of Object.entries(expected) as [string, number | null][]) {
              if (expVal === undefined) continue;
              const compName = field as DtComponent;
              const result = s.dt[compName]().toArray();
              if (ms === null) {
                expect(result[0], `${tc.name}.${field}`).toBeNull();
              } else {
                expect(result[0], `${tc.name}.${field}`).toBe(expVal);
              }
            }
          } finally {
            freeColumn(env.ctx, srcCol);
          }
        },
      );
    }
  });
});

// ── 3. Property: expr-path == pure-module for random timestamps ──────────────

describe('property: expr-path components == pure-module components', () => {

  it('timestamp UTC: 500 random values including pre-epoch', () => {
    // Generate a mix of positive and negative epoch-ms values.
    const seed = 42;
    const values: bigint[] = [];
    let lcg = seed;
    for (let i = 0; i < 500; i++) {
      // Simple LCG for determinism
      lcg = (lcg * 1664525 + 1013904223) & 0xffffffff;
      // Map to epoch-ms in range ±50 years around 1970 (fits in safe integer)
      const ms = ((lcg | 0) % (50 * 365 * 24 * 3600 * 1000));
      values.push(BigInt(ms));
    }

    const frame = new TestFrame(env, { ts: { dtype: 'timestamp', values } });
    try {
      for (const comp of ALL_COMPONENTS) {
        const plan = compile(col('ts').dt[comp](), frame);
        const { column } = plan.execute();
        const result = takeColumn(env.ctx, column!);

        for (let i = 0; i < values.length; i++) {
          const expected = timestampUtcToFields(values[i]!)[comp];
          expect(result[i], `ts[${i}] comp=${comp} ms=${values[i]}`).toBe(expected);
        }
      }
    } finally {
      frame.free();
    }
  });

  it('date32: 200 random day counts including pre-epoch', () => {
    const values: number[] = [];
    let lcg = 99;
    for (let i = 0; i < 200; i++) {
      lcg = (lcg * 1664525 + 1013904223) & 0xffffffff;
      values.push((lcg | 0) % 50000); // roughly ±50000 days around 1970
    }

    const frame = new TestFrame(env, { d: { dtype: 'date32', values } });
    const DATE32_COMPONENTS: DtComponent[] = ['year','month','day','hour','minute','second','millisecond','weekday','dayOfYear','quarter'];
    try {
      for (const comp of DATE32_COMPONENTS) {
        const plan = compile(col('d').dt[comp](), frame);
        const { column } = plan.execute();
        const result = takeColumn(env.ctx, column!);

        for (let i = 0; i < values.length; i++) {
          const fields = date32ToFields(values[i]!);
          const expected = fields[comp];
          expect(result[i], `d[${i}] comp=${comp} day=${values[i]}`).toBe(expected);
        }
      }
    } finally {
      frame.free();
    }
  });

  it('Series.dt timestamp UTC: components match pure-module for 100 random values', () => {
    const values: bigint[] = [];
    let lcg = 7777;
    for (let i = 0; i < 100; i++) {
      lcg = (lcg * 1664525 + 1013904223) & 0xffffffff;
      values.push(BigInt((lcg | 0) % (30 * 365 * 24 * 3600 * 1000)));
    }

    const srcCol = createColumn(env.ctx, 'timestamp', values);
    const s = new Series(env.ctx, 'ts', srcCol);
    try {
      for (const comp of ALL_COMPONENTS) {
        const result = s.dt[comp]().toArray();
        for (let i = 0; i < values.length; i++) {
          const expected = timestampUtcToFields(values[i]!)[comp];
          expect(result[i], `ts[${i}] comp=${comp}`).toBe(expected);
        }
      }
    } finally {
      freeColumn(env.ctx, srcCol);
    }
  });
});

// ── 4. Disallowed accessors: hour/minute/second/millisecond on date32 ────────

describe('Series.dt: disallowed accessors on date32 throw TypeError', () => {
  const DISALLOWED: DtComponent[] = ['hour', 'minute', 'second', 'millisecond'];

  for (const comp of DISALLOWED) {
    it(`date32.dt.${comp}() throws TypeError`, () => {
      const srcCol = createColumn(env.ctx, 'date32', [0]);
      const s = new Series(env.ctx, 'x', srcCol);
      try {
        expect(() => s.dt[comp]()).toThrow(TypeError);
        expect(() => s.dt[comp]()).toThrow(comp); // message names the accessor
      } finally {
        freeColumn(env.ctx, srcCol);
      }
    });
  }
});

// ── 5. Wrong dtype: dt on non-temporal column throws ─────────────────────────

describe('Series.dt: TypeError on non-temporal dtype', () => {
  const wrongDtypes = [
    { dtype: 'i32' as const, values: [1, 2, 3] },
    { dtype: 'f64' as const, values: [1.0, 2.0] },
    { dtype: 'bool' as const, values: [true, false] },
  ];

  for (const { dtype, values } of wrongDtypes) {
    it(`Series<${dtype}>.dt throws TypeError`, () => {
      const srcCol = createColumn(env.ctx, dtype, values);
      const s = new Series(env.ctx, 'x', srcCol);
      try {
        expect(() => s.dt).toThrow(TypeError);
        expect(() => s.dt).toThrow(dtype); // message names the dtype
      } finally {
        freeColumn(env.ctx, srcCol);
      }
    });
  }
});
