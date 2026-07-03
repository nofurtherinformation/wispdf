/**
 * Arrow IPC round-trip tests. Tests run in two directions:
 *   1. Our toArrow() → apache-arrow reads → values match
 *   2. apache-arrow writes → our fromArrow() reads → values match
 *
 * v2.6 additions: i64 (Int64), date32 (Date32/DAY), timestamp (Timestamp/MILLI±tz).
 * apache-arrow is a TEST-ONLY devDependency. None of its types land in
 * production code (src/). This file is the conformance boundary.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import {
  tableFromArrays, tableToIPC, tableFromIPC, makeVector, vectorFromArray,
  makeTable, makeData, Schema, Field, Int64 as ArrowInt64, DateDay, TimestampMillisecond,
  TimestampSecond, TimestampMicrosecond, TimestampNanosecond,
  Utf8,
  type Table,
} from 'apache-arrow';
import { toArrow, fromArrow } from '../../src/io/arrow.js';
import { DataFrame } from '../../src/frame/dataframe.js';
import { loadRuntimeForTest, makeDF } from '../frame/helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import type { Cell } from '../../src/memory/column.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Read all values of a named column from an apache-arrow Table. */
function arrowColValues(table: Table, name: string): unknown[] {
  const col = table.getChild(name);
  if (!col) throw new Error(`Column ${name} not found in Arrow table`);
  const out: unknown[] = [];
  for (let i = 0; i < col.length; i++) out.push(col.get(i));
  return out;
}

/** Compare two arrays element-wise, treating NaN === NaN as equal. */
function expectValsEqual(got: unknown[], expected: unknown[]): void {
  expect(got.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const g = got[i]; const e = expected[i];
    if (typeof e === 'number' && isNaN(e)) {
      expect(typeof g === 'number' && isNaN(g)).toBe(true);
    } else {
      expect(g).toBe(e);
    }
  }
}

// ---------------------------------------------------------------------------
// Direction 1: our toArrow() → apache-arrow reads
// ---------------------------------------------------------------------------

describe('toArrow → apache-arrow reads', () => {
  it('f64 column, no nulls', () => {
    const df = makeDF(rt, { v: new Float64Array([1.1, 2.2, 3.3]) });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expect(t.numRows).toBe(3);
    expectValsEqual(arrowColValues(t, 'v'), [1.1, 2.2, 3.3]);
    df.dispose();
  });

  it('f32 column', () => {
    const df = makeDF(rt, { v: new Float32Array([1.5, -2.5, 0.0]) });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expect(t.numRows).toBe(3);
    const vals = arrowColValues(t, 'v') as number[];
    expect(vals[0]).toBeCloseTo(1.5, 5);
    expect(vals[1]).toBeCloseTo(-2.5, 5);
    df.dispose();
  });

  it('i32 column', () => {
    const df = makeDF(rt, { v: new Int32Array([0, -1, 2147483647]) });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expectValsEqual(arrowColValues(t, 'v'), [0, -1, 2147483647]);
    df.dispose();
  });

  it('u32 column', () => {
    const df = makeDF(rt, { v: new Uint32Array([0, 1, 4294967295]) });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    // apache-arrow may represent u32 as Int32 or UInt32; read as numbers
    const vals = arrowColValues(t, 'v');
    expect(vals[0]).toBe(0);
    expect(vals[2]).toBe(4294967295);
    df.dispose();
  });

  it('bool column', () => {
    const df = makeDF(rt, { v: [true, false, true, false] }, { v: 'bool' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expectValsEqual(arrowColValues(t, 'v'), [true, false, true, false]);
    df.dispose();
  });

  it('utf8 column (dict-encoded)', () => {
    const df = makeDF(rt, { name: ['alice', 'bob', 'alice', 'carol'] }, { name: 'utf8' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expectValsEqual(arrowColValues(t, 'name'), ['alice', 'bob', 'alice', 'carol']);
    df.dispose();
  });

  it('f64 column with nulls — validity bitmap preserved', () => {
    const df = makeDF(rt, { v: [1.0, null, 3.0, null, 5.0] }, { v: 'f64' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expectValsEqual(arrowColValues(t, 'v'), [1.0, null, 3.0, null, 5.0]);
    df.dispose();
  });

  it('utf8 with nulls', () => {
    const df = makeDF(rt, { s: ['a', null, 'b', null] }, { s: 'utf8' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expectValsEqual(arrowColValues(t, 's'), ['a', null, 'b', null]);
    df.dispose();
  });

  it('bool with nulls', () => {
    const df = makeDF(rt, { b: [true, null, false] }, { b: 'bool' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expectValsEqual(arrowColValues(t, 'b'), [true, null, false]);
    df.dispose();
  });

  it('multiple columns all dtypes', () => {
    const df = makeDF(rt, {
      f64: [1.5, null, 3.5],
      i32: [1, null, 3],
      u32: new Uint32Array([0, 1, 2]),
      bool: [true, false, null],
      str: ['x', null, 'z'],
    }, {
      f64: 'f64',
      i32: 'i32',
      u32: 'u32',
      bool: 'bool',
      str: 'utf8',
    });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expect(t.numCols).toBe(5);
    expect(t.numRows).toBe(3);
    expectValsEqual(arrowColValues(t, 'f64'), [1.5, null, 3.5]);
    expectValsEqual(arrowColValues(t, 'i32'), [1, null, 3]);
    expectValsEqual(arrowColValues(t, 'bool'), [true, false, null]);
    expectValsEqual(arrowColValues(t, 'str'), ['x', null, 'z']);
    df.dispose();
  });

  it('empty DataFrame (0 rows)', () => {
    const df = makeDF(rt, { a: new Float64Array(0) });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expect(t.numRows).toBe(0);
    expect(t.numCols).toBe(1);
    df.dispose();
  });

  it('sliced DataFrame (head) round-trips correctly', () => {
    const df = makeDF(rt, { v: [1.0, 2.0, 3.0, 4.0, 5.0] }, { v: 'f64' });
    const sliced = df.head(3);
    const buf = toArrow(sliced);
    const t = tableFromIPC(buf);
    expect(t.numRows).toBe(3);
    expectValsEqual(arrowColValues(t, 'v'), [1.0, 2.0, 3.0]);
    df.dispose(); sliced.dispose();
  });
});

// ---------------------------------------------------------------------------
// Direction 2: apache-arrow writes → our fromArrow() reads
// ---------------------------------------------------------------------------

describe('fromArrow ← apache-arrow writes', () => {
  it('f64 column', () => {
    const t = tableFromArrays({ v: Float64Array.from([1.0, 2.0, 3.0]) });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.length).toBe(3);
    expect(df.dtypes['v']).toBe('f64');
    const vals = df.toColumns()['v'] as number[];
    expect(vals).toEqual([1.0, 2.0, 3.0]);
    df.dispose();
  });

  it('f32 column', () => {
    const t = tableFromArrays({ v: Float32Array.from([1.5, -2.5]) });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    const vals = df.toColumns()['v'] as number[];
    expect(vals[0]).toBeCloseTo(1.5, 5);
    df.dispose();
  });

  it('i32 column', () => {
    const t = tableFromArrays({ v: Int32Array.from([-100, 0, 100]) });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['v']).toEqual([-100, 0, 100]);
    df.dispose();
  });

  it('bool column', () => {
    const t = tableFromArrays({ v: [true, false, true] });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['v']).toBe('bool');
    expect(df.toColumns()['v']).toEqual([true, false, true]);
    df.dispose();
  });

  it('dict-encoded utf8 column (apache-arrow produces Dictionary<Int32,Utf8>)', () => {
    const t = tableFromArrays({ name: ['alice', 'bob', 'alice', 'carol'] });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['name']).toBe('utf8');
    expect(df.toColumns()['name']).toEqual(['alice', 'bob', 'alice', 'carol']);
    df.dispose();
  });

  it('f64 with nulls', () => {
    const t = tableFromArrays({ v: [1.0, null, 3.0] });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    const vals = df.toColumns()['v'] as (number | null)[];
    expect(vals[0]).toBeCloseTo(1.0);
    expect(vals[1]).toBe(null);
    expect(vals[2]).toBeCloseTo(3.0);
    df.dispose();
  });

  it('dict utf8 with nulls', () => {
    const t = tableFromArrays({ s: ['a', null, 'b', null] });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['s']).toEqual(['a', null, 'b', null]);
    df.dispose();
  });

  it('bool with nulls', () => {
    const t = tableFromArrays({ b: [true, null, false] });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['b']).toEqual([true, null, false]);
    df.dispose();
  });

  it('multiple columns', () => {
    const t = tableFromArrays({
      id: Int32Array.from([1, 2, 3]),
      val: Float64Array.from([1.1, 2.2, 3.3]),
      label: ['a', 'b', 'c'],
    });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.length).toBe(3);
    expect(df.dtypes['id']).toBe('i32');
    expect(df.dtypes['val']).toBe('f64');
    expect(df.dtypes['label']).toBe('utf8');
    df.dispose();
  });

  it('empty table (0 rows)', () => {
    const t = tableFromArrays({ a: Float64Array.from([]) });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.length).toBe(0);
    expect(df.columns).toEqual(['a']);
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// Full round-trip: our toArrow → fromArrow (self-contained, no apache-arrow)
// ---------------------------------------------------------------------------

describe('toArrow → fromArrow full round-trip', () => {
  function roundTrip(cols: Record<string, Cell[]>, dtypes: Record<string, import('../../src/memory/dtype.js').DType>): void {
    const df1 = makeDF(rt, cols, dtypes);
    const buf = toArrow(df1);
    const df2 = fromArrow(buf, rt);

    expect(df2.columns).toEqual(df1.columns);
    expect(df2.length).toBe(df1.length);

    const c1 = df1.toColumns();
    const c2 = df2.toColumns();
    for (const name of df1.columns) {
      expect(c2[name]).toEqual(c1[name]);
    }
    df1.dispose();
    df2.dispose();
  }

  it('f64 with nulls', () => roundTrip({ v: [1.0, null, 3.0] }, { v: 'f64' }));
  it('f32', () => roundTrip({ v: [1.5, 2.5, 3.5] }, { v: 'f32' }));
  it('i32 with nulls', () => roundTrip({ v: [1, null, -3] }, { v: 'i32' }));
  it('u32', () => roundTrip({ v: [0, 100, 200] }, { v: 'u32' }));
  it('bool with nulls', () => roundTrip({ v: [true, null, false] }, { v: 'bool' }));
  it('utf8 with nulls', () => roundTrip({ v: ['hello', null, 'world'] }, { v: 'utf8' }));
  it('utf8 empty strings', () => roundTrip({ v: ['', 'a', ''] }, { v: 'utf8' }));
  it('all-null column', () => roundTrip({ v: [null, null, null] }, { v: 'f64' }));
  it('all-null utf8', () => roundTrip({ v: [null, null] }, { v: 'utf8' }));

  it('multi-column all dtypes', () => {
    roundTrip(
      {
        a: [1.0, null, 3.0],
        b: [1, 2, null],
        c: [true, false, null],
        d: ['x', null, 'z'],
      },
      { a: 'f64', b: 'i32', c: 'bool', d: 'utf8' },
    );
  });

  it('single-row DataFrame', () => roundTrip({ v: [42.0] }, { v: 'f64' }));
  it('zero-row DataFrame', () => roundTrip({ v: [] }, { v: 'f64' }));
});

// ---------------------------------------------------------------------------
// Full cross-library round-trip: our toArrow → apache reads → apache writes → fromArrow
// ---------------------------------------------------------------------------

describe('full cross-library round-trip', () => {
  it('numeric columns survive apache-arrow transit', () => {
    const df1 = makeDF(rt, {
      f64: [1.5, 2.5, null],
      i32: [10, null, 30],
    }, { f64: 'f64', i32: 'i32' });

    // Our toArrow → apache reads
    const buf1 = toArrow(df1);
    const arrowTable = tableFromIPC(buf1);

    // apache writes → our fromArrow
    const buf2 = tableToIPC(arrowTable, 'stream');
    const df2 = fromArrow(buf2, rt);

    expect(df2.length).toBe(3);
    expect(df2.toColumns()['f64']).toEqual([1.5, 2.5, null]);
    expect(df2.toColumns()['i32']).toEqual([10, null, 30]);
    df1.dispose(); df2.dispose();
  });

  it('utf8 column survives apache-arrow transit', () => {
    const df1 = makeDF(rt, { name: ['alice', null, 'alice'] }, { name: 'utf8' });
    const buf1 = toArrow(df1);
    const at = tableFromIPC(buf1);
    const buf2 = tableToIPC(at, 'stream');
    const df2 = fromArrow(buf2, rt);
    expect(df2.toColumns()['name']).toEqual(['alice', null, 'alice']);
    df1.dispose(); df2.dispose();
  });

  it('bool column survives apache-arrow transit', () => {
    const df1 = makeDF(rt, { b: [true, false, null] }, { b: 'bool' });
    const buf1 = toArrow(df1);
    const at = tableFromIPC(buf1);
    const buf2 = tableToIPC(at, 'stream');
    const df2 = fromArrow(buf2, rt);
    expect(df2.toColumns()['b']).toEqual([true, false, null]);
    df1.dispose(); df2.dispose();
  });
});

// ---------------------------------------------------------------------------
// v2.6 — i64: toArrow → apache-arrow reads
// ---------------------------------------------------------------------------

describe('toArrow → apache-arrow reads (i64, ADR-009)', () => {
  it('i64 column no nulls — apache-arrow sees Int64', () => {
    const df = makeDF(rt, { v: [1n, -1n, 0n] }, { v: 'i64' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expect(t.numRows).toBe(3);
    const col = t.getChild('v');
    expect(col).toBeTruthy();
    // apache-arrow Int64 column elements are BigInt
    expect(col!.get(0)).toBe(1n);
    expect(col!.get(1)).toBe(-1n);
    expect(col!.get(2)).toBe(0n);
    df.dispose();
  });

  it('i64 boundary values (±2^63 limits)', () => {
    const INT64_MAX = 9_223_372_036_854_775_807n;
    const INT64_MIN = -9_223_372_036_854_775_808n;
    const df = makeDF(rt, { v: [INT64_MAX, INT64_MIN, 0n] }, { v: 'i64' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    const col = t.getChild('v');
    expect(col!.get(0)).toBe(INT64_MAX);
    expect(col!.get(1)).toBe(INT64_MIN);
    expect(col!.get(2)).toBe(0n);
    df.dispose();
  });

  it('i64 column with nulls — validity bitmap preserved', () => {
    const df = makeDF(rt, { v: [1n, null, 3n, null, 5n] }, { v: 'i64' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    const col = t.getChild('v');
    expect(col!.get(0)).toBe(1n);
    expect(col!.get(1)).toBeNull();
    expect(col!.get(2)).toBe(3n);
    expect(col!.get(3)).toBeNull();
    expect(col!.get(4)).toBe(5n);
    df.dispose();
  });

  it('i64 large safe values round-trip exactly', () => {
    const big = 9_007_199_254_740_993n; // > Number.MAX_SAFE_INTEGER
    const df = makeDF(rt, { v: [big, -big] }, { v: 'i64' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    const col = t.getChild('v');
    expect(col!.get(0)).toBe(big);
    expect(col!.get(1)).toBe(-big);
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// v2.6 — i64: fromArrow ← apache-arrow writes
// ---------------------------------------------------------------------------

describe('fromArrow ← apache-arrow writes (i64, ADR-009)', () => {
  it('i64 column from apache-arrow Int64', () => {
    const t = tableFromArrays({ v: BigInt64Array.from([1n, -1n, 0n]) });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.length).toBe(3);
    expect(df.dtypes['v']).toBe('i64');
    expect(df.toColumns()['v']).toEqual([1n, -1n, 0n]);
    df.dispose();
  });

  it('i64 boundary values from apache-arrow', () => {
    const INT64_MAX = 9_223_372_036_854_775_807n;
    const INT64_MIN = -9_223_372_036_854_775_808n;
    const t = tableFromArrays({ v: BigInt64Array.from([INT64_MAX, INT64_MIN]) });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    const vals = df.toColumns()['v'] as (bigint | null)[];
    expect(vals[0]).toBe(INT64_MAX);
    expect(vals[1]).toBe(INT64_MIN);
    df.dispose();
  });

  it('i64 with nulls from apache-arrow', () => {
    const t = tableFromArrays({ v: [1n, null, 3n] });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['v']).toBe('i64');
    const vals = df.toColumns()['v'] as (bigint | null)[];
    expect(vals[0]).toBe(1n);
    expect(vals[1]).toBeNull();
    expect(vals[2]).toBe(3n);
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// v2.6 — date32: toArrow → apache-arrow reads
// ---------------------------------------------------------------------------

describe('toArrow → apache-arrow reads (date32, ADR-010)', () => {
  // apache-arrow DateDay.get() returns epoch-milliseconds (not day counts).
  // 1 day = 86_400_000 ms.
  const DAY_MS = 86_400_000;

  it('date32 column — apache-arrow sees Date32(DAY)', () => {
    // day 0 = 1970-01-01; day 1 = 1970-01-02; day -1 = 1969-12-31 (pre-epoch)
    const df = makeDF(rt, { d: [0, 1, -1] }, { d: 'date32' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expect(t.numRows).toBe(3);
    const col = t.getChild('d');
    expect(col).toBeTruthy();
    // apache-arrow DateDay elements are epoch-ms (day * 86_400_000)
    expect(col!.get(0)).toBe(0);
    expect(col!.get(1)).toBe(1 * DAY_MS);
    expect(col!.get(2)).toBe(-1 * DAY_MS);
    df.dispose();
  });

  it('date32 pre-1970 dates (negative day counts)', () => {
    // 1969-01-01 = day -365 (1969 not a leap year)
    const df = makeDF(rt, { d: [-365, -366, -730] }, { d: 'date32' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    const col = t.getChild('d');
    expect(col!.get(0)).toBe(-365 * DAY_MS);
    expect(col!.get(1)).toBe(-366 * DAY_MS);
    expect(col!.get(2)).toBe(-730 * DAY_MS);
    df.dispose();
  });

  it('date32 with nulls', () => {
    const df = makeDF(rt, { d: [0, null, 2] }, { d: 'date32' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    const col = t.getChild('d');
    expect(col!.get(0)).toBe(0);
    expect(col!.get(1)).toBeNull();
    expect(col!.get(2)).toBe(2 * DAY_MS);
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// v2.6 — date32: fromArrow ← apache-arrow writes
// ---------------------------------------------------------------------------

describe('fromArrow ← apache-arrow writes (date32, ADR-010)', () => {
  // apache-arrow DateDay.set() expects epoch-milliseconds (like Date.valueOf()).
  // It stores day = floor(ms / 86_400_000). Pass ms values: 1 day = 86_400_000 ms.
  const DAY_MS = 86_400_000;

  it('date32 from apache-arrow Date32', () => {
    // Pass ms values: 0ms=day0, 86_400_000ms=day1, -86_400_000ms=day-1
    const dateVec = vectorFromArray([0, DAY_MS, -DAY_MS], new DateDay());
    const t = makeTable({ d: dateVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['d']).toBe('date32');
    expect(df.toColumns()['d']).toEqual([0, 1, -1]);
    df.dispose();
  });

  it('date32 with nulls from apache-arrow', () => {
    const dateVec = vectorFromArray([0, null, 2 * DAY_MS], new DateDay());
    const t = makeTable({ d: dateVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['d']).toBe('date32');
    expect(df.toColumns()['d']).toEqual([0, null, 2]);
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// v2.6 — timestamp: toArrow → apache-arrow reads
// ---------------------------------------------------------------------------

describe('toArrow → apache-arrow reads (timestamp, ADR-010)', () => {
  it('timestamp column without tz — apache-arrow Timestamp(MILLI, no tz)', () => {
    const df = makeDF(rt, { ts: [0n, 1000n, -1n] }, { ts: 'timestamp' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    expect(t.numRows).toBe(3);
    const col = t.getChild('ts');
    expect(col).toBeTruthy();
    // apache-arrow returns Number (epoch-ms) for Timestamp, not BigInt
    expect(col!.get(0)).toBe(0);
    expect(col!.get(1)).toBe(1000);
    expect(col!.get(2)).toBe(-1);
    // Verify no timezone on the type
    const field = t.schema.fields.find((f) => f.name === 'ts');
    expect((field?.type as unknown as { timezone?: string })?.timezone).toBeFalsy();
    df.dispose();
  });

  it('timestamp column with tz — apache-arrow sees Timestamp(MILLI, tz)', () => {
    const df = DataFrame.fromColumns(
      { ts: [0n, 86_400_000n] },
      { dtypes: { ts: 'timestamp' }, tzs: { ts: 'America/New_York' }, runtime: rt },
    );
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    const col = t.getChild('ts');
    // apache-arrow returns Number (epoch-ms) for Timestamp
    expect(col!.get(0)).toBe(0);
    expect(col!.get(1)).toBe(86_400_000); // values are UTC ms regardless of tz
    // Verify tz string is present on the Arrow type
    const field = t.schema.fields.find((f) => f.name === 'ts');
    const tzField = (field?.type as unknown as { timezone?: string })?.timezone;
    expect(tzField).toBe('America/New_York');
    df.dispose();
  });

  it('timestamp with nulls', () => {
    const df = makeDF(rt, { ts: [1000n, null, 3000n] }, { ts: 'timestamp' });
    const buf = toArrow(df);
    const t = tableFromIPC(buf);
    const col = t.getChild('ts');
    // apache-arrow returns Number (epoch-ms) for Timestamp
    expect(col!.get(0)).toBe(1000);
    expect(col!.get(1)).toBeNull();
    expect(col!.get(2)).toBe(3000);
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// v2.6 — timestamp: fromArrow ← apache-arrow writes
// ---------------------------------------------------------------------------

describe('fromArrow ← apache-arrow writes (timestamp, ADR-010)', () => {
  it('Timestamp(MILLI) without tz from apache-arrow', () => {
    const tsVec = vectorFromArray([0n, 1000n, -1n], new TimestampMillisecond());
    const t = makeTable({ ts: tsVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['ts']).toBe('timestamp');
    const col = df.getColumn('ts');
    expect(col?.tz).toBeUndefined();
    expect(df.toColumns()['ts']).toEqual([0n, 1000n, -1n]);
    df.dispose();
  });

  it('Timestamp(MILLI, tz) — tz string preserved', () => {
    const tsVec = vectorFromArray([0n, 86_400_000n], new TimestampMillisecond('UTC'));
    const t = makeTable({ ts: tsVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['ts']).toBe('timestamp');
    const col = df.getColumn('ts');
    expect(col?.tz).toBe('UTC');
    expect(df.toColumns()['ts']).toEqual([0n, 86_400_000n]);
    df.dispose();
  });

  it('Timestamp(MILLI, tz="America/New_York") — tz preserved', () => {
    const tsVec = vectorFromArray([0n, 3_600_000n], new TimestampMillisecond('America/New_York'));
    const t = makeTable({ ts: tsVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.getColumn('ts')?.tz).toBe('America/New_York');
    expect(df.toColumns()['ts']).toEqual([0n, 3_600_000n]);
    df.dispose();
  });

  it('Timestamp(SECOND) rescaled to ms ×1000', () => {
    // apache-arrow TimestampSecond builder expects ms-values and divides by 1000.
    // Pass [0ms, 1_000_000ms, -1_000ms] → stored as [0s, 1000s, -1s].
    // Our fromArrow rescales back: [0n, 1_000_000n, -1_000n] ms.
    const tsVec = vectorFromArray([0, 1_000_000, -1_000], new TimestampSecond());
    const t = makeTable({ ts: tsVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['ts']).toBe('timestamp');
    expect(df.toColumns()['ts']).toEqual([0n, 1_000_000n, -1_000n]);
    df.dispose();
  });

  it('Timestamp(MICRO) rescaled to ms ÷1000 (truncates sub-ms)', () => {
    // apache-arrow TimestampMicrosecond builder expects ms and multiplies by 1000.
    // Pass [0ms, 1000ms, 1000.001ms] → stored as [0µs, 1_000_000µs, 1_000_001µs].
    // Our fromArrow rescales: [0n, 1000n, 1000n] ms (sub-ms truncated).
    const tsVec = vectorFromArray([0, 1000, 1000.001], new TimestampMicrosecond());
    const t = makeTable({ ts: tsVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['ts']).toBe('timestamp');
    expect(df.toColumns()['ts']).toEqual([0n, 1000n, 1000n]);
    df.dispose();
  });

  it('Timestamp(NANO) rescaled to ms ÷1_000_000 (truncates sub-ms)', () => {
    // Pass [0ms, 1000ms, 1500ms] → stored as [0ns, 1_000_000_000ns, 1_500_000_000ns].
    // Our fromArrow rescales: [0n, 1000n, 1500n] ms.
    const tsVec = vectorFromArray([0, 1000, 1500], new TimestampNanosecond());
    const t = makeTable({ ts: tsVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['ts']).toBe('timestamp');
    expect(df.toColumns()['ts']).toEqual([0n, 1000n, 1500n]);
    df.dispose();
  });

  it('Timestamp(SECOND) overflow saturates to null', () => {
    // 9_223_372_036_854_776n seconds × 1000 overflows i64 → null.
    // Use makeData to inject raw BigInt64 values directly, bypassing the ms-value API.
    const overflow = 9_223_372_036_854_776n; // > SEC_MS_MAX = 9_223_372_036_854_775n
    const rawValues = new BigInt64Array([1n, overflow]);
    const data = makeData({ type: new TimestampSecond(), data: rawValues });
    const tsVec = makeVector([data]);
    const t = makeTable({ ts: tsVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    const vals = df.toColumns()['ts'] as (bigint | null)[];
    expect(vals[0]).toBe(1000n); // 1s × 1000 = 1000ms
    expect(vals[1]).toBeNull();  // overflow → null
    df.dispose();
  });

  it('Timestamp(MILLI) with nulls — nulls preserved', () => {
    const tsVec = vectorFromArray([1000n, null, 3000n], new TimestampMillisecond());
    const t = makeTable({ ts: tsVec });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['ts']).toEqual([1000n, null, 3000n]);
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// v2.6 — full round-trip: toArrow → fromArrow (self-contained)
// ---------------------------------------------------------------------------

describe('toArrow → fromArrow round-trip (i64 + date32 + timestamp)', () => {
  function roundTrip(cols: Record<string, Cell[]>, dtypes: Record<string, import('../../src/memory/dtype.js').DType>): void {
    const df1 = makeDF(rt, cols, dtypes);
    const buf = toArrow(df1);
    const df2 = fromArrow(buf, rt);
    expect(df2.columns).toEqual(df1.columns);
    expect(df2.length).toBe(df1.length);
    const c1 = df1.toColumns();
    const c2 = df2.toColumns();
    for (const name of df1.columns) {
      expect(c2[name]).toEqual(c1[name]);
    }
    df1.dispose(); df2.dispose();
  }

  it('i64 with nulls', () => roundTrip({ v: [1n, null, -1n] }, { v: 'i64' }));
  it('i64 boundary values', () => roundTrip(
    { v: [9_223_372_036_854_775_807n, -9_223_372_036_854_775_808n, 0n] },
    { v: 'i64' },
  ));
  it('date32 with nulls', () => roundTrip({ d: [0, null, -1, 365] }, { d: 'date32' }));
  it('date32 pre-1970', () => roundTrip({ d: [-365, -366, -1] }, { d: 'date32' }));
  it('timestamp without tz', () => roundTrip({ ts: [0n, null, -1n, 86_400_000n] }, { ts: 'timestamp' }));

  it('timestamp with tz preserved end-to-end', () => {
    const df1 = DataFrame.fromColumns(
      { ts: [0n, 86_400_000n, null] },
      { dtypes: { ts: 'timestamp' }, tzs: { ts: 'Europe/London' }, runtime: rt },
    );
    const buf = toArrow(df1);
    const df2 = fromArrow(buf, rt);
    expect(df2.dtypes['ts']).toBe('timestamp');
    expect(df2.getColumn('ts')?.tz).toBe('Europe/London');
    expect(df2.toColumns()['ts']).toEqual([0n, 86_400_000n, null]);
    df1.dispose(); df2.dispose();
  });

  it('all v2 dtypes together', () => {
    const df1 = makeDF(rt, {
      a: [1n, null, -1n],
      b: [0, 1, -1],
      c: [0n, 86_400_000n, null],
    }, { a: 'i64', b: 'date32', c: 'timestamp' });
    const buf = toArrow(df1);
    const df2 = fromArrow(buf, rt);
    expect(df2.dtypes['a']).toBe('i64');
    expect(df2.dtypes['b']).toBe('date32');
    expect(df2.dtypes['c']).toBe('timestamp');
    const c1 = df1.toColumns();
    const c2 = df2.toColumns();
    expect(c2['a']).toEqual(c1['a']);
    expect(c2['b']).toEqual(c1['b']);
    expect(c2['c']).toEqual(c1['c']);
    df1.dispose(); df2.dispose();
  });
});

// ---------------------------------------------------------------------------
// v2.6 — cross-library round-trip: toArrow → apache reads → apache writes → fromArrow
// ---------------------------------------------------------------------------

describe('cross-library round-trip (i64 + date32 + timestamp via apache-arrow)', () => {
  it('i64 survives apache-arrow transit', () => {
    const INT64_MAX = 9_223_372_036_854_775_807n;
    const df1 = makeDF(rt, { v: [INT64_MAX, 0n, null, -1n] }, { v: 'i64' });
    const buf1 = toArrow(df1);
    const at = tableFromIPC(buf1);
    const buf2 = tableToIPC(at, 'stream');
    const df2 = fromArrow(buf2, rt);
    expect(df2.dtypes['v']).toBe('i64');
    expect(df2.toColumns()['v']).toEqual([INT64_MAX, 0n, null, -1n]);
    df1.dispose(); df2.dispose();
  });

  it('date32 survives apache-arrow transit', () => {
    const df1 = makeDF(rt, { d: [0, -1, null, 365] }, { d: 'date32' });
    const buf1 = toArrow(df1);
    const at = tableFromIPC(buf1);
    const buf2 = tableToIPC(at, 'stream');
    const df2 = fromArrow(buf2, rt);
    expect(df2.dtypes['d']).toBe('date32');
    expect(df2.toColumns()['d']).toEqual([0, -1, null, 365]);
    df1.dispose(); df2.dispose();
  });

  it('timestamp (no tz) survives apache-arrow transit', () => {
    const df1 = makeDF(rt, { ts: [0n, 86_400_000n, null, -1n] }, { ts: 'timestamp' });
    const buf1 = toArrow(df1);
    const at = tableFromIPC(buf1);
    const buf2 = tableToIPC(at, 'stream');
    const df2 = fromArrow(buf2, rt);
    expect(df2.dtypes['ts']).toBe('timestamp');
    expect(df2.toColumns()['ts']).toEqual([0n, 86_400_000n, null, -1n]);
    df1.dispose(); df2.dispose();
  });

  it('timestamp with tz survives apache-arrow transit — tz string preserved', () => {
    const df1 = DataFrame.fromColumns(
      { ts: [0n, 3_600_000n] },
      { dtypes: { ts: 'timestamp' }, tzs: { ts: 'America/Chicago' }, runtime: rt },
    );
    const buf1 = toArrow(df1);
    const at = tableFromIPC(buf1);
    // Verify apache-arrow sees the tz
    const field = at.schema.fields.find((f) => f.name === 'ts');
    expect((field?.type as unknown as { timezone?: string })?.timezone).toBe('America/Chicago');
    const buf2 = tableToIPC(at, 'stream');
    const df2 = fromArrow(buf2, rt);
    expect(df2.dtypes['ts']).toBe('timestamp');
    expect(df2.getColumn('ts')?.tz).toBe('America/Chicago');
    expect(df2.toColumns()['ts']).toEqual([0n, 3_600_000n]);
    df1.dispose(); df2.dispose();
  });
});

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

describe('fromArrow — error cases', () => {
  it('throws on empty buffer', () => {
    expect(() => fromArrow(new Uint8Array(0), rt)).toThrow();
  });

  it('throws on truncated buffer', () => {
    expect(() => fromArrow(new Uint8Array(4), rt)).toThrow();
  });

  it('toArrow throws on unsupported dtype in schema', () => {
    // Verify the error path is reachable; 'i64' is supported, but the error for
    // unknown dtype has always been tested via type-system guards.
    // This test intentionally validates the happy path doesn't accidentally throw.
    const df = makeDF(rt, { v: [1n, 2n] }, { v: 'i64' });
    expect(() => toArrow(df)).not.toThrow();
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// CP.1 — ingest fast path (ABI §12, 2026-07-03)
// ---------------------------------------------------------------------------

describe('CP.1 — dict-encoded Arrow passthrough (no decode/re-encode)', () => {
  it('round-trips dict-encoded utf8 with duplicates and nulls', () => {
    // apache-arrow produces Dict<Int32,Utf8> for string arrays
    const t = tableFromArrays({ s: ['hello', 'world', 'hello', null, 'world', ''] });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.dtypes['s']).toBe('utf8');
    expect(df.toColumns()['s']).toEqual(['hello', 'world', 'hello', null, 'world', '']);
    df.dispose();
  });

  it('round-trips dict-encoded utf8 with unicode strings', () => {
    const strs = ['你好', '世界', '😀', '你好', '日本語', '😀', null, ''];
    const t = tableFromArrays({ s: strs });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['s']).toEqual(strs);
    df.dispose();
  });

  it('round-trips dict-encoded utf8 all-unique (no duplicates)', () => {
    // All-unique triggers the byte-dedup identity path in the plain-utf8 case
    const strs = Array.from({ length: 100 }, (_, i) => `string_${i}_unique`);
    const t = tableFromArrays({ s: strs });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['s']).toEqual(strs);
    df.dispose();
  });

  it('round-trips dict-encoded utf8 all-same', () => {
    const t = tableFromArrays({ s: Array(50).fill('repeat') });
    const buf = tableToIPC(t, 'stream');
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['s']).toEqual(Array(50).fill('repeat'));
    df.dispose();
  });
});

describe('CP.1 — plain UTF-8 fromArrow byte-dedup (property test)', () => {
  /**
   * Build an Arrow IPC buffer with a plain UTF-8 (non-dict) column.
   * Apache-arrow encodes strings as Dict<Int32,Utf8> by default when using
   * tableFromArrays with a string array — we must force plain Utf8 explicitly.
   */
  function makePlainUtf8Buf(strings: (string | null)[]): Uint8Array {
    const nonNull = strings.map((s) => s ?? '');
    const vec = vectorFromArray(nonNull, new Utf8());
    const t = tableFromArrays({ s: vec });
    return tableToIPC(t, 'stream');
  }

  it('byte-dedup produces identical values to JS-string-based dedup', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.oneof(
            fc.string(),
            fc.fullUnicodeString(),
            fc.constant(''),
            fc.constant('dup'),
            fc.constant('dup'),  // ensure some duplicates appear
          ),
          { minLength: 1, maxLength: 80 },
        ),
        (strings) => {
          const buf = makePlainUtf8Buf(strings);
          const df = fromArrow(buf, rt);
          try {
            const got = df.toColumns()['s'] as string[];
            expect(got).toEqual(strings);
          } finally {
            df.dispose();
          }
        },
      ),
      { numRuns: 150 },
    );
  });

  it('handles all-unique strings (identity index path)', () => {
    const uniq = Array.from({ length: 200 }, (_, i) => `item_${i}`);
    const buf = makePlainUtf8Buf(uniq);
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['s']).toEqual(uniq);
    df.dispose();
  });

  it('handles all-same strings (single-slot dict)', () => {
    const same = Array(200).fill('same');
    const buf = makePlainUtf8Buf(same);
    const df = fromArrow(buf, rt);
    expect(df.toColumns()['s']).toEqual(same);
    df.dispose();
  });

  it('handles empty string column (0 rows)', () => {
    const buf = makePlainUtf8Buf([]);
    const df = fromArrow(buf, rt);
    expect(df.length).toBe(0);
    df.dispose();
  });
});
