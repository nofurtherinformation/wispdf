/**
 * CSV reader tests: property round-trips, RFC-4180 edge cases, dtype inference,
 * null handling, option variants, and malformed-input errors.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import fc from 'fast-check';
import { fromCSV, ChunkParser } from '../../src/io/csv.js';
import { init, type DfRuntime } from '../../src/frame/runtime.js';
import { loadRuntimeForTest } from '../frame/helper.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

// ---------------------------------------------------------------------------
// ChunkParser unit tests
// ---------------------------------------------------------------------------

describe('ChunkParser', () => {
  function parseAll(text: string, delimiter = ','): string[][] {
    const rows: string[][] = [];
    const p = new ChunkParser(delimiter, (r) => rows.push([...r]));
    p.feed(text);
    p.finish();
    return rows;
  }

  it('parses simple CSV', () => {
    expect(parseAll('a,b,c\n1,2,3')).toEqual([['a', 'b', 'c'], ['1', '2', '3']]);
  });

  it('handles CRLF line endings', () => {
    expect(parseAll('a,b\r\n1,2\r\n')).toEqual([['a', 'b'], ['1', '2']]);
  });

  it('handles quoted fields', () => {
    expect(parseAll('"hello","world"')).toEqual([['hello', 'world']]);
  });

  it('handles escaped quotes (doubled)', () => {
    expect(parseAll('"say ""hi""","ok"')).toEqual([['say "hi"', 'ok']]);
  });

  it('handles embedded commas in quoted fields', () => {
    expect(parseAll('"a,b",c')).toEqual([['a,b', 'c']]);
  });

  it('handles embedded newlines in quoted fields', () => {
    expect(parseAll('"line1\nline2",end')).toEqual([['line1\nline2', 'end']]);
  });

  it('handles embedded CRLF in quoted fields', () => {
    expect(parseAll('"line1\r\nline2",end')).toEqual([['line1\r\nline2', 'end']]);
  });

  it('handles empty fields', () => {
    expect(parseAll(',,')).toEqual([['', '', '']]);
  });

  it('handles trailing newline (no extra empty row)', () => {
    expect(parseAll('a,b\n1,2\n')).toEqual([['a', 'b'], ['1', '2']]);
  });

  it('handles custom delimiter', () => {
    expect(parseAll('a|b|c\n1|2|3', '|')).toEqual([['a', 'b', 'c'], ['1', '2', '3']]);
  });

  it('handles multi-char delimiter', () => {
    expect(parseAll('a::b::c\n1::2::3', '::')).toEqual([['a', 'b', 'c'], ['1', '2', '3']]);
  });

  it('handles chunk-fed input', () => {
    const rows: string[][] = [];
    const p = new ChunkParser(',', (r) => rows.push([...r]));
    p.feed('a,b');
    p.feed('\n1');
    p.feed(',2\n');
    p.finish();
    expect(rows).toEqual([['a', 'b'], ['1', '2']]);
  });

  it('throws on unterminated quote', () => {
    expect(() => parseAll('"unterminated')).toThrow(/unterminated/);
  });
});

// ---------------------------------------------------------------------------
// fromCSV — basic parsing
// ---------------------------------------------------------------------------

describe('fromCSV — basic', () => {
  it('parses a simple CSV string', () => {
    const df = fromCSV('a,b,c\n1,2,3\n4,5,6', { runtime: rt });
    expect(df.length).toBe(2);
    expect(df.columns).toEqual(['a', 'b', 'c']);
    df.dispose();
  });

  it('header=false uses column_N names', () => {
    const df = fromCSV('1,2,3', { header: false, runtime: rt });
    expect(df.columns).toEqual(['column_0', 'column_1', 'column_2']);
    df.dispose();
  });

  it('header=false + columns option sets column names', () => {
    const df = fromCSV('1,2,3', { header: false, columns: ['x', 'y', 'z'], runtime: rt });
    expect(df.columns).toEqual(['x', 'y', 'z']);
    df.dispose();
  });

  it('skipRows skips leading rows', () => {
    const df = fromCSV('skip\na,b\n1,2', { skipRows: 1, runtime: rt });
    expect(df.length).toBe(1);
    expect(df.columns).toEqual(['a', 'b']);
    df.dispose();
  });

  it('maxRows limits data rows', () => {
    const df = fromCSV('a,b\n1,2\n3,4\n5,6', { maxRows: 2, runtime: rt });
    expect(df.length).toBe(2);
    df.dispose();
  });

  it('custom delimiter', () => {
    const df = fromCSV('a|b\n1|2', { delimiter: '|', runtime: rt });
    expect(df.columns).toEqual(['a', 'b']);
    df.dispose();
  });

  it('empty CSV returns empty DataFrame', () => {
    const df = fromCSV('', { runtime: rt });
    expect(df.length).toBe(0);
    df.dispose();
  });

  it('header-only CSV returns empty DataFrame', () => {
    const df = fromCSV('a,b,c', { runtime: rt });
    expect(df.length).toBe(0);
    expect(df.columns).toEqual(['a', 'b', 'c']);
    df.dispose();
  });

  it('throws on ragged rows', () => {
    expect(() => fromCSV('a,b\n1,2,3', { runtime: rt })).toThrow(/CSV parse error/);
  });
});

// ---------------------------------------------------------------------------
// fromCSV — null values
// ---------------------------------------------------------------------------

describe('fromCSV — null values', () => {
  it('empty field → null by default', () => {
    const df = fromCSV('a,b\n1,\n3,4', { runtime: rt });
    const cols = df.toColumns();
    expect(cols['b']![0]).toBe(null);
    df.dispose();
  });

  it('"null" → null by default', () => {
    const df = fromCSV('a\n1\nnull\n3', { runtime: rt });
    const cols = df.toColumns();
    expect(cols['a']![1]).toBe(null);
    df.dispose();
  });

  it('"NA" → null by default', () => {
    const df = fromCSV('a\n1\nNA\n3', { runtime: rt });
    const cols = df.toColumns();
    expect(cols['a']![1]).toBe(null);
    df.dispose();
  });

  it('custom nullValues', () => {
    const df = fromCSV('a\n1\nN/A\n3', { nullValues: ['N/A'], runtime: rt });
    const cols = df.toColumns();
    expect(cols['a']![1]).toBe(null);
    df.dispose();
  });

  it('null-only column infers utf8', () => {
    const df = fromCSV('a\n\n\n', { runtime: rt });
    expect(df.dtypes['a']).toBe('utf8');
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// fromCSV — type inference
// ---------------------------------------------------------------------------

describe('fromCSV — type inference', () => {
  it('integer column → i32', () => {
    const df = fromCSV('x\n1\n2\n3', { runtime: rt });
    expect(df.dtypes['x']).toBe('i32');
    df.dispose();
  });

  it('float column → f64', () => {
    const df = fromCSV('x\n1.5\n2.7', { runtime: rt });
    expect(df.dtypes['x']).toBe('f64');
    df.dispose();
  });

  it('i32 mixed with float → f64', () => {
    const df = fromCSV('x\n1\n2.5', { runtime: rt });
    expect(df.dtypes['x']).toBe('f64');
    df.dispose();
  });

  it('bool column (true/false) → bool', () => {
    const df = fromCSV('x\ntrue\nfalse\nTrue', { runtime: rt });
    expect(df.dtypes['x']).toBe('bool');
    const vals = df.toColumns()['x'];
    expect(vals).toEqual([true, false, true]);
    df.dispose();
  });

  it('string column → utf8', () => {
    const df = fromCSV('x\nhello\nworld', { runtime: rt });
    expect(df.dtypes['x']).toBe('utf8');
    df.dispose();
  });

  it('mixed (int then string) → utf8', () => {
    const df = fromCSV('x\n1\nhello', { runtime: rt });
    expect(df.dtypes['x']).toBe('utf8');
    df.dispose();
  });

  it('i32 overflow → f64', () => {
    const df = fromCSV('x\n2147483648', { runtime: rt }); // 2^31 > i32 max
    expect(df.dtypes['x']).toBe('f64');
    df.dispose();
  });

  it('dtypes option overrides inference', () => {
    const df = fromCSV('x\n1\n2', { dtypes: { x: 'f64' }, runtime: rt });
    expect(df.dtypes['x']).toBe('f64');
    const vals = df.toColumns()['x'];
    expect(vals![0]).toBe(1.0);
    df.dispose();
  });

  it('i32 boundary values round-trip', () => {
    const df = fromCSV(`x\n-2147483648\n2147483647\n0`, { runtime: rt });
    const vals = df.toColumns()['x'];
    expect(vals).toEqual([-2147483648, 2147483647, 0]);
    df.dispose();
  });
});

// ---------------------------------------------------------------------------
// fromCSV — round-trip property tests
// ---------------------------------------------------------------------------

describe('fromCSV — round-trip properties', () => {
  /** Escape a value for CSV output (simple reference implementation). */
  function toCsvCell(v: unknown): string {
    if (v === null || v === undefined) return '';
    const s = String(v);
    if (s.includes(',') || s.includes('"') || s.includes('\n') || s.includes('\r')) {
      return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
  }

  function toCsvText(names: string[], rows: unknown[][]): string {
    // Each row (including empty/null ones) ends with '\n' so the parser emits
    // the row rather than treating a trailing newline as the row terminator.
    let out = names.join(',') + '\n';
    for (const row of rows) out += row.map(toCsvCell).join(',') + '\n';
    return out;
  }

  it('integer round-trip (nulls preserved)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.option(fc.integer({ min: -1000, max: 1000 }), { nil: null }), {
          minLength: 0,
          maxLength: 20,
        }),
        (vals) => {
          const csv = toCsvText(['v'], vals.map((x) => [x]));
          const df = fromCSV(csv, { runtime: rt });
          try {
            const out = df.toColumns()['v'];
            expect(out).toEqual(vals);
          } finally {
            df.dispose();
          }
        },
      ),
    );
  });

  it('string round-trip (with special chars)', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.option(
            fc.string({ maxLength: 20 }).filter((s) => s !== '' && s !== 'null' && s !== 'NA'),
            { nil: null },
          ),
          { minLength: 1, maxLength: 15 },
        ),
        (vals) => {
          const csv = toCsvText(['s'], vals.map((x) => [x]));
          // Force utf8 dtype so that numeric-looking strings like "0" or "1"
          // are not inferred as i32/f64 by the type-inference ladder. The
          // inference ladder is tested separately; this property tests that
          // arbitrary strings survive the CSV encode/decode cycle when the
          // column dtype is explicitly declared as utf8.
          const df = fromCSV(csv, { dtypes: { s: 'utf8' }, runtime: rt });
          try {
            const out = df.toColumns()['s'];
            expect(out).toEqual(vals);
          } finally {
            df.dispose();
          }
        },
      ),
    );
  });

  it('float round-trip (finite values)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.option(fc.float({ min: -1e6, max: 1e6, noNaN: true }), { nil: null }), {
          minLength: 1,
          maxLength: 15,
        }),
        (vals) => {
          // Use String(x) for the shortest JavaScript representation that round-trips
          // exactly through Number parsing. toPrecision(10) loses precision for
          // values like -100304.5703125 (10 sig-digits truncates the decimal tail,
          // producing a difference that exceeds 5 decimal places tolerance).
          const csv = toCsvText(['f'], vals.map((x) => [x === null ? null : String(x)]));
          const df = fromCSV(csv, { dtypes: { f: 'f64' }, runtime: rt });
          try {
            const out = df.toColumns()['f'] as (number | null)[];
            for (let i = 0; i < vals.length; i++) {
              if (vals[i] === null) expect(out[i]).toBe(null);
              else expect(out[i]).toBeCloseTo(vals[i]!, 5);
            }
          } finally {
            df.dispose();
          }
        },
      ),
    );
  });
});
