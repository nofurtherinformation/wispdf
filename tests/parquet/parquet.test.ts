/**
 * databonk/parquet conformance + oracle tests (ADR-011).
 *
 * Test structure:
 *   1. parquet_roundtrip — fixture-driven: write → oracle validate → read → compare
 *   2. parquet_error     — write or oracle-generate out-of-profile file, assert throw
 *   3. smoke             — 1 M-row f64 round-trip (latency / no-crash)
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { createRequire } from 'module';
import { readParquet, writeParquet } from '../../src/parquet/index.js';
import { loadRuntimeForTest, makeDF } from '../frame/helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import type { DType } from '../../src/memory/dtype.js';

// parquet-wasm is a CJS devDep — load via createRequire.
const require = createRequire(import.meta.url);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const pw: any = require('parquet-wasm');
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const arrow: any = require('apache-arrow');

// hyparquet-writer used for out-of-profile file generation (INT96, DECIMAL, LIST).
import { parquetWriteBuffer } from 'hyparquet-writer/src/index.js';

import fixtureJson from '../conformance/fixtures/parquet.json';
import { createColumn } from '../../src/memory/column.js';
import { DataFrame } from '../../src/frame/dataframe.js';

// ---------------------------------------------------------------------------
// Fixture types (simplified)
// ---------------------------------------------------------------------------

interface FixtureColDef {
  dtype: string;
  data: (number | string | null)[];
  tz?: string | null;
}

interface FixtureFrame {
  columns: Record<string, FixtureColDef>;
  nrows: number;
}

interface RoundTripCase {
  name: string;
  layer: 'parquet_roundtrip';
  frame: FixtureFrame;
  write_opts: { compression?: string };
  expected: { columns: Record<string, FixtureColDef> };
}

interface ErrorCase {
  name: string;
  layer: 'parquet_error';
  op?: string;
  frame?: FixtureFrame;
  write_opts?: { compression?: string };
  generate_with_oracle?: { compression?: string; encoding?: string; data?: Record<string, FixtureColDef> };
  error_pattern: string;
}

type FixtureCase = RoundTripCase | ErrorCase;

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

// ---------------------------------------------------------------------------
// Helpers: fixture → DataFrame
// ---------------------------------------------------------------------------

/**
 * Convert fixture column data (strings for i64/timestamp, 0/1 for bool) to
 * the ColumnInput format that makeDF accepts.
 */
function fixtureColToInput(def: FixtureColDef): unknown[] | BigInt64Array | null[] {
  const dtype = def.dtype as DType;
  if (dtype === 'i64' || dtype === 'timestamp') {
    // Fixture encodes these as decimal strings; convert to BigInt.
    return def.data.map((v) => (v === null ? null : BigInt(v as string)));
  }
  // All other dtypes (f64, f32, i32, u32, bool, utf8, date32) pass through.
  // bool: fixture uses 0/1 — createBoolColumn treats any truthy value as true.
  return def.data as unknown[];
}

/** Build a DataFrame from a fixture frame definition. */
function buildFixtureFrame(frame: FixtureFrame): ReturnType<typeof makeDF> {
  const cols: Record<string, unknown> = {};
  const dtypes: Record<string, DType> = {};
  for (const [name, def] of Object.entries(frame.columns)) {
    cols[name] = fixtureColToInput(def);
    dtypes[name] = def.dtype as DType;
  }
  // makeDF uses runtime rt from outer scope.
  // ponytail: tz on timestamp columns is NOT set here; writeParquet reads it from col.tz.
  // We set tz via DataFrame.fromColumns which goes through makeDF without tz support.
  // For tz: build without makeDF and use DataFrame directly.
  const needsTz = Object.values(frame.columns).some(
    (def) => def.dtype === 'timestamp' && def.tz != null,
  );
  if (!needsTz) {
    return makeDF(rt, cols as Record<string, import('../../src/memory/column.js').ColumnInput>, dtypes);
  }
  // For timestamp+tz columns: build via createColumn directly.
  const seed = DataFrame.fromColumns({}, { runtime: rt });
  const namedCols: Array<{ name: string; col: unknown }> = [];
  for (const [name, def] of Object.entries(frame.columns)) {
    const data = fixtureColToInput(def);
    const tz = (def.tz != null && def.tz !== null) ? (def.tz as string) : undefined;
    const col = createColumn(rt.ctx, def.dtype as DType, data, tz);
    namedCols.push({ name, col });
  }
  const df = seed.buildResult(namedCols);
  seed.dispose();
  return df;
}

// ---------------------------------------------------------------------------
// Helpers: value comparison (fixture encoding → Cell values)
// ---------------------------------------------------------------------------

/** Convert a Cell value to fixture encoding (for comparison). */
function cellToFixture(v: unknown, dtype: DType): number | string | boolean | null {
  if (v === null) return null;
  if (dtype === 'i64' || dtype === 'timestamp') {
    // toColumns() returns bigint; fixture expects decimal string.
    return String(v as bigint);
  }
  if (dtype === 'bool') {
    // toColumns() returns boolean; fixture expects 0/1.
    return (v as boolean) ? 1 : 0;
  }
  return v as number | string;
}

/**
 * Assert that a readParquet DataFrame matches the fixture expected columns.
 * Disposes the DataFrame after comparison.
 */
function assertDfMatchesExpected(
  df: Awaited<ReturnType<typeof readParquet>>,
  expectedCols: Record<string, FixtureColDef>,
): void {
  const dfCols = df.toColumns();
  const dtypes = df.dtypes as Record<string, DType>;

  for (const [name, exp] of Object.entries(expectedCols)) {
    const dtype = dtypes[name];
    expect(dtype, `column '${name}' dtype`).toBe(exp.dtype);

    const got = dfCols[name]!;
    expect(got.length, `column '${name}' length`).toBe(exp.data.length);

    for (let i = 0; i < exp.data.length; i++) {
      const gotFixed = cellToFixture(got[i], dtype!);
      const expFixed = exp.data[i];
      if (expFixed === null) {
        expect(got[i], `${name}[${i}]`).toBeNull();
      } else if (dtype === 'f64' || dtype === 'f32') {
        // Use toBeCloseTo for floats to handle f32 precision.
        expect(gotFixed as number, `${name}[${i}]`).toBeCloseTo(expFixed as number, dtype === 'f32' ? 5 : 10);
      } else {
        expect(gotFixed, `${name}[${i}]`).toBe(expFixed);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers: parquet-wasm oracle validation
// ---------------------------------------------------------------------------

/** Read a Parquet Uint8Array with parquet-wasm and return an Apache Arrow table. */
function oracleRead(bytes: Uint8Array): unknown {
  const wasmTable = pw.readParquet(bytes);
  const ipcBytes = wasmTable.intoIPCStream();
  return arrow.tableFromIPC(ipcBytes);
}

/**
 * Validate Parquet bytes using parquet-wasm oracle.
 * Checks: no parse error, row count, column count.
 * Value comparison is covered by the readParquet round-trip.
 */
function assertOracleCanRead(bytes: Uint8Array, expectedRows: number, expectedCols: number): void {
  let arrowTable: unknown;
  expect(() => { arrowTable = oracleRead(bytes); }, 'oracle read should not throw').not.toThrow();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const t = arrowTable as any;
  expect(t.numRows, 'oracle row count').toBe(expectedRows);
  expect(t.numCols, 'oracle column count').toBe(expectedCols);
}

// ---------------------------------------------------------------------------
// Helpers: generate out-of-profile Parquet files for error cases
// ---------------------------------------------------------------------------

/**
 * Generate a Parquet file with GZIP compression using parquet-wasm.
 * GZIP is out-of-profile for readParquet (ADR-011).
 */
function oracleWriteGzip(data: Record<string, FixtureColDef>): Uint8Array {
  // Only handles f64 for simplicity (fixture only uses {x: f64}).
  const col = Object.values(data)[0]!;
  const name = Object.keys(data)[0]!;
  const f64arr = new Float64Array(col.data.filter((v) => v !== null) as number[]);
  const arrowTable = arrow.tableFromArrays({ [name]: f64arr });
  const ipcBytes = arrow.tableToIPC(arrowTable, 'stream');
  const wasmTable = pw.Table.fromIPCStream(ipcBytes);
  const props = new pw.WriterPropertiesBuilder().setCompression(pw.Compression.GZIP).build();
  return pw.writeParquet(wasmTable, props) as Uint8Array;
}

/**
 * Generate a Parquet file with LZ4 compression via hyparquet-writer.
 * LZ4 is also out-of-profile (used for ZSTD fixture case — parquet-wasm 0.7.2
 * does not compile ZSTD; LZ4 tests the same codec-rejection path in readParquet).
 */
function oracleWriteOutOfProfileCodec(data: Record<string, FixtureColDef>): Uint8Array {
  const name = Object.keys(data)[0]!;
  const col = Object.values(data)[0]!;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ab = parquetWriteBuffer({
    schema: [{ name: 'root', num_children: 1 }, { name, type: 'DOUBLE', repetition_type: 'OPTIONAL' }],
    columnData: [{ name, data: (col.data.filter((v) => v !== null) as number[]) }],
    codec: 'LZ4',
  } as any);
  return new Uint8Array(ab instanceof ArrayBuffer ? ab : (ab as any).buffer);
}

/** Generate a Parquet file with INT96 physical timestamp column (empty, 0 rows). */
function makeInt96ParquetBytes(): Uint8Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ab = parquetWriteBuffer({
    schema: [{ name: 'root', num_children: 1 }, { name: 'ts', type: 'INT96', repetition_type: 'OPTIONAL' }],
    columnData: [{ name: 'ts', data: [] }],
    codec: 'UNCOMPRESSED',
  } as any);
  return new Uint8Array(ab instanceof ArrayBuffer ? ab : (ab as any).buffer);
}

/** Generate a Parquet file with DECIMAL logical type (empty, 0 rows). */
function makeDecimalParquetBytes(): Uint8Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ab = parquetWriteBuffer({
    schema: [
      { name: 'root', num_children: 1 },
      { name: 'd', type: 'INT32', repetition_type: 'OPTIONAL', logical_type: { type: 'DECIMAL', precision: 10, scale: 2 } },
    ],
    columnData: [{ name: 'd', data: [] }],
    codec: 'UNCOMPRESSED',
  } as any);
  return new Uint8Array(ab instanceof ArrayBuffer ? ab : (ab as any).buffer);
}

/** Generate a Parquet file with a LIST (nested/repeated) column (empty, 0 rows). */
function makeListParquetBytes(): Uint8Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ab = parquetWriteBuffer({
    schema: [
      { name: 'root', num_children: 1 },
      { name: 'arr', num_children: 1, repetition_type: 'OPTIONAL', logical_type: { type: 'LIST' } },
      { name: 'list', num_children: 1, repetition_type: 'REPEATED' },
      { name: 'element', type: 'INT32', repetition_type: 'OPTIONAL' },
    ],
    columnData: [],
    codec: 'UNCOMPRESSED',
  } as any);
  return new Uint8Array(ab instanceof ArrayBuffer ? ab : (ab as any).buffer);
}

// ---------------------------------------------------------------------------
// Test runner — drive from fixture
// ---------------------------------------------------------------------------

const roundTripCases = fixtureJson.cases.filter(
  (c) => c.layer === 'parquet_roundtrip',
) as unknown as RoundTripCase[];

const errorCases = fixtureJson.cases.filter(
  (c) => c.layer === 'parquet_error',
) as unknown as ErrorCase[];

describe('parquet conformance: round-trip', () => {
  for (const tc of roundTripCases) {
    it(tc.name, async () => {
      const df = buildFixtureFrame(tc.frame);
      const opts = tc.write_opts.compression
        ? { compression: tc.write_opts.compression as 'snappy' | 'uncompressed' }
        : {};

      // Write
      const bytes = writeParquet(df, opts);
      expect(bytes).toBeInstanceOf(Uint8Array);
      expect(bytes.length).toBeGreaterThan(0);

      // Oracle: parquet-wasm must be able to parse the output.
      assertOracleCanRead(
        bytes,
        tc.frame.nrows,
        Object.keys(tc.frame.columns).length,
      );

      // Read back via our implementation
      const df2 = await readParquet(bytes, rt);
      expect(df2.length).toBe(tc.frame.nrows);

      // Compare expected columns
      assertDfMatchesExpected(df2, tc.expected.columns);

      // Check tz survives round-trip for timestamp columns.
      for (const [name, expDef] of Object.entries(tc.expected.columns)) {
        if (expDef.dtype === 'timestamp' && expDef.tz != null) {
          const col = df2.getColumn(name);
          expect((col as { tz?: string } | null)?.tz ?? undefined, `tz of '${name}'`).toBe(expDef.tz);
        }
      }

      df.dispose();
      df2.dispose();
    });
  }
});

describe('parquet conformance: error cases', () => {
  it('parquet_write__invalid_codec_throws', () => {
    const tc = errorCases.find((c) => c.name === 'parquet_write__invalid_codec_throws')!;
    const df = buildFixtureFrame(tc.frame!);
    expect(() =>
      writeParquet(df, { compression: tc.write_opts?.compression as 'snappy' }),
    ).toThrow(tc.error_pattern);
    df.dispose();
  });

  it('parquet_read__gzip_throws', async () => {
    const tc = errorCases.find((c) => c.name === 'parquet_read__gzip_throws')!;
    const bytes = oracleWriteGzip(tc.generate_with_oracle!.data!);
    await expect(readParquet(bytes, rt)).rejects.toThrow(tc.error_pattern);
  });

  it('parquet_read__zstd_throws (oracle uses LZ4; parquet-wasm 0.7.2 lacks ZSTD)', async () => {
    const tc = errorCases.find((c) => c.name === 'parquet_read__zstd_throws')!;
    // parquet-wasm 0.7.2 does not compile ZSTD; LZ4 is also out-of-profile and
    // exercises the same codec-rejection path in readParquet.
    const bytes = oracleWriteOutOfProfileCodec(tc.generate_with_oracle!.data!);
    await expect(readParquet(bytes, rt)).rejects.toThrow(tc.error_pattern);
  });

  it('parquet_read__int96_timestamp_throws', async () => {
    const tc = errorCases.find((c) => c.name === 'parquet_read__int96_timestamp_throws')!;
    const bytes = makeInt96ParquetBytes();
    await expect(readParquet(bytes, rt)).rejects.toThrow(tc.error_pattern);
  });

  it('parquet_read__decimal_column_throws', async () => {
    const tc = errorCases.find((c) => c.name === 'parquet_read__decimal_column_throws')!;
    const bytes = makeDecimalParquetBytes();
    await expect(readParquet(bytes, rt)).rejects.toThrow(tc.error_pattern);
  });

  it('parquet_read__list_column_throws', async () => {
    const tc = errorCases.find((c) => c.name === 'parquet_read__list_column_throws')!;
    const bytes = makeListParquetBytes();
    await expect(readParquet(bytes, rt)).rejects.toThrow(tc.error_pattern);
  });
});

describe('parquet smoke: 1M-row f64', () => {
  it('write and read back 1 million rows without error', async () => {
    const N = 1_000_000;
    const data = new Float64Array(N);
    for (let i = 0; i < N; i++) data[i] = i * 0.5;
    const df = makeDF(rt, { v: data }, { v: 'f64' });

    const bytes = writeParquet(df);
    expect(bytes.length).toBeGreaterThan(N * 4); // raw > 4 MB for 1M f64 uncompressed

    const df2 = await readParquet(bytes, rt);
    expect(df2.length).toBe(N);
    expect(df2.dtypes['v']).toBe('f64');

    df.dispose();
    df2.dispose();
  }, 30_000); // 30 s timeout for 1M rows
});
