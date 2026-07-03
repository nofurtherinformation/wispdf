/**
 * databonk/parquet — Parquet I/O subpath export (ADR-011).
 *
 * Exported API:
 *   readParquet(bytes, rt)  → Promise<DataFrame>   (async; hyparquet reader)
 *   writeParquet(df, opts?) → Uint8Array            (sync; hyparquet-writer)
 *
 * Supported profile (ADR-011 §supported-profile):
 *   dtypes:      f64 f32 i32 u32 i64 bool utf8 date32 timestamp (+ nulls everywhere)
 *   compression: snappy and uncompressed
 *   utf8:        dictionary-encoded when beneficial
 *   timestamp:   INT64 + TIMESTAMP(MILLIS, isAdjustedToUTC); tz round-trips in
 *                file key_value_metadata as "databonk:tz:<colname>"
 *
 * Anything outside the profile raises a clear "unsupported" Error naming the feature —
 * never a silent wrong result (ADR-011 §Consequences).
 *
 * Dependencies (runtime):
 *   hyparquet        — reader (native Snappy decompressor included)
 *   hyparquet-writer — writer (native Snappy included, default UNCOMPRESSED)
 * Neither dependency leaks into the main "." entry — this subpath is the firewall.
 */

// ---------------------------------------------------------------------------
// Imports — hyparquet (reader) + hyparquet-writer (writer)
// The 'node' sub-export of hyparquet is used for Node compatibility; the
// default export works in both Node and browser contexts via the exports map.
// ---------------------------------------------------------------------------
// Use sub-path exports so TypeScript NodeNext resolves the "types" field via
// the "./src/*.js" entry in each package's exports map — the root "." entry uses
// a "browser"/"default" split that tsc doesn't match when building DTS.
// NOTE: internal 'src/index.js' subpath instead of the public 'hyparquet' entry —
// the root entry's browser/default exports split defeats NodeNext DTS resolution
// (see bead dataframe-dh9.7). hyparquet is EXACT-pinned in package.json for this
// reason; when bumping the pin, retry the public specifier first.
import { parquetMetadataAsync, parquetRead, parquetSchema } from 'hyparquet/src/index.js';
import { parquetWriteBuffer } from 'hyparquet-writer/src/index.js';

// Internal databonk imports (NOT re-exported — parquet subpath does not widen
// the main-entry surface).
import { DataFrame } from '../frame/dataframe.js';
import type { DfRuntime } from '../frame/runtime.js';
import type { NamedColumn } from '../frame/groupby.js';
import type { DType } from '../memory/dtype.js';
import { createColumn, columnToArray, type ColumnInput } from '../memory/column.js';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Write options for {@link writeParquet}. */
export interface ParquetWriteOptions {
  /**
   * Compression codec.  Only `'snappy'` and `'uncompressed'` are supported
   * (ADR-011).  Defaults to `'uncompressed'`.
   */
  readonly compression?: 'snappy' | 'uncompressed';
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Supported compression codecs (Parquet names). */
const SUPPORTED_CODECS = new Set<string | undefined>(['UNCOMPRESSED', 'SNAPPY', undefined]);

/**
 * Custom hyparquet parsers that keep temporal values as raw primitives rather
 * than converting them to `Date` objects.  This lets us store them in wasm
 * memory as i64/i32 without a double conversion.
 */
const RAW_PARSERS = {
  timestampFromMilliseconds: (ms: bigint) => ms,
  timestampFromMicroseconds: (us: bigint) => us / 1000n,
  timestampFromNanoseconds: (ns: bigint) => ns / 1_000_000n,
  dateFromDays: (d: number) => d,
  stringFromBytes: (b: Uint8Array | null) =>
    b != null ? new TextDecoder().decode(b) : null,
  jsonFromBytes: (b: Uint8Array | null) =>
    b != null ? JSON.parse(new TextDecoder().decode(b)) : null,
  bsonFromBytes: (b: Uint8Array | null) => b,
  geometryFromBytes: (b: Uint8Array | null) => b,
  geographyFromBytes: (b: Uint8Array | null) => b,
  uuidFromBytes: (b: Uint8Array | null) => b,
} as Record<string, (v: unknown) => unknown>;

/** Wrap a Uint8Array / ArrayBuffer in the hyparquet AsyncBuffer shape. */
function makeAsyncBuffer(
  input: Uint8Array | ArrayBuffer,
): { byteLength: number; slice(s: number, e: number): Promise<ArrayBuffer> } {
  const ab: ArrayBuffer =
    input instanceof ArrayBuffer
      ? input
      : // Extract exactly the bytes of this typed-array view (handles byteOffset).
        // Cast needed: TypedArray.buffer is ArrayBuffer|SharedArrayBuffer per TS,
        // but slice() always returns a plain ArrayBuffer.
        (input.buffer as ArrayBuffer).slice(input.byteOffset, input.byteOffset + input.byteLength);
  return {
    byteLength: ab.byteLength,
    slice: (s: number, e: number) => Promise.resolve((ab as ArrayBuffer).slice(s, e)),
  };
}

// ---------------------------------------------------------------------------
// Schema inspection helpers (for readParquet)
// ---------------------------------------------------------------------------

interface ColInfo {
  name: string;
  dtype: DType;
  tz?: string; // for timestamp columns only
}

/**
 * Map one Parquet SchemaElement (a direct child of the root, i.e. a column)
 * to a databonk ColInfo.  Throws a descriptive "unsupported" error for anything
 * outside the ADR-011 profile.
 */
function mapSchemaElement(
  el: Record<string, unknown>,
  tzByCol: Map<string, string>,
): ColInfo {
  const name = el['name'] as string;
  const type = el['type'] as string | undefined;
  const lt = el['logical_type'] as Record<string, unknown> | undefined;
  const ct = el['converted_type'] as string | undefined;
  const numChildren = el['num_children'] as number | undefined;
  const rep = el['repetition_type'] as string | undefined;

  // ── Nested / repeated structures ────────────────────────────────────────
  if (numChildren != null && numChildren > 0) {
    const ltType = lt?.['type'] as string | undefined;
    if (ltType === 'LIST')
      throw new Error(
        `unsupported Parquet feature: LIST column '${name}' — nested/repeated types are not supported by databonk/parquet (ADR-011)`,
      );
    if (ltType === 'MAP')
      throw new Error(
        `unsupported Parquet feature: MAP column '${name}' — nested types are not supported (ADR-011)`,
      );
    throw new Error(
      `unsupported Parquet feature: nested/group column '${name}' — databonk/parquet only supports flat columns (ADR-011)`,
    );
  }
  if (rep === 'REPEATED')
    throw new Error(
      `unsupported Parquet feature: REPEATED field '${name}' — only OPTIONAL/REQUIRED are supported (ADR-011)`,
    );

  // ── Per-physical-type mapping ────────────────────────────────────────────
  switch (type) {
    case 'DOUBLE':
      return { name, dtype: 'f64' };

    case 'FLOAT':
      return { name, dtype: 'f32' };

    case 'BOOLEAN':
      return { name, dtype: 'bool' };

    case 'INT32': {
      const ltType = lt?.['type'] as string | undefined;
      if (ltType === 'DATE' || ct === 'DATE')
        return { name, dtype: 'date32' };
      if (ltType === 'DECIMAL' || ct === 'DECIMAL')
        throw new Error(
          `unsupported Parquet feature: DECIMAL column '${name}' (ADR-011 — only the databonk numeric dtypes are supported)`,
        );
      // INTEGER logical type: check signedness
      if (ltType === 'INTEGER') {
        const isSigned = lt?.['isSigned'] as boolean | undefined;
        return { name, dtype: isSigned === false ? 'u32' : 'i32' };
      }
      // Converted-type unsigned aliases
      if (ct === 'UINT_32') return { name, dtype: 'u32' };
      // Default INT32 = signed i32
      return { name, dtype: 'i32' };
    }

    case 'INT64': {
      const ltType = lt?.['type'] as string | undefined;
      if (ltType === 'DECIMAL' || ct === 'DECIMAL')
        throw new Error(
          `unsupported Parquet feature: DECIMAL column '${name}' (ADR-011)`,
        );
      if (ltType === 'TIMESTAMP' || ct === 'TIMESTAMP_MILLIS') {
        const unit = lt?.['unit'] as string | undefined;
        if (unit != null && unit !== 'MILLIS')
          throw new Error(
            `unsupported Parquet feature: TIMESTAMP with unit '${unit}' in column '${name}' — only MILLIS is supported (ADR-011)`,
          );
        if (ct === 'TIMESTAMP_MICROS')
          throw new Error(
            `unsupported Parquet feature: TIMESTAMP_MICROS column '${name}' — only MILLIS is supported (ADR-011)`,
          );
        const tz = tzByCol.get(name);
        // exactOptionalPropertyTypes: only include tz key when it has a value.
        return tz !== undefined ? { name, dtype: 'timestamp', tz } : { name, dtype: 'timestamp' };
      }
      return { name, dtype: 'i64' };
    }

    case 'INT96':
      throw new Error(
        `unsupported Parquet feature: INT96 timestamp column '${name}' — only TIMESTAMP(INT64, MILLIS) is supported (ADR-011)`,
      );

    case 'FIXED_LEN_BYTE_ARRAY':
      throw new Error(
        `unsupported Parquet feature: FIXED_LEN_BYTE_ARRAY column '${name}' (ADR-011)`,
      );

    case 'BYTE_ARRAY': {
      const ltType = lt?.['type'] as string | undefined;
      if (ltType === 'STRING' || ct === 'UTF8')
        return { name, dtype: 'utf8' };
      // Decimal over byte array
      if (ltType === 'DECIMAL' || ct === 'DECIMAL')
        throw new Error(
          `unsupported Parquet feature: DECIMAL column '${name}' (ADR-011)`,
        );
      throw new Error(
        `unsupported Parquet feature: BYTE_ARRAY column '${name}' without STRING/UTF8 annotation is not supported (ADR-011)`,
      );
    }

    default:
      throw new Error(
        `unsupported Parquet feature: unknown physical type '${String(type ?? 'GROUP')}' in column '${name}' (ADR-011)`,
      );
  }
}

// ---------------------------------------------------------------------------
// Chunk accumulation for multi-row-group files
// ---------------------------------------------------------------------------

type DecodedChunk = unknown[] | ArrayBufferView;

/**
 * Merge column chunks from multiple row groups into a single typed or JS array
 * suitable for `createColumn`.  Single-chunk files (the common case) return
 * the chunk directly to preserve TypedArray fast paths.
 */
function mergeChunks(parts: DecodedChunk[]): ColumnInput {
  if (parts.length === 0) return [] as unknown as ColumnInput;
  if (parts.length === 1) return parts[0] as unknown as ColumnInput;

  // Multiple row groups: flatten to a JS array (createColumn slow path is fine here).
  const total = parts.reduce((s, p) => s + (p as ArrayLike<unknown>).length, 0);
  const out = new Array<unknown>(total);
  let off = 0;
  for (const p of parts) {
    const len = (p as ArrayLike<unknown>).length;
    for (let i = 0; i < len; i++) out[off++] = (p as ArrayLike<unknown>)[i];
  }
  return out as unknown as ColumnInput;
}

// ---------------------------------------------------------------------------
// readParquet
// ---------------------------------------------------------------------------

/**
 * Read a Parquet file into a databonk {@link DataFrame}.
 *
 * @param bytes - The raw Parquet file bytes.
 * @param rt    - The databonk runtime (from `init()` / `defaultRuntime()`).
 *
 * @throws {Error} if the file uses any out-of-profile feature (ADR-011):
 *   compression other than SNAPPY/UNCOMPRESSED, nested/repeated columns,
 *   INT96 timestamps, DECIMAL, FIXED_LEN_BYTE_ARRAY, etc.
 */
export async function readParquet(
  bytes: Uint8Array | ArrayBuffer,
  rt: DfRuntime,
): Promise<DataFrame> {
  const asyncBuf = makeAsyncBuffer(bytes);

  // 1. Parse metadata — includes schema and row-group codec info.
  const metadata = await parquetMetadataAsync(asyncBuf);

  // 2. Verify compression codecs across all row groups (fail fast before reading).
  for (const rg of (metadata as unknown as { row_groups?: unknown[] }).row_groups ?? []) {
    for (const cc of (rg as { columns?: unknown[] }).columns ?? []) {
      const codec = (cc as { meta_data?: { codec?: string } }).meta_data?.codec;
      if (!SUPPORTED_CODECS.has(codec)) {
        throw new Error(
          `unsupported Parquet compression codec: '${codec}'. ` +
          `databonk/parquet supports only SNAPPY and UNCOMPRESSED (ADR-011). ` +
          `For GZIP/ZSTD/BROTLI/LZ4, re-compress the file first.`,
        );
      }
    }
  }

  // 3. Extract per-column tz from file key_value_metadata (ADR-011 round-trip).
  const tzByCol = new Map<string, string>();
  const kvMeta = (metadata as unknown as { key_value_metadata?: Array<{ key: string; value: string }> }).key_value_metadata ?? [];
  const TZ_PREFIX = 'databonk:tz:';
  for (const { key, value } of kvMeta) {
    if (key.startsWith(TZ_PREFIX) && value) {
      tzByCol.set(key.slice(TZ_PREFIX.length), value);
    }
  }

  // 4. Parse the schema — map each column to a databonk dtype (throws on unsupported).
  const schemaTree = parquetSchema(metadata);
  // Use 'unknown' cast to avoid SchemaTree structural typing issues with exactOptionalPropertyTypes.
  const colInfos: ColInfo[] = (schemaTree.children as unknown as Array<{ element: Record<string, unknown> }>).map(
    (c) => mapSchemaElement(c.element, tzByCol),
  );

  // 5. Read column chunks via onChunk (column-native; avoids row materialization).
  const chunksByName = new Map<string, DecodedChunk[]>();
  for (const { name } of colInfos) chunksByName.set(name, []);

  await parquetRead({
    file: asyncBuf,
    metadata,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    parsers: RAW_PARSERS as any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    onChunk(chunk: any) {
      const arr = chunksByName.get(chunk.columnName);
      if (arr) arr.push(chunk.columnData as DecodedChunk);
    },
  } as Parameters<typeof parquetRead>[0]);

  // 6. Build databonk columns from the accumulated chunks.
  const namedCols: NamedColumn[] = [];
  for (const { name, dtype, tz } of colInfos) {
    const parts = chunksByName.get(name) ?? [];
    const data = mergeChunks(parts);
    const col = createColumn(rt.ctx, dtype, data, tz);
    namedCols.push({ name, col });
  }

  // 7. Build the DataFrame via buildResult (public API on a seed frame).
  //    The seed frame carries the runtime; buildResult adopts the named columns.
  const seedDf = DataFrame.fromColumns({}, { runtime: rt });
  const df = seedDf.buildResult(namedCols);
  seedDf.dispose();
  return df;
}

// ---------------------------------------------------------------------------
// writeParquet helpers
// ---------------------------------------------------------------------------

/** Parquet SchemaElement shape accepted by hyparquet-writer. */
interface SchemaEl {
  name: string;
  num_children?: number;
  type?: string;
  repetition_type?: string;
  logical_type?: Record<string, unknown>;
  converted_type?: string;
}

/** ColumnSource shape accepted by hyparquet-writer. */
interface ColSrc {
  name: string;
  data: unknown[];
}

/**
 * Build the Parquet schema element and JS data array for one databonk column.
 * Returns `{ el, data, tzKey }` where `tzKey` is set for timestamp-with-tz columns
 * so the caller can add it to `kvMetadata`.
 */
function buildColWrite(
  ctx: { ctx: import('../memory/context.js').MemoryContext }['ctx'],
  colName: string,
  col: import('../memory/column.js').Column,
  n: number,
): { el: SchemaEl; data: unknown[]; tzKey?: { key: string; value: string } } {
  // Extract JS values (with nulls where the validity bit = 0).
  // columnToArray handles all dtypes including utf8 dictionary decode, bigint i64/ts, etc.
  const values = columnToArray(ctx, col) as unknown[];

  let el: SchemaEl;
  let tzKey: { key: string; value: string } | undefined;

  switch (col.dtype) {
    case 'f64':
      el = { name: colName, type: 'DOUBLE', repetition_type: 'OPTIONAL' };
      break;
    case 'f32':
      el = { name: colName, type: 'FLOAT', repetition_type: 'OPTIONAL' };
      break;
    case 'i32':
      el = { name: colName, type: 'INT32', repetition_type: 'OPTIONAL' };
      break;
    case 'u32':
      // Use converted_type UINT_32 (Parquet 1.0 annotation) rather than the newer
      // INTEGER logical type: hyparquet-writer zigzag-encodes the bitWidth i8 as i32
      // which makes parquet-wasm read bitWidth=64 instead of 32 and reject the file.
      // UINT_32 is universally supported and hyparquet reads it via `ct === 'UINT_32'`.
      el = {
        name: colName,
        type: 'INT32',
        repetition_type: 'OPTIONAL',
        converted_type: 'UINT_32',
      };
      break;
    case 'i64':
      el = { name: colName, type: 'INT64', repetition_type: 'OPTIONAL' };
      break;
    case 'bool':
      el = { name: colName, type: 'BOOLEAN', repetition_type: 'OPTIONAL' };
      break;
    case 'utf8':
      el = {
        name: colName,
        type: 'BYTE_ARRAY',
        repetition_type: 'OPTIONAL',
        logical_type: { type: 'STRING' },
        converted_type: 'UTF8',
      };
      break;
    case 'date32':
      el = {
        name: colName,
        type: 'INT32',
        repetition_type: 'OPTIONAL',
        logical_type: { type: 'DATE' },
      };
      break;
    case 'timestamp': {
      el = {
        name: colName,
        type: 'INT64',
        repetition_type: 'OPTIONAL',
        logical_type: {
          type: 'TIMESTAMP',
          isAdjustedToUTC: true,
          // hyparquet-writer expects the Thrift union shape { MILLIS: {} }
          unit: { MILLIS: {} },
        },
      };
      if (col.tz) {
        tzKey = { key: `databonk:tz:${colName}`, value: col.tz };
      }
      break;
    }
    default: {
      // Should not happen for supported dtypes, but be explicit.
      const d: never = col.dtype;
      throw new Error(`unsupported databonk dtype for Parquet write: '${String(d)}'`);
    }
  }

  // exactOptionalPropertyTypes: include tzKey only when it has a value.
  return tzKey !== undefined ? { el, data: values, tzKey } : { el, data: values };
}

// ---------------------------------------------------------------------------
// writeParquet
// ---------------------------------------------------------------------------

/**
 * Encode a databonk {@link DataFrame} as a Parquet file (Uint8Array).
 *
 * All databonk dtypes are supported (ADR-011 §supported-profile).
 * Only `'snappy'` and `'uncompressed'` are valid `compression` values —
 * any other string throws a clear "unsupported" error.
 *
 * @param df   - The DataFrame to encode.
 * @param opts - Optional write options (compression, default `'uncompressed'`).
 */
export function writeParquet(
  df: DataFrame,
  opts: ParquetWriteOptions = {},
): Uint8Array {
  const { compression = 'uncompressed' } = opts;

  if (compression !== 'snappy' && compression !== 'uncompressed') {
    throw new Error(
      `unsupported Parquet compression codec: '${compression}'. ` +
      `databonk/parquet writeParquet only accepts 'snappy' or 'uncompressed' (ADR-011).`,
    );
  }

  const codec = compression === 'snappy' ? 'SNAPPY' : 'UNCOMPRESSED';
  const names = df.columns as string[];
  const n = df.length;
  const ctx = df.ctx;

  // Build schema (hyparquet-writer expects root element first, then children).
  const schema: SchemaEl[] = [{ name: 'root', num_children: names.length }];
  const columnData: ColSrc[] = [];
  const kvMetadata: Array<{ key: string; value: string }> = [];

  for (const name of names) {
    const col = df.getColumn(name)!;
    const { el, data, tzKey } = buildColWrite(ctx, name, col, n);
    schema.push(el);
    columnData.push({ name, data });
    if (tzKey) kvMetadata.push(tzKey);
  }

  // Cast to any to avoid exactOptionalPropertyTypes conflicts with hyparquet-writer's
  // untyped JS options object.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const writeOpts: any = { schema, columnData, codec };
  if (kvMetadata.length > 0) writeOpts.kvMetadata = kvMetadata;
  const resultAb = parquetWriteBuffer(writeOpts);

  // hyparquet-writer returns an ArrayBuffer; wrap in Uint8Array.
  return new Uint8Array(resultAb as unknown as ArrayBuffer);
}
