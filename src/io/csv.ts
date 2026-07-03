/**
 * RFC-4180 CSV parser with typed column inference.
 *
 * Key features:
 *   - Quoted fields, escaped quotes (""), embedded newlines and delimiters.
 *   - CRLF and bare LF line endings; lone CR treated as data.
 *   - Streaming-friendly internal design: `ChunkParser` is a stateful, chunk-fed
 *     parser (feed(s)/finish()). `fromCSV` drives it synchronously over the full text.
 *     ReadableStream adapter is a v2 nice-to-have (see ponytail note below).
 *
 * Type inference ladder (applied column-by-column over a sample of up to
 * `INFER_ROWS` rows after null-value stripping):
 *
 *   1. i32   — integer in [-(2^31), 2^31-1], no decimal point, no exponent.
 *   2. i64   — integer text exceeding Number.MAX_SAFE_INTEGER (|x| > 2^53-1, per ADR-009).
 *              Promotes when any non-null cell is a valid strict-integer string whose
 *              absolute value is above 9_007_199_254_740_991 (2^53−1) but within i64 range.
 *              Once promoted, the column stays i64 even if later cells are in safe range.
 *              Cells whose BigInt value overflows i64 → utf8 fallback.
 *   3. f64   — any string parseable by `Number()` that is finite (or ±Infinity),
 *              i.e. `!isNaN(n)` after trimming.
 *   4. bool  — case-insensitive "true" / "false".
 *   5. utf8  — fallback.
 *
 * Explicit dtype overrides (`dtypes` option):
 *   - `'i64'`       — parse via BigInt(text); throws on non-integer strings.
 *   - `'date32'`    — parse via strict 'yyyy-MM-dd'; throws on invalid format.
 *   - `'timestamp'` — parse via strict ISO-8601 with 'Z' or explicit UTC offset;
 *                     throws on ambiguous local-time strings (no silent tz guessing).
 *
 * Null values: the `nullValues` option (default `['', 'null', 'NA']`) is
 * checked BEFORE inference. A null-value cell becomes `null` in any dtype column.
 *
 * v2 nice-to-have (not shipped): ReadableStream adapter wrapping ChunkParser.
 * // ponytail: fromCSVStream(stream, opts) → v2 when ReadableStream usage is confirmed
 */

import { DataFrame, type FrameOptions } from '../frame/dataframe.js';
import type { DType } from '../memory/dtype.js';
import type { Cell } from '../memory/column.js';
import type { DfRuntime } from '../frame/runtime.js';
import { civilToDays } from '../temporal/civil.js';

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface FromCsvOptions {
  /** Column delimiter (default: ','). */
  readonly delimiter?: string;
  /** If true (default), treat first row as column headers. */
  readonly header?: boolean;
  /**
   * Override column names (used when header=false, or to rename columns).
   * When header=true, these override the parsed header names.
   */
  readonly columns?: readonly string[];
  /** Force specific dtypes (column name → dtype). Skips inference for those columns. */
  readonly dtypes?: Readonly<Record<string, DType>>;
  /**
   * Values treated as null (case-sensitive). Default: `['', 'null', 'NA']`.
   * An empty string means the empty field is null (the default for CSV).
   */
  readonly nullValues?: readonly string[];
  /** Skip this many rows from the top (before header if present). */
  readonly skipRows?: number;
  /** Maximum number of data rows to read (not counting header or skipped rows). */
  readonly maxRows?: number;
  /** Runtime to use. If omitted, uses the default runtime. */
  readonly runtime?: DfRuntime;
}

const DEFAULT_NULL_VALUES = ['', 'null', 'NA'];
const INFER_ROWS = 100; // sample size for type inference

// ---------------------------------------------------------------------------
// RFC-4180 chunk-fed parser
// ---------------------------------------------------------------------------

/**
 * Stateful RFC-4180 CSV parser.
 * Feed text chunks with `feed(s)` and call `finish()` to flush the last row.
 * Each completed row is passed to the `onRow` callback.
 *
 * Design: simple state machine over a single `remaining` string. Performance is
 * not critical here (CSV ingestion is I/O-bound); correctness is.
 */
export class ChunkParser {
  private buf = '';       // unparsed bytes from previous chunk(s)
  private row: string[] = [];
  private field = '';
  private inQuote = false;
  private readonly delim: string;
  private readonly onRow: (row: string[]) => void;

  constructor(delimiter: string, onRow: (row: string[]) => void) {
    this.delim = delimiter;
    this.onRow = onRow;
  }

  feed(chunk: string): void {
    this.buf += chunk;
    this.parse(false);
  }

  finish(): void {
    this.parse(true);
  }

  private parse(eof: boolean): void {
    const d = this.delim;
    let i = 0;
    const s = this.buf;
    const len = s.length;

    while (i < len) {
      const ch = s[i]!;

      if (this.inQuote) {
        if (ch === '"') {
          // Peek next character
          if (i + 1 < len) {
            if (s[i + 1] === '"') {
              // Escaped quote — append one quote and advance past both
              this.field += '"';
              i += 2;
            } else {
              // End of quoted field
              this.inQuote = false;
              i++;
            }
          } else if (eof) {
            // End of input while in quote — close it (malformed but tolerate)
            this.inQuote = false;
            i++;
          } else {
            // Need more data to decide if this is "" or end-of-quote
            break;
          }
        } else {
          this.field += ch;
          i++;
        }
      } else {
        if (ch === '"') {
          this.inQuote = true;
          i++;
        } else if (s.startsWith(d, i)) {
          // Field separator
          this.row.push(this.field);
          this.field = '';
          i += d.length;
        } else if (ch === '\r') {
          if (i + 1 < len) {
            if (s[i + 1] === '\n') {
              // CRLF
              this.emitRow();
              i += 2;
            } else {
              // Lone CR — treat as data (not a standard line ending)
              this.field += ch;
              i++;
            }
          } else if (eof) {
            // Lone CR at end of input
            this.emitRow();
            i++;
          } else {
            // Can't decide — need more data
            break;
          }
        } else if (ch === '\n') {
          this.emitRow();
          i++;
        } else {
          this.field += ch;
          i++;
        }
      }
    }

    this.buf = s.slice(i); // Keep unparsed remainder for next chunk

    if (eof && (this.row.length > 0 || this.field.length > 0)) {
      if (this.inQuote) {
        throw new Error(
          `CSV parse error: unterminated quoted field starting near "${this.field.slice(0, 40)}"`,
        );
      }
      this.emitRow();
    }
  }

  private emitRow(): void {
    this.row.push(this.field);
    this.field = '';
    this.onRow(this.row);
    this.row = [];
  }
}

// ---------------------------------------------------------------------------
// Type inference
// ---------------------------------------------------------------------------

const I32_MIN = -(2 ** 31);
const I32_MAX = 2 ** 31 - 1;

// i64 inference constants (ADR-009 CSV promotion rule).
const I64_MAX_N = 9_223_372_036_854_775_807n;
const I64_MIN_N = -9_223_372_036_854_775_808n;
const SAFE_MAX_N = 9_007_199_254_740_991n; // Number.MAX_SAFE_INTEGER as BigInt (2^53-1)

/**
 * True if `s` is a strict-integer string (no decimal/exponent, no leading zeros
 * except '0' itself, optional leading minus). Used by both i32 and i64 checkers.
 */
function isStrictInt(s: string): boolean {
  if (s.length === 0) return false;
  let i = 0;
  if (s[0] === '-') i = 1;
  if (i === s.length) return false; // bare '-'
  if (s.length - i > 1 && s[i] === '0') return false; // leading zeros
  for (; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 48 || c > 57) return false;
  }
  return true;
}

function isI32(s: string): boolean {
  if (!isStrictInt(s)) return false;
  const n = Number(s);
  return n >= I32_MIN && n <= I32_MAX && Number.isInteger(n);
}

/**
 * True if `s` is a strict-integer string whose absolute value exceeds
 * Number.MAX_SAFE_INTEGER (2^53-1) but fits within i64 range.
 * These values MUST be represented as BigInt — they cannot be safely stored as JS numbers.
 * Encountering such a value in the sample triggers i64 column promotion (ADR-009).
 */
function isI64Overflow(s: string): boolean {
  if (!isStrictInt(s)) return false;
  try {
    const v = BigInt(s);
    const abs = v < 0n ? -v : v;
    return abs > SAFE_MAX_N && v >= I64_MIN_N && v <= I64_MAX_N;
  } catch {
    return false;
  }
}

/** True if `s` is a valid i64 decimal integer string (any value in i64 range). */
function isAnyI64(s: string): boolean {
  if (!isStrictInt(s)) return false;
  try {
    const v = BigInt(s);
    return v >= I64_MIN_N && v <= I64_MAX_N;
  } catch {
    return false;
  }
}

function isF64(s: string): boolean {
  if (s.length === 0) return false;
  // Whitespace-only strings coerce to 0 via Number(), but they are not numeric
  // values — they should fall through to utf8 inference.
  if (s.trim().length === 0) return false;
  const n = Number(s);
  return !isNaN(n);
}

function isBool(s: string): boolean {
  const l = s.toLowerCase();
  return l === 'true' || l === 'false';
}

type InferState = 'i32' | 'i64' | 'f64' | 'bool' | 'utf8';

/**
 * Promote the inference state given a new non-null cell value.
 *
 * Inference ladder (ADR-009 v2 extension):
 *   i32 → i64  when a strict-integer string exceeds Number.MAX_SAFE_INTEGER
 *   i32 → f64  when a float-parseable string is seen
 *   i64 → f64  when a float (non-integer) string is seen while in i64 state
 *   * → bool   when only "true"/"false" strings appear
 *   * → utf8   fallback
 */
function promote(state: InferState, cell: string): InferState {
  if (state === 'utf8') return 'utf8';
  if (state === 'i32') {
    if (isI32(cell)) return 'i32';
    if (isI64Overflow(cell)) return 'i64'; // big integer promotes to i64 (ADR-009)
    if (isF64(cell)) return 'f64';
    if (isBool(cell)) return 'bool';
    return 'utf8';
  }
  if (state === 'i64') {
    // In i64 state: accept any valid i64 integer (small or large).
    if (isAnyI64(cell)) return 'i64';
    // A non-integer float demotes to f64 (precision may be lost for large i64 values).
    if (isF64(cell)) return 'f64';
    if (isBool(cell)) return 'bool';
    return 'utf8';
  }
  if (state === 'f64') {
    if (isF64(cell)) return 'f64';
    if (isBool(cell)) return 'bool';
    return 'utf8';
  }
  // state === 'bool'
  if (isBool(cell)) return 'bool';
  if (isI32(cell)) return 'utf8'; // can't demote bool→i32 (ambiguous)
  return 'utf8';
}

function inferDtypeFromSample(sample: string[], nullSet: Set<string>): DType {
  let state: InferState = 'i32';
  let seenNonNull = false;
  for (const cell of sample) {
    if (nullSet.has(cell)) continue;
    seenNonNull = true;
    state = promote(state, cell);
    if (state === 'utf8') return 'utf8'; // can't get worse
  }
  if (!seenNonNull) return 'utf8'; // all-null column → utf8 (safe default)
  return state as DType;
}

// ---------------------------------------------------------------------------
// Explicit dtype parse helpers (ADR-009/010: i64, date32, timestamp)
// ---------------------------------------------------------------------------

/**
 * Parse a cell as i64 by converting the decimal string to BigInt (ADR-009).
 * Throws a descriptive error on non-integer strings (e.g. "3.14", "hello").
 */
function parseCellI64(cell: string): bigint {
  try {
    return BigInt(cell);
  } catch {
    throw new Error(
      `CSV parse error: cannot parse '${cell}' as i64 — expected an integer decimal string.`,
    );
  }
}

// Strict 'yyyy-MM-dd' regex (4+ digit year, two-digit month/day). No time, no tz.
const DATE32_RE = /^(-?\d{4,})-(\d{2})-(\d{2})$/;

/**
 * Parse a cell as date32 (days since 1970-01-01 UTC) from strict 'yyyy-MM-dd' format (ADR-010).
 * Reuses `civilToDays` from src/temporal/civil.ts for the calendar math.
 * Throws a descriptive error for invalid format or out-of-range month/day values.
 */
function parseCellDate32(cell: string): number {
  const m = DATE32_RE.exec(cell);
  if (!m) {
    throw new Error(
      `CSV parse error: cannot parse '${cell}' as date32 — expected strict 'yyyy-MM-dd' format.`,
    );
  }
  const y = Number(m[1]);
  const mo = Number(m[2]);
  const d = Number(m[3]);
  if (mo < 1 || mo > 12 || d < 1 || d > 31) {
    throw new Error(
      `CSV parse error: invalid date components in '${cell}' (month=${mo}, day=${d}).`,
    );
  }
  return civilToDays(y, mo, d);
}

// Strict ISO-8601 timestamp requires explicit 'Z' or UTC offset (±HH:MM or ±HHMM).
// Bare 'yyyy-MM-ddTHH:MM:SS' without tz is ambiguous local-time and is rejected.
const HAS_TZ_RE = /Z$|[+-]\d{2}:?\d{2}$/;

/**
 * Parse a cell as a timestamp (epoch milliseconds, BigInt) from strict ISO-8601 format (ADR-010).
 * Requires an explicit 'Z' or UTC offset (e.g. '+05:30', '-0700'). Rejects ambiguous
 * local-time strings (no 'Z'/offset) with a helpful error — no silent tz guessing.
 */
function parseCellTimestamp(cell: string): bigint {
  if (!HAS_TZ_RE.test(cell)) {
    throw new Error(
      `CSV parse error: timestamp '${cell}' has no timezone (Z or UTC offset required). ` +
      `Ambiguous local-time strings are rejected to avoid silent tz guessing — ` +
      `use '${cell}Z' for UTC or append an explicit offset like '+00:00'.`,
    );
  }
  const ms = Date.parse(cell);
  if (isNaN(ms)) {
    throw new Error(
      `CSV parse error: cannot parse '${cell}' as ISO-8601 timestamp.`,
    );
  }
  return BigInt(ms);
}

function parseCell(cell: string, dtype: DType, nullSet: Set<string>): Cell {
  if (nullSet.has(cell)) return null;
  switch (dtype) {
    case 'i64':
      return parseCellI64(cell);
    case 'date32':
      return parseCellDate32(cell);
    case 'timestamp':
      return parseCellTimestamp(cell);
    case 'i32':
    case 'u32':
    case 'f64':
    case 'f32':
      return Number(cell);
    case 'bool':
      return cell.toLowerCase() === 'true';
    case 'utf8':
      return cell;
    default:
      return cell;
  }
}

// ---------------------------------------------------------------------------
// fromCSV()
// ---------------------------------------------------------------------------

/**
 * Parse a CSV string into a DataFrame.
 *
 * @param text   - Full CSV text (a chunk-fed path for streaming is internal).
 * @param opts   - Parsing and inference options.
 *
 * @throws If there are ragged rows (inconsistent column count) or unterminated quotes.
 */
export function fromCSV(text: string, opts: FromCsvOptions = {}): DataFrame {
  const delimiter  = opts.delimiter ?? ',';
  const hasHeader  = opts.header !== false;
  const skipRows   = opts.skipRows ?? 0;
  const maxRows    = opts.maxRows ?? Infinity;
  const nullSet    = new Set<string>(opts.nullValues ?? DEFAULT_NULL_VALUES);
  const forceDtype = opts.dtypes ?? {};

  // Collect raw rows
  const rawRows: string[][] = [];
  const parser = new ChunkParser(delimiter, (row) => rawRows.push(row));
  parser.feed(text);
  parser.finish();

  const rtOpts = (rt: DfRuntime | undefined): FrameOptions => rt ? { runtime: rt } : {};

  if (rawRows.length === 0) {
    // Empty file — return empty DataFrame
    return DataFrame.fromColumns({}, rtOpts(opts.runtime));
  }

  // Apply skipRows
  const startIdx = skipRows;
  if (startIdx >= rawRows.length) {
    return DataFrame.fromColumns({}, rtOpts(opts.runtime));
  }

  // Extract header
  let colNames: string[];
  let dataStart: number;

  if (hasHeader) {
    colNames = rawRows[startIdx]!;
    dataStart = startIdx + 1;
  } else {
    const firstData = rawRows[startIdx]!;
    colNames = firstData.map((_, i) => opts.columns?.[i] ?? `column_${i}`);
    dataStart = startIdx;
  }

  // Override names from opts.columns
  if (opts.columns) {
    for (let i = 0; i < opts.columns.length; i++) {
      colNames[i] = opts.columns[i]!;
    }
  }

  const numCols = colNames.length;

  // Collect data rows (up to maxRows), validate width
  const dataRows = rawRows.slice(dataStart, dataStart + (isFinite(maxRows) ? maxRows : rawRows.length));

  for (let r = 0; r < dataRows.length; r++) {
    if (dataRows[r]!.length !== numCols) {
      throw new Error(
        `CSV parse error: row ${dataStart + r + 1} has ${dataRows[r]!.length} fields, ` +
        `expected ${numCols} (columns: ${colNames.join(', ')})`,
      );
    }
  }

  if (dataRows.length === 0) {
    const emptyCols: Record<string, Cell[]> = {};
    const emptyDtypes: Record<string, DType> = {};
    for (const name of colNames) {
      emptyCols[name] = [];
      emptyDtypes[name] = forceDtype[name] ?? 'utf8';
    }
    const emptyOpts: FrameOptions = { ...rtOpts(opts.runtime), dtypes: emptyDtypes };
    return DataFrame.fromColumns(emptyCols as Record<string, import('../memory/column.js').ColumnInput>, emptyOpts);
  }

  // Infer dtypes using first INFER_ROWS rows
  const sample = dataRows.slice(0, INFER_ROWS);
  const dtypes: Record<string, DType> = {};
  for (let c = 0; c < numCols; c++) {
    const name = colNames[c]!;
    if (forceDtype[name]) {
      dtypes[name] = forceDtype[name];
    } else {
      const colSample = sample.map((row) => row[c]!);
      dtypes[name] = inferDtypeFromSample(colSample, nullSet);
    }
  }

  // Parse all cells
  const colData: Record<string, Cell[]> = {};
  for (const name of colNames) colData[name] = new Array(dataRows.length);

  for (let r = 0; r < dataRows.length; r++) {
    const row = dataRows[r]!;
    for (let c = 0; c < numCols; c++) {
      const name = colNames[c]!;
      colData[name]![r] = parseCell(row[c]!, dtypes[name]!, nullSet);
    }
  }

  const finalOpts: FrameOptions = { ...rtOpts(opts.runtime), dtypes };
  return DataFrame.fromColumns(colData as Record<string, import('../memory/column.js').ColumnInput>, finalOpts);
}
