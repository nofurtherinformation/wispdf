/** Series — one named column, zero-copy over a frame's buffers (spec §4 `df.col('a')`).
 * Borrows (does not own) its parent's column; valid while that frame lives. Read-only. */

import type { MemoryContext } from '../memory/context.js';
import { DTYPES, type DType } from '../memory/dtype.js';
import { createColumn, columnToArray, type Cell, type Column } from '../memory/column.js';
import type { ColumnView } from '../memory/views.js';
import { extractComponents, type DtComponent } from '../temporal/vectorize.js';
import {
  writeDictionary,
  decodeDictionary,
  freeDictionary,
} from '../memory/dictionary.js';
import { getBit, validityBytes } from '../memory/bitmap.js';

/** dt accessor components allowed for date32 columns (dtypes.md §10). */
const DATE32_ALLOWED: ReadonlySet<DtComponent> = new Set<DtComponent>([
  'year', 'month', 'day', 'weekday', 'dayOfYear', 'quarter',
]);

/**
 * Series.dt accessor proxy. Returned by `series.dt`; provides per-field
 * methods that return a new Series (dtype=i32) with the extracted values.
 *
 * Rules (dtypes.md §10, ADR-010, ADR-012):
 *   - timestamp columns: all 9 fields are valid.
 *   - date32 columns: year/month/day/weekday/dayOfYear/quarter are valid;
 *     hour/minute/second/millisecond throw a TypeError naming the op.
 *   - tz metadata on a timestamp column gives local-time components (ADR-010).
 */
export class SeriesDtProxy {
  constructor(private readonly series: Series) {}

  private extract(c: DtComponent): Series {
    const { dtype } = this.series;
    if (dtype === 'date32') {
      if (!DATE32_ALLOWED.has(c)) {
        throw new TypeError(
          `dt.${c} is not supported for date32 columns ` +
          `(time-of-day fields are always 0 for date32; ` +
          `use year/month/day/weekday/dayOfYear/quarter, or cast to timestamp first).`,
        );
      }
    } else if (dtype !== 'timestamp') {
      throw new TypeError(
        `dt accessor requires a date32 or timestamp column, got ${dtype}.`,
      );
    }

    const col = this.series.col;
    const ctx = this.series['ctx'];
    const len = col.length;
    const tz = (dtype === 'timestamp' && col.tz) ? col.tz : undefined;

    // Build validity bitmap for extractComponents (null = all-valid).
    // Series columns always have validityBitOffset === 0 (normalised at creation).
    let validityBitmap: Uint8Array | null = null;
    if (col.validityPtr !== 0) {
      const offset = col.validityBitOffset;
      const vbytes = Math.ceil((offset + len) / 8);
      const raw = ctx.viewOf({
        ptr: col.validityPtr,
        length: vbytes,
        dtype: 'u8',
      }) as Uint8Array;

      if (offset === 0) {
        // Fast path: share the view directly (extractComponents only reads it)
        validityBitmap = raw;
      } else {
        // Shift bits to a zero-aligned copy (rare — non-zero offset from slices).
        const shifted = new Uint8Array(Math.ceil(len / 8));
        for (let i = 0; i < len; i++) {
          const srcBit = offset + i;
          if ((raw[srcBit >> 3]! >> (srcBit & 7)) & 1) {
            shifted[i >> 3]! |= 1 << (i & 7);
          }
        }
        validityBitmap = shifted;
      }
    }

    // View source data and run extraction.
    let resultData: Int32Array;
    if (dtype === 'date32') {
      const view = ctx.viewOf({
        ptr: col.dataPtr,
        length: len,
        dtype: 'i32',
      }) as Int32Array;
      ({ data: resultData } = extractComponents(view, validityBitmap, c));
    } else {
      const view = ctx.viewOf({
        ptr: col.dataPtr,
        length: len,
        dtype: 'i64',
      }) as BigInt64Array;
      ({ data: resultData } = extractComponents(view, validityBitmap, c, tz));
    }

    // Build i32 output values array, propagating nulls from source validity.
    const outputValues: Array<number | null> = new Array(len);
    if (validityBitmap !== null) {
      for (let i = 0; i < len; i++) {
        const valid = ((validityBitmap[i >> 3] ?? 0) >> (i & 7)) & 1;
        outputValues[i] = valid ? (resultData[i] ?? 0) : null;
      }
    } else {
      for (let i = 0; i < len; i++) {
        outputValues[i] = resultData[i] ?? 0;
      }
    }

    const outCol = createColumn(ctx, 'i32', outputValues);
    return new Series(ctx, `${this.series.name}.dt.${c}`, outCol);
  }

  year(): Series        { return this.extract('year'); }
  month(): Series       { return this.extract('month'); }
  day(): Series         { return this.extract('day'); }
  hour(): Series        { return this.extract('hour'); }
  minute(): Series      { return this.extract('minute'); }
  second(): Series      { return this.extract('second'); }
  millisecond(): Series { return this.extract('millisecond'); }
  weekday(): Series     { return this.extract('weekday'); }
  dayOfYear(): Series   { return this.extract('dayOfYear'); }
  quarter(): Series     { return this.extract('quarter'); }
}

/**
 * Series.str accessor proxy. Returned by `series.str`; provides string operations
 * over `utf8` columns (dtypes.md §13). Throws TypeError if the column is not utf8.
 */
export class SeriesStrProxy {
  constructor(private readonly series: Series) {}

  /**
   * Substring via `JS String.prototype.slice` semantics (dtypes.md §13).
   *
   * - Negative `start`/`end`: count from the end of the string.
   * - `end` omitted: slice to the end of the string.
   * - Out-of-range indices clamp (same as JS).
   * - Null rows propagate null; the op never inspects null-row string values.
   * - UTF-16 code-unit indexing.
   *   **Surrogate-pair caveat:** a supplementary character occupies two code units;
   *   `slice` may split a surrogate pair (well-defined in JS, unusual in UTF-8).
   *
   * Implementation: applied to dictionary values once (O(unique)), then indices
   * remapped (O(rows)) — no per-row string work.
   *
   * Returns a new `utf8` Series.
   */
  slice(start: number, end?: number): Series {
    const { dtype, col, name } = this.series;
    const ctx = this.series['ctx'];
    if (dtype !== 'utf8') {
      throw new TypeError(
        `str.slice requires a utf8 column, got ${dtype}.`,
      );
    }
    const len = col.length;

    // Decode source dictionary once (memoized per slot via decodeSlot).
    const srcStrings = col.dict ? decodeDictionary(ctx, col.dict) : [];
    const srcCount = srcStrings.length;

    // Apply slice to each unique value; re-dedup into result uniques.
    const slotRemap = new Int32Array(srcCount);
    const resultUniques: string[] = [];
    const index = new Map<string, number>();
    for (let k = 0; k < srcCount; k++) {
      const sliced = end === undefined
        ? srcStrings[k]!.slice(start)
        : srcStrings[k]!.slice(start, end);
      let j = index.get(sliced);
      if (j === undefined) {
        j = resultUniques.length;
        resultUniques.push(sliced);
        index.set(sliced, j);
      }
      slotRemap[k] = j;
    }

    // Build output as a plain Cell array: remap each non-null row, propagate nulls.
    // We use createColumn to encode the result dict (handles all wasm allocs cleanly).
    const srcIdx = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i32' }) as Int32Array;
    const outValues: Array<string | null> = new Array(len);

    if (col.validityPtr === 0) {
      // All valid.
      for (let i = 0; i < len; i++) {
        outValues[i] = resultUniques[slotRemap[srcIdx[i]!]!]!;
      }
    } else {
      // Propagate nulls.
      const bitOff = col.validityBitOffset;
      const vbytes = validityBytes(bitOff + len);
      const vmap = ctx.viewOf({ ptr: col.validityPtr, length: vbytes, dtype: 'u8' }) as Uint8Array;
      for (let i = 0; i < len; i++) {
        if (getBit(vmap, bitOff + i)) {
          outValues[i] = resultUniques[slotRemap[srcIdx[i]!]!]!;
        } else {
          outValues[i] = null;
        }
      }
    }

    const outCol = createColumn(ctx, 'utf8', outValues);
    return new Series(ctx, `${name}.str.slice(${start}${end === undefined ? '' : `, ${end}`})`, outCol);
  }
}

export class Series {

  readonly name: string;

  readonly dtype: DType;

  readonly length: number;

  private readonly ctx: MemoryContext;
  private readonly column: Column;

  constructor(ctx: MemoryContext, name: string, column: Column) {
    this.ctx = ctx;
    this.name = name;
    this.dtype = column.dtype;
    this.length = column.length;
    this.column = column;
  }

  get col(): Column {
    return this.column;
  }

  toArray(): Cell[] {
    return columnToArray(this.ctx, this.column);
  }

  get(i: number): Cell {
    if (i < 0 || i >= this.length) return null;
    return this.toArray()[i] ?? null;
  }

  values(): ColumnView {
    return this.ctx.viewOf({
      ptr: this.column.dataPtr,
      length: this.length,
      dtype: DTYPES[this.dtype].view,
    });
  }

  /**
   * dt accessor namespace for date32 / timestamp columns (dtypes.md §10, ADR-010, ADR-012).
   * Returns a {@link SeriesDtProxy} with accessor methods for each calendar field.
   * Throws TypeError for disallowed fields (e.g. hour on date32).
   */
  get dt(): SeriesDtProxy {
    if (this.dtype !== 'date32' && this.dtype !== 'timestamp') {
      throw new TypeError(
        `dt accessor requires a date32 or timestamp column, got ${this.dtype}.`,
      );
    }
    return new SeriesDtProxy(this);
  }

  /**
   * str accessor namespace for `utf8` columns (dtypes.md §13).
   * Returns a {@link SeriesStrProxy} with string methods.
   * Throws TypeError if the column is not `utf8`.
   */
  get str(): SeriesStrProxy {
    if (this.dtype !== 'utf8') {
      throw new TypeError(
        `str accessor requires a utf8 column, got ${this.dtype}.`,
      );
    }
    return new SeriesStrProxy(this);
  }

  [Symbol.iterator](): IterableIterator<Cell> {
    return this.toArray()[Symbol.iterator]();
  }

  toString(): string {
    return `Series '${this.name}' (${this.dtype}, ${this.length} rows)`;
  }
}
