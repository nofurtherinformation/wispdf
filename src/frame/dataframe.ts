/**
 * DataFrame — the public columnar frame (spec §4). Every op returns a NEW frame; buffers are
 * shared zero-copy where no data changes (select/drop/untouched withColumn, head/tail/slice)
 * and reference-counted ({@link OwnedColumn}) so `dispose()` is order-independent. Expression
 * ops (filter/withColumn) run the P3.1 compiler (fast path); filterFn/mapFn are the ADR-003
 * slow path. Use `scope(fn)` to dispose a batch of intermediates.
 */

import type { LoadOptions } from '../memory/loader.js';
import type { MemoryContext } from '../memory/context.js';
import { type DType } from '../memory/dtype.js';
import {
  createColumn,
  columnToArray,
  sliceColumn,
  type Cell,
  type Column,
  type ColumnInput,
} from '../memory/column.js';
import {
  Expr,
  compile,
  compileFilter,
  type FrameView,
  type KernelWasm,
} from '../expr/index.js';
import { defaultRuntime, type DfRuntime } from './runtime.js';
import { OwnedColumn } from './owned.js';
import { inferDType } from './infer.js';
import { Series } from './series.js';
import { gatherColumn } from './gather.js';
import { sortPerm } from './sort.js';
import { makeRowCursor, type Row } from './rowproxy.js';
import { GroupBy, type GroupBySource, type NamedColumn } from './groupby.js';
import { joinFrames, type JoinOptions } from './join.js';
import { FrameError, unknownColumn } from './errors.js';
import { formatCell, formatTable, alignFor, ELLIPSIS, type Align } from './printer.js';

interface Entry {
  readonly name: string;
  readonly col: Column;
  readonly owner: OwnedColumn;
}

export interface FrameOptions {

  readonly dtypes?: Readonly<Record<string, DType>>;

  readonly runtime?: DfRuntime;

  /**
   * IANA timezone strings for `timestamp` columns (ADR-010 §10 tz metadata).
   * Keys are column names; values are IANA tz strings (e.g. `"America/New_York"`).
   * Applied only to columns whose dtype is `'timestamp'`; ignored for other dtypes.
   * Used by Arrow/CSV IO layers to propagate tz metadata from the source format.
   */
  readonly tzs?: Readonly<Record<string, string>>;
}

export interface SortOptions {

  readonly descending?: boolean | readonly boolean[];
}

export interface WithColumnOptions {

  readonly dtype?: DType;
}

const NUMERIC = new Set<DType>(['f64', 'f32', 'i32', 'u32', 'i64']);

export class DataFrame implements FrameView, GroupBySource {

  readonly length: number;
  private readonly _rt: DfRuntime;
  private readonly entries: Entry[];
  private readonly byName: Map<string, Entry>;
  private disposed = false;

  private constructor(rt: DfRuntime, entries: Entry[], length: number) {
    this._rt = rt;
    this.entries = entries;
    this.length = length;
    this.byName = new Map(entries.map((e) => [e.name, e]));
  }

  get ctx(): MemoryContext {
    return this._rt.ctx;
  }
  get wasm(): KernelWasm {
    return this._rt.wasm;
  }
  get rt(): DfRuntime {
    return this._rt;
  }
  getColumn(name: string): Column | undefined {
    return this.byName.get(name)?.col;
  }
  dtypeOf(name: string): DType | undefined {
    return this.byName.get(name)?.col.dtype;
  }
  columnNames(): readonly string[] {
    return this.entries.map((e) => e.name);
  }

  buildResult(named: NamedColumn[]): DataFrame {
    const length = named.length > 0 ? named[0]!.col.length : this.length;
    return DataFrame.fromRoots(this._rt, named, length);
  }

  static fromColumns(cols: Readonly<Record<string, ColumnInput>>, opts: FrameOptions = {}): DataFrame {
    const rt = opts.runtime ?? defaultRuntime();
    const named: NamedColumn[] = [];
    let length = -1;
    for (const [name, input] of Object.entries(cols)) {
      const dtype = opts.dtypes?.[name] ?? inferDType(input);
      const tz = dtype === 'timestamp' ? opts.tzs?.[name] : undefined;
      const col = createColumn(rt.ctx, dtype, input, tz);
      if (length === -1) length = col.length;
      else if (col.length !== length) {
        throw new FrameError(`column '${name}' has length ${col.length}, expected ${length}.`);
      }
      named.push({ name, col });
    }
    return DataFrame.fromRoots(rt, named, length === -1 ? 0 : length);
  }

  static fromRecords(
    records: ReadonlyArray<Readonly<Record<string, Cell>>>,
    opts: FrameOptions = {},
  ): DataFrame {
    const rt = opts.runtime ?? defaultRuntime();
    const names: string[] = [];
    const seen = new Set<string>();
    for (const rec of records) {
      for (const k of Object.keys(rec)) {
        if (!seen.has(k)) {
          seen.add(k);
          names.push(k);
        }
      }
    }
    const cols: Record<string, ColumnInput> = {};
    for (const name of names) {
      const values = new Array<Cell>(records.length);
      for (let i = 0; i < records.length; i++) {
        const v = records[i]![name];
        values[i] = v === undefined ? null : v;
      }
      cols[name] = values as ColumnInput;
    }
    return DataFrame.fromColumns(cols, opts);
  }

  private static fromRoots(rt: DfRuntime, named: NamedColumn[], length: number): DataFrame {
    const entries = named.map((n) => ({
      name: n.name,
      col: n.col,
      owner: new OwnedColumn(rt.ctx, n.col),
    }));
    return new DataFrame(rt, entries, length);
  }

  get shape(): readonly [number, number] {
    return [this.length, this.entries.length];
  }

  get columns(): readonly string[] {
    return this.columnNames();
  }

  get dtypes(): Readonly<Record<string, DType>> {
    const out: Record<string, DType> = {};
    for (const e of this.entries) out[e.name] = e.col.dtype;
    return out;
  }

  col(name: string): Series {
    const e = this.entryOf(name);
    return new Series(this._rt.ctx, name, e.col);
  }

  select(names: readonly string[]): DataFrame {
    return new DataFrame(this._rt, this.shareEntries(names), this.length);
  }

  drop(names: readonly string[]): DataFrame {
    const dropSet = new Set(names);
    for (const n of names) if (!this.byName.has(n)) throw unknownColumn(n, this.columnNames());
    const keep = this.entries.filter((e) => !dropSet.has(e.name)).map((e) => e.name);
    return new DataFrame(this._rt, this.shareEntries(keep), this.length);
  }

  withColumn(name: string, value: Expr | ColumnInput, opts: WithColumnOptions = {}): DataFrame {
    const col = this.materializeColumn(value, opts);
    const owner = new OwnedColumn(this._rt.ctx, col);
    const replacement: Entry = { name, col, owner };

    const next: Entry[] = [];
    let replaced = false;
    for (const e of this.entries) {
      if (e.name === name) {
        next.push(replacement);
        replaced = true;
      } else {
        next.push(this.retainEntry(e));
      }
    }
    if (!replaced) next.push(replacement);
    return new DataFrame(this._rt, next, this.length);
  }

  assign(name: string, value: Expr | ColumnInput, opts?: WithColumnOptions): DataFrame {
    return this.withColumn(name, value, opts);
  }

  filter(predicate: Expr): DataFrame {
    const sel = compileFilter(predicate, this).execute();
    try {
      const named: NamedColumn[] = this.entries.map((e) => ({
        name: e.name,
        col: sel.compact(e.col),
      }));
      return DataFrame.fromRoots(this._rt, named, sel.count);
    } finally {
      sel.free();
    }
  }

  filterFn(fn: (row: Row) => boolean): DataFrame {
    const cursor = makeRowCursor(this._rt.ctx, this.entries);
    const kept: number[] = [];
    for (let i = 0; i < this.length; i++) if (fn(cursor.at(i))) kept.push(i);
    return this.gatherRows(kept);
  }

  mapFn<T>(fn: (row: Row) => T): T[] {
    const cursor = makeRowCursor(this._rt.ctx, this.entries);
    const out = new Array<T>(this.length);
    for (let i = 0; i < this.length; i++) out[i] = fn(cursor.at(i));
    return out;
  }

  sortValues(by: string | readonly string[], opts: SortOptions = {}): DataFrame {
    const keys = typeof by === 'string' ? [by] : [...by];
    if (keys.length === 0) throw new FrameError('sortValues requires at least one key.');
    const keyCols = keys.map((k) => this.entryOf(k).col);
    const desc = Array.isArray(opts.descending)
      ? keys.map((_, i) => (opts.descending as readonly boolean[])[i] === true)
      : keys.map(() => opts.descending === true);

    const perm = sortPerm(this._rt, keyCols, desc);
    try {
      const named: NamedColumn[] = this.entries.map((e) => ({
        name: e.name,
        col: gatherColumn(this._rt, e.col, perm.ptr, perm.len),
      }));
      return DataFrame.fromRoots(this._rt, named, perm.len);
    } finally {
      this._rt.ctx.viewOf.forget({ ptr: perm.ptr, length: perm.len, dtype: 'i32' });
      this._rt.ctx.mod.free(perm.ptr);
    }
  }

  groupby(keys: string | readonly string[]): GroupBy {
    const list = typeof keys === 'string' ? [keys] : [...keys];
    if (list.length === 0) throw new FrameError('groupby requires at least one key.');
    return new GroupBy(this, list);
  }

  join(other: DataFrame, opts: JoinOptions): DataFrame {
    return joinFrames(this, other, opts);
  }

  head(n = 5): DataFrame {
    return this.sliceRange(0, n);
  }

  tail(n = 5): DataFrame {
    return this.sliceRange(Math.max(0, this.length - n), this.length);
  }

  slice(start: number, end: number = this.length): DataFrame {
    return this.sliceRange(start, end);
  }

  private sliceRange(start: number, end: number): DataFrame {
    const s = Math.max(0, Math.min(start, this.length));
    const e = Math.max(s, Math.min(end, this.length));
    const entries = this.entries.map((entry) => ({
      name: entry.name,
      col: sliceColumn(entry.col, s, e),
      owner: entry.owner.retain(),
    }));
    return new DataFrame(this._rt, entries, e - s);
  }

  toColumns(): Record<string, Cell[]> {
    const out: Record<string, Cell[]> = {};
    for (const e of this.entries) out[e.name] = columnToArray(this._rt.ctx, e.col);
    return out;
  }

  toRecords(): Array<Record<string, Cell>> {
    const cols = this.toColumns();
    const names = this.columnNames();
    const rows = new Array<Record<string, Cell>>(this.length);
    for (let i = 0; i < this.length; i++) {
      const rec: Record<string, Cell> = {};
      for (const name of names) rec[name] = cols[name]![i]!;
      rows[i] = rec;
    }
    return rows;
  }

  describe(): DataFrame {
    const stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
    const numeric = this.entries.filter((e) => NUMERIC.has(e.col.dtype));
    const cols: Record<string, ColumnInput> = { statistic: stats };
    const dtypes: Record<string, DType> = { statistic: 'utf8' };
    for (const e of numeric) {
      cols[e.name] = describeColumn(columnToArray(this._rt.ctx, e.col));
      dtypes[e.name] = 'f64';
    }
    return DataFrame.fromColumns(cols, { runtime: this._rt, dtypes });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const e of this.entries) e.owner.release();
  }

  toString(): string {
    return this.render();
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.render();
  }

  private entryOf(name: string): Entry {
    const e = this.byName.get(name);
    if (!e) throw unknownColumn(name, this.columnNames());
    return e;
  }

  private retainEntry(e: Entry): Entry {
    e.owner.retain();
    return { name: e.name, col: e.col, owner: e.owner };
  }

  private shareEntries(names: readonly string[]): Entry[] {
    return names.map((name) => this.retainEntry(this.entryOf(name)));
  }

  private materializeColumn(value: Expr | ColumnInput, opts: WithColumnOptions): Column {
    if (value instanceof Expr) {
      const res = compile(value, this).execute();
      if (res.kind === 'column' && res.column) return res.column;
      const s = res.scalar!;
      const filled = new Array<Cell>(this.length).fill(s.value);
      return createColumn(this._rt.ctx, s.dtype, filled as ColumnInput);
    }
    const dtype = opts.dtype ?? inferDType(value);
    if (value.length !== this.length) {
      throw new FrameError(
        `withColumn value has length ${value.length}, expected ${this.length}.`,
      );
    }
    return createColumn(this._rt.ctx, dtype, value);
  }

  private gatherRows(indices: number[]): DataFrame {
    const { ctx } = this._rt;
    const n = indices.length;
    const idxPtr = ctx.mod.alloc(Math.max(n * 4, 1));
    try {
      const view = ctx.viewOf({ ptr: idxPtr, length: n, dtype: 'i32' }) as Int32Array;
      for (let k = 0; k < n; k++) view[k] = indices[k]!;
      const named: NamedColumn[] = this.entries.map((e) => ({
        name: e.name,
        col: gatherColumn(this._rt, e.col, idxPtr, n),
      }));
      return DataFrame.fromRoots(this._rt, named, n);
    } finally {
      ctx.viewOf.forget({ ptr: idxPtr, length: n, dtype: 'i32' });
      ctx.mod.free(idxPtr);
    }
  }

  private render(): string {
    const names = this.columnNames();
    if (names.length === 0) return `Empty DataFrame [${this.length} rows × 0 columns]`;
    const dtypes = this.entries.map((e) => e.col.dtype);
    const aligns: Align[] = dtypes.map(alignFor);

    const HEAD = 5;
    const TAIL = 5;
    const bodyRows: string[][] = [];
    const truncated = this.length > HEAD + TAIL;
    const emit = (start: number, end: number): void => {
      const slices = this.entries.map((e) =>
        columnToArray(this._rt.ctx, sliceColumn(e.col, start, end)),
      );
      for (let r = 0; r < end - start; r++) {
        bodyRows.push(slices.map((cells, c) => formatCell(cells[r]!, dtypes[c]!)));
      }
    };
    if (truncated) {
      emit(0, HEAD);
      bodyRows.push(names.map(() => ELLIPSIS));
      emit(this.length - TAIL, this.length);
    } else {
      emit(0, this.length);
    }

    const footer = `[${this.length} rows × ${names.length} columns]`;
    return formatTable(names, dtypes, bodyRows, aligns, footer);
  }
}

export function scope<T>(fn: (track: <F extends DataFrame>(df: F) => F) => T): T {
  const frames: DataFrame[] = [];
  const track = <F extends DataFrame>(df: F): F => {
    frames.push(df);
    return df;
  };
  try {
    return fn(track);
  } finally {
    for (const df of frames) df.dispose();
  }
}

function describeColumn(cells: Cell[]): (number | null)[] {
  const finite: number[] = [];
  for (const v of cells) {
    if (v === null) continue;
    // i64 columns: convert bigint to number (may lose precision for |x| > 2^53)
    const n = typeof v === 'bigint' ? Number(v) : (v as number);
    if (Number.isFinite(n)) finite.push(n);
  }
  const count = finite.length;
  if (count === 0) return [0, null, null, null, null, null, null, null];
  const sorted = finite.slice().sort((a, b) => a - b);
  const sum = finite.reduce((s, x) => s + x, 0);
  const mean = sum / count;
  const std =
    count >= 2
      ? Math.sqrt(finite.reduce((s, x) => s + (x - mean) * (x - mean), 0) / (count - 1))
      : null;
  const q = (p: number): number => {
    if (sorted.length === 1) return sorted[0]!;
    const idx = p * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const frac = idx - lo;
    return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * frac;
  };
  return [count, mean, std, sorted[0]!, q(0.25), q(0.5), q(0.75), sorted[sorted.length - 1]!];
}
