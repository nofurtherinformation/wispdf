/** Series — one named column, zero-copy over a frame's buffers (spec §4 `df.col('a')`).
 * Borrows (does not own) its parent's column; valid while that frame lives. Read-only. */

import type { MemoryContext } from '../memory/context.js';
import { DTYPES, type DType } from '../memory/dtype.js';
import { columnToArray, type Cell, type Column } from '../memory/column.js';
import type { ColumnView } from '../memory/views.js';

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

  [Symbol.iterator](): IterableIterator<Cell> {
    return this.toArray()[Symbol.iterator]();
  }

  toString(): string {
    return `Series '${this.name}' (${this.dtype}, ${this.length} rows)`;
  }
}
