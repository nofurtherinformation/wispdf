/**
 * Buffer ownership (spec §4 "shared/COW internally"). Ops return NEW frames that share
 * physical buffers wherever no data changes; sharing is made safe by reference-counting the
 * owning allocation ({@link OwnedColumn}). Each frame holds one ref; `dispose()` releases it,
 * and buffers are freed at zero refs — so slices outliving parents, or parents disposed before
 * a `select` child, are still correct. This is the v1 ownership story.
 */

import type { MemoryContext } from '../memory/context.js';
import { freeColumn, type Column } from '../memory/column.js';

export class OwnedColumn {

  readonly col: Column;
  private readonly ctx: MemoryContext;
  private refs: number;

  constructor(ctx: MemoryContext, col: Column) {
    this.ctx = ctx;
    this.col = col;
    this.refs = 1;
  }

  retain(): this {
    this.refs++;
    return this;
  }

  release(): void {
    if (this.refs <= 0) return;
    if (--this.refs === 0) freeColumn(this.ctx, this.col);
  }
}
