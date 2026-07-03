/**
 * Row proxy for the lambda escape hatch (ADR-003): one reusable proxy object moved across
 * rows (no per-row allocation), reading live viewOf buffers with memoized utf8 decode. This
 * is the documented SLOW PATH — the expression API (filter/withColumn) is the fast path.
 */

import type { MemoryContext } from '../memory/context.js';
import { DTYPES } from '../memory/dtype.js';
import { getBit, validityBytes } from '../memory/bitmap.js';
import { decodeSlot } from '../memory/dictionary.js';
import type { Cell, Column } from '../memory/column.js';

export type Row = Record<string, Cell>;

export interface RowCursor {

  at(i: number): Row;
}

export function makeRowCursor(
  ctx: MemoryContext,
  columns: ReadonlyArray<{ name: string; col: Column }>,
): RowCursor {
  let cur = 0;
  const proxy: Row = {};

  for (const { name, col } of columns) {
    const len = col.length;
    const bitOff = col.validityBitOffset;
    const validity =
      col.validityPtr === 0
        ? null
        : (ctx.viewOf({
            ptr: col.validityPtr,
            length: validityBytes(bitOff + len),
            dtype: 'u8',
          }) as Uint8Array);

    if (col.dtype === 'utf8') {
      const idx = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i32' }) as Int32Array;
      const dict = col.dict!;
      Object.defineProperty(proxy, name, {
        enumerable: true,
        get(): Cell {
          if (validity && !getBit(validity, bitOff + cur)) return null;
          return decodeSlot(ctx, dict, idx[cur]!);
        },
      });
    } else if (col.dtype === 'bool') {
      const data = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'bool' }) as Uint8Array;
      Object.defineProperty(proxy, name, {
        enumerable: true,
        get(): Cell {
          if (validity && !getBit(validity, bitOff + cur)) return null;
          return data[cur] !== 0;
        },
      });
    } else if (col.dtype === 'i64') {
      const data = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i64' }) as BigInt64Array;
      Object.defineProperty(proxy, name, {
        enumerable: true,
        get(): Cell {
          if (validity && !getBit(validity, bitOff + cur)) return null;
          return data[cur]!; // bigint
        },
      });
    } else {
      const data = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: DTYPES[col.dtype].view });
      Object.defineProperty(proxy, name, {
        enumerable: true,
        get(): Cell {
          if (validity && !getBit(validity, bitOff + cur)) return null;
          return data[cur] as number;
        },
      });
    }
  }

  return {
    at(i: number): Row {
      cur = i;
      return proxy;
    },
  };
}
