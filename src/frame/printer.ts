/** Aligned table pretty-printer (spec §4): name header, dtype sub-header, head/tail rows with
 * an `…` gap, and a [rows × cols] footer. Numeric right-aligned; nulls render as `null`. */

import type { Cell } from '../memory/column.js';
import type { DType } from '../memory/dtype.js';

export type Align = 'left' | 'right';

export const ELLIPSIS = '…';

export function formatCell(cell: Cell, dtype: DType): string {
  if (cell === null) return 'null';
  if (typeof cell === 'boolean') return cell ? 'true' : 'false';
  if (typeof cell === 'string') return cell.length > 24 ? `${cell.slice(0, 23)}${ELLIPSIS}` : cell;
  if (dtype === 'date32' && typeof cell === 'number') return new Date(cell * 86_400_000).toISOString().slice(0, 10);
  if (typeof cell === 'bigint') return dtype === 'timestamp' ? new Date(Number(cell)).toISOString() : String(cell); // no 'n' suffix per spec

  if (Number.isNaN(cell)) return 'NaN';
  if (cell === Infinity) return 'inf';
  if (cell === -Infinity) return '-inf';
  if (Number.isInteger(cell)) return String(cell);
  if (dtype === 'f32' || dtype === 'f64') {
    const p = (cell as number).toPrecision(6);
    return p.includes('.') && !p.includes('e') ? p.replace(/0+$/, '').replace(/\.$/, '') : p;
  }
  return String(cell);
}


export function alignFor(dtype: DType): Align {
  return dtype === 'utf8' || dtype === 'bool' ? 'left' : 'right';
}

function pad(s: string, width: number, align: Align): string {
  const gap = width - s.length;
  if (gap <= 0) return s;
  return align === 'right' ? ' '.repeat(gap) + s : s + ' '.repeat(gap);
}

export function formatTable(
  names: readonly string[],
  dtypes: readonly string[],
  bodyRows: readonly string[][],
  aligns: readonly Align[],
  footer: string,
): string {
  const cols = names.length;
  const widths = new Array<number>(cols);
  for (let c = 0; c < cols; c++) {
    let w = Math.max(names[c]!.length, dtypes[c]!.length);
    for (const row of bodyRows) w = Math.max(w, row[c]!.length);
    widths[c] = w;
  }

  const line = (cells: readonly string[]): string =>
    cells.map((s, c) => pad(s, widths[c]!, aligns[c]!)).join('  ');

  const sep = widths.map((w) => '─'.repeat(w)).join('  ');
  const out: string[] = [];
  out.push(line(names));
  out.push(line(dtypes));
  out.push(sep);
  for (const row of bodyRows) out.push(line(row));
  if (footer) out.push(footer);
  return out.join('\n');
}
