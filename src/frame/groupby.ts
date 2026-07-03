/**
 * GroupBy + agg (spec §4; ADR-005 hash grouping). Group: hash key columns (`hash_dt` +
 * `hash_combine`) and assign first-occurrence ids via `group_build`; string keys hash their
 * dictionary indices (no unification needed within one frame); null keys form one group
 * (dtypes.md §4.5). Aggregate: each operand is materialised once over the frame then reduced
 * per group by one JS scatter pass (no per-group boundary chatter). skipna + result-dtype per §4.3.
 */

import { DTYPES, type DType } from '../memory/dtype.js';
import { validityBytes, getBit } from '../memory/bitmap.js';
import { decodeSlot, type Dictionary } from '../memory/dictionary.js';
import { createColumn, freeColumn, type Cell, type Column, type ColumnInput } from '../memory/column.js';
import { Expr, aggResult, compile, inferType, type AggOp, type FrameView } from '../expr/index.js';
import { groupBuild, type HashExports } from '../kernels/hash/index.js';
import type { DfRuntime } from './runtime.js';
import { rawKernel } from './runtime.js';
import { alignValidity, freeAligned } from './util.js';
import { FrameError, unknownColumn } from './errors.js';
import type { DataFrame } from './dataframe.js';

export type AggName = AggOp | 'size';

export type AggRequest = AggName | AggName[] | Expr;

export type AggSpec = Record<string, AggRequest>;

export interface NamedColumn {
  readonly name: string;
  readonly col: Column;
}

export interface GroupBySource extends FrameView {
  readonly rt: DfRuntime;
  buildResult(named: NamedColumn[]): DataFrame;
}

const AGG_NAMES = new Set<AggName>([
  'sum', 'mean', 'min', 'max', 'count', 'size', 'nunique', 'std', 'var', 'first', 'last',
]);

export class GroupBy {
  private readonly src: GroupBySource;
  private readonly keys: string[];

  constructor(src: GroupBySource, keys: string[]) {
    this.src = src;
    this.keys = keys;
    for (const k of keys) {
      if (src.dtypeOf(k) === undefined) throw unknownColumn(k, src.columnNames());
    }
  }

  agg(spec: AggSpec): DataFrame {
    const { rt } = this.src;
    const { ctx } = rt;
    const len = this.src.length;
    const keyCols = this.keys.map((k) => this.src.getColumn(k)!);

    const plan = this.buildPlan(spec);
    const groups = computeGroups(rt, keyCols);
    const owned: Column[] = [];
    try {
      const firstRow = groups.firstRow;
      const out: NamedColumn[] = [];

      for (let k = 0; k < this.keys.length; k++) {
        const col = gatherFirst(rt, keyCols[k]!, firstRow);
        owned.push(col);
        out.push({ name: this.keys[k]!, col });
      }

      const gids = ctx.viewOf({ ptr: groups.groupIdsPtr, length: len, dtype: 'i32' }) as Int32Array;
      for (const a of plan) {
        const col = scatterReduce(rt, a, gids, groups.groupCount, len);
        owned.push(col);
        out.push({ name: a.outName, col });
      }

      const result = this.src.buildResult(out);

      owned.length = 0;
      return result;
    } finally {
      for (const c of owned) freeColumn(ctx, c);
      for (const a of plan) if (a.ownsOperand && a.operand) freeColumn(ctx, a.operand);
      ctx.viewOf.forget({ ptr: groups.groupIdsPtr, length: len, dtype: 'i32' });
      ctx.mod.free(groups.groupIdsPtr);
    }
  }

  private buildPlan(spec: AggSpec): AggPlan[] {
    const plan: AggPlan[] = [];
    for (const [key, req] of Object.entries(spec)) {
      if (req instanceof Expr) {
        plan.push(this.exprPlan(key, req));
      } else if (Array.isArray(req)) {
        for (const op of req) plan.push(this.columnPlan(`${key}_${op}`, key, op));
      } else {
        plan.push(this.columnPlan(key, key, req));
      }
    }
    return plan;
  }

  private columnPlan(outName: string, colName: string, op: AggName): AggPlan {
    if (!AGG_NAMES.has(op)) throw new FrameError(`unknown aggregation '${op}' for '${colName}'.`);
    if (op === 'size') {
      return { outName, op, operand: null, operandDtype: 'i32', ownsOperand: false, resultDtype: 'i32' };
    }
    const col = this.src.getColumn(colName);
    if (!col) throw unknownColumn(colName, this.src.columnNames());
    const resultDtype = aggDtype(op, col.dtype);
    return { outName, op, operand: col, operandDtype: col.dtype, ownsOperand: false, resultDtype };
  }

  private exprPlan(outName: string, expr: Expr): AggPlan {
    const node = expr.node;
    if (node.kind !== 'agg') {
      throw new FrameError(
        `agg value for '${outName}' must be a top-level aggregation (e.g. col('a').sum()).`,
      );
    }
    const op = node.op;
    const operandDtype = inferType(node.operand, this.src);
    const resultDtype = aggResult(op, operandDtype);
    const res = compile(node.operand, this.src).execute();
    if (res.kind !== 'column' || !res.column) {
      throw new FrameError(`aggregation operand for '${outName}' did not produce a column.`);
    }
    return { outName, op, operand: res.column, operandDtype, ownsOperand: true, resultDtype };
  }
}

function aggDtype(op: AggName, d: DType): DType {
  return op === 'size' ? 'i32' : aggResult(op, d);
}

interface AggPlan {
  readonly outName: string;
  readonly op: AggName;

  readonly operand: Column | null;
  readonly operandDtype: DType;
  readonly ownsOperand: boolean;
  readonly resultDtype: DType;
}

interface Groups {
  readonly groupIdsPtr: number;
  readonly groupCount: number;
  readonly firstRow: Int32Array;
}

function computeGroups(rt: DfRuntime, keyCols: Column[]): Groups {
  const { ctx, wasm } = rt;
  const len = keyCols.length > 0 ? keyCols[0]!.length : 0;
  const hashPtr = ctx.mod.alloc(Math.max(len * 8, 1));
  try {
    for (let k = 0; k < keyCols.length; k++) {
      const col = keyCols[k]!;
      const key = hashInput(rt, col, len);
      const av = alignValidity(ctx, col);
      try {
        if (k === 0) {
          rawKernel(wasm, `hash_${key.tok}`)(key.ptr, av.ptr, hashPtr, len);
        } else {
          const tmp = ctx.mod.alloc(Math.max(len * 8, 1));
          try {
            rawKernel(wasm, `hash_${key.tok}`)(key.ptr, av.ptr, tmp, len);
            rawKernel(wasm, 'hash_combine')(hashPtr, tmp, len);
          } finally {
            ctx.viewOf.forget({ ptr: tmp, length: len, dtype: 'u8' });
            ctx.mod.free(tmp);
          }
        }
      } finally {
        freeAligned(ctx, av);
        key.free();
      }
    }

    const groupIdsPtr = ctx.mod.alloc(Math.max(len * 4, 1));
    const groupCount = groupBuild(wasm as unknown as HashExports, hashPtr, len, groupIdsPtr);

    const gids = ctx.viewOf({ ptr: groupIdsPtr, length: len, dtype: 'i32' }) as Int32Array;
    const firstRow = new Int32Array(groupCount);
    let seen = 0;
    for (let i = 0; i < len && seen < groupCount; i++) {
      if (gids[i] === seen) firstRow[seen++] = i;
    }
    return { groupIdsPtr, groupCount, firstRow };
  } finally {
    ctx.viewOf.forget({ ptr: hashPtr, length: len, dtype: 'u8' });
    ctx.mod.free(hashPtr);
  }
}

function hashInput(rt: DfRuntime, col: Column, len: number): { ptr: number; tok: string; free(): void } {
  const { ctx } = rt;
  if (col.dtype === 'utf8') return { ptr: col.dataPtr, tok: 'i32', free() {} };
  if (col.dtype === 'bool') {
    const ptr = ctx.mod.alloc(Math.max(len * 4, 1));
    const src = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'bool' }) as Uint8Array;
    const dst = ctx.viewOf({ ptr, length: len, dtype: 'i32' }) as Int32Array;
    for (let i = 0; i < len; i++) dst[i] = src[i]!;
    return {
      ptr,
      tok: 'i32',
      free() {
        ctx.viewOf.forget({ ptr, length: len, dtype: 'i32' });
        ctx.mod.free(ptr);
      },
    };
  }
  return { ptr: col.dataPtr, tok: DTYPES[col.dtype].wasm, free() {} };
}

function gatherFirst(rt: DfRuntime, col: Column, firstRow: Int32Array): Column {
  const { ctx } = rt;
  const n = firstRow.length;
  const validity =
    col.validityPtr === 0
      ? null
      : (ctx.viewOf({
          ptr: col.validityPtr,
          length: validityBytes(col.validityBitOffset + col.length),
          dtype: 'u8',
        }) as Uint8Array);
  const bitOff = col.validityBitOffset;

  if (col.dtype === 'utf8') {
    const idx = ctx.viewOf({ ptr: col.dataPtr, length: col.length, dtype: 'i32' }) as Int32Array;
    const dict = col.dict!;
    const out = new Array<string | null>(n);
    for (let g = 0; g < n; g++) {
      const r = firstRow[g]!;
      out[g] = validity && !getBit(validity, bitOff + r) ? null : decodeSlot(ctx, dict, idx[r]!);
    }
    return createColumn(ctx, 'utf8', out);
  }

  if (col.dtype === 'i64') {
    const data = ctx.viewOf({ ptr: col.dataPtr, length: col.length, dtype: 'i64' }) as BigInt64Array;
    const out = new Array<bigint | null>(n);
    for (let g = 0; g < n; g++) {
      const r = firstRow[g]!;
      out[g] = validity && !getBit(validity, bitOff + r) ? null : data[r]!;
    }
    return createColumn(ctx, 'i64', out as unknown as ColumnInput);
  }

  const isBool = col.dtype === 'bool';
  const data = ctx.viewOf({ ptr: col.dataPtr, length: col.length, dtype: DTYPES[col.dtype].view });
  const out = new Array<number | boolean | null>(n);
  for (let g = 0; g < n; g++) {
    const r = firstRow[g]!;
    if (validity && !getBit(validity, bitOff + r)) out[g] = null;
    else out[g] = isBool ? data[r] !== 0 : (data[r] as number);
  }
  return createColumn(ctx, col.dtype, out as unknown as ColumnInput);
}

function scatterReduce(
  rt: DfRuntime,
  plan: AggPlan,
  gids: Int32Array,
  groupCount: number,
  len: number,
): Column {
  const { ctx } = rt;
  const { op } = plan;

  if (op === 'size') {
    const out = new Array<number>(groupCount).fill(0);
    for (let i = 0; i < len; i++) { const g = gids[i]!; out[g] = out[g]! + 1; }
    return createColumn(ctx, 'i32', out);
  }

  const col = plan.operand!;
  const validity =
    col.validityPtr === 0
      ? null
      : (ctx.viewOf({
          ptr: col.validityPtr,
          length: validityBytes(col.validityBitOffset + len),
          dtype: 'u8',
        }) as Uint8Array);
  const bitOff = col.validityBitOffset;
  const valid = (i: number): boolean => validity === null || getBit(validity, bitOff + i);

  if (op === 'count') {
    const out = new Array<number>(groupCount).fill(0);
    for (let i = 0; i < len; i++) if (valid(i)) { const g = gids[i]!; out[g] = out[g]! + 1; }
    return createColumn(ctx, 'i32', out);
  }

  if (op === 'first' || op === 'last') {
    return reduceFirstLast(rt, plan, gids, groupCount, len, valid);
  }
  if (op === 'nunique') {
    return reduceNunique(rt, plan, gids, groupCount, len, valid);
  }
  if (op === 'std' || op === 'var') {
    return reduceStdVar(rt, plan, gids, groupCount, len, valid);
  }

  // i64 operations require bigint arithmetic
  if (col.dtype === 'i64') {
    const data = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i64' }) as BigInt64Array;
    if (op === 'sum') {
      const acc = new Array<bigint>(groupCount).fill(0n);
      for (let i = 0; i < len; i++) if (valid(i)) { const g = gids[i]!; acc[g] = acc[g]! + data[i]!; }
      return createColumn(ctx, 'i64', acc as unknown as ColumnInput);
    }
    if (op === 'mean') {
      const acc = new Float64Array(groupCount);
      const cnt = new Int32Array(groupCount);
      for (let i = 0; i < len; i++) {
        if (!valid(i)) continue;
        const g = gids[i]!;
        acc[g] = acc[g]! + Number(data[i]!);
        cnt[g] = cnt[g]! + 1;
      }
      const out = new Array<number | null>(groupCount);
      for (let g = 0; g < groupCount; g++) out[g] = cnt[g]! > 0 ? acc[g]! / cnt[g]! : null;
      return createColumn(ctx, 'f64', out as unknown as ColumnInput);
    }
    // min/max for i64
    const extI64 = new Array<bigint | null>(groupCount).fill(null);
    const isMin = op === 'min';
    for (let i = 0; i < len; i++) {
      if (!valid(i)) continue;
      const g = gids[i]!;
      const x = data[i]!;
      if (extI64[g] === null) extI64[g] = x;
      else extI64[g] = isMin ? (x < extI64[g]! ? x : extI64[g]!) : (x > extI64[g]! ? x : extI64[g]!);
    }
    return createColumn(ctx, 'i64', extI64 as unknown as ColumnInput);
  }

  const data = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: DTYPES[col.dtype].view }) as
    | Float64Array | Float32Array | Int32Array | Uint32Array;
  const out = new Array<number | null>(groupCount).fill(null);

  if (op === 'sum') {
    const acc = new Float64Array(groupCount);
    for (let i = 0; i < len; i++) if (valid(i)) { const g = gids[i]!; acc[g] = acc[g]! + data[i]!; }
    for (let g = 0; g < groupCount; g++) out[g] = acc[g]!;
    return createColumn(ctx, plan.resultDtype, out as unknown as ColumnInput);
  }
  if (op === 'mean') {
    const acc = new Float64Array(groupCount);
    const cnt = new Int32Array(groupCount);
    for (let i = 0; i < len; i++) {
      if (!valid(i)) continue;
      const g = gids[i]!;
      acc[g] = acc[g]! + data[i]!;
      cnt[g] = cnt[g]! + 1;
    }
    for (let g = 0; g < groupCount; g++) out[g] = cnt[g]! > 0 ? acc[g]! / cnt[g]! : null;
    return createColumn(ctx, 'f64', out as unknown as ColumnInput);
  }

  const cntNonNull = new Int32Array(groupCount);
  const cntFinite = new Int32Array(groupCount);
  const ext = new Float64Array(groupCount);
  const isMin = op === 'min';
  for (let i = 0; i < len; i++) {
    if (!valid(i)) continue;
    const g = gids[i]!;
    cntNonNull[g] = cntNonNull[g]! + 1;
    const x = data[i]!;
    if (Number.isNaN(x)) continue;
    if (cntFinite[g] === 0) ext[g] = x;
    else ext[g] = isMin ? Math.min(ext[g]!, x) : Math.max(ext[g]!, x);
    cntFinite[g] = cntFinite[g]! + 1;
  }
  for (let g = 0; g < groupCount; g++) {
    out[g] = cntNonNull[g] === 0 ? null : cntFinite[g] === 0 ? NaN : ext[g]!;
  }
  return createColumn(ctx, plan.resultDtype, out as unknown as ColumnInput);
}

function reduceStdVar(
  rt: DfRuntime,
  plan: AggPlan,
  gids: Int32Array,
  groupCount: number,
  len: number,
  valid: (i: number) => boolean,
): Column {
  const { ctx } = rt;
  const col = plan.operand!;
  const isI64 = col.dtype === 'i64';
  const data = isI64
    ? ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i64' }) as BigInt64Array
    : ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: DTYPES[col.dtype].view }) as
        Float64Array | Float32Array | Int32Array | Uint32Array;
  const sum = new Float64Array(groupCount);
  const cnt = new Int32Array(groupCount);
  for (let i = 0; i < len; i++) {
    if (!valid(i)) continue;
    const g = gids[i]!;
    sum[g] = sum[g]! + (isI64 ? Number((data as BigInt64Array)[i]!) : (data as Float64Array)[i]!);
    cnt[g] = cnt[g]! + 1;
  }
  const mean = new Float64Array(groupCount);
  for (let g = 0; g < groupCount; g++) mean[g] = cnt[g]! > 0 ? sum[g]! / cnt[g]! : 0;
  const sse = new Float64Array(groupCount);
  for (let i = 0; i < len; i++) {
    if (!valid(i)) continue;
    const g = gids[i]!;
    const x = isI64 ? Number((data as BigInt64Array)[i]!) : (data as Float64Array)[i]!;
    const d = x - mean[g]!;
    sse[g] = sse[g]! + d * d;
  }
  const out = new Array<number | null>(groupCount);
  const wantStd = plan.op === 'std';
  for (let g = 0; g < groupCount; g++) {
    if (cnt[g]! < 2) out[g] = null;
    else {
      const v = sse[g]! / (cnt[g]! - 1);
      out[g] = wantStd ? Math.sqrt(v) : v;
    }
  }
  return createColumn(ctx, 'f64', out as unknown as ColumnInput);
}

function reduceNunique(
  rt: DfRuntime,
  plan: AggPlan,
  gids: Int32Array,
  groupCount: number,
  len: number,
  valid: (i: number) => boolean,
): Column {
  const { ctx } = rt;
  const col = plan.operand!;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const sets: Array<Set<any> | undefined> = new Array(groupCount);
  const data =
    col.dtype === 'utf8' || col.dtype === 'bool'
      ? (ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: col.dtype === 'utf8' ? 'i32' : 'bool' }))
      : col.dtype === 'i64'
        ? (ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i64' }) as BigInt64Array)
        : (ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: DTYPES[col.dtype].view }));
  for (let i = 0; i < len; i++) {
    if (!valid(i)) continue;
    const g = gids[i]!;
    (sets[g] ??= new Set()).add(data[i]!);
  }
  const out = new Array<number>(groupCount);
  for (let g = 0; g < groupCount; g++) out[g] = sets[g] ? sets[g]!.size : 0;
  return createColumn(ctx, 'i32', out);
}

function reduceFirstLast(
  rt: DfRuntime,
  plan: AggPlan,
  gids: Int32Array,
  groupCount: number,
  len: number,
  valid: (i: number) => boolean,
): Column {
  const { ctx } = rt;
  const col = plan.operand!;
  const wantFirst = plan.op === 'first';
  const chosen = new Int32Array(groupCount).fill(-1);
  for (let i = 0; i < len; i++) {
    if (!valid(i)) continue;
    const g = gids[i]!;
    if (wantFirst) {
      if (chosen[g] === -1) chosen[g] = i;
    } else {
      chosen[g] = i;
    }
  }
  const out = new Array<Cell>(groupCount);
  if (col.dtype === 'utf8') {
    const idx = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i32' }) as Int32Array;
    const dict = col.dict as Dictionary;
    for (let g = 0; g < groupCount; g++) {
      out[g] = chosen[g] === -1 ? null : decodeSlot(ctx, dict, idx[chosen[g]!]!);
    }
    return createColumn(ctx, 'utf8', out as unknown as ColumnInput);
  }
  if (col.dtype === 'i64') {
    const data = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: 'i64' }) as BigInt64Array;
    for (let g = 0; g < groupCount; g++) {
      const r = chosen[g]!;
      out[g] = r === -1 ? null : data[r]!;
    }
    return createColumn(ctx, 'i64', out as unknown as ColumnInput);
  }

  const isBool = col.dtype === 'bool';
  const data = ctx.viewOf({ ptr: col.dataPtr, length: len, dtype: DTYPES[col.dtype].view });
  for (let g = 0; g < groupCount; g++) {
    const r = chosen[g]!;
    out[g] = r === -1 ? null : isBool ? data[r] !== 0 : (data[r] as number);
  }
  return createColumn(ctx, col.dtype, out as unknown as ColumnInput);
}
