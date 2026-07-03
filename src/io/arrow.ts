/**
 * Arrow IPC stream encode/decode for databonk DataFrames (ADR-002, ADR-009, ADR-010).
 *
 * Our layout is Arrow-compatible by design:
 *   - Numeric columns: contiguous TypedArray → Arrow primitive buffer (zero-transform).
 *   - Validity bitmaps: LSB-first, 1=valid — already Arrow format (zero-transform).
 *   - utf8 columns: i32 indices + (i32 offsets + u8 bytes) dict → Arrow Dict<Int32, Utf8>.
 *   - bool: u8[n] per-element → bit-pack to Arrow's 1-bit-per-element format (one transform).
 *   - i64 → Arrow Int(64, signed); date32 → Arrow Date(DAY); timestamp → Arrow Timestamp(MILLI, tz?).
 *
 * IPC stream per message: [int32 -1] [int32 meta_size] [meta bytes, pad→8B] [body bytes, pad→8B]
 * EOS: [int32 -1] [int32 0]
 *
 * FlatBuffers field-index constants — sourced from the Arrow FlatBuffers IDL:
 *   Message:          version(0), header_type(1), header(2), bodyLength(3)
 *   Schema:           endianness(0), fields(1)
 *   Field:            name(0), nullable(1), type_type(2), type(3), dictionary(4), children(5)
 *   RecordBatch:      length(0), nodes(1), buffers(2)
 *   DictionaryBatch:  id(0), data(1), isDelta(2)
 *   DictionaryEncoding: id(0), indexType(1), isOrdered(2)
 *   Int:              bitWidth(0), isSigned(1)
 *   FloatingPoint:    precision(0)
 *   Date:             unit(0) — DateUnit: DAY=0, MILLISECOND=1 (FlatBuffers default=1)
 *   Timestamp:        unit(0) — TimeUnit: SECOND=0, MILLISECOND=1, MICROSECOND=2, NANOSECOND=3;
 *                     timezone(1) — optional IANA tz string
 *
 * MessageHeader union tags: Schema=1, DictionaryBatch=2, RecordBatch=3
 * Type union tags:          Int=2, FloatingPoint=3, Utf8=5, Bool=6, Date=8, Timestamp=10, LargeUtf8=20
 * MetadataVersion V5=4
 * FloatingPoint.Precision: HALF=0, SINGLE=1, DOUBLE=2
 */

import type { MemoryContext } from '../memory/context.js';
import { DTYPES, type DType } from '../memory/dtype.js';
import { validityBytes, getBit, setBit } from '../memory/bitmap.js';
import { columnToArray, createColumn, type Column, type ColumnInput } from '../memory/column.js';
import { writeDictionary, writeDictionaryFromRawBytes } from '../memory/dictionary.js';
import { DataFrame, type FrameOptions } from '../frame/dataframe.js';
import type { DfRuntime } from '../frame/runtime.js';
import { FBBuilder, FBTable, fbRoot } from './fb.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const META_V5 = 4; // MetadataVersion.V5
const MH_SCHEMA = 1;
const MH_DICT = 2;
const MH_RECORD = 3;
const TYPE_INT = 2;
const TYPE_FLOAT = 3;
const TYPE_UTF8 = 5;
const TYPE_BOOL = 6;
const TYPE_DATE = 8;       // Arrow Date type (Date32/Date64)
const TYPE_TIMESTAMP = 10; // Arrow Timestamp type
const TYPE_LARGE_UTF8 = 20;
const PREC_SINGLE = 1;
const PREC_DOUBLE = 2;
const STRUCT16 = 16; // FieldNode / Buffer are each 16 bytes

// DateUnit (Arrow FlatBuffers IDL): DAY=0, MILLISECOND=1; FlatBuffers default = MILLISECOND (1).
const DATE_UNIT_DAY = 0;
const DATE_UNIT_MILLI_DEFAULT = 1; // FlatBuffers default for Date.unit — needed to force-write DAY=0

// TimeUnit (Arrow FlatBuffers IDL): SECOND=0(default), MILLISECOND=1, MICROSECOND=2, NANOSECOND=3.
const TIME_UNIT_SECOND = 0; // FlatBuffers default for Timestamp.unit
const TIME_UNIT_MILLI = 1;
const TIME_UNIT_MICRO = 2;
const TIME_UNIT_NANO = 3;

// Saturation bounds for SECOND→MILLI rescaling (saturation-to-null on overflow, ADR-010).
// INT64_MAX = 9_223_372_036_854_775_807n; INT64_MAX / 1000n = 9_223_372_036_854_775n (truncate).
// INT64_MIN = -9_223_372_036_854_775_808n; INT64_MIN / 1000n = -9_223_372_036_854_775n (truncate toward 0).
const SEC_MS_MAX = 9_223_372_036_854_775n; // max safe seconds before ×1000 overflows i64
const SEC_MS_MIN = -9_223_372_036_854_775n; // min safe seconds before ×1000 underflows i64

// ---------------------------------------------------------------------------
// Encode helpers
// ---------------------------------------------------------------------------

const pad8 = (n: number): number => (n + 7) & ~7;

/** Repack bool column's u8[n] to Arrow bit-packed (same LSB-first as validity). */
function packBools(src: Uint8Array, len: number): Uint8Array {
  const out = new Uint8Array(Math.ceil(len / 8));
  for (let i = 0; i < len; i++) if ((src[i] ?? 0) !== 0) out[i >> 3]! |= 1 << (i & 7);
  return out;
}

/** Expand Arrow bit-packed bools to our u8[n] storage. */
function unpackBools(src: Uint8Array, len: number): Uint8Array {
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = (src[i >> 3]! >> (i & 7)) & 1;
  return out;
}

/**
 * Copy the validity bitmap for a column into a fresh Uint8Array starting at bit 0.
 * Handles sliced columns (validityBitOffset > 0). Returns null if all-valid.
 */
function extractValidity(ctx: MemoryContext, col: Column): Uint8Array | null {
  if (col.validityPtr === 0) return null;
  const totalBits = col.validityBitOffset + col.length;
  const raw = ctx.viewOf({ ptr: col.validityPtr, length: validityBytes(totalBits), dtype: 'u8' }) as Uint8Array;
  if (col.validityBitOffset === 0) return raw.slice(0, validityBytes(col.length));
  const out = new Uint8Array(validityBytes(col.length));
  for (let i = 0; i < col.length; i++) if (getBit(raw, col.validityBitOffset + i)) setBit(out, i);
  return out;
}

// ---------------------------------------------------------------------------
// FlatBuffers builders for Arrow messages
// ---------------------------------------------------------------------------

function emptyTable(fb: FBBuilder): number {
  fb.startTable(); return fb.endTable();
}

function buildInt(fb: FBBuilder, bits: number, signed: boolean): number {
  fb.startTable();
  fb.addFieldI32(0, bits, 0);
  fb.addFieldBool(1, signed, false);
  return fb.endTable();
}

function buildFloat(fb: FBBuilder, prec: number): number {
  fb.startTable();
  fb.addFieldI16(0, prec, 0);
  return fb.endTable();
}

function buildDictEncoding(fb: FBBuilder, id: number): number {
  const intT = buildInt(fb, 32, true);
  fb.startTable();
  fb.addFieldI64(0, id);
  fb.addFieldOffset(1, intT);
  return fb.endTable();
}

/** Build Arrow Date32(DAY) FlatBuffers type table. */
function buildDate(fb: FBBuilder): number {
  // Date.unit: FlatBuffers default = MILLISECOND (1). We want DAY (0), so we must write it.
  fb.startTable();
  fb.addFieldI16(0, DATE_UNIT_DAY, DATE_UNIT_MILLI_DEFAULT);
  return fb.endTable();
}

/** Build Arrow Timestamp(MILLISECOND, tz?) FlatBuffers type table. */
function buildTimestamp(fb: FBBuilder, tz?: string): number {
  // Timestamp.unit: FlatBuffers default = SECOND (0). We write MILLISECOND (1).
  // timezone (field 1): optional IANA string; omitted if tz is absent.
  const tzOff = tz ? fb.createString(tz) : null;
  fb.startTable();
  fb.addFieldI16(0, TIME_UNIT_MILLI, TIME_UNIT_SECOND);
  if (tzOff !== null) fb.addFieldOffset(1, tzOff);
  return fb.endTable();
}

function buildField(
  fb: FBBuilder,
  name: string,
  dtype: DType,
  dictId: number | null,
  colTz?: string,
): number {
  const nameOff = fb.createString(name);
  let typeTag: number, typeT: number;
  let dictOff: number | null = null;

  switch (dtype) {
    case 'f64':  typeTag = TYPE_FLOAT; typeT = buildFloat(fb, PREC_DOUBLE); break;
    case 'f32':  typeTag = TYPE_FLOAT; typeT = buildFloat(fb, PREC_SINGLE); break;
    case 'i32':  typeTag = TYPE_INT;   typeT = buildInt(fb, 32, true);      break;
    case 'u32':  typeTag = TYPE_INT;   typeT = buildInt(fb, 32, false);     break;
    case 'i64':  typeTag = TYPE_INT;   typeT = buildInt(fb, 64, true);      break;  // ADR-009
    case 'bool': typeTag = TYPE_BOOL;  typeT = emptyTable(fb);              break;
    case 'utf8':
      typeTag = TYPE_UTF8; typeT = emptyTable(fb);
      if (dictId !== null) dictOff = buildDictEncoding(fb, dictId);
      break;
    case 'date32':
      typeTag = TYPE_DATE; typeT = buildDate(fb); break;           // ADR-010: Date32(DAY)
    case 'timestamp':
      typeTag = TYPE_TIMESTAMP; typeT = buildTimestamp(fb, colTz); break; // ADR-010: Timestamp(MILLI, tz?)
    default:
      throw new Error(`Arrow toArrow: unsupported dtype '${String(dtype)}'`);
  }

  const childrenOff = fb.createOffsetVector([]); // no children for our dtypes

  fb.startTable();
  fb.addFieldOffset(0, nameOff);
  fb.addFieldBool(1, true, false);  // nullable = true
  fb.addFieldI8(2, typeTag);         // type_type
  fb.addFieldOffset(3, typeT);       // type
  if (dictOff !== null) fb.addFieldOffset(4, dictOff);
  fb.addFieldOffset(5, childrenOff);
  return fb.endTable();
}

/**
 * Build the FlatBuffers bytes for a Schema Message (bodyLength=0).
 * `tzs` carries the optional IANA timezone string per column (for timestamp columns).
 */
function buildSchemaMeta(
  names: string[],
  dtypes: DType[],
  dictIds: (number | null)[],
  tzs: (string | undefined)[],
): Uint8Array {
  const fb = new FBBuilder(1024);
  const fieldOffs = names.map((n, i) => buildField(fb, n, dtypes[i]!, dictIds[i]!, tzs[i]));
  const fieldsVec = fb.createOffsetVector(fieldOffs);
  fb.startTable();
  fb.addFieldI16(0, 0); // endianness = Little
  fb.addFieldOffset(1, fieldsVec);
  const schemaOff = fb.endTable();

  fb.startTable();
  fb.addFieldI16(0, META_V5);
  fb.addFieldI8(1, MH_SCHEMA);
  fb.addFieldOffset(2, schemaOff);
  // bodyLength=0: omitted (default)
  const msgOff = fb.endTable();
  return fb.finish(msgOff);
}

interface BodyPlan {
  nodes: Array<{ length: number; nullCount: number }>;
  buffers: Array<{ offset: number; length: number }>;
  body: Uint8Array;
}

/** Build FlatBuffers bytes for a RecordBatch Message. */
function buildRBMeta(plan: BodyPlan): Uint8Array {
  return buildBatchMeta(plan, MH_RECORD, -1);
}

/** Build FlatBuffers bytes for a DictionaryBatch Message. */
function buildDictMeta(plan: BodyPlan, id: number): Uint8Array {
  return buildBatchMeta(plan, MH_DICT, id);
}

function buildBatchMeta(plan: BodyPlan, headerType: number, dictId: number): Uint8Array {
  const fb = new FBBuilder(512);

  // FieldNode structs (16 bytes each): i64 length + i64 null_count
  const nodeData = new Uint8Array(plan.nodes.length * STRUCT16);
  const ndv = new DataView(nodeData.buffer);
  for (let i = 0; i < plan.nodes.length; i++) {
    ndv.setInt32(i * 16 + 0, plan.nodes[i]!.length, true);   // length (lo)
    ndv.setInt32(i * 16 + 8, plan.nodes[i]!.nullCount, true); // null_count (lo)
  }
  const nodesVec = fb.createStructVector(nodeData, plan.nodes.length);

  // Buffer structs (16 bytes each): i64 offset + i64 length
  const bufData = new Uint8Array(plan.buffers.length * STRUCT16);
  const bdv = new DataView(bufData.buffer);
  for (let i = 0; i < plan.buffers.length; i++) {
    bdv.setInt32(i * 16 + 0, plan.buffers[i]!.offset, true);
    bdv.setInt32(i * 16 + 8, plan.buffers[i]!.length, true);
  }
  const bufsVec = fb.createStructVector(bufData, plan.buffers.length);

  const rowLen = plan.nodes.length > 0 ? plan.nodes[0]!.length : 0;
  fb.startTable();
  fb.addFieldI64(0, rowLen);
  fb.addFieldOffset(1, nodesVec);
  fb.addFieldOffset(2, bufsVec);
  const rbOff = fb.endTable();

  let innerOff = rbOff;
  if (headerType === MH_DICT) {
    fb.startTable();
    fb.addFieldI64(0, dictId);
    fb.addFieldOffset(1, rbOff);
    // isDelta = false (default, omit)
    innerOff = fb.endTable();
  }

  fb.startTable();
  fb.addFieldI16(0, META_V5);
  fb.addFieldI8(1, headerType);
  fb.addFieldOffset(2, innerOff);
  fb.addFieldI64(3, plan.body.byteLength);
  const msgOff = fb.endTable();
  return fb.finish(msgOff);
}

// ---------------------------------------------------------------------------
// Body assembly
// ---------------------------------------------------------------------------

interface BodyBuilder {
  parts: Uint8Array[];
  size: number;
}

function pushBuffer(bb: BodyBuilder, data: Uint8Array | null): { offset: number; length: number } {
  if (!data || data.byteLength === 0) return { offset: 0, length: 0 };
  const offset = bb.size;
  bb.parts.push(data);
  const padded = pad8(data.byteLength);
  if (padded > data.byteLength) bb.parts.push(new Uint8Array(padded - data.byteLength));
  bb.size += padded;
  return { offset, length: data.byteLength };
}

function buildBody(parts: Uint8Array[]): Uint8Array {
  const tot = parts.reduce((s, p) => s + p.byteLength, 0);
  const out = new Uint8Array(tot);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.byteLength; }
  return out;
}

// ---------------------------------------------------------------------------
// IPC stream writer
// ---------------------------------------------------------------------------

function ipcMessage(meta: Uint8Array, body: Uint8Array): Uint8Array {
  // Arrow IPC stream format (post-v0.15 / non-legacy):
  //   [int32 -1]  [int32 padded_meta_size]  [meta bytes padded to 8B]  [body bytes]
  //
  // The size field contains the PADDED metadata length (including trailing zero
  // bytes up to 8-byte alignment). This is what apache-arrow writes and expects:
  //   prefixSize = 8 (4-byte continuation + 4-byte size field)
  //   alignedSize = (rawLen + prefixSize + 7) & ~7
  //   written size = alignedSize - prefixSize = rawLen + padding
  //
  // Writing the unpadded rawLen was our prior bug: apache-arrow read rawLen bytes
  // of meta (stopping before the padding zeros), left the stream misaligned, and
  // then read the next message's continuation marker from the padding bytes instead
  // of from the actual next message.
  const rawLen = meta.byteLength;
  const padded = pad8(rawLen);        // padded metadata size (what goes in the size field)
  const total = 4 + 4 + padded + body.byteLength;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setInt32(0, -1, true);           // continuation marker (0xFFFFFFFF)
  dv.setInt32(4, padded, true);       // PADDED metadata size (apache-arrow convention)
  out.set(meta, 8);                   // metadata bytes (trailing zero padding is already 0)
  out.set(body, 8 + padded);          // body follows padded metadata
  return out;
}

// ---------------------------------------------------------------------------
// toArrow() — public API
// ---------------------------------------------------------------------------

/**
 * Encode a DataFrame to an Arrow IPC stream (Uint8Array).
 *
 * Stream: Schema message → DictionaryBatch per utf8 column → RecordBatch → EOS.
 * Dictionary-encoded columns use Arrow Dict<Int32, Utf8>; our offsets/bytes
 * buffers pass through directly (ADR-002 "nearly free").
 * Bool columns are repacked from u8 to bit-packed (the ONE real transform).
 */
export function toArrow(df: DataFrame): Uint8Array {
  const ctx = df.ctx;
  const names = df.columns as string[];
  const n = df.length;
  const dtypes = names.map((nm) => df.dtypeOf(nm)!);
  const cols = names.map((nm) => df.getColumn(nm)!);

  // Assign ascending dict IDs to utf8 columns
  const dictIds: (number | null)[] = dtypes.map(() => null);
  let nextDictId = 0;
  for (let i = 0; i < dtypes.length; i++) {
    if (dtypes[i] === 'utf8') dictIds[i] = nextDictId++;
  }

  // Collect tz metadata for timestamp columns (ADR-010: tz = display metadata only).
  const tzs: (string | undefined)[] = cols.map((col) =>
    col.dtype === 'timestamp' ? col.tz : undefined,
  );

  const messages: Uint8Array[] = [];

  // 1. Schema
  messages.push(ipcMessage(buildSchemaMeta(names, dtypes, dictIds, tzs), new Uint8Array(0)));

  // 2. DictionaryBatch per utf8 column
  for (let i = 0; i < cols.length; i++) {
    if (dtypes[i] !== 'utf8') continue;
    const col = cols[i]!;
    const dict = col.dict!;
    const count = dict.count;
    const id = dictIds[i]!;

    const offsetsSrc = ctx.viewOf({ ptr: dict.offsetsPtr, length: count + 1, dtype: 'i32' }) as Int32Array;
    const bytesSrc = dict.bytesLen > 0
      ? (ctx.viewOf({ ptr: dict.bytesPtr, length: dict.bytesLen, dtype: 'u8' }) as Uint8Array)
      : new Uint8Array(0);

    const bb: BodyBuilder = { parts: [], size: 0 };
    const vBuf  = pushBuffer(bb, null);                                         // no nulls in dict
    const oBuf  = pushBuffer(bb, new Uint8Array(offsetsSrc.buffer, offsetsSrc.byteOffset, offsetsSrc.byteLength).slice());
    const dBuf  = pushBuffer(bb, bytesSrc.slice());
    const body  = buildBody(bb.parts);

    const plan: BodyPlan = {
      nodes: [{ length: count, nullCount: 0 }],
      buffers: [vBuf, oBuf, dBuf],
      body,
    };
    messages.push(ipcMessage(buildDictMeta(plan, id), body));
  }

  // 3. RecordBatch
  const rb: BodyBuilder = { parts: [], size: 0 };
  const rbNodes: BodyPlan['nodes'] = [];
  const rbBufs: BodyPlan['buffers'] = [];

  for (let i = 0; i < cols.length; i++) {
    const col = cols[i]!;
    const dtype = dtypes[i]!;

    // Count nulls
    let nullCount = 0;
    if (col.validityPtr !== 0) {
      const totalBits = col.validityBitOffset + n;
      const vraw = ctx.viewOf({ ptr: col.validityPtr, length: validityBytes(totalBits), dtype: 'u8' }) as Uint8Array;
      for (let k = 0; k < n; k++) if (!getBit(vraw, col.validityBitOffset + k)) nullCount++;
    }
    rbNodes.push({ length: n, nullCount });
    rbBufs.push(pushBuffer(rb, extractValidity(ctx, col)));

    if (dtype === 'utf8') {
      // Indices buffer (i32[n])
      const idx = ctx.viewOf({ ptr: col.dataPtr, length: n, dtype: 'i32' }) as Int32Array;
      rbBufs.push(pushBuffer(rb, new Uint8Array(idx.buffer, idx.byteOffset, idx.byteLength).slice()));
    } else if (dtype === 'bool') {
      const boolV = ctx.viewOf({ ptr: col.dataPtr, length: n, dtype: 'bool' }) as Uint8Array;
      rbBufs.push(pushBuffer(rb, packBools(boolV, n)));
    } else {
      const info = DTYPES[dtype];
      const v = ctx.viewOf({ ptr: col.dataPtr, length: n, dtype: info.view });
      rbBufs.push(pushBuffer(rb, new Uint8Array(v.buffer, v.byteOffset, n * info.size).slice()));
    }
  }

  const rbBody = buildBody(rb.parts);
  const rbPlan: BodyPlan = { nodes: rbNodes, buffers: rbBufs, body: rbBody };
  messages.push(ipcMessage(buildRBMeta(rbPlan), rbBody));

  // 4. EOS
  const eos = new Uint8Array(8);
  new DataView(eos.buffer).setInt32(0, -1, true);
  messages.push(eos);

  // Concat all
  const total = messages.reduce((s, m) => s + m.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const m of messages) { out.set(m, off); off += m.byteLength; }
  return out;
}

// ---------------------------------------------------------------------------
// fromArrow() — IPC reader
// ---------------------------------------------------------------------------

class IpcReader {
  private pos = 0;
  private dv: DataView;

  constructor(private buf: Uint8Array) {
    this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  /** null = EOS */
  next(): { meta: Uint8Array; body: Uint8Array } | null {
    if (this.pos + 8 > this.buf.byteLength) return null;
    const cont = this.dv.getUint32(this.pos, true);
    if (cont !== 0xffff_ffff) throw new Error(`Arrow: expected continuation (0xFFFFFFFF) at offset ${this.pos}`);
    const metaSize = this.dv.getInt32(this.pos + 4, true);
    this.pos += 8;
    if (metaSize === 0) return null; // EOS

    // metaSize is the PADDED metadata length (apache-arrow convention: size field includes
    // trailing zero padding to 8-byte alignment). We read exactly metaSize bytes and
    // advance by pad8(metaSize) which is a no-op since metaSize is already aligned.
    const meta = this.buf.slice(this.pos, this.pos + metaSize);
    this.pos += pad8(metaSize); // advance past (already-padded) metadata

    const root = fbRoot(meta);
    const bodyLen = root.getInt64Num(3);
    // bodyLength in metadata is already a multiple of 8 (Arrow spec);
    // pad8(bodyLen) is a no-op for compliant streams but is defensive.
    const body = bodyLen > 0 ? this.buf.slice(this.pos, this.pos + bodyLen) : new Uint8Array(0);
    this.pos += pad8(bodyLen);
    return { meta, body };
  }
}

interface ArrowField {
  name: string;
  dtype: DType;
  dictId: number | null; // null if not dict-encoded
  /** For non-dict Utf8: true if the body has [validity, offsets, bytes] (3 buffers). */
  plainUtf8: boolean;
  /** IANA tz string for Arrow Timestamp columns; undefined for all other types. */
  tz?: string;
  /**
   * Arrow TimeUnit for Timestamp columns: SECOND=0, MILLISECOND=1, MICROSECOND=2, NANOSECOND=3.
   * Undefined for non-timestamp columns. Used by fromArrow to rescale to ms when needed.
   */
  arrowUnit?: number;
}

function parseSchema(meta: Uint8Array): ArrowField[] {
  const msg = fbRoot(meta);
  if (msg.getUint8(1, 0) !== MH_SCHEMA) throw new Error('Arrow: first message is not Schema');
  const schema = msg.getTable(2);
  if (!schema) throw new Error('Arrow: missing schema table');

  const n = schema.vectorLen(1);
  const fields: ArrowField[] = [];
  for (let i = 0; i < n; i++) {
    const f = schema.vectorTable(1, i);
    const name = f.getString(0) ?? `f${i}`;
    const typeTag = f.getUint8(2, 0);
    const typeT = f.getTable(3);
    const dictEnc = f.getTable(4);
    const dictId = dictEnc ? dictEnc.getInt64Num(0) : null;

    let dtype: DType;
    let plainUtf8 = false;
    let tz: string | undefined;
    let arrowUnit: number | undefined;

    switch (typeTag) {
      case TYPE_INT: {
        const bits = typeT?.getInt32(0, 32) ?? 32;
        const signed = typeT?.getBool(1, false) ?? false;
        if (bits === 64) {
          if (!signed) throw new Error(
            `Arrow fromArrow: unsupported Arrow type UInt64. ` +
            `databonk supports Int64(signed) as 'i64'.`,
          );
          dtype = 'i64'; // ADR-009: Arrow Int64(signed) → databonk i64
        } else {
          dtype = (bits <= 32 && !signed) ? 'u32' : 'i32';
        }
        break;
      }
      case TYPE_FLOAT: {
        const prec = typeT?.getInt16(0, 0) ?? 0;
        dtype = prec === PREC_DOUBLE ? 'f64' : 'f32';
        break;
      }
      case TYPE_BOOL: dtype = 'bool'; break;
      case TYPE_UTF8:
      case TYPE_LARGE_UTF8:
        dtype = 'utf8';
        plainUtf8 = dictId === null;
        break;
      case TYPE_DATE: {
        // Date.unit FlatBuffers default = MILLISECOND (1). We only support Date32 = DAY (0).
        const unit = typeT?.getInt16(0, DATE_UNIT_MILLI_DEFAULT) ?? DATE_UNIT_MILLI_DEFAULT;
        if (unit !== DATE_UNIT_DAY) throw new Error(
          `Arrow fromArrow: unsupported Date unit ${unit} (only Date32(DAY)=0 is supported).`,
        );
        dtype = 'date32'; // ADR-010: Arrow Date(DAY) → databonk date32
        break;
      }
      case TYPE_TIMESTAMP: {
        // Timestamp.unit FlatBuffers default = SECOND (0). timezone (field 1) is optional.
        const unit = typeT?.getInt16(0, TIME_UNIT_SECOND) ?? TIME_UNIT_SECOND;
        const tzStr = typeT?.getString(1) ?? undefined;
        dtype = 'timestamp'; // ADR-010: Arrow Timestamp(unit, tz?) → databonk timestamp
        arrowUnit = unit;
        tz = tzStr;
        break;
      }
      default:
        throw new Error(
          `Arrow fromArrow: unsupported Arrow type tag ${typeTag}. ` +
          `Supported: Int32, Int64, UInt32, Float32, Float64, Bool, Utf8/Dict<Int32,Utf8>, ` +
          `Date32(DAY), Timestamp(any unit, optional tz).`,
        );
    }
    fields.push({
      name,
      dtype,
      dictId,
      plainUtf8,
      ...(tz !== undefined ? { tz } : {}),
      ...(arrowUnit !== undefined ? { arrowUnit } : {}),
    });
  }
  return fields;
}

// ---------------------------------------------------------------------------
// Byte-hash dedup for plain UTF-8 ingest (ABI §12, CP.1 fast path)
// ---------------------------------------------------------------------------

/**
 * FNV-1a 32-bit hash over `bytes[start..end)`.
 * Empty range → fixed non-zero value (0x811c9dc5, the FNV offset basis).
 */
function fnv1aBytes(bytes: Uint8Array, start: number, end: number): number {
  let h = 0x811c9dc5 >>> 0;
  for (let i = start; i < end; i++) {
    h ^= bytes[i]!;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/**
 * True if `a[as..ae)` and `b[bs..be)` are identical byte sequences
 * (caller already checked length equality).
 */
function bytesRangeEqual(
  a: Uint8Array, as_: number, ae: number,
  b: Uint8Array, bs: number,
): boolean {
  const len = ae - as_;
  for (let i = 0; i < len; i++) {
    if (a[as_ + i] !== b[bs + i]) return false;
  }
  return true;
}

/**
 * Dedup result: unique byte ranges in first-seen order + per-row index array.
 * `dictOffsets` and `dictBytes` are in Arrow/databonk dictionary layout
 * (offsets[k]..offsets[k+1] = byte range of slot k in bytes).
 */
interface ByteDedupResult {
  dictOffsets: Int32Array;
  dictBytes:   Uint8Array;
  dictCount:   number;
  indices:     Int32Array;
}

/** Next power of two ≥ n (n must be > 0). */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Deduplicate a plain-UTF-8 Arrow column on raw bytes — no TextDecoder or
 * TextEncoder. Hash each row's byte range with FNV-1a; use a flat open-
 * addressing hash table (Int32Array) for O(1) amortised lookup — much faster
 * than a JS `Map` for large numeric keys.
 *
 * For the all-unique case (e.g. 85K unique GEOIDs): the hot path never
 * enters a collision branch; the compaction step is skipped entirely and the
 * Arrow offsets/bytes are returned as-is (no second copy).
 *
 * O(n × avgLen + uniqueCount × avgLen): one byte pass for hashing + one for
 * compaction (skipped for all-unique).
 */
function buildDictFromPlainUtf8(
  rawBytes: Uint8Array,
  offsets:  Int32Array,
  rowLen:   number,
): ByteDedupResult {
  const indices = new Int32Array(rowLen);

  if (rowLen === 0) {
    return {
      dictOffsets: new Int32Array([0]),
      dictBytes:   new Uint8Array(0),
      dictCount:   0,
      indices,
    };
  }

  // Flat open-addressing hash table: ht[slot] = dict-slot-index, or -1 (empty).
  // Sized to 2× rowLen so load factor ≤ 0.5 keeps expected probes near 1.
  const htCap  = nextPow2(rowLen * 2);
  const htMask = htCap - 1;
  const ht     = new Int32Array(htCap).fill(-1); // -1 = empty
  // hash values stored alongside ht to avoid re-hashing on collision probe
  const htHash = new Uint32Array(htCap);

  // Arrays tracking unique string byte ranges in rawBytes (first-seen order).
  // For n unique strings these stay as compact Int32Arrays — no boxing.
  let dictCount = 0;
  // Use Float64Array as a growable buffer for (start, end) pairs to avoid
  // boxing overhead of a plain number[]. We resize by doubling.
  let slotCap = Math.min(rowLen, 256);
  let slotS   = new Float64Array(slotCap); // slot k start in rawBytes
  let slotE   = new Float64Array(slotCap); // slot k end   in rawBytes

  for (let k = 0; k < rowLen; k++) {
    const s = offsets[k]!;
    const e = offsets[k + 1]!;
    const h = fnv1aBytes(rawBytes, s, e);
    const len = e - s;

    // Open-address probe: find either a match or an empty slot.
    let slot = (h >>> 0) & htMask;
    let found = -1;
    for (;;) {
      const di = ht[slot]!;
      if (di < 0) break;                        // empty — this string is new
      if (htHash[slot] === (h >>> 0)) {
        // Same hash: byte-compare to confirm (collision guard)
        const ds = slotS[di]!;
        const de = slotE[di]!;
        if (de - ds === len && bytesRangeEqual(rawBytes, s, e, rawBytes, ds)) {
          found = di;
          break;
        }
      }
      slot = (slot + 1) & htMask;               // linear probe
    }

    if (found >= 0) {
      indices[k] = found;
    } else {
      // New unique string: record it and insert into the hash table.
      const idx = dictCount++;
      if (idx >= slotCap) {
        slotCap <<= 1;
        const ns = new Float64Array(slotCap); ns.set(slotS); slotS = ns;
        const ne = new Float64Array(slotCap); ne.set(slotE); slotE = ne;
      }
      slotS[idx] = s;
      slotE[idx] = e;
      ht[slot]     = idx;
      htHash[slot] = h >>> 0;
      indices[k]   = idx;
    }
  }

  // All-unique fast path: the Arrow bytes and offsets ARE the dictionary verbatim
  // (first-seen order = row order when every row is unique). Skip the compaction
  // copy: return aliases instead. writeDictionaryFromRawBytes copies them to wasm.
  if (dictCount === rowLen) {
    return { dictOffsets: offsets, dictBytes: rawBytes, dictCount, indices };
  }

  // General path: build compact dictOffsets + dictBytes from the unique ranges.
  const dictOffsets = new Int32Array(dictCount + 1);
  let acc = 0;
  for (let k = 0; k < dictCount; k++) {
    dictOffsets[k] = acc;
    acc += slotE[k]! - slotS[k]!;
  }
  dictOffsets[dictCount] = acc;

  const dictBytes = new Uint8Array(acc);
  let pos = 0;
  for (let k = 0; k < dictCount; k++) {
    const ss = slotS[k]!;
    const ee = slotE[k]!;
    dictBytes.set(rawBytes.subarray(ss, ee), pos);
    pos += ee - ss;
  }

  return { dictOffsets, dictBytes, dictCount, indices };
}

// ---------------------------------------------------------------------------

interface DictEntry { offsets: Int32Array; bytes: Uint8Array; count: number }

function parseDictBatch(meta: Uint8Array, body: Uint8Array): { id: number; entry: DictEntry } {
  const msg = fbRoot(meta);
  const db = msg.getTable(2);
  if (!db) throw new Error('Arrow: DictionaryBatch missing');
  const id = db.getInt64Num(0);
  const rb = db.getTable(1);
  if (!rb) throw new Error('Arrow: DictionaryBatch missing inner RecordBatch');

  const dictLen = rb.getInt64Num(0);

  // Buffer 0: validity (ignored, dict entries never null)
  // Buffer 1: offsets i32[dictLen+1]
  // Buffer 2: bytes  u8[bytesLen]
  const getBufOff = (bi: number): number => rb.i32at(rb.structVectorElemPos(2, bi, STRUCT16));
  const getBufLen = (bi: number): number => rb.i32at(rb.structVectorElemPos(2, bi, STRUCT16) + 8);

  const nBufs = rb.vectorLen(2);
  if (nBufs < 2) throw new Error('Arrow: DictionaryBatch RecordBatch needs ≥2 buffers');

  const oOff = getBufOff(1); const oLen = getBufLen(1);
  const bOff = nBufs >= 3 ? getBufOff(2) : 0;
  const bLen = nBufs >= 3 ? getBufLen(2) : 0;

  const offsets = new Int32Array(body.buffer, body.byteOffset + oOff, dictLen + 1).slice();
  const bytes   = bLen > 0 ? body.slice(bOff, bOff + bLen) : new Uint8Array(0);

  return { id, entry: { offsets, bytes, count: dictLen } };
}

function readBody(body: Uint8Array, off: number, len: number): Uint8Array {
  return len > 0 ? body.slice(off, off + len) : new Uint8Array(0);
}

/**
 * Parse a RecordBatch message and produce raw Column objects owned by wasm memory.
 *
 * CP.1 fast paths (ABI §12):
 *   - utf8 dict-encoded: `writeDictionaryFromRawBytes` — no TextDecoder/TextEncoder.
 *   - utf8 plain: `buildDictFromPlainUtf8` (byte-hash dedup) + `writeDictionaryFromRawBytes`.
 *   - numeric/bool: unchanged — single alloc + TypedArray.set bulk-copy.
 *
 * Returns `{ columns, length }`. Caller must free columns that are not adopted
 * by a DataFrame.
 */
function parseRecordBatch(
  meta: Uint8Array,
  body: Uint8Array,
  fields: ArrowField[],
  dicts: Map<number, DictEntry>,
  ctx: MemoryContext,
): { columns: Column[]; length: number } {
  const msg = fbRoot(meta);
  if (msg.getUint8(1, 0) !== MH_RECORD) throw new Error('Arrow: expected RecordBatch');
  const rb = msg.getTable(2);
  if (!rb) throw new Error('Arrow: missing RecordBatch table');

  const rowLen = rb.getInt64Num(0);
  const nNodes = rb.vectorLen(1);
  if (nNodes !== fields.length) {
    throw new Error(`Arrow: field count mismatch (schema=${fields.length}, batch=${nNodes})`);
  }

  const getBufOff = (bi: number): number => rb.i32at(rb.structVectorElemPos(2, bi, STRUCT16));
  const getBufLen = (bi: number): number => rb.i32at(rb.structVectorElemPos(2, bi, STRUCT16) + 8);

  const columns: Column[] = [];
  let bi = 0;

  for (let fi = 0; fi < fields.length; fi++) {
    const { dtype, dictId } = fields[fi]!;

    // Validity buffer (bi)
    const vOff = getBufOff(bi); const vLen = getBufLen(bi); bi++;
    const hasValidity = vLen > 0;

    if (dtype === 'utf8') {
      if (dictId !== null) {
        // ── Dict-encoded fast path (ABI §12): bulk-copy bytes + offsets, no decode/encode ──
        const iOff = getBufOff(bi); bi++;
        const entry = dicts.get(dictId);
        if (!entry) throw new Error(`Arrow: missing dictionary id=${dictId}`);

        const dict = writeDictionaryFromRawBytes(ctx, entry.bytes, entry.offsets, entry.count);

        const dataPtr = ctx.mod.alloc(Math.max(rowLen * 4, 1));
        const idxView = ctx.viewOf({ ptr: dataPtr, length: rowLen, dtype: 'i32' }) as Int32Array;
        idxView.set(new Int32Array(body.buffer, body.byteOffset + iOff, rowLen));

        let validityPtr = 0;
        if (hasValidity) {
          validityPtr = ctx.mod.alloc(validityBytes(rowLen));
          // vLen from IPC metadata may be padded to 8 bytes (apache-arrow convention);
          // copy only the actual bytes needed for the bitmap (validityBytes(rowLen)).
          (ctx.viewOf({ ptr: validityPtr, length: validityBytes(rowLen), dtype: 'u8' }) as Uint8Array)
            .set(body.subarray(vOff, vOff + validityBytes(rowLen)));
        }
        columns.push({ dtype: 'utf8', length: rowLen, dataPtr, validityPtr, validityBitOffset: 0, dict, owned: true });

      } else {
        // ── Plain UTF-8 fast path (ABI §12): byte-hash dedup, no TextDecoder/TextEncoder ──
        const oOff2 = getBufOff(bi); bi++;
        const dOff2 = getBufOff(bi); const dLen2 = getBufLen(bi); bi++;

        const arrowOffsets = new Int32Array(body.buffer, body.byteOffset + oOff2, rowLen + 1);
        const arrowBytes   = body.subarray(dOff2, dOff2 + dLen2);

        // Dedup on raw bytes: O(n×avgLen), no JS string allocation in the hot loop.
        const { dictOffsets, dictBytes, dictCount, indices } =
          buildDictFromPlainUtf8(arrowBytes, arrowOffsets, rowLen);

        // Bulk-copy the compact dict into wasm (no TextEncoder).
        const dict = writeDictionaryFromRawBytes(ctx, dictBytes, dictOffsets, dictCount);

        const dataPtr = ctx.mod.alloc(Math.max(rowLen * 4, 1));
        (ctx.viewOf({ ptr: dataPtr, length: rowLen, dtype: 'i32' }) as Int32Array).set(indices);

        let validityPtr = 0;
        if (hasValidity) {
          validityPtr = ctx.mod.alloc(validityBytes(rowLen));
          // vLen may be padded; copy only actual bitmap bytes.
          (ctx.viewOf({ ptr: validityPtr, length: validityBytes(rowLen), dtype: 'u8' }) as Uint8Array)
            .set(body.subarray(vOff, vOff + validityBytes(rowLen)));
        }
        columns.push({ dtype: 'utf8', length: rowLen, dataPtr, validityPtr, validityBitOffset: 0, dict, owned: true });
      }

    } else if (dtype === 'bool') {
      const datOff = getBufOff(bi); const datLen = getBufLen(bi); bi++;
      const packed = body.subarray(datOff, datOff + datLen);
      const u8data = unpackBools(packed, rowLen);

      const dataPtr = ctx.mod.alloc(Math.max(rowLen, 1));
      (ctx.viewOf({ ptr: dataPtr, length: rowLen, dtype: 'bool' }) as Uint8Array).set(u8data);

      let validityPtr = 0;
      if (hasValidity) {
        validityPtr = ctx.mod.alloc(validityBytes(rowLen));
        // vLen may be padded to 8 bytes (apache-arrow convention); copy only actual bits.
        (ctx.viewOf({ ptr: validityPtr, length: validityBytes(rowLen), dtype: 'u8' }) as Uint8Array)
          .set(body.subarray(vOff, vOff + validityBytes(rowLen)));
      }
      columns.push({ dtype: 'bool', length: rowLen, dataPtr, validityPtr, validityBitOffset: 0, dict: null, owned: true });

    } else {
      // ── Numeric fast path: single alloc + TypedArray.set bulk-copy ──
      const info = DTYPES[dtype];
      const datOff = getBufOff(bi); bi++;
      const dataPtr = ctx.mod.alloc(Math.max(rowLen * info.size, 1));
      const view = ctx.viewOf({ ptr: dataPtr, length: rowLen, dtype: info.view });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const srcTyped = new (info.ctor as any)(body.buffer as ArrayBuffer, body.byteOffset + datOff, rowLen);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (view as any).set(srcTyped);

      let validityPtr = 0;
      if (hasValidity) {
        validityPtr = ctx.mod.alloc(validityBytes(rowLen));
        // vLen may be padded to 8 bytes (apache-arrow convention); copy only actual bits.
        (ctx.viewOf({ ptr: validityPtr, length: validityBytes(rowLen), dtype: 'u8' }) as Uint8Array)
          .set(body.subarray(vOff, vOff + validityBytes(rowLen)));
      }
      columns.push({ dtype, length: rowLen, dataPtr, validityPtr, validityBitOffset: 0, dict: null, owned: true });
    }
  }
  return { columns, length: rowLen };
}

/**
 * Rescale Arrow Timestamp column values to milliseconds.
 *
 * Arrow Timestamp can carry any TimeUnit. databonk stores timestamps as ms (ADR-010).
 * - SECOND → ms: ×1000 with saturation-to-null on i64 overflow (documented).
 *   Safe range: |seconds| ≤ 9_223_372_036_854_775 (≈ ±292 million years). Out-of-range
 *   values (which would overflow i64 when multiplied by 1000) become null.
 * - MICROSECOND → ms: ÷1000 (integer truncation; sub-ms precision silently dropped).
 * - NANOSECOND → ms: ÷1_000_000 (integer truncation; sub-ms precision silently dropped).
 * - MILLISECOND: no-op (values already in ms).
 */
function rescaleTimestampToMs(vals: (bigint | null)[], unit: number): (bigint | null)[] {
  if (unit === TIME_UNIT_MILLI) return vals;
  return vals.map((v) => {
    if (v === null) return null;
    switch (unit) {
      case TIME_UNIT_SECOND:
        // Saturation-to-null: if v is outside safe range, ×1000 would overflow i64.
        if (v > SEC_MS_MAX || v < SEC_MS_MIN) return null;
        return v * 1000n;
      case TIME_UNIT_MICRO:
        return v / 1000n; // truncate sub-ms
      case TIME_UNIT_NANO:
        return v / 1_000_000n; // truncate sub-ms
      default:
        return null; // unknown unit → null
    }
  });
}

/**
 * Decode an Arrow IPC stream buffer into a DataFrame.
 *
 * Supported Arrow types: Int32, Int64, UInt32, Float32, Float64, Bool,
 * Dict<Int32, Utf8> (written by our toArrow), plain Utf8 (builds a dict internally),
 * Date32(DAY) (→ date32), Timestamp(any unit, optional tz) (→ timestamp, rescaled to ms).
 * Any other type throws a clear error naming the type tag.
 */
export function fromArrow(buf: Uint8Array, rt: DfRuntime): DataFrame {
  const reader = new IpcReader(buf);

  // Schema
  const schemaMsg = reader.next();
  if (!schemaMsg) throw new Error('Arrow fromArrow: empty or truncated buffer');
  const fields = parseSchema(schemaMsg.meta);

  // Collect DictionaryBatches, stop at RecordBatch
  const dicts = new Map<number, DictEntry>();
  let rbMsg: { meta: Uint8Array; body: Uint8Array } | null = null;

  for (;;) {
    const msg = reader.next();
    if (!msg) break;
    const root = fbRoot(msg.meta);
    const hType = root.getUint8(1, 0);
    if (hType === MH_DICT) {
      const { id, entry } = parseDictBatch(msg.meta, msg.body);
      dicts.set(id, entry);
    } else if (hType === MH_RECORD) {
      rbMsg = msg;
      break;
    }
    // Skip unknown message types
  }

  if (!rbMsg) throw new Error('Arrow fromArrow: no RecordBatch found in stream');

  const { columns: rawCols, length: rowLen } =
    parseRecordBatch(rbMsg.meta, rbMsg.body, fields, dicts, rt.ctx);

  // CP.1 fast path: adopt pre-built wasm columns directly — no JS-array round-trip.
  //
  // The general contract: `parseRecordBatch` produced owned wasm columns with
  // correct dtype/dataPtr/validityPtr/dict. We hand them straight to
  // `DataFrame._adoptColumns`. The only exception is a non-MILLISECOND Arrow
  // Timestamp, which must be rescaled via a JS round-trip (rare edge case).
  const named: Array<{ name: string; col: Column }> = [];

  for (let i = 0; i < fields.length; i++) {
    const { name, dtype, tz, arrowUnit } = fields[i]!;
    let col = rawCols[i]!;

    if (dtype === 'timestamp' && arrowUnit !== undefined && arrowUnit !== TIME_UNIT_MILLI) {
      // Rare: rescale non-ms Arrow Timestamp to ms via a JS round-trip (ADR-010).
      const jsVals = columnToArray(rt.ctx, col) as (bigint | null)[];
      const rescaled = rescaleTimestampToMs(jsVals, arrowUnit);
      freeColumn(rt.ctx, col);
      col = createColumn(rt.ctx, 'timestamp', rescaled as ColumnInput, tz);
    } else if (dtype === 'timestamp' && tz !== undefined) {
      // Attach tz metadata (display/accessor only, ADR-010) without touching wasm.
      col = { ...col, tz };
    }

    named.push({ name, col });
  }

  return DataFrame._adoptColumns(rt, named, rowLen);
}

// ---------------------------------------------------------------------------
// Free a raw (owned) column that won't be adopted by a DataFrame
// ---------------------------------------------------------------------------

import { freeColumn } from '../memory/column.js';
// freeDictionary is re-exported from memory/index but not needed directly here —
// freeColumn already calls freeDictionary for utf8 columns.

