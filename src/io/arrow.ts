/**
 * Arrow IPC stream encode/decode for wispdf DataFrames (ADR-002).
 *
 * Our layout is Arrow-compatible by design:
 *   - Numeric columns: contiguous TypedArray → Arrow primitive buffer (zero-transform).
 *   - Validity bitmaps: LSB-first, 1=valid — already Arrow format (zero-transform).
 *   - utf8 columns: i32 indices + (i32 offsets + u8 bytes) dict → Arrow Dict<Int32, Utf8>.
 *   - bool: u8[n] per-element → bit-pack to Arrow's 1-bit-per-element format (one transform).
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
 *
 * MessageHeader union tags: Schema=1, DictionaryBatch=2, RecordBatch=3
 * Type union tags:          Int=2, FloatingPoint=3, Utf8=5, Bool=6, LargeUtf8=20
 * MetadataVersion V5=4
 * FloatingPoint.Precision: HALF=0, SINGLE=1, DOUBLE=2
 */

import type { MemoryContext } from '../memory/context.js';
import { DTYPES, type DType } from '../memory/dtype.js';
import { validityBytes, getBit, setBit } from '../memory/bitmap.js';
import { columnToArray, type Column } from '../memory/column.js';
import { writeDictionary } from '../memory/dictionary.js';
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
const TYPE_LARGE_UTF8 = 20;
const PREC_SINGLE = 1;
const PREC_DOUBLE = 2;
const STRUCT16 = 16; // FieldNode / Buffer are each 16 bytes

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

function buildField(fb: FBBuilder, name: string, dtype: DType, dictId: number | null): number {
  const nameOff = fb.createString(name);
  let typeTag: number, typeT: number;
  let dictOff: number | null = null;

  switch (dtype) {
    case 'f64': typeTag = TYPE_FLOAT; typeT = buildFloat(fb, PREC_DOUBLE); break;
    case 'f32': typeTag = TYPE_FLOAT; typeT = buildFloat(fb, PREC_SINGLE); break;
    case 'i32': typeTag = TYPE_INT;   typeT = buildInt(fb, 32, true);      break;
    case 'u32': typeTag = TYPE_INT;   typeT = buildInt(fb, 32, false);     break;
    case 'bool': typeTag = TYPE_BOOL; typeT = emptyTable(fb);               break;
    case 'utf8':
      typeTag = TYPE_UTF8; typeT = emptyTable(fb);
      if (dictId !== null) dictOff = buildDictEncoding(fb, dictId);
      break;
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
 */
function buildSchemaMeta(names: string[], dtypes: DType[], dictIds: (number | null)[]): Uint8Array {
  const fb = new FBBuilder(1024);
  const fieldOffs = names.map((n, i) => buildField(fb, n, dtypes[i]!, dictIds[i]!));
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

  const messages: Uint8Array[] = [];

  // 1. Schema
  messages.push(ipcMessage(buildSchemaMeta(names, dtypes, dictIds), new Uint8Array(0)));

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
    switch (typeTag) {
      case TYPE_INT: {
        const bits = typeT?.getInt32(0, 32) ?? 32;
        const signed = typeT?.getBool(1, false) ?? false;
        dtype = (bits <= 32 && !signed) ? 'u32' : 'i32';
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
      default:
        throw new Error(
          `Arrow fromArrow: unsupported Arrow type tag ${typeTag}. ` +
          `Supported: Int(32), UInt32, Float32, Float64, Bool, Utf8, Dict<Int32,Utf8>.`,
        );
    }
    fields.push({ name, dtype, dictId, plainUtf8 });
  }
  return fields;
}

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
 * Caller must free these columns if they won't be handed to DataFrame.
 */
function parseRecordBatch(
  meta: Uint8Array,
  body: Uint8Array,
  fields: ArrowField[],
  dicts: Map<number, DictEntry>,
  ctx: MemoryContext,
): Column[] {
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
    const { dtype, dictId, plainUtf8 } = fields[fi]!;

    // Validity buffer (bi)
    const vOff = getBufOff(bi); const vLen = getBufLen(bi); bi++;
    const hasValidity = vLen > 0;

    if (dtype === 'utf8') {
      if (dictId !== null) {
        // Dict-encoded: next buffer = i32 indices
        const iOff = getBufOff(bi); bi++;
        const entry = dicts.get(dictId);
        if (!entry) throw new Error(`Arrow: missing dictionary id=${dictId}`);

        // Build our dictionary
        const dec = new TextDecoder();
        const uniques: string[] = [];
        for (let k = 0; k < entry.count; k++) {
          const s = entry.offsets[k]!; const e = entry.offsets[k + 1]!;
          uniques.push(e > s ? dec.decode(entry.bytes.subarray(s, e)) : '');
        }
        const dict = writeDictionary(ctx, uniques);

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
        // Plain Utf8: offsets buffer + bytes buffer
        const oOff2 = getBufOff(bi); bi++;
        const dOff2 = getBufOff(bi); const dLen2 = getBufLen(bi); bi++;

        const offsets = new Int32Array(body.buffer, body.byteOffset + oOff2, rowLen + 1);
        const rawBytes = body.subarray(dOff2, dOff2 + dLen2);
        const dec = new TextDecoder();

        // Build dictionary from unique strings
        const idxMap = new Map<string, number>();
        const uniques: string[] = [];
        const indices = new Int32Array(rowLen);
        for (let k = 0; k < rowLen; k++) {
          const s = offsets[k]!; const e = offsets[k + 1]!;
          const str = dec.decode(rawBytes.subarray(s, e));
          let idx = idxMap.get(str);
          if (idx === undefined) { idx = uniques.length; uniques.push(str); idxMap.set(str, idx); }
          indices[k] = idx;
        }
        const dict = writeDictionary(ctx, uniques);
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
      // Numeric: f64, f32, i32, u32
      const info = DTYPES[dtype];
      const datOff = getBufOff(bi); bi++;
      const dataPtr = ctx.mod.alloc(Math.max(rowLen * info.size, 1));
      const view = ctx.viewOf({ ptr: dataPtr, length: rowLen, dtype: info.view });
      const srcTyped = new info.ctor(body.buffer as ArrayBuffer, body.byteOffset + datOff, rowLen);
      (view as unknown as { set(src: ArrayLike<number>): void }).set(srcTyped);

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
  return columns;
}

/**
 * Decode an Arrow IPC stream buffer into a DataFrame.
 *
 * Supported Arrow types: Int32, UInt32, Float32, Float64, Bool,
 * Dict<Int32, Utf8> (written by our toArrow), plain Utf8 (builds a dict internally).
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

  const rawCols = parseRecordBatch(rbMsg.meta, rbMsg.body, fields, dicts, rt.ctx);

  // Convert raw wasm columns to JS arrays and build DataFrame
  // (using columnToArray to avoid exposing internal Column constructors)
  const colData: Record<string, import('../memory/column.js').ColumnInput> = {};
  const dtypes: Record<string, DType> = {};

  for (let i = 0; i < fields.length; i++) {
    const name = fields[i]!.name;
    colData[name] = columnToArray(rt.ctx, rawCols[i]!) as import('../memory/column.js').ColumnInput;
    dtypes[name] = fields[i]!.dtype;
    // Free the intermediate wasm allocation
    freeRawColumn(rt.ctx, rawCols[i]!);
  }

  const opts: FrameOptions = { runtime: rt, dtypes };
  return DataFrame.fromColumns(colData, opts);
}

// ---------------------------------------------------------------------------
// Free a raw (owned) column that won't be adopted by a DataFrame
// ---------------------------------------------------------------------------

import { freeDictionary } from '../memory/dictionary.js';
import { freeColumn } from '../memory/column.js';

function freeRawColumn(ctx: MemoryContext, col: Column): void {
  freeColumn(ctx, col);
}
