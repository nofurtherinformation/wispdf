/**
 * Minimal FlatBuffers encoder + decoder for Arrow IPC messages.
 *
 * ## FlatBuffers binary layout (little-endian throughout)
 *
 * The builder works RIGHT-TO-LEFT: data is prepended by decrementing `head`.
 * "posFromEnd" = `space.length - head` (bytes built so far from the right end).
 * In the FINAL buffer `space.slice(head)`:
 *   - Earlier writes  → higher posFromStart (further RIGHT) = larger index
 *   - Later writes    → lower posFromStart  (further LEFT)  = smaller index
 *
 * Since children are built FIRST (higher posFromStart = further right), and
 * parents LATER (lower posFromStart = further left), forward references from
 * parent fields to child objects are POSITIVE relative offsets.
 *
 * ### Relative forward offsets
 *
 * A 4-byte slot in a parent table/vector containing a relative offset to a
 * child satisfies:
 *
 *   relOff = posFromEnd(slot) - posFromEnd(child)   (always positive)
 *
 * which the reader converts back via:
 *
 *   posFromStart(child) = posFromStart(slot) + relOff
 *
 * ### vtable layout
 *
 *   [u16 vtable_size][u16 object_size][u16 field_0_offset][u16 field_1_offset]…
 *
 * `field_N_offset` is the byte distance from the TABLE OBJECT START to field N.
 * The table object itself starts with a 4-byte int32 "soffset":
 *
 *   soffset  = posFromEnd(vtable_start) - posFromEnd(table_start)
 *            = posFromStart(table_start) - posFromStart(vtable_start)  (positive)
 *
 * The reader finds the vtable as: `vtable_pos = table_pos - soffset`.
 *
 * ### Root object
 *
 * `finish(rootOff)` writes a u32 at position 0 of the final buffer:
 *   root_u32 = posFromStart(root_table) = finalBufLen - rootOff
 * where `rootOff` = posFromEnd of the root table at the time `endTable()` returned.
 * The reader uses this u32 directly as an absolute position.
 */

// ---------------------------------------------------------------------------
// FBBuilder
// ---------------------------------------------------------------------------

export class FBBuilder {
  private space: Uint8Array;
  private dv: DataView;
  private head: number;

  // Current table state: accumulated field positions (posFromEnd)
  private vtableFields = new Map<number, number>();
  private objectStart = 0;

  constructor(initialSize = 512) {
    this.space = new Uint8Array(initialSize);
    this.dv = new DataView(this.space.buffer);
    this.head = initialSize;
  }

  /** Bytes written so far (= posFromEnd of the current write head). */
  private offset(): number {
    return this.space.length - this.head;
  }

  /** Grow the buffer, preserving existing content at the right end. */
  private grow(neededExtra: number): void {
    const newLen = Math.max(this.space.length * 2, this.space.length + neededExtra);
    const newSpace = new Uint8Array(newLen);
    const dataLen = this.space.length - this.head;
    newSpace.set(this.space.subarray(this.head), newLen - dataLen);
    this.head = newLen - dataLen;
    this.space = newSpace;
    this.dv = new DataView(newSpace.buffer);
  }

  /**
   * Ensure `size`-byte alignment + `additionalBytes` available space.
   * Prepends zero-padding as needed.
   */
  private prep(size: number, additionalBytes: number): void {
    const alignSize = ((~(this.offset() + additionalBytes)) + 1) & (size - 1);
    if (this.head < size + alignSize + additionalBytes) {
      this.grow(size + alignSize + additionalBytes);
    }
    for (let i = 0; i < alignSize; i++) this.space[--this.head] = 0;
  }

  // -------------------------------------------------------------------------
  // Primitive write methods (each decrements `head`)
  // -------------------------------------------------------------------------

  writeU8(v: number): void {
    this.prep(1, 0);
    this.space[--this.head] = v & 0xff;
  }

  writeU16(v: number): void {
    this.prep(2, 0);
    this.head -= 2;
    this.dv.setUint16(this.head, v, true);
  }

  writeI32(v: number): void {
    this.prep(4, 0);
    this.head -= 4;
    this.dv.setInt32(this.head, v, true);
  }

  writeU32(v: number): void {
    this.prep(4, 0);
    this.head -= 4;
    this.dv.setUint32(this.head, v, true);
  }

  /** Write int64 as two LE int32s (lo at lower address, hi at higher). */
  writeI64(lo: number, hi = 0): void {
    this.prep(8, 0);
    this.head -= 8;
    this.dv.setInt32(this.head, lo, true);
    this.dv.setInt32(this.head + 4, hi, true);
  }

  // -------------------------------------------------------------------------
  // Compound builders (return posFromEnd of the created structure)
  // -------------------------------------------------------------------------

  /** FlatBuffers string: u32 length + utf-8 bytes + null terminator. */
  createString(s: string): number {
    const enc = new TextEncoder().encode(s);
    this.prep(4, enc.length + 1);
    this.space[--this.head] = 0; // null terminator
    for (let i = enc.length - 1; i >= 0; i--) this.space[--this.head] = enc[i]!;
    this.writeU32(enc.length);
    return this.offset();
  }

  /** FlatBuffers byte vector: u32 length + raw bytes. */
  createByteVector(data: Uint8Array): number {
    this.prep(4, data.length);
    for (let i = data.length - 1; i >= 0; i--) this.space[--this.head] = data[i]!;
    this.writeU32(data.length);
    return this.offset();
  }

  /**
   * FlatBuffers offset vector (vector of table/string references).
   * `offsets` are posFromEnd values returned by earlier create calls.
   */
  createOffsetVector(offsets: number[]): number {
    this.prep(4, offsets.length * 4);
    for (let i = offsets.length - 1; i >= 0; i--) {
      // Compute relative forward offset AFTER the upcoming head decrement:
      // posFromEnd(slot) = this.offset() + 4  (after decrement by 4)
      const relOff = this.offset() + 4 - offsets[i]!;
      this.head -= 4;
      this.dv.setUint32(this.head, relOff, true);
    }
    this.writeU32(offsets.length);
    return this.offset();
  }

  /**
   * FlatBuffers inline-struct vector (structs have no vtable; data is raw bytes).
   * `data` = serialised structs concatenated, `count` = element count.
   * Written into an 8-byte aligned region (Arrow structs need 8-byte alignment
   * for the embedded int64 fields).
   */
  createStructVector(data: Uint8Array, count: number): number {
    this.prep(8, data.length);
    for (let i = data.length - 1; i >= 0; i--) this.space[--this.head] = data[i]!;
    this.writeU32(count);
    return this.offset();
  }

  // -------------------------------------------------------------------------
  // Table builder
  // -------------------------------------------------------------------------

  /** Begin a new table. Call `addField*` then `endTable()`. */
  startTable(): void {
    this.vtableFields = new Map();
    this.objectStart = this.offset();
  }

  // Scalar field helpers — each skips writing if the value equals its default
  // (FlatBuffers omits default-valued fields to save space).

  addFieldBool(idx: number, value: boolean, def = false): void {
    if (value === def) return;
    this.writeU8(value ? 1 : 0);
    this.vtableFields.set(idx, this.offset());
  }

  addFieldI8(idx: number, value: number, def = 0): void {
    if (value === def) return;
    this.writeU8(value & 0xff);
    this.vtableFields.set(idx, this.offset());
  }

  addFieldI16(idx: number, value: number, def = 0): void {
    if (value === def) return;
    this.writeU16(value);
    this.vtableFields.set(idx, this.offset());
  }

  addFieldI32(idx: number, value: number, def = 0): void {
    if (value === def) return;
    this.writeI32(value);
    this.vtableFields.set(idx, this.offset());
  }

  addFieldI64(idx: number, lo: number, hi = 0): void {
    if (lo === 0 && hi === 0) return;
    this.writeI64(lo, hi);
    this.vtableFields.set(idx, this.offset());
  }

  /**
   * Add an offset field (reference to a child table, vector, or string).
   * `childOff` = posFromEnd returned by a prior create/endTable call.
   *
   * After `head -= 4`, `this.offset()` = posFromEnd(slot).
   * relOff = posFromEnd(slot) - posFromEnd(child)  — always positive.
   */
  addFieldOffset(idx: number, childOff: number): void {
    this.prep(4, 0);
    this.head -= 4;
    const relOff = this.offset() - childOff; // posFromEnd(slot) - posFromEnd(child)
    this.dv.setUint32(this.head, relOff, true);
    this.vtableFields.set(idx, this.offset());
  }

  /**
   * Finalize a table. Returns posFromEnd of the table (for use as a child
   * reference in a parent table or offset vector).
   */
  endTable(): number {
    if (this.vtableFields.size === 0) {
      // Empty table: just the soffset and a minimal vtable
    }
    const numFields = this.vtableFields.size === 0
      ? 0
      : Math.max(...this.vtableFields.keys()) + 1;

    // objectSize = (bytes for soffset) + (bytes for inline fields)
    // = tableOff - objectStart  (computed below after writing soffset)

    // 1. Write the soffset slot (4 bytes); will be patched to the vtable later
    this.writeI32(0); // placeholder
    const tableOff = this.offset(); // posFromEnd of table start (= soffset field)

    // objectSize = tableOff - objectStart (fields + soffset)
    const objectSize = tableOff - this.objectStart;

    // 2. Build vtable: vtableSize + objectSize + field offsets
    //    Written right-to-left: field[N-1] first, field[0] last, then sizes.
    const vtableSize = 4 + numFields * 2; // u16 sizes (2 each) + N field offsets

    for (let i = numFields - 1; i >= 0; i--) {
      const fieldPosFromEnd = this.vtableFields.get(i);
      // Field offset within object = how many bytes from the table start to the field
      // = posFromEnd(table) - posFromEnd(field)  (positive; field was written earlier)
      const fieldOff = fieldPosFromEnd !== undefined ? tableOff - fieldPosFromEnd : 0;
      this.writeU16(fieldOff);
    }
    this.writeU16(objectSize);
    this.writeU16(vtableSize);

    const vtableOff = this.offset(); // posFromEnd of vtable start

    // 3. Patch the soffset:
    //    soffset = posFromStart(table) - posFromStart(vtable)
    //            = (finalBufLen - tableOff) - (finalBufLen - vtableOff)
    //            = vtableOff - tableOff   (positive; vtable is to the LEFT)
    const soffset = vtableOff - tableOff;
    // table soffset is at space index: space.length - tableOff
    this.dv.setInt32(this.space.length - tableOff, soffset, true);

    return tableOff;
  }

  /**
   * Write the final root offset and return the completed FlatBuffers buffer.
   *
   * FlatBuffers root at position 0 = absolute posFromStart of the root table.
   * After prepending 4 bytes (head decremented by 4):
   *   finalBufLen = this.offset()
   *   posFromStart(root) = finalBufLen - rootOff
   */
  finish(rootOff: number): Uint8Array {
    this.prep(4, 0);
    this.head -= 4;
    // this.offset() is now the final buffer length
    this.dv.setUint32(this.head, this.offset() - rootOff, true);
    return this.space.slice(this.head);
  }
}

// ---------------------------------------------------------------------------
// FBTable — reader for a FlatBuffers table
// ---------------------------------------------------------------------------

/**
 * Read a FlatBuffers table from a DataView at absolute position `pos`.
 *
 * Field access:
 *   fieldPos(idx) → absolute position of field data (0 if absent / default)
 *
 * The vtable is located via the soffset at `pos`:
 *   vtable_pos = pos - soffset
 * Field values are at `pos + vtable_field_offset`.
 *
 * For offset fields (tables, strings, vectors): the 4 bytes at the field
 * position hold a relative FORWARD offset; the target is at `field_pos + relOff`.
 */
export class FBTable {
  constructor(
    private readonly dv: DataView,
    /** Absolute byte position of this table's soffset in `dv`. */
    readonly pos: number,
  ) {}

  /** Absolute position of field `idx` in this table, or 0 if absent. */
  fieldPos(idx: number): number {
    const { dv, pos } = this;
    const soffset = dv.getInt32(pos, true);
    const vtPos = pos - soffset;
    const vtSize = dv.getUint16(vtPos, true);
    const vtFieldOff = 4 + idx * 2;
    if (vtFieldOff >= vtSize) return 0;
    const fieldRelOff = dv.getUint16(vtPos + vtFieldOff, true);
    return fieldRelOff === 0 ? 0 : pos + fieldRelOff;
  }

  getInt8 (idx: number, def = 0):    number  { const p = this.fieldPos(idx); return p ? this.dv.getInt8(p)               : def; }
  getUint8(idx: number, def = 0):    number  { const p = this.fieldPos(idx); return p ? this.dv.getUint8(p)              : def; }
  getInt16(idx: number, def = 0):    number  { const p = this.fieldPos(idx); return p ? this.dv.getInt16(p, true)        : def; }
  getInt32(idx: number, def = 0):    number  { const p = this.fieldPos(idx); return p ? this.dv.getInt32(p, true)        : def; }
  getBool (idx: number, def = false): boolean { const p = this.fieldPos(idx); return p ? this.dv.getUint8(p) !== 0       : def; }

  /** Read int64 as [lo32, hi32]. */
  getInt64(idx: number): [number, number] {
    const p = this.fieldPos(idx);
    if (p === 0) return [0, 0];
    return [this.dv.getInt32(p, true), this.dv.getInt32(p + 4, true)];
  }

  /** Read int64 as a JS number (safe for values < 2^53). */
  getInt64Num(idx: number): number {
    const [lo, hi] = this.getInt64(idx);
    return hi * 0x100000000 + (lo >>> 0);
  }

  /** Follow an offset field → absolute position of the target. */
  private follow(p: number): number {
    return p + this.dv.getUint32(p, true);
  }

  getString(idx: number): string | null {
    const p = this.fieldPos(idx);
    if (p === 0) return null;
    const strPos = this.follow(p);
    const len = this.dv.getUint32(strPos, true);
    return new TextDecoder().decode(new Uint8Array(this.dv.buffer, this.dv.byteOffset + strPos + 4, len));
  }

  getTable(idx: number): FBTable | null {
    const p = this.fieldPos(idx);
    if (p === 0) return null;
    return new FBTable(this.dv, this.follow(p));
  }

  vectorLen(idx: number): number {
    const p = this.fieldPos(idx);
    if (p === 0) return 0;
    return this.dv.getUint32(this.follow(p), true);
  }

  /** Get the nested table at index `i` in an offset-vector field. */
  vectorTable(idx: number, i: number): FBTable {
    const p = this.fieldPos(idx);
    const vecPos = this.follow(p);
    const elemPos = vecPos + 4 + i * 4;
    return new FBTable(this.dv, elemPos + this.dv.getUint32(elemPos, true));
  }

  /**
   * Absolute position of the i-th struct element in an inline-struct vector.
   * (Structs are laid out inline — no relative offsets, just raw bytes.)
   */
  structVectorElemPos(idx: number, i: number, elemSize: number): number {
    const p = this.fieldPos(idx);
    if (p === 0) return 0;
    const vecPos = this.follow(p);
    return vecPos + 4 + i * elemSize;
  }

  /** Read int32 at an absolute position (for struct fields). */
  i32at(absPos: number): number {
    return this.dv.getInt32(absPos, true);
  }

  /** Read int64 as JS number at an absolute position. */
  i64at(absPos: number): number {
    const lo = this.dv.getInt32(absPos, true);
    const hi = this.dv.getInt32(absPos + 4, true);
    return hi * 0x100000000 + (lo >>> 0);
  }
}

/** Read the root FlatBuffers table from a buffer. */
export function fbRoot(buf: Uint8Array): FBTable {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // Position 0 holds the absolute position of the root table
  const rootPos = dv.getUint32(0, true);
  return new FBTable(dv, rootPos);
}
