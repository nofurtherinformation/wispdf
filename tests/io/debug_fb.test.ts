import { describe, it, expect, beforeAll } from 'vitest';
import { toArrow } from '../../src/io/arrow.js';
import { DataFrame } from '../../src/frame/dataframe.js';
import { loadRuntimeForTest } from '../frame/helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import * as arrow from 'apache-arrow';

let rt: DfRuntime;
beforeAll(async () => { rt = await loadRuntimeForTest(); });

describe('IPC stream debug', () => {
  it('prints all message headers', () => {
    const df = DataFrame.fromColumns({ v: new Float64Array([1.1, 2.2]) }, { runtime: rt });
    const buf = toArrow(df);
    df.dispose();
    
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    console.log('Total IPC size:', buf.byteLength);
    
    function hex(off: number, len: number) {
      return Array.from(buf.slice(off, off + len)).map((b: number) => b.toString(16).padStart(2,'0')).join(' ');
    }
    
    let pos = 0;
    let msgNum = 0;
    while (pos + 8 <= buf.byteLength && msgNum < 5) {
      const cont = dv.getUint32(pos, true);
      const metaSize = dv.getInt32(pos + 4, true);
      console.log(`\n--- Message ${msgNum} at IPC pos=${pos} ---`);
      console.log(`  cont=0x${cont.toString(16)}, metaSize=${metaSize}`);
      console.log(`  IPC header bytes: ${hex(pos, 8)}`);
      if (metaSize === 0) { console.log('  (EOS)'); break; }
      if (metaSize < 0 || metaSize > 100000) {
        console.log('  ERROR: invalid metaSize!');
        console.log(`  Next 16 bytes after header: ${hex(pos + 8, 16)}`);
        break;
      }
      console.log(`  First 16 bytes of meta: ${hex(pos + 8, 16)}`);
      const padded = (metaSize + 7) & ~7;
      pos += 8 + padded;
      // Print how many body bytes follow (based on size alone)
      const remaining = buf.byteLength - pos;
      console.log(`  After meta: ${remaining} bytes remaining`);
      msgNum++;
    }
    
    expect(buf.byteLength).toBeGreaterThan(50);
  });
});
