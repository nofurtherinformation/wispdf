/**
 * End-to-end pipeline integration tests: chained ops mirroring the spec §5 gate
 * (filter → groupby → sum) plus a mixed utf8 pipeline and the expression fast path
 * composed with slicing / sorting.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { loadRuntimeForTest, makeDF } from '../frame/helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import { col } from '../../src/expr/index.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

describe('filter → groupby → sum', () => {
  it('matches a naive JS reference (utf8 group key)', () => {
    const n = 400;
    const a = new Float64Array(n);
    const g: string[] = [];
    for (let i = 0; i < n; i++) {
      a[i] = ((i * 2654435761) % 1000) / 1000; // deterministic in [0,1)
      g.push('g' + (i % 7));
    }
    const df = makeDF(rt, { a, g }, { a: 'f64', g: 'utf8' });

    const result = df
      .filter(col('a').gt(0.5))
      .groupby('g')
      .agg({ a: 'sum' });

    // JS reference
    const ref = new Map<string, number>();
    for (let i = 0; i < n; i++) {
      if (a[i]! > 0.5) ref.set(g[i]!, (ref.get(g[i]!) ?? 0) + a[i]!);
    }
    for (const rec of result.toRecords()) {
      expect(rec.a as number).toBeCloseTo(ref.get(rec.g as string)!, 9);
    }
    expect(result.shape[0]).toBe(ref.size);
    result.dispose(); df.dispose();
  });
});

describe('chained pipeline', () => {
  it('withColumn → filter → sortValues → head', () => {
    const df = makeDF(
      rt,
      { x: [5, 3, 9, 1, 7, 2], label: ['e', 'c', 'i', 'a', 'g', 'b'] },
      { x: 'i32', label: 'utf8' },
    );
    const out = df
      .withColumn('x2', col('x').mul(2))
      .filter(col('x').gt(1))
      .sortValues('x')
      .head(3);
    expect(out.toRecords()).toEqual([
      { x: 2, label: 'b', x2: 4 },
      { x: 3, label: 'c', x2: 6 },
      { x: 5, label: 'e', x2: 10 },
    ]);
    out.dispose(); df.dispose();
  });

  it('join then groupby', () => {
    const left = makeDF(rt, { k: ['a', 'b', 'a', 'b'], v: [1, 2, 3, 4] }, { k: 'utf8', v: 'f64' });
    const right = makeDF(rt, { k: ['a', 'b'], w: [10, 20] }, { k: 'utf8', w: 'f64' });
    const joined = left.join(right, { on: 'k' });
    const rolled = joined.groupby('k').agg({ v: 'sum', w: 'first' });
    const recs = rolled.toRecords().sort((p, q) => (p.k! < q.k! ? -1 : 1));
    expect(recs).toEqual([
      { k: 'a', v: 4, w: 10 },
      { k: 'b', v: 6, w: 20 },
    ]);
    rolled.dispose(); joined.dispose(); left.dispose(); right.dispose();
  });
});
