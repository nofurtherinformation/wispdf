/**
 * Type + runtime surface tests for the public P3.2 API. The static assertions
 * (`expectTypeOf`) are checked by `vitest typecheck`; the runtime assertions run under the
 * normal suite. The authoritative type gate is the tsup `dts` build of `src/index.ts`.
 */

import { describe, it, expect, beforeAll, expectTypeOf } from 'vitest';
import { loadRuntimeForTest } from '../frame/helper.js';
import type { DfRuntime } from '../../src/frame/runtime.js';
import { DataFrame, GroupBy, Series, col } from '../../src/index.js';
import type { Cell } from '../../src/index.js';

let rt: DfRuntime;
beforeAll(async () => {
  rt = await loadRuntimeForTest();
});

describe('public API types', () => {
  it('construction + core ops keep the right static types', () => {
    const df = DataFrame.fromColumns({ a: [1, 2, 3], g: ['x', 'y', 'x'] }, { runtime: rt });

    expectTypeOf(df).toEqualTypeOf<DataFrame>();
    expectTypeOf(df.select(['a'])).toEqualTypeOf<DataFrame>();
    expectTypeOf(df.filter(col('a').gt(1))).toEqualTypeOf<DataFrame>();
    expectTypeOf(df.withColumn('b', col('a').mul(2))).toEqualTypeOf<DataFrame>();
    expectTypeOf(df.sortValues('a')).toEqualTypeOf<DataFrame>();
    expectTypeOf(df.head()).toEqualTypeOf<DataFrame>();
    expectTypeOf(df.groupby('g')).toEqualTypeOf<GroupBy>();
    expectTypeOf(df.groupby('g').agg({ a: 'sum' })).toEqualTypeOf<DataFrame>();
    expectTypeOf(df.col('a')).toEqualTypeOf<Series>();
    expectTypeOf(df.shape).toEqualTypeOf<readonly [number, number]>();
    expectTypeOf(df.toRecords()).toEqualTypeOf<Array<Record<string, Cell>>>();
    expectTypeOf(df.mapFn((r) => r.a as number)).toEqualTypeOf<number[]>();

    // runtime sanity
    expect(df.shape).toEqual([3, 2]);
    expect(df.col('a')).toBeInstanceOf(Series);
    df.dispose();
  });
});
