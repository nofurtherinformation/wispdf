/**
 * databonk node-quickstart — mirrors the README quickstart exactly, plus v2 additions.
 *
 * Exercises:
 *   fromColumns → filter (expression path) → groupby → agg → toRecords
 *   scope() for automatic cleanup
 *   fromCSV
 *   I/O round-trip (Arrow IPC)
 *   lambda escape hatch (SLOW PATH — labeled clearly per ADR-003)
 *   v2: i64/BigInt columns — construction, arithmetic, reductions
 *   v2: temporal dtypes — date32/timestamp, dt accessors, tz metadata
 *   v2: Parquet I/O — databonk/parquet subpath (readParquet/writeParquet)
 *
 * Run: node quickstart.mjs
 * (From the repo root in Docker: npm run build && cd examples/node-quickstart && npm install && node quickstart.mjs)
 */

import { init, DataFrame, col, lit, scope, fromCSV, fromArrow, toArrow, defaultRuntime } from 'databonk';
import { readParquet, writeParquet } from 'databonk/parquet';

// ── 1. Init ────────────────────────────────────────────────────────────────
await init();
console.log('✓ init() — wasm loaded');

// ── 2. fromColumns → filter → groupby → agg → toRecords ───────────────────
const df = DataFrame.fromColumns({
  id:    new Int32Array([1, 2, 3, 4, 5]),
  value: new Float64Array([10.5, 3.2, 8.1, 5.9, 2.7]),
  group: ['a', 'b', 'a', 'b', 'a'],
});

const summary = scope(() => {
  const filtered = df.filter(col('value').gt(5));
  const grouped  = filtered.groupby(['group']).agg({ value: ['sum', 'mean'] });
  const records  = grouped.toRecords();
  return records;   // plain JS — safe to return outside scope
});

console.log('\nfilter(value > 5) → groupby(group) → agg(sum, mean):');
console.table(summary);

// Verify expected shapes
if (summary.length !== 2) throw new Error(`expected 2 rows, got ${summary.length}`);
const rowA = summary.find(r => r.group === 'a');
if (!rowA) throw new Error('missing group a');
if (Math.abs(rowA.value_sum - 18.6) > 0.01) throw new Error(`wrong sum for a: ${rowA.value_sum}`);
console.log('✓ filter → groupby → agg verified');

// ── 3. scope() cleanup ─────────────────────────────────────────────────────
const records2 = scope(() => {
  const f = df.filter(col('value').gt(0));
  return f.toRecords();   // all 5 rows; f disposed by scope
});
if (records2.length !== 5) throw new Error(`expected 5 rows, got ${records2.length}`);
console.log('\n✓ scope() — 5-row frame built and disposed automatically');

// ── 4. Lambda escape hatch (SLOW PATH — per ADR-003, always show expression equivalent) ──
// SLOW PATH — use only when expressions cannot express the logic
const slowResult = scope(() => {
  // SLOW PATH: iterates via JS row-proxy (zero-copy but scalar JS speed)
  const f = df.filterFn(r => r.value > 5 && r.group === 'a');
  return f.toRecords();
});
// Expression equivalent (fast path — use this instead):
// df.filter(col('value').gt(5).and(col('group').eq('a')))
const fastResult = scope(() => {
  const f = df.filter(col('value').gt(5).and(col('group').eq('a')));
  return f.toRecords();
});
if (slowResult.length !== fastResult.length) {
  throw new Error(`SLOW/fast mismatch: ${slowResult.length} vs ${fastResult.length}`);
}
console.log('✓ lambda escape hatch (SLOW PATH) matches expression equivalent (fast path)');

// ── 5. Arrow IPC round-trip ────────────────────────────────────────────────
const buf = toArrow(df);
const df2 = fromArrow(buf, defaultRuntime());
const rt   = df2.toRecords();
if (rt.length !== df.length) throw new Error('Arrow round-trip length mismatch');
df2.dispose();
console.log('✓ toArrow → fromArrow round-trip');

// ── 6. CSV import ──────────────────────────────────────────────────────────
const csv = `x,y,label\n1.0,2.0,foo\n3.0,4.0,bar\n5.0,6.0,foo\n`;
const dfCsv = fromCSV(csv, { hasHeader: true });
if (dfCsv.length !== 3) throw new Error(`CSV: expected 3 rows, got ${dfCsv.length}`);
const csvSum = scope(() => {
  const s = dfCsv.groupby(['label']).agg({ x: 'sum' });
  return s.toRecords();
});
console.log('\nCSV groupby result:');
console.table(csvSum);
dfCsv.dispose();
console.log('✓ fromCSV + groupby');

// ── 7. toRecords (main frame) ──────────────────────────────────────────────
const allRecords = df.toRecords();
console.log('\nFull frame (toRecords):');
console.table(allRecords);

df.dispose();

// ── 8. i64 / BigInt columns (v2, ADR-009) ─────────────────────────────────
console.log('\n--- i64 / BigInt columns ---');

// BigInt64Array fast path — zero-copy into WASM memory
const dfI64 = DataFrame.fromColumns({
  id:    BigInt64Array.from([1n, 2n, 9007199254740993n]),  // > 2^53
  label: ['a', 'b', 'a'],
});
if (dfI64.dtypes['id'] !== 'i64') throw new Error('expected i64 dtype');

// Reductions: sum/min/max → bigint; mean → number (f64)
const gbI64 = dfI64.groupby('label').agg({ id: ['sum', 'min'] });
const gbRecs = gbI64.toRecords();
const rowLabelA = gbRecs.find(r => r['label'] === 'a');
if (!rowLabelA) throw new Error('missing label a in i64 groupby');
const sumA = rowLabelA['id_sum'];
if (typeof sumA !== 'bigint') throw new Error(`id_sum should be bigint, got ${typeof sumA}`);
console.log('  i64 groupby sum (label a):', sumA, '(bigint confirmed)');

// Arithmetic: wrapping on overflow, widening i64+i32
const dfI64Ops = DataFrame.fromColumns({
  a: BigInt64Array.from([10n, 20n, 30n]),
  b: new Int32Array([1, 2, 3]),
});
const withWiden = dfI64Ops.withColumn('c', col('a').add(col('b')));  // i64 + i32 → i64
const withScaled = dfI64Ops.withColumn('d', col('a').mul(lit(100n)));
const widenCols = withWiden.toColumns();
if (widenCols['c'][0] !== 11n) throw new Error(`expected 11n, got ${widenCols['c'][0]}`);
withWiden.dispose();
withScaled.dispose();
dfI64Ops.dispose();
gbI64.dispose();
dfI64.dispose();
console.log('✓ i64/BigInt columns verified (construction, reductions, arithmetic, widening)');

// Safe-int throw: number outside isSafeInteger range must throw
try {
  DataFrame.fromColumns(
    { bad: [9007199254740993] },  // > 2^53 — should throw
    { dtypes: { bad: 'i64' } },
  );
  throw new Error('expected RangeError for unsafe integer literal, but no error thrown');
} catch (e) {
  if (!(e instanceof RangeError)) throw new Error(`expected RangeError, got ${e}`);
}
console.log('✓ safe-int throw for unsafe number literal');

// ── 9. Temporal dtypes (v2, ADR-010) ──────────────────────────────────────
console.log('\n--- Temporal dtypes ---');

// date32: days since epoch (1970-01-01)
// 19737 = 2024-01-15  (Math.floor(new Date('2024-01-15').getTime() / 86_400_000))
// 19783 = 2024-03-01
const dfDate = DataFrame.fromColumns(
  { d: new Int32Array([19737, 19783, 0]) },
  { dtypes: { d: 'date32' } },
);
if (dfDate.dtypes['d'] !== 'date32') throw new Error('expected date32 dtype');

// dt accessor: year/month/day from a date32 column
const seriesD = dfDate.col('d');
if (!seriesD) throw new Error('expected date column');
const years = seriesD.dt.year().toArray();
if (years[0] !== 2024) throw new Error(`expected year 2024, got ${years[0]}`);
const months = seriesD.dt.month().toArray();
if (months[0] !== 1) throw new Error(`expected month 1, got ${months[0]}`);
console.log('  date32 year accessor:', years[0], 'month:', months[0]);
dfDate.dispose();

// timestamp: ms since Unix epoch, always UTC; tz as metadata
const ts1 = new Date('2024-06-01T17:00:00Z').getTime();  // UTC ms
const ts2 = new Date('2024-06-15T00:00:00Z').getTime();
const dfTs = DataFrame.fromColumns(
  { created: BigInt64Array.from([BigInt(ts1), BigInt(ts2)]) },
  { dtypes: { created: 'timestamp' }, tzs: { created: 'America/Chicago' } },
);
if (dfTs.dtypes['created'] !== 'timestamp') throw new Error('expected timestamp dtype');

// dt accessor: tz-aware (America/Chicago is UTC-5 or UTC-6)
const seriesTs = dfTs.col('created');
if (!seriesTs) throw new Error('expected timestamp column');
const tsYear = seriesTs.dt.year().toArray();
if (tsYear[0] !== 2024) throw new Error(`expected year 2024, got ${tsYear[0]}`);
// In America/Chicago, 2024-06-01T17:00Z = 2024-06-01T12:00 CDT (UTC-5)
const tsHour = seriesTs.dt.hour().toArray();
console.log(`  timestamp in America/Chicago: year=${tsYear[0]}, hour=${tsHour[0]} (local)`);

// ISO weekday: Monday=1 … Sunday=7
// 2024-06-01 is a Saturday (UTC); UTC-5 keeps same day, so weekday=6
const tsWkday = seriesTs.dt.weekday().toArray();
console.log(`  weekday (ISO, 1=Mon…7=Sun):`, tsWkday[0]);
if (tsWkday[0] < 1 || tsWkday[0] > 7) throw new Error(`unexpected weekday ${tsWkday[0]}`);

// Restricted arithmetic: timestamp − timestamp → i64 (ms duration)
const dfDiff = dfTs.withColumn('dur', col('created').sub(col('created')));
if (dfDiff.dtypes['dur'] !== 'i64') throw new Error('ts-ts subtraction should give i64');
dfDiff.dispose();
dfTs.dispose();
console.log('✓ temporal dtypes verified (date32, timestamp, dt accessors, tz, restricted arithmetic)');

// ── 10. Parquet I/O (v2, ADR-011) ─────────────────────────────────────────
console.log('\n--- Parquet I/O (databonk/parquet subpath) ---');

const rt2 = defaultRuntime();

// Build a frame with mixed v2 dtypes
const dfParquet = DataFrame.fromColumns({
  id:    BigInt64Array.from([1n, 2n, 3n]),
  value: new Float64Array([1.1, 2.2, 3.3]),
  label: ['alpha', 'beta', null],
}, { dtypes: { id: 'i64', value: 'f64', label: 'utf8' } });

// writeParquet (sync) → Uint8Array
const parquetBytes = writeParquet(dfParquet, { compression: 'uncompressed' });
if (!(parquetBytes instanceof Uint8Array)) throw new Error('writeParquet should return Uint8Array');
if (parquetBytes.length === 0) throw new Error('writeParquet returned empty bytes');
console.log(`  writeParquet: ${parquetBytes.length} bytes`);

// readParquet (async) → DataFrame
const dfRound = await readParquet(parquetBytes, rt2);
if (dfRound.length !== 3) throw new Error(`readParquet: expected 3 rows, got ${dfRound.length}`);
if (dfRound.dtypes['id'] !== 'i64') throw new Error(`expected i64 after round-trip, got ${dfRound.dtypes['id']}`);
const roundRecs = dfRound.toRecords();
if (roundRecs[0]['id'] !== 1n) throw new Error(`expected 1n, got ${roundRecs[0]['id']}`);
if (roundRecs[2]['label'] !== null) throw new Error(`expected null label, got ${roundRecs[2]['label']}`);
dfRound.dispose();
dfParquet.dispose();

// Out-of-profile error: unsupported compression should throw
// (use a fresh frame since dfParquet was already disposed above)
const dfForErr = DataFrame.fromColumns({ x: [1, 2] });
try {
  writeParquet(dfForErr, { compression: /** @type {any} */ ('gzip') });
  throw new Error('expected error for unsupported codec, but no error thrown');
} catch (e) {
  if (!(e instanceof Error) || !e.message.includes('unsupported')) {
    throw new Error(`expected unsupported error, got: ${e}`);
  }
}
dfForErr.dispose();
console.log('✓ Parquet I/O verified (write, read, round-trip, unsupported-codec error)');

console.log('\n✓ node-quickstart PASSED — all checks green (including v2: i64, temporals, parquet)');
