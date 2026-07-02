/**
 * skidi node-quickstart — mirrors the README quickstart exactly.
 *
 * Exercises:
 *   fromColumns → filter (expression path) → groupby → agg → toRecords
 *   scope() for automatic cleanup
 *   fromCSV
 *   I/O round-trip (Arrow IPC)
 *   lambda escape hatch (SLOW PATH — labeled clearly per ADR-003)
 *
 * Run: node quickstart.mjs
 * (From the repo root in Docker: npm run build && cd examples/node-quickstart && npm install && node quickstart.mjs)
 */

import { init, DataFrame, col, scope, fromCSV, fromArrow, toArrow, defaultRuntime } from 'skidi';

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

console.log('\n✓ node-quickstart PASSED — all checks green');
