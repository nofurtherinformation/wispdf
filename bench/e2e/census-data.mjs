/**
 * Synthetic census tract data generator for the databonk perf repro bench.
 *
 * Produces two Arrow IPC buffers that mimic the pathological real-census case:
 *   - names table: 85,395 rows × {GEOID: utf8 (PLAIN), name: utf8 (PLAIN)}
 *   - pops  table: same 85,395 GEOIDs (shuffled order) × 72 numeric cols (mix f64/i32)
 *
 * GEOIDs: '1400000US' + 2-digit state (from 52 weighted codes) + 3-digit county
 *         + 6-digit tract.  All 85,395 GEOIDs are unique (one per census tract).
 *
 * Arrow encoding: utf8 columns use plain (non-dictionary) Utf8 encoding — this is
 * what real ACS/census Arrow files look like and is the pathological case for
 * databonk's fromArrow(), which must build a dictionary from ~85K unique strings.
 *
 * Numeric columns: 36 f64 + 36 i32, columns 0,7,14,21,28,35,42,49 get ~2% nulls.
 * Seeded with mulberry32 so the output is fully reproducible.
 */

import { tableFromArrays, tableToIPC, vectorFromArray, makeVector, Float64, Int32, Utf8 } from 'apache-arrow';
import { mulberry32 } from '../datasets.mjs';

// ─── Census geography constants ──────────────────────────────────────────────

/**
 * 52 US state/territory FIPS codes with rough tract-count weights.
 * Weights are proportional to real ACS tract counts (California ~8800 tracts, etc.)
 * Total 85,395 tracts.
 */
const STATE_FIPS = [
  '01', '02', '04', '05', '06', '08', '09', '10', '11', '12',
  '13', '15', '16', '17', '18', '19', '20', '21', '22', '23',
  '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
  '34', '35', '36', '37', '38', '39', '40', '41', '42', '44',
  '45', '46', '47', '48', '49', '50', '51', '53', '54', '55',
  '56', '72',
];

/**
 * Approximate real ACS 2020 tract counts per state (same order as STATE_FIPS).
 * Raw weights are scaled to sum to exactly 85,395.
 *
 * Raw proportions based on Census 2020 5-year ACS tract counts; last state absorbs rounding.
 */
const STATE_WEIGHTS_RAW = [
  1182, 167, 1526, 686, 8884, 1249, 874, 218, 179, 4245,
  1969, 351, 298, 3124, 1511, 825, 861, 1115, 1148, 358,
  1406, 1478, 2813, 1338, 664, 1393, 281, 532, 686, 259,
  2482, 499, 4919, 1896, 209, 2952, 1046, 834, 2939, 253,
  1103, 167, 1473, 4935, 588, 255, 1907, 1458, 484, 1368,
  135, 2404,
];

const TARGET_ROWS = 85_395;
const RAW_SUM = STATE_WEIGHTS_RAW.reduce((a, b) => a + b, 0);

// Scale weights so floor-sum is near TARGET, then fix the remainder in the last element.
const STATE_WEIGHTS = STATE_WEIGHTS_RAW.map(w => Math.floor(w * TARGET_ROWS / RAW_SUM));
const scaledSum = STATE_WEIGHTS.reduce((a, b) => a + b, 0);
STATE_WEIGHTS[STATE_WEIGHTS.length - 1] += TARGET_ROWS - scaledSum;

const TOTAL_ROWS = STATE_WEIGHTS.reduce((a, b) => a + b, 0);
if (TOTAL_ROWS !== TARGET_ROWS) {
  throw new Error(`STATE_WEIGHTS sum = ${TOTAL_ROWS}, expected ${TARGET_ROWS}`);
}

// ─── State/county/tract names for readable GEOID strings ─────────────────────

const STATE_NAMES = [
  'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
  'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia',
  'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
  'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
  'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
  'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
  'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
  'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
  'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Puerto Rico',
];

// ─── PRNG ─────────────────────────────────────────────────────────────────────

const rng = mulberry32(0xdeadcafe);

// ─── Generate unique GEOIDs ───────────────────────────────────────────────────

/**
 * Build 85,395 unique census-tract GEOIDs in STATE_WEIGHTS order.
 * Within each state, counties are allocated in blocks of ~8 tracts.
 * County codes: 001, 003, 005, ... (odd codes, matching real FIPS).
 * Tract codes: 000100–999900 step 100 (6-digit, matching real ACS format).
 */
function buildGeoIds() {
  const geoids = new Array(TOTAL_ROWS);
  let row = 0;

  for (let si = 0; si < STATE_FIPS.length; si++) {
    const stateFips = STATE_FIPS[si];
    const stateCount = STATE_WEIGHTS[si];

    // Distribute tracts into counties: ~8 tracts per county
    const tractsPerCounty = 8;
    let remaining = stateCount;
    let county = 1; // county FIPS start at 001

    while (remaining > 0) {
      const countyCount = Math.min(tractsPerCounty, remaining);
      const countyCode = String(county).padStart(3, '0');
      for (let t = 0; t < countyCount; t++) {
        const tractNum = (t + 1) * 100; // 100, 200, ... up to 99900
        const tractCode = String(tractNum).padStart(6, '0');
        geoids[row++] = `1400000US${stateFips}${countyCode}${tractCode}`;
      }
      county += 2; // odd county codes (001, 003, 005, ...)
      remaining -= countyCount;
    }
  }

  return geoids;
}

/**
 * Build human-readable "Census Tract X.YY, County Name, State" strings.
 * Each tract gets a unique float tract number like 101.00, 201.00, etc.
 */
function buildNames(geoids) {
  const COUNTIES = [
    'Jefferson', 'Washington', 'Franklin', 'Lincoln', 'Madison',
    'Jackson', 'Monroe', 'Adams', 'Harrison', 'Polk',
  ];

  return geoids.map((geoid, i) => {
    const siIdx = Math.floor(i / 8) % STATE_NAMES.length;
    const stateAbbr = STATE_NAMES[siIdx % STATE_NAMES.length];
    const countyName = COUNTIES[Math.floor(i / 8) % COUNTIES.length];
    // Tract number: base on last 6 chars of GEOID, convert to float like "101.00"
    const tractDigits = parseInt(geoid.slice(-6), 10);
    const tractNum = (tractDigits / 100).toFixed(2);
    return `Census Tract ${tractNum}, ${countyName} County, ${stateAbbr}`;
  });
}

// ─── Build numeric columns ─────────────────────────────────────────────────

const NULL_RATE = 0.02; // ~2% nulls in some columns
const NULL_COL_INDICES = new Set([0, 7, 14, 21, 28, 35, 42, 49]); // 8 of 72 cols get nulls

/**
 * Generate 72 numeric columns: 36 f64 + 36 i32.
 * Columns at NULL_COL_INDICES indices get ~2% nulls (null represented as null).
 * Returns Map<colName, Array<number|null>>
 */
function buildNumericCols(n) {
  const cols = new Map();
  for (let ci = 0; ci < 72; ci++) {
    const isF64 = ci < 36;
    const hasNulls = NULL_COL_INDICES.has(ci);
    const arr = new Array(n);
    for (let i = 0; i < n; i++) {
      if (hasNulls && rng() < NULL_RATE) {
        arr[i] = null;
        rng(); // consume value slot
      } else {
        arr[i] = isF64 ? rng() * 100000 : Math.floor(rng() * 10000);
      }
    }
    const name = isF64 ? `pop_f${ci}` : `pop_i${ci - 36}`;
    cols.set(name, arr);
  }
  return cols;
}

// ─── Arrow IPC encoding (PLAIN utf8, not dictionary) ────────────────────────

/**
 * Encode a plain Utf8 column as Arrow IPC bytes.
 * We use apache-arrow's vectorFromArray which produces Utf8 (non-dict) when
 * passed a string array with Arrow Utf8 type — this is the pathological case
 * since real census Arrow files come as plain utf8, not Dict<Int32,Utf8>.
 */
function makeUtf8Vector(strings) {
  return vectorFromArray(strings, new Utf8());
}

/**
 * Build Arrow IPC buffer for the names table:
 *   Schema: GEOID (Utf8), name (Utf8)
 */
export function buildNamesArrow(geoids, names) {
  const table = tableFromArrays({
    GEOID: makeUtf8Vector(geoids),
    name:  makeUtf8Vector(names),
  });
  return tableToIPC(table, 'stream');
}

/**
 * Build Arrow IPC buffer for the pops table:
 *   Schema: GEOID (Utf8) + 72 numeric cols (f64/i32)
 */
export function buildPopsArrow(geoids, numericCols) {
  const arrays = { GEOID: makeUtf8Vector(geoids) };
  for (const [name, vals] of numericCols) {
    const isF64 = name.startsWith('pop_f');
    arrays[name] = isF64
      ? vectorFromArray(vals, new Float64())
      : vectorFromArray(vals, new Int32());
  }
  const table = tableFromArrays(arrays);
  return tableToIPC(table, 'stream');
}

// ─── Fisher-Yates shuffle ─────────────────────────────────────────────────────

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ─── Public entry point ──────────────────────────────────────────────────────

/**
 * Generate both Arrow IPC buffers.
 * Returns { namesBuf: Uint8Array, popsBuf: Uint8Array, popColNames: string[] }
 */
export function generateCensusData() {
  const geoids = buildGeoIds();
  const names = buildNames(geoids);

  // pops table uses same GEOIDs in shuffled order
  const shuffledGeoids = shuffle([...geoids]);
  const numericCols = buildNumericCols(TOTAL_ROWS);

  const namesBuf = buildNamesArrow(geoids, names);
  const popsBuf  = buildPopsArrow(shuffledGeoids, numericCols);

  const popColNames = [...numericCols.keys()];

  return { namesBuf, popsBuf, popColNames };
}
