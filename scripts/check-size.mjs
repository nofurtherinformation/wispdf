#!/usr/bin/env node
/**
 * Size gate per spec §5:
 *   - Gzipped JS entry (dist/index.js or dist/index.cjs, whichever is larger) must be ≤ 30 KB (ADR-012)
 *   - Any dist/*.wasm must be ≤ 75 KB gzipped
 *
 * Exits 1 if any limit is exceeded.
 */

import { readFileSync, readdirSync } from 'fs';
import { resolve, join } from 'path';
import { gzipSync } from 'zlib';

const DIST = resolve(process.cwd(), 'dist');
const JS_LIMIT_BYTES = 30 * 1024;   // 30 KB (v2 surface, ADR-012; was 25 KB for the v1 profile)
const WASM_LIMIT_BYTES = 75 * 1024; // 75 KB

let failed = false;

function gzippedSize(filePath) {
  const content = readFileSync(filePath);
  return gzipSync(content).length;
}

function fmt(bytes) {
  return `${(bytes / 1024).toFixed(2)} KB`;
}

// Check JS entry points
const jsEntries = ['index.js', 'index.cjs'];
for (const entry of jsEntries) {
  const filePath = join(DIST, entry);
  let size;
  try {
    size = gzippedSize(filePath);
  } catch {
    // File may not exist (e.g. if only ESM build)
    continue;
  }
  const status = size <= JS_LIMIT_BYTES ? 'OK' : 'FAIL';
  console.log(`[size-gate] ${entry}: ${fmt(size)} gzipped (limit ${fmt(JS_LIMIT_BYTES)}) — ${status}`);
  if (size > JS_LIMIT_BYTES) {
    failed = true;
  }
}

// Check wasm files anywhere in dist/
let files;
try {
  files = readdirSync(DIST, { recursive: true });
} catch {
  files = [];
}

for (const file of files) {
  const name = typeof file === 'string' ? file : file.toString();
  if (!name.endsWith('.wasm')) continue;
  const filePath = join(DIST, name);
  const size = gzippedSize(filePath);
  const status = size <= WASM_LIMIT_BYTES ? 'OK' : 'FAIL';
  console.log(`[size-gate] ${name}: ${fmt(size)} gzipped (limit ${fmt(WASM_LIMIT_BYTES)}) — ${status}`);
  if (size > WASM_LIMIT_BYTES) {
    failed = true;
  }
}

if (failed) {
  console.error('[size-gate] FAILED: one or more files exceed size limits.');
  process.exit(1);
} else {
  console.log('[size-gate] All size checks passed.');
}
