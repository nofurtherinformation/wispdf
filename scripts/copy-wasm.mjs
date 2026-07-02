#!/usr/bin/env node
/**
 * Copy the built wasm binaries into dist/ after tsup (which cleans dist/).
 *
 * The binaries are produced by wasm/rust/build.sh into wasm/dist/. Placing them
 * next to the JS bundle lets the loader resolve them relative to import.meta.url
 * in the published package, and lets scripts/check-size.mjs apply the size gate
 * to the real binaries.
 */
import { copyFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const srcDir = join(root, 'wasm', 'dist');
const dstDir = join(root, 'dist');
const files = ['scalar.wasm', 'simd.wasm', 'simd-threads.wasm'];

mkdirSync(dstDir, { recursive: true });

for (const f of files) {
  const src = join(srcDir, f);
  if (!existsSync(src)) {
    console.error(
      `[copy-wasm] missing ${src} — run "npm run build:wasm" (bash wasm/rust/build.sh) first.`,
    );
    process.exit(1);
  }
  copyFileSync(src, join(dstDir, f));
  console.log(`[copy-wasm] ${f} -> dist/${f}`);
}
