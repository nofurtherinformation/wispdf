/**
 * postinstall: copy databonk's wasm binaries to public/ so Vite's dev server and
 * build can serve them at the root path (e.g. /simd.wasm).
 *
 * In a production setup you would instead use Vite's assetsInclude config and
 * reference wasm via `new URL('../node_modules/databonk/dist/simd.wasm', import.meta.url)`
 * — Vite then copies and hashes the file automatically. See docs/bundlers.md.
 */
import { copyFileSync, mkdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dir = dirname(fileURLToPath(import.meta.url));
const databonkDist = join(__dir, '..', 'node_modules', 'databonk', 'dist');
const publicDir = join(__dir, '..', 'public');

mkdirSync(publicDir, { recursive: true });

const files = ['scalar.wasm', 'simd.wasm'];
let anyMissing = false;
for (const f of files) {
  const src = join(databonkDist, f);
  if (!existsSync(src)) {
    console.warn(`[copy-wasm] skipping ${f} — not found at ${src}`);
    anyMissing = true;
    continue;
  }
  copyFileSync(src, join(publicDir, f));
  console.log(`[copy-wasm] ${f} -> public/${f}`);
}
if (anyMissing) {
  console.warn('[copy-wasm] some wasm files were missing — run `npm run build` in the repo root first.');
}
