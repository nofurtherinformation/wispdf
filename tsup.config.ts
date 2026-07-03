import { defineConfig } from 'tsup';

export default defineConfig({
  // workers is a separate entry (subpath export "wispdf/workers") so the opt-in
  // parallel mode never counts against the 25 KB main-entry size gate (spec §1).
  entry: { index: 'src/index.ts', workers: 'src/workers/index.ts' },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: false,
  treeshake: true,
  minify: true,
  outDir: 'dist',
});
