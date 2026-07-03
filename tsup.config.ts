import { defineConfig } from 'tsup';

export default defineConfig({
  // workers is a separate entry (subpath export "databonk/workers") so the opt-in
  // parallel mode never counts against the 25 KB main-entry size gate (spec §1).
  // parquet is a separate entry (subpath export "databonk/parquet", ADR-011) so the
  // hyparquet/hyparquet-writer runtime deps stay out of the main bundle entirely.
  entry: { index: 'src/index.ts', workers: 'src/workers/index.ts', parquet: 'src/parquet/index.ts' },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: false,
  treeshake: true,
  minify: true,
  outDir: 'dist',
});
