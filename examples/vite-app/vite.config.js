import { defineConfig } from 'vite';

export default defineConfig({
  // Prevent Vite from pre-bundling wispdf (it uses dynamic import() for wasm).
  optimizeDeps: {
    exclude: ['wispdf'],
  },
  // Treat .wasm files as URL assets (copies them to output, does NOT bundle them).
  assetsInclude: ['**/*.wasm'],
  build: {
    // Raise the chunk size warning limit; wispdf's combined JS+wasm is small but
    // the warning threshold is conservative by default.
    chunkSizeWarningLimit: 600,
    target: 'es2022',
  },
  server: {
    // Cross-origin isolation headers required for enableThreads() in the browser.
    // Omit if you don't use the parallel mode.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
