# WASM loading — Vite / webpack / Node.js

databonk ships three `.wasm` binaries inside the `dist/` directory of the published
package:

| File | When used |
|---|---|
| `dist/simd.wasm` | SIMD-capable engines (Chrome ≥ 91, Firefox ≥ 89, Safari ≥ 16.4, Node ≥ 16) |
| `dist/scalar.wasm` | Fallback for older Safari and any SIMD-incapable engine |
| `dist/simd-threads.wasm` | Opt-in parallel mode only (`enableThreads()`); requires SAB + COOP/COEP |

The JS entry feature-detects SIMD and loads the right binary. The loader needs
to locate the `.wasm` file at runtime. How you provide that URL depends on your
toolchain.

---

## Node.js (≥ 18)

No configuration required. The loader resolves the wasm path relative to the
installed package directory using `import.meta.url`. After `npm install`:

```typescript
import { init, DataFrame, col } from 'databonk';

await init();   // auto-detects SIMD, loads the right wasm
const df = DataFrame.fromColumns({ x: [1, 2, 3] });
console.log(df.toString());
```

---

## Vite (≥ 5)

Vite handles `.wasm` assets automatically when you use the `?url` query or the
`assetsInclude` pattern. The easiest approach is to let Vite inline-process the
binary:

```typescript
// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  assetsInclude: ['**/*.wasm'],   // treat .wasm as a URL asset (copies to output)
  optimizeDeps: {
    exclude: ['databonk'],           // do not pre-bundle — databonk uses dynamic import()
  },
});
```

Then in your app:

```typescript
import { init, DataFrame, col } from 'databonk';
await init();
```

The loader resolves the `.wasm` URL via `new URL('./scalar.wasm', import.meta.url)`
which Vite handles correctly. If you see a "Failed to fetch" error, confirm:

1. `assetsInclude` covers `.wasm` (or use the `vite-plugin-wasm` plugin).
2. `optimizeDeps.exclude` lists `'databonk'`.

### Inline base64 fallback (no asset pipeline)

If your Vite config cannot serve `.wasm` assets (e.g. a monorepo where the
`dist/` is not under the public root), you can pass the wasm binary as a
`Uint8Array` directly:

```typescript
import { useRuntime, runtimeFromExports } from 'databonk';
import simdWasmBytes from 'databonk/dist/simd.wasm?arraybuffer'; // Vite raw import

const { instance } = await WebAssembly.instantiate(simdWasmBytes, {});
useRuntime(runtimeFromExports(instance.exports, true /* isSIMD */));
```

---

## webpack (≥ 5)

webpack 5 has built-in async WASM support. Enable it in your config:

```javascript
// webpack.config.js
module.exports = {
  experiments: {
    asyncWebAssembly: true,
  },
  module: {
    rules: [
      {
        test: /\.wasm$/,
        type: 'asset/resource',   // copy wasm to output directory
      },
    ],
  },
};
```

Then import and initialise as usual:

```typescript
import { init, DataFrame } from 'databonk';
await init();
```

### Inline base64 fallback for webpack

If asset serving is unavailable, use `asset/inline` to embed the binary:

```javascript
{ test: /\.wasm$/, type: 'asset/inline' }
```

The loader accepts a `Uint8Array` directly — see the Vite inline example above.

---

## Threads / COOP + COEP

The opt-in parallel mode (`enableThreads()`) uses `SharedArrayBuffer` and
requires cross-origin isolation. See [docs/threads.md](threads.md) for full
setup instructions for Vite, webpack, nginx, and Cloudflare Pages.

---

## Bundle size reference

Sizes gzipped, measured 2026-07-02 on the 0.1.0 build (Node 22, Apple M-series):

| Asset | Gzipped |
|---|---|
| `dist/index.js` (ESM) | 22.4 KB |
| `dist/index.cjs` (CJS) | 22.7 KB |
| `dist/simd.wasm` | 18.6 KB |
| `dist/scalar.wasm` | 15.0 KB |
| `dist/simd-threads.wasm` | 18.4 KB |
