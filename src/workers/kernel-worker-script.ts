/**
 * Inline kernel-worker script (ADR-006 / P5.1).
 *
 * This string is passed to `new Worker(script, { eval: true })` in Node.js or
 * wrapped in a Blob URL for browser Workers. It must be self-contained CommonJS
 * (no ESM import/export) because Node eval workers run as CJS.
 *
 * Protocol:
 *   host → worker: { type: 'init',   requestId, bytes: Uint8Array, memory: WebAssembly.Memory }
 *   host → worker: { type: 'kernel', requestId, fn: string, args: number[] }
 *   host → worker: { type: 'meta',   requestId, ops: Array<{ fn: string; args: number[] }> }
 *   worker → host: { type: 'ready',  requestId }
 *   worker → host: { type: 'result', requestId, value: number }           (kernel)
 *   worker → host: { type: 'values', requestId, values: number[] }        (meta)
 *   worker → host: { type: 'error',  requestId, error: string }
 *
 * Invariant: workers NEVER call alloc/free/realloc (ADR-006).  They only call
 * stateless kernel functions on caller-provided pointer ranges.  All allocation
 * is done by the main thread before dispatching work.
 */

export const KERNEL_WORKER_SCRIPT: string = /* js */ `
;(function workerMain() {
  var isNode = (
    typeof process !== 'undefined' &&
    typeof process.versions === 'object' &&
    typeof process.versions.node === 'string'
  );

  /* Capture 'require' via a typeof guard so that the literal text
   * 'require(' never appears in this file — packaging tools like publint
   * use a regex heuristic (require\\s*\\() to classify CJS vs ESM, and a
   * false-positive here would misclassify the ESM workers bundle.
   * The guard is safe: in a Node.js eval worker this is always a function;
   * in a browser worker it is undefined and the isNode guard short-circuits. */
  var _workerRequire = typeof require === 'function' ? require : null;

  var wasm = null; /* WebAssembly exports, set on 'init' */

  /* ------------------------------------------------------------------ */
  /* send a message back to the main thread                               */
  /* ------------------------------------------------------------------ */
  function send(msg) {
    if (isNode) {
      _workerRequire('worker_threads').parentPort.postMessage(msg);
    } else {
      /* global postMessage available in browser Worker scope */
      postMessage(msg);
    }
  }

  /* ------------------------------------------------------------------ */
  /* message handler                                                      */
  /* ------------------------------------------------------------------ */
  function handle(msg) {
    var type = msg.type;
    var requestId = msg.requestId;

    /* ---- init: receive wasm bytes + shared memory, instantiate ---- */
    if (type === 'init') {
      var bytes = msg.bytes;
      /* Accept Uint8Array or ArrayBuffer */
      var src = (bytes instanceof Uint8Array) ? bytes.buffer : bytes;
      WebAssembly
        .instantiate(src, { env: { memory: msg.memory } })
        .then(function(r) {
          wasm = r.instance.exports;
          /* Workers NEVER call __wasm_init_memory — only the main thread does.
           * All our static-mut arena variables are zero-initialised (BSS),
           * so the shared linear memory already contains the correct values
           * written by the main thread's instantiation. */
          send({ type: 'ready', requestId: requestId });
        })
        .catch(function(e) {
          send({ type: 'error', requestId: requestId, error: String(e) });
        });
      return;
    }

    if (!wasm) {
      send({ type: 'error', requestId: requestId, error: 'worker not initialised' });
      return;
    }

    /* ---- kernel: single wasm kernel call, returns one number ---- */
    if (type === 'kernel') {
      try {
        var result = wasm[msg.fn].apply(null, msg.args);
        send({ type: 'result', requestId: requestId, value: +result });
      } catch(e) {
        send({ type: 'error', requestId: requestId, error: String(e) });
      }
      return;
    }

    /* ---- meta: batch of kernel calls, returns one number per op ---- */
    if (type === 'meta') {
      try {
        var ops = msg.ops;
        var values = new Array(ops.length);
        for (var i = 0; i < ops.length; i++) {
          values[i] = +wasm[ops[i].fn].apply(null, ops[i].args);
        }
        send({ type: 'values', requestId: requestId, values: values });
      } catch(e) {
        send({ type: 'error', requestId: requestId, error: String(e) });
      }
      return;
    }
  }

  /* ------------------------------------------------------------------ */
  /* wire up message listener                                             */
  /* ------------------------------------------------------------------ */
  if (isNode) {
    _workerRequire('worker_threads').parentPort.on('message', handle);
  } else {
    /* browser Web Worker */
    self.onmessage = function(e) { handle(e.data); };
  }
})();
`;
