/** Frame-layer errors (spec §4): unknown column suggests the nearest name (shared
 * Levenshtein), dtype mismatch names both dtypes + the op. FrameError extends Error. */

import type { DType } from '../memory/dtype.js';
import { nearest } from '../expr/index.js';

export class FrameError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'FrameError';
  }
}

export function unknownColumn(name: string, known: readonly string[]): FrameError {
  const near = nearest(name, known);
  const suffix = near ? ` Did you mean '${near}'?` : '';
  const list = known.length ? ` Available columns: ${known.map((k) => `'${k}'`).join(', ')}.` : '';
  return new FrameError(`unknown column '${name}'.${suffix}${list}`);
}

export function dtypeMismatch(op: string, left: DType, right: DType, hint?: string): FrameError {
  const tail = hint ? ` ${hint}` : '';
  return new FrameError(`dtype mismatch in '${op}': ${left} vs ${right}.${tail}`);
}

export function unsupportedDtype(op: string, dtype: DType, hint?: string): FrameError {
  const tail = hint ? ` ${hint}` : '';
  return new FrameError(`operation '${op}' is not supported for dtype ${dtype}.${tail}`);
}
