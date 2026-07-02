import { describe, it, expect } from 'vitest';
import { hello, VERSION } from '../src/index.js';

describe('hello', () => {
  it('returns default greeting', () => {
    expect(hello()).toBe(`Hello, world! dataframe v${VERSION}`);
  });

  it('returns custom greeting', () => {
    expect(hello('dataframe')).toBe(`Hello, dataframe! dataframe v${VERSION}`);
  });

  it('VERSION is a semver string', () => {
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
