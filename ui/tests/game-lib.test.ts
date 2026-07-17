import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gameLib } from '../src/game-bindings';

describe('source/lib', () => {
  it('exports the two functions content calls as G.*', () => {
    expect(typeof gameLib.engineTick).toBe('function');
    expect(typeof gameLib.spaSupportInject).toBe('function');
  });

  it('is DOM-free — that is the only reason it can be shared', () => {
    const src = readFileSync(resolve(__dirname, '../../source/lib/cat_engine.js'), 'utf8');
    // `window.RTI_CAT_ENGINE` in the no-module fallback is the one allowed
    // mention; strip it before asserting.
    const body = src.replace(/window\.RTI_CAT_ENGINE/g, '');
    expect(body).not.toMatch(/\bdocument\./);
    expect(body).not.toMatch(/\bd3\./);
    expect(body).not.toMatch(/\bwindow\./);
  });
});
