import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const gameJs = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'html', 'game.js'), 'utf8');

describe('old-shell widget lifecycle', () => {
  it('mounts newly displayed content after the engine has completed on-display', () => {
    const start = gameJs.indexOf('window.onDisplayContent = function () {');
    const end = gameJs.indexOf('\n  };', start);
    const body = gameJs.slice(start, end);

    expect(start).toBeGreaterThan(-1);
    expect(body).toMatch(/setTimeout\(function \(\) \{[\s\S]*mountWidgets\(document,/);
    expect(body.indexOf('setTimeout(')).toBeLessThan(body.indexOf('mountWidgets(document,'));
  });
});
