import { createRequire } from 'node:module';
import { describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);
const { auditCompiledGame } = require('../../vendor/dendrynexus-ten/lib/audit.js') as {
  auditCompiledGame(game: object, options: object): {
    violations: unknown[];
    unknownWidgets: string[];
    unknownDerives: string[];
  };
};

describe('compiled-game audit', () => {
  it('keeps marker validation active in scenes exempted from the browser-global rule', () => {
    const game = {
      scenes: {
        root: {
          onArrival: [{ $code: 'window.location.href' }],
          content: ['<div data-widget="missing" data-props=\'{"deriveFrom":"missing"}\'></div>'],
        },
      },
    };
    const result = auditCompiledGame(game, {
      allowBrowserGlobals: ['root'],
      widgetNames: ['known'],
      deriveNames: ['known'],
    });

    expect(result.violations).toEqual([]);
    expect(result.unknownWidgets).toEqual(['missing']);
    expect(result.unknownDerives).toEqual(['missing']);
    expect(auditCompiledGame(game, { widgetNames: ['known'] }).violations).toHaveLength(1);
  });
});
