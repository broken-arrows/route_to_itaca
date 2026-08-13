// @vitest-environment node
import { describe, expect, it } from 'vitest';
import config from '../vite.config';

describe('Vite development dependency interop', () => {
  it('pre-bundles every CommonJS engine module imported by the Desk', () => {
    const includes = config.optimizeDeps?.include ?? [];

    expect(includes).toContain('dendrynexus-ten/lib/engine.js');
    expect(includes).toContain('dendrynexus-ten/lib/ui/content/html.js');
    expect(includes).toContain('dendrynexus-ten/lib/persistence.js');
  });
});
