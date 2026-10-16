// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { createRenderer, convertLine } from 'dendrynexus-ten/lib/ui/content/html.js';
import { DendryAdapter } from '../src/engine/adapter';

describe('independent HTML renderers', () => {
  it('formats nested content in Node with independent callbacks and plain output', () => {
    const a = createRenderer({ formatText: (text) => text.replace('ERC', 'A') });
    const b = createRenderer({ formatText: (text) => text.replace('ERC', 'B') });
    const content = ['ERC ', { type: 'emphasis-1', content: ['ERC'] }];
    expect(a.convert([{ type: 'paragraph', content }])).toBe('<p>A <em>A</em></p>');
    expect(b.convertLine(content)).toBe('B <em>B</em>');
    expect(a.convertLine(content)).toBe('A <em>A</em>');
    expect(convertLine(content)).toBe('ERC <em>ERC</em>');
    expect(a.convertLine({ type: 'magic', content: '<b>ERC</b>' })).toBe('<b>ERC</b>');
    expect(typeof globalThis.window).toBe('undefined');
  });

  it('ignores an unrelated legacy global, including for plain exports', () => {
    Object.assign(globalThis, { window: { displayText: () => { throw new Error('global used'); } } });
    try {
      expect(convertLine('ERC')).toBe('ERC');
      expect(createRenderer({ formatText: (text) => text.toLowerCase() }).convertLine('ERC')).toBe('erc');
    } finally {
      delete (globalThis as { window?: unknown }).window;
    }
  });

  it('keeps adapter glossaries isolated through content, choices and saved-page restoration', () => {
    const make = (display: string) => DendryAdapter.fromJSONText(JSON.stringify({
      scenes: {
        root: { id: 'root', content: [{ type: 'paragraph', content: ['ERC'] }], options: [{ id: '@next' }] },
        next: { id: 'next', title: ['ERC'], content: [], options: [] },
      }, qualities: {}, qdisplays: {},
      data: {
        glossary: { terms: [{ id: 'erc', match: ['ERC'], display }] },
        localization: { registryFields: [{ registry: 'glossary', paths: ['terms.*.display'] }] },
      },
    }));
    const a = make('First');
    const b = make('Second');
    a.beginGame(); b.beginGame();
    expect(a.renderView('root')).toContain('>First</span>');
    expect(b.renderView('root')).toContain('>Second</span>');
    const saved = JSON.parse(JSON.stringify(a.engine.getExportableState()));
    a.engine.setState(saved);
    expect(a.renderView('root')).toContain('>First</span>');
    expect(a.beginGame().choices[0].title).toContain('>First</span>');
    const cached = a.glossary;
    expect(a.glossary).toBe(cached);
    a.refreshLocale('ca', { First: 'Primer' });
    expect(a.renderView('root')).toContain('>Primer</span>');
    expect(b.renderView('root')).toContain('>Second</span>');
    a.refreshLocale('en', {});
    expect(a.renderView('root')).toContain('>First</span>');
  });
});
