import { describe, it, expect } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';
import { miniGame } from './fixtures/mini-game';

// The compiler leaves `game.data` undefined when source/data/ has no
// registries (compiler.data-registry.test.ts). DendryAdapter.data/.glossary
// must survive that — every existing fixture (miniGame included) predates
// this task and carries no `data` key at all.
describe('DendryAdapter.data / .glossary', () => {
  it('is an empty object/array when the compiled game carries no data registries', () => {
    const adapter = DendryAdapter.fromJSONText(JSON.stringify(miniGame));
    expect(adapter.data).toEqual({});
    expect(adapter.glossary).toEqual([]);
  });

  it('exposes game.json.data.glossary.terms verbatim once compiled in', () => {
    const withData = {
      ...miniGame,
      data: { glossary: { terms: [{ id: 'ciu', match: ['CiU'], colour: 'ciu' }] } },
    };
    const adapter = DendryAdapter.fromJSONText(JSON.stringify(withData));
    expect(adapter.data.glossary).toEqual({
      terms: [{ id: 'ciu', match: ['CiU'], colour: 'ciu' }],
    });
    expect(adapter.glossary).toEqual([{ id: 'ciu', match: ['CiU'], colour: 'ciu' }]);
  });

  it('other data registries survive too — .data is not glossary-only', () => {
    const withData = { ...miniGame, data: { achievements: { list: [1, 2, 3] } } };
    const adapter = DendryAdapter.fromJSONText(JSON.stringify(withData));
    expect(adapter.data.achievements).toEqual({ list: [1, 2, 3] });
    expect(adapter.glossary).toEqual([]); // no glossary registry present
  });
});
