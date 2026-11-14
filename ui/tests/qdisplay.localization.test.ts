import { beforeAll, describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { extractFiles } from 'dendrynexus-ten/lib/localization-extract.js';
import { DendryAdapter } from '../src/engine/adapter';
import { parseQdisplayHtml } from '../src/components/brief/useBand';

const root = join(import.meta.dirname, '../..');
const displays = ['faction_strength', 'dissent', 'politics_trust', 'politics_trust_sentence', 'relationships', 'events2012'];
const sources = displays.map(id => ({
  name: join('source', 'qdisplays', `${id}.qdisplay.dry`),
  contents: readFileSync(join(root, 'source/qdisplays', `${id}.qdisplay.dry`), 'utf8'),
}));
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
let game: string;

beforeAll(async () => {
  game = await new Promise<string>((resolve, reject) => compileGame([
    ...sources,
    { name: join('source', 'scenes', 'root.scene.dry'), contents:
      'title: Root\non-arrival: {! Q.visits = (Q.visits || 0) + 1; Q.level = 35; !}\n\n' +
      'Strength: [+ level : faction_strength +]\n\nDissent: [+ level : dissent +]\n\n' +
      '[+ level : politics_trust_sentence +]\n\n- @root: Stay\n' },
  ], (error, compiled) => {
    if (error) return reject(error);
    convertGameToJSON(compiled, 0, (error2, text) => error2 ? reject(error2) : resolve(text));
  }));
});

describe('qdisplay localization through the shared engine', () => {
  it('extracts independent keys for identical English in different display contexts', async () => {
    const result = await extractFiles(sources, { catalogs: { en: {}, ca: {} } });
    expect(result.errors).toEqual([]);
    const keys = result.units.map((unit: { key: string }) => unit.key);
    const adapter = DendryAdapter.fromJSONText(game);
    adapter.beginGame([1, 2, 3, 4]);
    const strength = adapter.qdisplay(35, 'faction_strength');
    const dissent = adapter.qdisplay(35, 'dissent');
    expect(parseQdisplayHtml(strength).label).toBe(parseQdisplayHtml(dissent).label);
    expect(strength).not.toBe(dissent);
    expect(keys).toContain(strength);
    expect(keys).toContain(dissent);
  });

  it('renders distinct agreement in scenes and direct calls, retaining captured values across saves and locale cycles', () => {
    const adapter = DendryAdapter.fromJSONText(game);
    const english = adapter.beginGame([1, 2, 3, 4]);
    const strength = adapter.qdisplay(35, 'faction_strength');
    const dissent = adapter.qdisplay(35, 'dissent');
    const catalog = {
      [strength]: strength.replace('>high<', '>alta<'),
      [dissent]: dissent.replace('>high<', '>alt<'),
      'Strength: {level}': 'Força: {level}',
      'Dissent: {level}': 'Dissens: {level}',
    };
    // Historical text must keep its original band even after the live value changes.
    adapter.qualities.level = 0;
    const saved = clone(adapter.engine.state);
    const random = adapter.engine.random.getState();
    const translated = adapter.refreshLocale('ca', catalog)!;
    expect(translated.html).toContain('Força: ' + catalog[strength]);
    expect(translated.html).toContain('Dissens: ' + catalog[dissent]);
    expect(adapter.qdisplay(35, 'faction_strength')).toBe(catalog[strength]);
    expect(adapter.qdisplay(35, 'dissent')).toBe(catalog[dissent]);
    expect(adapter.engine.state.qualities).toEqual(saved.qualities);
    expect(adapter.engine.random.getState()).toEqual(random);
    const restored = DendryAdapter.fromJSONText(game);
    restored.setLocale('ca', catalog);
    restored.engine.setState(clone(saved));
    expect(restored.currentFrame().html).toBe(translated.html);
    expect(restored.qualities.visits).toBe(1);
    expect(restored.qualities.level).toBe(0);
    expect(restored.refreshLocale('en', {})!.html).toBe(english.html);
  });

  it('keeps strength/dissent and compact/sentence trust on their existing shared thresholds', () => {
    const adapter = DendryAdapter.fromJSONText(game);
    adapter.beginGame([1, 2, 3, 4]);
    for (const value of [-1, 0, 4.999, 5, 14.999, 15, 29.999, 30, 49.999, 50, 100]) {
      expect(parseQdisplayHtml(adapter.qdisplay(value, 'faction_strength')).band)
        .toBe(parseQdisplayHtml(adapter.qdisplay(value, 'dissent')).band);
    }
    for (const value of [0, 15, 15.01, 30, 40, 60, 70, 85, 100]) {
      expect(parseQdisplayHtml(adapter.qdisplay(value, 'politics_trust')).band)
        .toBe(parseQdisplayHtml(adapter.qdisplay(value, 'politics_trust_sentence')).band);
    }
  });

  it('localizes direct formatted qdisplay calls without corrupting their inline nodes', () => {
    const adapter = DendryAdapter.fromJSONText(game);
    adapter.beginGame([1, 2, 3, 4]);
    expect(adapter.qdisplay(10, 'events2012')).toContain("<em>'spanish-ize'</em>");
    adapter.setLocale('ca', {
      '"We must <em>{emphasis1}</em> Catalan children." Spanish Minister of Education José Ignacio Wert says.':
        'Cal <em>{emphasis1}</em> els nens catalans.',
    });
    expect(adapter.qdisplay(10, 'events2012')).toBe("Cal <em>'spanish-ize'</em> els nens catalans.");
  });

  it('renders the maintained Catalan catalogs for every active qualitative display', () => {
    const adapter = DendryAdapter.fromJSONText(readFileSync(join(root, 'out/game.json'), 'utf8'));
    adapter.beginGame([1, 2, 3, 4]);
    const catalog = JSON.parse(readFileSync(join(root, 'source/locales/ca/content.json'), 'utf8'));
    const names = ['faction_strength', 'dissent', 'social_dissent', 'independence_movement', 'cat_spa_relations',
      'relationships', 'control', 'public_debt_qualifier', 'roadmap', 'politics_trust', 'politics_trust_sentence'];
    const definitions = adapter.engine.game.qdisplays as Record<string, { content: { min?: number; max?: number }[] }>;
    const readings = names.flatMap(name => definitions[name].content.map(range => {
      const value = range.min === undefined ? range.max! - 1 : range.max === undefined ? range.min + 1 : (range.min + range.max) / 2;
      return { name, value, key: adapter.qdisplay(value, name) };
    }));
    adapter.setLocale('ca', catalog);
    for (const reading of readings) {
      expect(typeof catalog[reading.key], reading.name).toBe('string');
      expect(adapter.qdisplay(reading.value, reading.name), reading.name).toBe(catalog[reading.key]);
      expect(parseQdisplayHtml(adapter.qdisplay(reading.value, reading.name)).band)
        .toBe(parseQdisplayHtml(reading.key).band);
    }
  });
});
