import { describe, expect, it, vi, afterEach } from 'vitest';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { DendryAdapter } from '../src/engine/adapter';
import { loadContentCatalog, clearContentCatalogCacheForTest } from '../src/locales/content';

const require = createRequire(import.meta.url);
const L = require('dendrynexus-ten/lib/localization.js');
const { compileGame, convertGameToJSON } = require('dendrynexus-ten/lib/parsers/compiler.js');
const { extractFiles } = require('dendrynexus-ten/lib/localization-extract.js');
const read = (path: string) => JSON.parse(readFileSync(join(import.meta.dirname, '../../', path), 'utf8'));
const parties = read('source/locales/ca/parties.json');
const declarations = read('source/data/localization.json');
const key = '{party} wants to negotiate with {other}.';
const catalog = L.mergeNames({ [key]: '{party:definite:cap} vol negociar amb {other:definite}.' }, { party: parties });

afterEach(() => { vi.unstubAllGlobals(); clearContentCatalogCacheForTest(); });

describe('named party references', () => {
  it.each([['ciu', 'CiU'], ['pdcat', 'El PDeCAT'], ['cup', 'La CUP']])('renders %s with sentence capitalization', (id, label) => {
    const recipe = L.text(key, { party: L.nameText('party', id), other: L.nameText('party', 'cup') });
    expect(L.render(recipe, 'ca', catalog)).toBe(`${label} vol negociar amb la CUP.`);
    expect(L.render(recipe, 'en', null)).toBe(`${id} wants to negotiate with cup.`);
  });

  it('supports different forms of the same reference, apostrophes, and bare labels', () => {
    const source = '{party} / {party} / {party} / {party} / {party}';
    const localized = L.mergeNames({ [source]: '{party:bare} / {party:definite} / {party:de} / {party:a} / {party:per}' }, { party: parties });
    expect(L.render(L.text(source, { party: L.nameText('party', 'pdcat') }), 'ca', localized))
      .toBe('PDeCAT / el PDeCAT / del PDeCAT / al PDeCAT / pel PDeCAT');
    const apostrophe = L.mergeNames({ '{x}': '{x:de:cap}' }, { party: { example: { bare: 'Exemple', de: "de l’Exemple" } } });
    expect(L.render(L.text('{x}', { x: L.nameText('party', 'example') }), 'ca', apostrophe)).toBe('De l’Exemple');
    expect(L.render(L.text('{x}', { x: L.nameText('party', 'erc') }), 'ca', L.mergeNames({ '{x}': '{x:de}' }, { party: parties }))).toBe('d’ERC');
    // The same ID may be used in technical markup: unformatted values are exact.
    const technical = L.text('<span data-party="{party}">{party}</span>', { party: L.nameText('party', 'pdcat') });
    expect(L.render(technical, 'ca', localized)).toBe('<span data-party="pdcat">pdcat</span>');
  });

  it('falls back safely for old plain recipes, unknown identities, and missing forms', () => {
    expect(L.render(L.text(key, { party: 'pdcat', other: 'cup' }), 'ca', catalog)).toBe('pdcat wants to negotiate with cup.');
    expect(L.render(L.text(key, { party: L.nameText('party', 'unknown'), other: L.nameText('party', 'cup') }), 'ca', catalog))
      .toBe('unknown wants to negotiate with cup.');
    const bad = L.mergeNames({ '{party}': '{party:typo}' }, { party: parties });
    expect(L.render(L.text('{party}', { party: L.nameText('party', 'pdcat') }), 'ca', bad)).toBe('pdcat');
    expect(() => L.mergeNames({}, { party: { x: { bare: '<b>x</b>' } } })).toThrow();
    expect(L.render(L.text('{party}', { party: L.nameText('party', 'pdcat') }), 'ca', {
      '{party}': '{party:definite}', '@name/party/pdcat/definite': { bad: 'entry' },
    })).toBe('pdcat');
    // Existing nested translations still work when their enclosing sentence is untranslated.
    expect(L.render(L.text('A {label}.', { label: L.text('government') }), 'ca', {
      'A {label}.': null, government: 'govern',
    })).toBe('A govern.');
  });

  it('captures declared inserts and generated references across locale switches and save/load', async () => {
    const files = [{ name: join('scenes', 'root.scene.dry'), contents: `title: Test
on-arrival: {! Q.parlament_current_ciu = "pdcat"; Q.arrivals = (Q.arrivals || 0) + 1; G.setText(Q, "announcement", "{party} acts.", {party:G.nameText("party", Q.parlament_current_ciu)}); !}

[+ parlament_current_ciu +] wants to negotiate.

[+ announcement +]
` }];
    const compiled = await new Promise<any>((resolve, reject) => compileGame(files, (e: Error, g: any) => e ? reject(e) : resolve(g)));
    compiled.data = { localization: declarations };
    const json = await new Promise<string>((resolve, reject) => convertGameToJSON(compiled, 0, (e: Error, s: string) => e ? reject(e) : resolve(s)));
    const a = DendryAdapter.fromJSONText(json);
    a.beginGame();
    const engine = a.engine as any;
    engine.state.qualities.parlament_current_ciu = 'junts';
    const saved = JSON.parse(JSON.stringify(engine.getExportableState()));
    const rng = engine.random.getState();
    const translations = L.mergeNames({
      '{parlament_current_ciu} wants to negotiate.': '{parlament_current_ciu:definite:cap} vol negociar.',
      '{party} acts.': '{party:definite:cap} actua.',
    }, { party: parties });
    engine.setLocale('ca', translations);
    engine.refreshPresentation();
    const frame = a.refreshLocale('ca', translations)!;
    expect(frame.html).toContain('El PDeCAT vol negociar.');
    expect(frame.html).toContain('El PDeCAT actua.');
    expect(engine.state.qualities.announcement).toBe('pdcat acts.');
    expect(engine.getExportableState()).toEqual(saved);
    expect(engine.random.getState()).toEqual(rng);
    const b = DendryAdapter.fromJSONText(json);
    b.engine.setLocale('ca', translations);
    (b.engine as any).setState(saved);
    expect(b.refreshLocale('ca', translations)!.html).toContain('El PDeCAT actua.');
    expect((b.engine as any).state.qualities.arrivals).toBe(1);
    b.engine.setLocale('en', {});
    (b.engine as any).refreshPresentation();
    expect(b.refreshLocale('en', {})!.html).toContain('pdcat acts.');
  });

  it('extracts named schemas and rejects forms on ordinary strings and unknown form names', async () => {
    const files = [{ name: join('scenes', 'root.scene.dry'), contents: `title: Root
on-arrival: {! G.setText(Q, "message", "{party} acts.", {party:G.nameText("party", Q.parlament_current_ciu)}); G.text("{plain} acts.", {plain:"cup"}); !}

[+ parlament_current_ciu +] acts.
` }];
    const namingDeclarations = { nameBindings: declarations.nameBindings, nameForms: declarations.nameForms };
    const result = await extractFiles(files, { declarations: namingDeclarations, catalogs: { ca: {
      '{party} acts.': '{party:definite:cap} actua.',
      '{parlament_current_ciu} acts.': '{parlament_current_ciu:definite:cap} actua.',
      '{plain} acts.': '{plain:definite} actua.',
    } } });
    expect(result.errors).toHaveLength(1);
    expect(result.errors[0].message).toContain('requires a named reference: plain');
    const invalid = await extractFiles(files, { declarations: namingDeclarations, catalogs: { ca: { '{party} acts.': '{party:typo} actua.' } } });
    expect(invalid.errors.some((e: any) => e.message.includes('Unknown name form typo'))).toBe(true);
  });

  it('loads the canonical naming catalog in the new UI', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ [key]: '{party:definite:cap} vol negociar amb {other:definite}.' }) })));
    const loaded = await loadContentCatalog('ca');
    expect(L.render(L.text(key, { party: L.nameText('party', 'pdcat'), other: L.nameText('party', 'cup') }), 'ca', loaded))
      .toBe('El PDeCAT vol negociar amb la CUP.');
  });
});
