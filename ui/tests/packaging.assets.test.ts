// @vitest-environment node
import { createRequire } from 'node:module';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readdirSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

const { validateGameAssets, publishRuntimeAssets } = createRequire(import.meta.url)(
  '../../vendor/dendrynexus-ten/lib/cli/game-assets.js',
) as {
  validateGameAssets(root: string, options?: { extraSources: string[] }): {
    imageCount: number; referenceCount: number; locales: string[]; warnings: string[];
  };
  publishRuntimeAssets(root: string, dest: string): void;
};
const roots: string[] = [];
afterEach(() => roots.splice(0).forEach(root => rmSync(root, { recursive: true, force: true })));

function fixture(content: unknown = { cardImage: 'img/portrait.jpg' }) {
  const root = mkdtempSync(path.join(tmpdir(), 'rti-packaging-'));
  roots.push(root);
  const write = (name: string, value: string | object) => {
    const file = path.join(root, name);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, typeof value === 'string' ? value : JSON.stringify(value));
  };
  write('out/game.json', { info: { languages: ['en', 'ca'] }, scenes: { root: content } });
  write('source/lib/index.js', '// game library');
  write('source/img/portrait.jpg', 'portrait bytes');
  for (const locale of ['en', 'ca']) {
    write('source/locales/' + locale + '/content.json', { Hello: locale === 'en' ? 'Hello' : null });
    write('source/locales/' + locale + '/ui.json', { app: { title: locale === 'en' ? 'Game' : 'Joc' } });
  }
  return { root, write };
}

describe('runtime asset packaging', () => {
  it('accepts incomplete translation catalogs and summarizes English fallback without dropping runtime files', () => {
    const { root } = fixture();
    const result = validateGameAssets(root);
    expect(result.imageCount).toBe(1);
    expect(result.referenceCount).toBe(1);
    expect(result.warnings).toEqual([expect.stringContaining('ca: 1 untranslated content entries, 0 untranslated UI entries')]);
    publishRuntimeAssets(root, path.join(root, 'dist'));
    expect(JSON.parse(readFileSync(path.join(root, 'dist/locales/ca/content.json'), 'utf8'))).toEqual({ Hello: null });
  });

  it.each(['out/game.json', 'source/lib/index.js', 'source/img', 'source/locales/en/content.json',
    'source/locales/ca/content.json', 'source/locales/ca/ui.json'])('fails when required input %s is missing', name => {
    const { root } = fixture();
    rmSync(path.join(root, name), { recursive: true, force: true });
    expect(() => validateGameAssets(root)).toThrow(/Missing required/);
  });

  it('rejects malformed catalogs and an empty art tree', () => {
    const { root, write } = fixture();
    write('source/locales/ca/content.json', '{broken');
    expect(() => validateGameAssets(root)).toThrow(/Invalid JSON.*content.json/);
    write('source/locales/ca/content.json', {});
    rmSync(path.join(root, 'source/img/portrait.jpg'));
    expect(() => validateGameAssets(root)).toThrow(/Empty required art tree/);
  });

  it('reports missing paths with their compiled property and JS producer locations', () => {
    const { root, write } = fixture({ faceImage: 'img/missing.jpg' });
    write('source/lib/index.js', 'const icon = "img/absent.svg";');
    expect(() => validateGameAssets(root)).toThrow(/img\/missing.jpg.*scenes.root.faceImage/);
    expect(() => validateGameAssets(root)).toThrow(/img\/absent.svg.*source\/lib\/index.js:1/);
  });

  it('validates literal HTML, CSS and image paths containing spaces', () => {
    const { root, write } = fixture({ content: '<img src="img/portrait copy.jpg">', background: 'img/portrait.jpg' });
    write('source/img/portrait copy.jpg', 'copy');
    write('ui/src/card.vue', '<img src="img/missing.jpg">');
    expect(() => validateGameAssets(root, { extraSources: [path.join(root, 'ui/src')] })).toThrow(/ui\/src\/card.vue:1/);
    write('ui/src/card.vue', '<style>div { background: url("img/portrait copy.jpg"); }</style>');
    expect(validateGameAssets(root, { extraSources: [path.join(root, 'ui/src')] }).referenceCount).toBe(2);
  });

  it('detects filename case mismatches even on a case-insensitive filesystem', () => {
    const { root } = fixture({ cardImage: 'img/Portrait.jpg' });
    expect(() => validateGameAssets(root)).toThrow(/img\/Portrait.jpg/);
  });

  it('does not treat remote URLs containing /img/ as local files', () => {
    const { root } = fixture({ cardImage: 'img/portrait.jpg',
      content: '<img src="https://example.com/img/remote.jpg"><img src="//example.com/img/other.png">' });
    expect(validateGameAssets(root).referenceCount).toBe(1);
  });

  it('warns about unresolved dynamic paths while checking the static ones', () => {
    const { root, write } = fixture();
    write('source/lib/index.js', 'const logo = `img/parties/${party}.png`;');
    expect(validateGameAssets(root).warnings).toContainEqual(expect.stringContaining('dynamic image references require runtime checks'));
  });

  it('copies only runtime catalogs and replaces stale generated files without touching shell source', () => {
    const { root, write } = fixture();
    write('source/locales/_reports/dispositions.json', { private: true });
    write('source/locales/ca/notes.txt', 'development artifact');
    write('dist/img/deleted.jpg', 'stale');
    write('dist/locales/fr/content.json', {});
    write('dist/index.html', 'hand-maintained shell');
    publishRuntimeAssets(root, path.join(root, 'dist'));
    expect(readdirSync(path.join(root, 'dist/img'))).toEqual(['portrait.jpg']);
    expect(readdirSync(path.join(root, 'dist/locales')).sort()).toEqual(['ca', 'en']);
    expect(readdirSync(path.join(root, 'dist/locales/ca')).sort()).toEqual(['content.json', 'ui.json']);
    expect(readFileSync(path.join(root, 'dist/index.html'), 'utf8')).toBe('hand-maintained shell');
  });

  it('validates inputs before replacing the existing generated output', () => {
    const { root, write } = fixture();
    write('dist/img/previous.jpg', 'previous successful build');
    rmSync(path.join(root, 'source/locales/ca/ui.json'));
    expect(() => publishRuntimeAssets(root, path.join(root, 'dist'))).toThrow(/Missing required/);
    expect(existsSync(path.join(root, 'dist/img/previous.jpg'))).toBe(true);
  });

  it('rejects an output directory outside its owning project', () => {
    const { root } = fixture();
    expect(() => publishRuntimeAssets(root, path.join(root, '..', 'outside'))).toThrow(/inside project/);
  });

  it('rejects publishing over canonical inputs before deleting any assets', () => {
    const { root } = fixture();
    expect(() => publishRuntimeAssets(root, path.join(root, 'source'))).toThrow(/overlaps canonical source/);
    expect(existsSync(path.join(root, 'source/img/portrait.jpg'))).toBe(true);
  });
});
