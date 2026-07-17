import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

// dendrynexus-ten/lib/cli/utils.js has no ambient type declarations (only
// engine.js / html.js / parsers/compiler.js do, in
// src/engine/dendrynexus.d.ts — a shared file this test-only change must not
// touch, and a local `declare module` augmentation doesn't work here: this
// file already has top-level imports, which makes TS treat any nested
// `declare module` as an augmentation of an existing (typed) module rather
// than a fresh ambient declaration, and utils.js has neither). Suppress the
// resulting "implicitly has an any type" diagnostic here instead; the
// callback in run() below is typed explicitly so the rest of the file stays
// fully checked.
// @ts-expect-error dendrynexus-ten/lib/cli/utils.js is untyped (see above)
import { fetchContent } from 'dendrynexus-ten/lib/cli/utils.js';

// This exercises the REAL fetchContent -> walkDir -> CONTENT_PATTERN chain
// against a REAL directory on disk (dendrynexus-ten/lib/cli/utils.js). Every
// other compiler test in this suite calls compileGame directly with an
// in-memory file array, so the filesystem walk and its read-filter regex
// (CONTENT_PATTERN, utils.js:259) have never actually executed in CI. That
// filter is the only thing standing between a compile and slurping the whole
// source/ tree -- including the 101 MB art tree about to land in
// source/img/ -- into memory as UTF-8 strings. A regex mistake there is
// silent: no error, just missing content in the compiled game. See
// .superpowers/sdd/p25-task-1-report.md's code review for the full finding.
describe('fetchContent read-filter (real fs walk, real CONTENT_PATTERN)', () => {
  let root: string;
  let sourceDir: string;

  beforeAll(() => {
    // A real fixture tree, built on real disk, shaped like the source/ dir
    // is about to look once source/lib, source/img and source/locales exist
    // alongside compiled content.
    root = mkdtempSync(path.join(tmpdir(), 'rti-content-filter-'));
    sourceDir = path.join(root, 'source');

    const write = (relPath: string, contents: string) => {
      const full = path.join(sourceDir, relPath);
      mkdirSync(path.dirname(full), { recursive: true });
      writeFileSync(full, contents);
    };

    write('info.dry', 'title: T\nauthor: A\n');
    write(path.join('scenes', 'root.scene.dry'), 'title: Root\n\nHello.\n');
    write(path.join('scenes', 'nested', 'deep.scene.dry'), 'title: Deep\n\nNested.\n');
    write(path.join('qdisplays', 'dissent.qdisplay.dry'), 'title: Dissent\n\nA qdisplay.\n');
    write(path.join('data', 'glossary.json'), JSON.stringify({ terms: [{ id: 'ciu' }] }));
    // Shipped, not compiled -- see utils.js:250-252's rationale comment.
    write(path.join('lib', 'cat_engine.js'), '// engine lib, not game content\n');
    write(path.join('lib', 'index.js'), '// index, not game content\n');
    write(path.join('img', 'parties', 'logo.png'), 'not-a-real-png-just-needs-to-exist-on-disk');
    write(path.join('locales', 'en', 'ui.json'), JSON.stringify({ ok: true }));
  });

  afterAll(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function run(): Promise<{ name: string; contents: string }[]> {
    return new Promise((resolve, reject) => {
      fetchContent(
        sourceDir,
        (err: Error | null, files: { name: string; contents: string }[]) =>
          err ? reject(err) : resolve(files),
      );
    });
  }

  // Path suffix relative to sourceDir, with separators normalized to '/'.
  // fetchContent hands back NATIVE absolute paths (backslashes on Windows --
  // exactly why CONTENT_PATTERN itself has a [\\/] character class), so
  // asserting on raw `f.name` would make this test pass on Windows and lie
  // about Linux/CI, or vice versa. Assert on this instead.
  function relSuffixes(files: { name: string }[]): string[] {
    return files.map((f) => path.relative(sourceDir, f.name).split(path.sep).join('/')).sort();
  }

  it('reads every .dry file, including the top-level info.dry and arbitrarily nested scenes', async () => {
    const files = await run();
    const dryFiles = relSuffixes(files.filter((f) => f.name.endsWith('.dry')));
    // This is the catastrophic direction: a false negative here silently
    // drops game content from the compiled artifact with no error.
    expect(dryFiles).toEqual([
      'info.dry',
      'qdisplays/dissent.qdisplay.dry',
      'scenes/nested/deep.scene.dry',
      'scenes/root.scene.dry',
    ]);
  });

  it('reads source/data/glossary.json', async () => {
    const files = await run();
    const jsonFiles = relSuffixes(files.filter((f) => f.name.endsWith('.json')));
    expect(jsonFiles).toEqual(['data/glossary.json']);

    const glossary = files.find((f) => f.name.endsWith('glossary.json'));
    expect(glossary && JSON.parse(glossary.contents)).toEqual({ terms: [{ id: 'ciu' }] });
  });

  it('skips source/lib/**, source/img/** and source/locales/** entirely', async () => {
    const files = await run();
    const suffixes = relSuffixes(files);

    expect(suffixes.some((s) => s.startsWith('lib/'))).toBe(false);
    expect(suffixes.some((s) => s.startsWith('img/'))).toBe(false);
    expect(suffixes.some((s) => s.startsWith('locales/'))).toBe(false);

    // Belt and braces by basename, so an over-broad regex match on any of
    // these specific shipped-not-compiled files is caught even if the
    // startsWith checks above were somehow satisfied by coincidence.
    const basenames = files.map((f) => path.basename(f.name));
    expect(basenames).not.toContain('cat_engine.js');
    expect(basenames).not.toContain('index.js');
    expect(basenames).not.toContain('logo.png');
    expect(basenames).not.toContain('ui.json');
  });

  it('reads exactly the 5 content files out of the 9-file fixture -- no more, no less', async () => {
    const files = await run();
    expect(relSuffixes(files)).toEqual([
      'data/glossary.json',
      'info.dry',
      'qdisplays/dissent.qdisplay.dry',
      'scenes/nested/deep.scene.dry',
      'scenes/root.scene.dry',
    ]);
  });
});
