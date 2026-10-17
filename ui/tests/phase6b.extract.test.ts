import { describe, expect, it } from 'vitest';
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { extractFiles, extractProject } from 'dendrynexus-ten/lib/localization-extract.js';
import { compile } from 'dendrynexus-ten/lib/parsers/content.js';
import { collectUnits, captureContent, renderContent } from 'dendrynexus-ten/lib/localization.js';
import { convertLine } from 'dendrynexus-ten/lib/ui/content/html.js';

const scene = (body: string, arrival = '') => ({
  name: join('source', 'scenes', 'sample.scene.dry'),
  contents: `title: Sample\n${arrival ? `on-arrival: {!\n${arrival}\n!}\n` : ''}\n${body}\n`,
});
const js = (contents: string) => ({ name: join('source', 'lib', 'sample.js'), contents });
const catalogs = { en: {}, ca: { 'An existing key': 'Una traducció que cal conservar' } };

describe('Phase 6B localization extraction and discovery', () => {
  it.each([
    ['Our relations with [+ parlament_current_icv +] are not good enough.', 'Our relations with CSQP are not good enough.'],
    ['Our relations with **CSQP** are not good enough.', 'Our relations with <strong>CSQP</strong> are not good enough.'],
  ])('preserves spaces before inline content in JavaScript conditions: %s', async (body, expected) => {
    const authored = `[? if {! return true; !}: ${body} ?]`;
    const compiled = await new Promise<any>((resolve, reject) =>
      compile(authored, false, (err: unknown, value: unknown) => err ? reject(err) : resolve(value)),
    );
    const snapshot = captureContent(compiled, {
      resolveDependency: (dependency: any) => dependency.type === 'predicate' ? true : 'CSQP',
    });
    expect(convertLine(renderContent(snapshot, 'en', null))).toBe(expected);
    const result = await extractFiles([{
      name: join('source', 'scenes', 'spacing.scene.dry'),
      contents: `title: Sample\nunavailable-subtitle: ${authored}\n\nBody.\n`,
    }], { catalogs });
    expect(result.errors).toEqual([]);
    expect(result.units.some((unit: any) => unit.key.startsWith('Our relations with '))).toBe(true);
  });

  it('validates selector context against every source schema sharing an English key', async () => {
    const result = await extractFiles([js('G.text("Ready", { gender: Q.president_gender }); G.text("Ready");')], {
      catalogs: { ca: { Ready: { select: 'gender', cases: { f: 'Preparada', other: 'Preparat' } } } },
    });
    expect(result.units.find((unit: any) => unit.key === 'Ready')?.schemas).toHaveLength(2);
    expect(result.errors.some((error: any) => error.message.includes('Unknown selector gender'))).toBe(true);
  });
  it('uses the runtime canonical units, including insert gender context, clauses and formatting', async () => {
    const authored = `President [+ president +] has **arrived**. [? if majority: A majority. ?]`;
    const compiled = await new Promise<any>((resolve, reject) =>
      compile(authored, true, (err: unknown, value: unknown) => err ? reject(err) : resolve(value)),
    );
    const declarations = { grammarBindings: { president: 'president_gender' } };
    const expected = collectUnits(compiled, declarations);
    const result = await extractFiles([scene(authored)], { declarations, catalogs });
    for (const unit of expected) {
      const extracted = result.units.find((entry: any) => entry.key === unit.key);
      expect(extracted).toBeDefined();
      expect(extracted.parameters).toEqual(expect.arrayContaining(unit.params));
      expect(extracted.context).toContain('president_gender');
    }
    expect(result.errors).toEqual([]);
    expect(result.units.some((unit: any) => unit.key.includes('{president}'))).toBe(true);
    expect(result.units.some((unit: any) => unit.key.includes('{clause1}'))).toBe(true);
  });

  it('extracts marked JS templates and qdisplay/declared registry outputs, not internal identifiers', async () => {
    const files = [
      scene('Now: [+ government +]; [+ month : month +].', `
        G.setText(Q, "government", "Government has {seats} seats.", { seats: Q.seats });
        Q.destination = "post_event";
        Q.party = "erc";
        Q.leader = "Artur Mas";
      `),
      { name: join('source', 'qdisplays', 'month.qdisplay.dry'), contents: '\n(1..1) January\n(2..2) February\n' },
      { name: join('source', 'data', 'terms.json'), contents: JSON.stringify({ terms: [{ id: 'erc', image: 'img/logo.jpg', tooltip: { description: 'A republican party.' } }] }) },
    ];
    const result = await extractFiles(files, {
      declarations: { registryFields: [{ registry: 'terms', paths: ['terms.*.tooltip.description'] }] }, catalogs,
    });
    const keys = result.units.map((unit: any) => unit.key);
    expect(keys).toContain('Government has {seats} seats.');
    expect(keys).toContain('January');
    expect(keys).toContain('A republican party.');
    for (const internal of ['post_event', 'erc', 'Artur Mas', 'img/logo.jpg']) expect(keys).not.toContain(internal);
    expect(result.findings.some((finding: any) => finding.quality === 'government')).toBe(false);
    expect(result.errors).toEqual([]);
  });

  it('preserves translations and orphaned entries while reporting new/untranslated keys', async () => {
    const existing = { select: 'president_gender', cases: { f: 'La presidenta {president}.', other: 'El president {president}.' } };
    const result = await extractFiles([scene('President [+ president +].\n\nA new sentence.')], {
      declarations: { grammarBindings: { president: 'president_gender' } },
      catalogs: { en: {}, ca: { 'President {president}.': existing, Retired: 'Retirat' } },
    });
    expect(result.catalogs.ca['President {president}.']).toEqual(existing);
    expect(result.catalogs.ca.Retired).toBe('Retirat');
    expect(result.catalogs.ca['A new sentence.']).toBeNull();
    expect(result.catalogReports.ca.orphanedKeys).toEqual(['Retired']);
    expect(result.catalogReports.ca.untranslatedKeys).toContain('A new sentence.');
    expect(result.errors).toEqual([]);
  });

  it('reports writes, copies, local concatenation, joins and replacements that reach inserts', async () => {
    const result = await extractFiles([scene('Result: [+ archived +].', `
      var reasons = ["Too few seats.", "Relations are poor."];
      Q.reason_text = reasons.join(" ");
      Q.government = "A government led by " + Q.president;
      Q.government = Q.government.replaceAll("cdc", "pdecat");
      Q.archived = Q.government + Q.reason_text;
      Q.old = Q.archived;
    `)], { declarations: { displayQualities: ['old'] }, catalogs });
    const kinds = result.findings.map((finding: any) => finding.kind);
    expect(kinds).toContain('unmarked-join');
    expect(kinds).toContain('unmarked-composition');
    expect(kinds).toContain('unmarked-replacement');
    expect(kinds).toContain('unmarked-copy');
    const government = result.findings.find((finding: any) => finding.quality === 'government');
    expect(government.displayUses.length).toBeGreaterThan(0);
    expect(result.catalogs.ca['A government led by ']).toBeUndefined();
    expect(result.errors).toEqual([]);
  });

  it('keeps reasoned dispositions on moved code and reopens changed evidence', async () => {
    const initial = await extractFiles([js('Q.foo = "Some text.";')], { declarations: { displayQualities: ['foo'] }, catalogs });
    const finding = initial.findings[0];
    const dispositions = { findings: { [finding.id]: { fingerprint: finding.fingerprint, status: 'migration-pending', reason: 'Migrate in a later content pass.' } } };
    const moved = await extractFiles([js('\n\nQ.foo = "Some text.";')], {
      declarations: { displayQualities: ['foo'] }, catalogs, dispositions, previousReport: initial,
    });
    expect(moved.findings[0].id).toBe(finding.id);
    expect(moved.findings[0].change).toBe('moved');
    expect(moved.findings[0].review.stale).toBe(false);
    const changed = await extractFiles([js('Q.foo = "Different text.";')], {
      declarations: { displayQualities: ['foo'] }, catalogs, dispositions, previousReport: moved,
    });
    expect(changed.findings[0].id).toBe(finding.id);
    expect(changed.findings[0].change).toBe('changed');
    expect(changed.findings[0].review.stale).toBe(true);
    const repeated = await extractFiles([js('Q.foo = "Different text.";')], {
      declarations: { displayQualities: ['foo'] }, catalogs, previousReport: changed,
    });
    expect(repeated.findings[0].change).toBe('unchanged');
  });

  it('reports unresolved dynamic qualities and helper outputs without treating candidates as confirmed keys', async () => {
    const result = await extractFiles([scene('Result: [+ foo +].', `
      Q[Q.party + "_description"] = "Something visible.";
      Q.foo = G.describeGovernment(Q);
    `), js('function describeGovernment(Q) { return "A government."; }')], { catalogs });
    expect(result.limitations.some((limit: any) => limit.reason.includes('Dynamic quality write'))).toBe(true);
    const finding = result.findings.find((entry: any) => entry.quality === 'foo');
    expect(finding.related[0].helper).toBe('describeGovernment');
    expect(result.units.map((unit: any) => unit.key)).not.toContain('A government.');
  });

  it('validates marked placeholders, selector fallback/parameters and protected markup', async () => {
    const result = await extractFiles([
      js('G.text("Missing {name}.", {}); G.text("There are {seats} seats.", {seats: 1}); G.text("Unclosed {name", {name: "Test"});'),
      scene('**Formatted text.**'),
    ], {
      catalogs: { ca: {
        'There are {seats} seats.': { plural: 'seats', cases: { one: 'Hi ha {seats} escó.' } },
        '<strong>{emphasis1}</strong>': '<em>{emphasis1}</em>',
      } },
    });
    expect(result.errors.some((error: any) => error.message.includes('Unknown placeholder {name}'))).toBe(true);
    expect(result.errors.some((error: any) => error.message.includes('Malformed placeholder'))).toBe(true);
    expect(result.errors.some((error: any) => error.message.includes('other'))).toBe(true);
    // Outer pure formatting is intentionally not a catalog key; validate a mixed run separately.
    const markup = await extractFiles([scene('Read **this** now.')], {
      catalogs: { ca: { 'Read <strong>{emphasis1}</strong> now.': 'Llegiu <em>{emphasis1}</em> ara.' } },
    });
    expect(markup.errors.some((error: any) => error.message.includes('markup'))).toBe(true);
  });

  it('accepts explicit registry template schemas for plural widget text', async () => {
    const result = await extractFiles([{ name: join('source', 'data', 'localization.json'), contents: JSON.stringify({ widgetText: { seats: '{seats} seats' } }) }], {
      declarations: { registryFields: [{ registry: 'localization', paths: ['widgetText.*'], parameters: ['seats'] }] },
      catalogs: { ca: { '{seats} seats': { plural: 'seats', cases: { '=0': '{seats} escons', one: '{seats} escó', other: '{seats} escons' } } } },
    });
    expect(result.errors).toEqual([]);
    expect(result.units[0].parameters).toEqual(['seats']);
  });

  it('dry-runs a project without filesystem changes, then writes additive catalogs/report and retains reviews', async () => {
    const project = mkdtempSync(join(tmpdir(), 'phase6b-extract-'));
    try {
      mkdirSync(join(project, 'source', 'scenes'), { recursive: true });
      mkdirSync(join(project, 'source', 'locales', 'ca'), { recursive: true });
      mkdirSync(join(project, 'source', 'locales', '_reports'), { recursive: true });
      writeFileSync(join(project, 'source', 'info.dry'), 'title: Sample\nauthor: Test\nlanguages: en ca\n');
      writeFileSync(join(project, 'source', 'scenes', 'root.scene.dry'), 'title: Root\n\nA sentence.\n');
      const catalogPath = join(project, 'source', 'locales', 'ca', 'content.json');
      const reportPath = join(project, 'source', 'locales', '_reports', 'extraction.json');
      const reviewPath = join(project, 'source', 'locales', '_reports', 'dispositions.json');
      const before = JSON.stringify({ Retired: 'Retirat' });
      const reviewBefore = JSON.stringify({ version: 1, findings: {} });
      writeFileSync(catalogPath, before);
      writeFileSync(reviewPath, reviewBefore);
      const preview = await extractProject(project, { dryRun: true });
      expect(preview.catalogs.ca['A sentence.']).toBeNull();
      expect(readFileSync(catalogPath, 'utf8')).toBe(before);
      expect(readdirSync(join(project, 'source', 'locales', '_reports'))).toEqual(['dispositions.json']);
      const applied = await extractProject(project);
      expect(applied.errors).toEqual([]);
      expect(JSON.parse(readFileSync(catalogPath, 'utf8')).Retired).toBe('Retirat');
      expect(JSON.parse(readFileSync(reportPath, 'utf8')).catalogs).toBeUndefined();
      expect(readFileSync(reviewPath, 'utf8')).toBe(reviewBefore);
    } finally {
      rmSync(project, { recursive: true, force: true });
    }
  });
});
