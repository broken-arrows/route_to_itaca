import { describe, expect, it } from 'vitest';
import { markGlossary } from '../src/glossary/mark';
import type { GlossaryTerm } from '../src/glossary/mark';

const TERMS: GlossaryTerm[] = [
  { id: 'ciu', match: ['CiU', 'ciu'], display: 'CiU', colour: 'ciu', bold: true },
  { id: 'erc', match: ['ERC'], display: 'ERC', colour: 'erc' },
];

describe('markGlossary', () => {
  it('wraps a match in a neutral span — no colour, no inline style', () => {
    expect(markGlossary('CiU governs.', TERMS)).toBe(
      '<span class="term" data-term="ciu">CiU</span> governs.',
    );
  });

  it('applies the canonical display form', () => {
    expect(markGlossary('ciu governs.', TERMS)).toContain('>CiU</span>');
  });

  it('never marks inside a tag', () => {
    expect(markGlossary('<img alt="ERC">', TERMS)).toBe('<img alt="ERC">');
  });

  it('never marks inside an existing span or strong', () => {
    const html = '<span style="color:red">ERC</span>';
    expect(markGlossary(html, TERMS)).toBe(html);
  });

  it('respects the zero-width-space escape hatch', () => {
    expect(markGlossary('​ERC', TERMS)).toBe('​ERC');
  });

  it('respects the -- escape hatch', () => {
    expect(markGlossary('--ERC', TERMS)).toBe('--ERC');
  });

  it('matches on word boundaries only', () => {
    expect(markGlossary('ERCX', TERMS)).toBe('ERCX');
  });

  it('is a no-op with an empty glossary — never throws on every text run', () => {
    expect(markGlossary('CiU', [])).toBe('CiU');
  });

  // The JS `\b` boundary is ASCII-only — these match words were silently dead
  // before the Unicode lookaround boundaries (see mark.ts's comment).
  const UNICODE_TERMS: GlossaryTerm[] = [
    { id: '_ngel_ros', match: ['Àngel Ros'], colour: 'psc' },
    { id: '_te_', match: ['¡TE!'], colour: 'te' },
    { id: 'na_', match: ['NA+'], colour: 'na' },
  ];

  it('matches a word starting with an accented letter', () => {
    expect(markGlossary('Says Àngel Ros today.', UNICODE_TERMS)).toContain(
      'data-term="_ngel_ros"',
    );
  });

  it('matches a word wrapped in punctuation', () => {
    expect(markGlossary('Enter ¡TE! now.', UNICODE_TERMS)).toContain('data-term="_te_"');
  });

  it('matches a word ending in a regex metachar without treating it as an operator', () => {
    const out = markGlossary('NA+ holds Navarra.', UNICODE_TERMS);
    expect(out).toContain('data-term="na_"');
    expect(out).toContain('>NA+</span>');
  });

  it('still refuses to match inside a longer Unicode word', () => {
    expect(markGlossary("l'aÀngel Rosada", UNICODE_TERMS)).toBe("l'aÀngel Rosada");
  });

  it('returns identical output on repeated calls with the same terms array (memoized path)', () => {
    const first = markGlossary('CiU governs.', TERMS);
    const second = markGlossary('CiU governs.', TERMS);
    expect(second).toBe(first);
  });
});
