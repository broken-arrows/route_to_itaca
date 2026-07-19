/**
 * The glossary MARKER. Pure, DOM-free, and deliberately presentation-neutral:
 * it emits `<span class="term" data-term="ciu">CiU</span>` and nothing else —
 * no colour, no inline style. The DATA is the game's
 * (`source/data/glossary.json`, compiled to `game.json.data.glossary`); the
 * LOOK is each UI's (see `ui/src/components/Prose.vue`).
 *
 * It has to be string surgery: engine prose arrives as an HTML string (dendry
 * calls `window.displayText(text)` on every rendered text run — see
 * `main.ts` and vendor/dendrynexus-ten/lib/ui/content/html.js:14 — so
 * something must find the words in that string). What this version drops
 * relative to the old shell's `applyWholesome` (out/html/game.js) is its
 * habit of deciding what a match should look like: no colour, no bold, no
 * tooltip trigger classing — just a neutral marker the UI decorates.
 *
 * This runs on EVERY text run the engine renders. It must never throw.
 */
export interface GlossaryTerm {
  id: string;
  match: string[];
  display?: string;
  colour?: string;
  bold?: boolean;
  tooltip?: {
    title: string;
    subtitle?: string;
    img?: string;
    infoDesc?: string;
    q?: Record<string, string>;
  };
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// Segments the string into tags, already-marked spans/strongs, and bare text
// — so we only ever touch bare text. Mirrors applyWholesome's segmentation
// exactly (same trust boundary: never rewrite inside existing markup).
const SEGMENTS = /(<(?:span|strong)[^>]*>.*?<\/(?:span|strong)>|<[^>]+>|[^<]+)/g;

// Unicode-aware word boundary. `\b` is ASCII-only in JS: a match word that
// starts or ends with an accented letter ("Àngel Ros", "JxSí", "BComú") or
// punctuation ("¡TE!", "NA+") sits next to a non-\w character, so `\b` never
// fires there and the term silently never matches — 9 of the harvested match
// words were dead this way (in BOTH UIs; out/html/game.js applyWholesome
// carries the same fix). Lookarounds against Unicode letters/digits are the
// boundary `\b` should have been.
const NOT_WORD_BEFORE = '(?<![\\p{L}\\p{N}_])';
const NOT_WORD_AFTER = '(?![\\p{L}\\p{N}_])';

// This runs on EVERY text run the engine renders (window.displayText), so the
// term map and the ~200-alternation regex must not be rebuilt per call. The
// glossary array is static per game load (one JSON.parse in the adapter), so
// its identity is a correct cache key; a new game load is a new array and a
// clean rebuild.
interface Compiled {
  byWord: Map<string, GlossaryTerm>;
  re: RegExp;
}
const compiledCache = new WeakMap<GlossaryTerm[], Compiled | null>();

function compile(terms: GlossaryTerm[]): Compiled | null {
  let compiled = compiledCache.get(terms);
  if (compiled !== undefined) return compiled;

  const byWord = new Map<string, GlossaryTerm>();
  for (const t of terms) for (const w of t.match) byWord.set(w, t);
  if (byWord.size === 0) {
    compiledCache.set(terms, null);
    return null;
  }
  const words = [...byWord.keys()].sort((a, b) => b.length - a.length).map(escapeRe);
  compiled = {
    byWord,
    re: new RegExp(`${NOT_WORD_BEFORE}(${words.join('|')})${NOT_WORD_AFTER}`, 'gu'),
  };
  compiledCache.set(terms, compiled);
  return compiled;
}

export function markGlossary(html: string, terms: GlossaryTerm[]): string {
  if (!html || terms.length === 0) return html;

  const compiled = compile(terms);
  if (!compiled) return html;
  const { byWord, re } = compiled;

  return html.replace(SEGMENTS, (segment) => {
    if (segment.startsWith('<')) return segment;
    return segment.replace(re, (match, _g, offset: number) => {
      // Author escape hatches, both preserved from applyWholesome:
      // a zero-width space immediately before the match, or a literal `--`.
      if (segment[offset - 1] === '​') return match;
      if (segment.slice(offset - 2, offset) === '--') return match;

      const term = byWord.get(match)!;
      return `<span class="term" data-term="${term.id}">${term.display ?? match}</span>`;
    });
  });
}
