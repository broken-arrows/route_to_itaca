/**
 * `(value, qdisplayId)` -> `{ band, label }` — a TOKEN and a PLAIN WORD.
 *
 * WHY THIS EXISTS. Brief row view-models (`source/lib/brief.js`) carry a raw
 * number plus the id of the qdisplay that classifies it (`stampDisplay` /
 * `valueDisplay`), never a band word: thresholds live in
 * `source/qdisplays/*.dry` and nowhere else. The only thing that can apply
 * them is the engine, and it answers in HTML. Every widget would otherwise
 * have to parse that HTML itself. This is the single place that does it.
 *
 * WHAT `adapter.qdisplay()` ACTUALLY EMITS — measured against a booted
 * adapter (`DendryAdapter.fromJSONText(out/game.json)` + `beginGame`), not
 * assumed. Five distinct shapes, all of which this parser handles:
 *
 *   1. the common case — a `<span>`:
 *        qdisplay(45,'social_dissent')
 *        -> '<span class="q-band" data-scale="social_dissent"
 *                 data-band="medium">medium</span>'
 *   2. a `<p>`, NOT a span (`international_opinion` only):
 *        qdisplay(3,'international_opinion')
 *        -> '<p class="q-band" data-scale="international_opinion"
 *               data-band="supportive">"Catalonia would be a most welcome
 *               addition…"</p>'
 *      -> so the parser must not hardcode the tag name.
 *   3. TRAILING TEXT OUTSIDE the element (`politics_trust` only — its lines
 *      end in the preposition that follows the word in prose):
 *        qdisplay(50,'politics_trust')
 *        -> '<span … data-band="neutral">neutral</span> about'
 *      -> so the label is the element's INNER text, never the whole string
 *         ('neutral', not 'neutral about').
 *   4. NO markup at all, for qdisplays that were never banded
 *      (`relationships`, `month`):  qdisplay(64,'relationships') -> 'warm'.
 *   5. NO markup and no word, when the value matches no range at all:
 *        qdisplay(99,'roadmap') -> '99'  (the raw value, stringified).
 *
 * And two ways it FAILS, both of which throw rather than return anything:
 *   - an unknown qdisplay id           -> Error('Assertion failed.')
 *   - `undefined` as the value         -> TypeError (…reading 'toString')
 * Both are caught here: a Brief row must never take a sheet down.
 *
 * No DOM is used on purpose — this runs identically in a widget, in a plain
 * unit test and in node, and the input is engine-authored HTML from a fixed
 * generator, not arbitrary markup.
 */
import { useGameStore } from '../../stores/game';

export interface Band {
  /**
   * The qdisplay's own `data-band` token (`medium_high`, `hostile`,
   * `unset`, …) — what the Desk's one band ramp is keyed on.
   *
   * For an un-banded qdisplay (shape 4 above) there is no token to read, so
   * the word itself is slugged into one: `relationships` yields `warm`,
   * `hostile`, `very_friendly` — exactly the tokens the design's bench
   * stamps use. Empty string when there is nothing to classify at all.
   */
  band: string;
  /** The word, as plain text: no tags, no entities, whitespace collapsed. */
  label: string;
}

const EMPTY: Band = Object.freeze({ band: '', label: '' });

/** `class="… q-band …"`, single or double quoted, in an attribute run. */
const HAS_Q_BAND = /\bclass\s*=\s*(?:"[^"]*\bq-band\b[^"]*"|'[^']*\bq-band\b[^']*')/i;
const DATA_BAND = /\bdata-band\s*=\s*(?:"([^"]*)"|'([^']*)')/i;

const NAMED_ENTITIES: Record<string, string> = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ',
};

function decodeEntities(s: string): string {
  return s.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, body: string) => {
    if (body[0] !== '#') return NAMED_ENTITIES[body.toLowerCase()] ?? whole;
    const cp = body[1] === 'x' || body[1] === 'X'
      ? Number.parseInt(body.slice(2), 16)
      : Number.parseInt(body.slice(1), 10);
    return Number.isFinite(cp) && cp > 0 ? String.fromCodePoint(cp) : whole;
  });
}

/** Tags out, entities in, whitespace collapsed. */
function plainText(html: string): string {
  return decodeEntities(html.replace(/<[^>]*>/g, ' ')).replace(/\s+/g, ' ').trim();
}

/** A word with no `data-band` of its own still needs a token to ink by. */
function slug(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

/**
 * Pure half of `useBand` — exported so the parser can be pinned against the
 * five real emitted shapes without booting an adapter.
 */
export function parseQdisplayHtml(html: string): Band {
  if (typeof html !== 'string' || html === '') return EMPTY;
  // Built per call: a /g regex carries `lastIndex` between calls, and this
  // one is exited early on the first match.
  const element = /<([a-z][\w-]*)\b([^>]*)>([\s\S]*?)<\/\1\s*>/gi;
  let m: RegExpExecArray | null;
  while ((m = element.exec(html)) !== null) {
    if (!HAS_Q_BAND.test(m[2])) continue;
    const label = plainText(m[3]);
    const attr = DATA_BAND.exec(m[2]);
    const band = (attr?.[1] ?? attr?.[2] ?? '').trim();
    return { band: band || slug(label), label };
  }
  const label = plainText(html);
  return label ? { band: slug(label), label } : EMPTY;
}

/**
 * The composable widgets use. `band(value, qdisplayId)` is a pure function of
 * its two arguments plus the game's (load-static) qdisplay table — it does NOT
 * take a `void game.q` tick dependency, because it reads no live state: the
 * VALUE is what changes, and whatever produced it (a `deriveFrom` row set,
 * rebuilt by WidgetHost on every tick) already owns that dependency. Reading
 * `game.adapter` does give any caller the one dependency that matters here —
 * the adapter arriving at boot.
 */
export function useBand(): {
  band: (value: unknown, qdisplayId?: string | null) => Band;
} {
  const game = useGameStore();

  function band(value: unknown, qdisplayId?: string | null): Band {
    const adapter = game.adapter;
    // `null` is the row contract's "nothing to classify here" (brief.js:87 —
    // your own bench, an unassigned relation); a null/absent qdisplay id is
    // the same statement about the scale (`valueDisplay: null` on a minister's
    // name or a seat count). Neither is an error.
    if (!adapter || !qdisplayId || value === null || value === undefined) return EMPTY;
    let html: string;
    try {
      html = adapter.qdisplay(value, qdisplayId);
    } catch (err) {
      // Unknown qdisplay id, or a value the engine cannot stringify. A bad
      // row must not blank the sheet around it.
      console.warn(`useBand: qdisplay(${String(value)}, "${qdisplayId}") threw`, err);
      return EMPTY;
    }
    return parseQdisplayHtml(html);
  }

  return { band };
}
