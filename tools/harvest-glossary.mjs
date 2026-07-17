// One-shot migration: out/html/data.js -> source/data/glossary.json
// Run: node tools/harvest-glossary.mjs
//
// data.js has TWO tables keyed by the SAME search words: tooltipList (95) and
// colourList (102). They are one table. Merge them; report any entry that
// appears in only one, any word claimed by two different ids, and any colour
// that is a raw hex rather than a token — those are the cases where a silent
// merge would lose data.
//
// Two deliberate departures from a naive "one pass over each list, key by
// slugify(words[0])" merge, discovered by actually reading every entry (the
// brief's own warning: "do not eyeball it") rather than trusting the shape of
// the two tables to line up:
//
// 1. Colour buckets are built from colourList ALONE (never bridged through
//    tooltipList). colourList sometimes has TWO entries for one concept that
//    tooltipList treats as ONE — e.g. "ICV"/"ICV-EUiA"/"icv" share a single
//    tooltip, but colourList renders bare "ICV" as itself and "ICV-EUiA"/"icv"
//    as the expanded name (two different `transform` values); same for
//    "Omnium" vs "Omnium Cultural". A single JSON term can only hold one
//    `display` value, so merging these into one id would silently change what
//    one of the two literal word-variants renders as. Fix: colour buckets are
//    formed from colourList's own word-overlaps only; tooltip content is then
//    attached to EVERY colour bucket its search words touch (duplicated, not
//    shared by reference — cheap, and there is no third bucket in this data
//    that needs true reference sharing).
// 2. `allegiances` (a `(Q) => [...]` closure with real branching — if/else,
//    string equality, numeric comparisons — on ~19 leader entries) is NOT
//    migrated into JSON, and is reported below rather than silently dropped.
//    A JSON registry can hold data, not code; encoding arbitrary Q-conditional
//    branching declaratively would mean inventing a condition-expression
//    mini-language, which this project already tried once for a similar
//    reason (source/data/brief.json, rejected — see
//    docs/design/LEARNINGS.md 2026-07-13) and for the identical reason: real
//    branching logic belongs in a renderer, not a data file. Those closures
//    are ported verbatim into out/html/game.js as game-local code; every id
//    that had one is named below so nobody mistakes its absence from
//    glossary.json for accidental data loss.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync('out/html/data.js', 'utf8');
const ctx = { window: {} };
vm.createContext(ctx);
vm.runInContext(src + '\n;globalThis.__t = tooltipList; globalThis.__c = colourList;', ctx);
const tooltips = vm.runInContext('__t', ctx);
const colours = vm.runInContext('__c', ctx);

const idFor = (word) => word.toLowerCase().replace(/[^a-z0-9]+/g, '_');
const warnings = [];

function addWords(list, words) {
  for (const w of words) if (!list.includes(w)) list.push(w);
}

// ---- Step 1: colour buckets — union-find over colourList's OWN words only ----
const parent = new Map();
function find(w) {
  if (!parent.has(w)) parent.set(w, w);
  let r = w;
  while (parent.get(r) !== r) r = parent.get(r);
  let cur = w;
  while (parent.get(cur) !== r) {
    const next = parent.get(cur);
    parent.set(cur, r);
    cur = next;
  }
  return r;
}
function union(a, b) {
  const ra = find(a);
  const rb = find(b);
  if (ra !== rb) parent.set(ra, rb);
}
for (const c of colours) {
  for (const w of c.words) find(w);
  for (let i = 1; i < c.words.length; i++) union(c.words[0], c.words[i]);
}

const colourGroups = new Map(); // union-find root -> { colourEntries, words }
function cGroupFor(root) {
  let g = colourGroups.get(root);
  if (!g) {
    g = { colourEntries: [], words: [] };
    colourGroups.set(root, g);
  }
  return g;
}
for (const c of colours) {
  const g = cGroupFor(find(c.words[0]));
  g.colourEntries.push(c);
  addWords(g.words, c.words);
}
for (const g of colourGroups.values()) {
  if (g.colourEntries.length > 1) {
    warnings.push(
      `MERGE: ${g.colourEntries.length} colourList entries share a literal search word and were merged into one id — verify this is intentional: ${g.colourEntries.map((e) => JSON.stringify(e.words)).join(' / ')}`,
    );
  }
}

// ---- Step 2: seed one term per colour bucket ----
const KNOWN_COLOUR_KEYS = new Set(['words', 'colour', 'style', 'transform']);
const termsById = new Map();
const idForColourBucket = new Map(); // union-find root -> id, for step 3 lookups

for (const [root, g] of colourGroups) {
  const c = g.colourEntries[0];
  for (const k of Object.keys(c)) {
    if (!KNOWN_COLOUR_KEYS.has(k))
      warnings.push(`UNHANDLED FIELD: colourList entry for "${c.words[0]}" has unrecognised key "${k}" — not migrated, check by hand`);
  }
  const id = idFor(c.words[0]);
  idForColourBucket.set(root, id);
  termsById.set(id, {
    id,
    match: [...g.words],
    display: c.transform ?? undefined,
    // "var(--ciu)" -> "ciu". A raw hex (e.g. CUP's "#b8a12b") is kept verbatim
    // and flagged: it means the old UI had no token for it.
    colour: c.colour?.startsWith('var(--') ? c.colour.slice(6, -1) : c.colour,
    bold: /font-weight:\s*bold/.test(c.style ?? '') || undefined,
  });
}

// ---- Step 3: attach tooltip content to every colour bucket its words touch;
// tooltip-only entities (touch no colour bucket at all) become their own term.
const KNOWN_TOOLTIP_KEYS = new Set([
  'searchString', 'mainText', 'subText', 'img', 'ledBy', 'ideology', 'infoDesc', 'allegiances',
]);
const allegianceIds = new Set();

for (const t of tooltips) {
  for (const k of Object.keys(t)) {
    if (!KNOWN_TOOLTIP_KEYS.has(k))
      warnings.push(`UNHANDLED FIELD: tooltipList entry for "${t.searchString[0]}" has unrecognised key "${k}" — not migrated, check by hand`);
  }

  const touchedIds = new Set();
  for (const w of t.searchString) {
    if (!parent.has(w)) continue;
    const root = find(w);
    const id = idForColourBucket.get(root);
    if (id) touchedIds.add(id);
  }

  const q = {};
  for (const k of ['ledBy', 'ideology']) if (t[k]) q[k] = t[k];
  const tooltip = {
    title: t.mainText,
    subtitle: t.subText || undefined,
    img: t.img || undefined,
    infoDesc: t.infoDesc || undefined,
    q: Object.keys(q).length ? q : undefined,
  };
  const hasAllegiances = typeof t.allegiances === 'function';

  if (touchedIds.size === 0) {
    // Pure tooltip-only entity: no colour presence anywhere (e.g. Pablo Iglesias).
    const id = idFor(t.searchString[0]);
    const term = termsById.get(id) ?? { id, match: [] };
    addWords(term.match, t.searchString);
    term.tooltip = tooltip;
    termsById.set(id, term);
    if (hasAllegiances) allegianceIds.add(id);
    continue;
  }

  // A word in the tooltip's own list that isn't already claimed by ANY of the
  // colour buckets it touches. Safe to fold in when there's exactly one
  // touched bucket (e.g. tooltip's "pp_bcn" alongside colour's "ppbcn"); with
  // more than one touched bucket, which one it belongs to is a real
  // ambiguity — don't guess, warn instead.
  const uncovered = t.searchString.filter((w) => {
    for (const id of touchedIds) if (termsById.get(id).match.includes(w)) return false;
    return true;
  });
  if (uncovered.length && touchedIds.size > 1) {
    warnings.push(
      `AMBIGUOUS WORD: ${JSON.stringify(uncovered)} in tooltip "${t.mainText}" touches ${touchedIds.size} different colour ids (${[...touchedIds].join(', ')}) — not assigned to any, check by hand`,
    );
  }

  for (const id of touchedIds) {
    const term = termsById.get(id);
    term.tooltip = tooltip; // shared content, duplicated per bucket (JSON, not a reference)
    if (touchedIds.size === 1 && uncovered.length) addWords(term.match, uncovered);
    if (hasAllegiances) allegianceIds.add(id);
  }
}

const terms = [...termsById.values()].sort((a, b) => a.id.localeCompare(b.id));

// Report, do not silently drop.
const seen = new Map();
for (const t of terms)
  for (const w of t.match) {
    if (seen.has(w)) warnings.push(`COLLISION: "${w}" claimed by ${seen.get(w)} and ${t.id}`);
    seen.set(w, t.id);
  }
for (const t of terms) {
  if (!t.tooltip) warnings.push(`colour-only (no tooltip): ${t.id}`);
  if (!t.colour) warnings.push(`tooltip-only (no colour):  ${t.id}`);
  // Token names come from CSS custom-property names (e.g. --ua-psc), which
  // may contain hyphens — only flag values that aren't var(--...) at all.
  if (t.colour && !/^[a-z0-9_-]+$/.test(t.colour)) warnings.push(`RAW COLOUR (no token): ${t.id} = ${t.colour}`);
}
for (const id of allegianceIds) {
  warnings.push(`ALLEGIANCES (not migrated — executable Q-conditional logic, kept as code in out/html/game.js): ${id}`);
}

for (const w of warnings) console.warn(w);

mkdirSync('source/data', { recursive: true });
writeFileSync('source/data/glossary.json', JSON.stringify({ terms }, null, 2) + '\n');
console.log(`wrote ${terms.length} terms`);
console.log(`${warnings.length} warnings total`);
