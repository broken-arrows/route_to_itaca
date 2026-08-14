<script setup lang="ts">
/**
 * One Brief sheet: the shared paper chrome plus the scene's rendered content.
 *
 * Content arrives as HTML from `renderView` and goes through Prose, NOT a bare
 * v-html: Prose is what marks glossary terms and mounts widgets (WidgetHost).
 * A bare v-html here would silently strip both — the same class of regression
 * as the window.displayText hole (LEARNINGS 2026-07-13).
 */
import Prose from '../Prose.vue';

defineProps<{ title: string; context: string; html: string }>();
</script>

<template>
  <div class="sheet">
    <span class="clip"></span>
    <header class="sheet-head">
      <h2>{{ title }}</h2>
      <span class="context">{{ context }}</span>
    </header>
    <div class="sheet-body">
      <Prose :html="html" />
    </div>
  </div>
</template>

<style scoped>
.sheet {
  position: relative;
  z-index: 2;
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  background: var(--paper-0);
  border: 1px solid #e0d9c8;
  box-shadow: 0 3px 12px rgba(60, 45, 20, 0.14);
  padding: 24px 22px 16px;
  overflow: hidden;
}
.clip {
  position: absolute;
  left: 50%;
  top: -8px;
  transform: translateX(-50%);
  width: 104px;
  height: 19px;
  background: #c9c0aa;
  border: 1px solid #a89e8c;
  border-radius: 6px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.18);
}
.sheet-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 3px double #c6bda8;
  padding-bottom: 8px;
  margin-bottom: 12px;
}
.sheet-head h2 {
  font-family: var(--font-news);
  font-weight: 800;
  font-size: 21px;
  font-variant: small-caps;
  color: #2e2a22;
  margin: 0;
}
.context {
  font-family: var(--font-title);
  font-weight: 600;
  font-size: 11.5px;
  letter-spacing: 0.12em;
  color: #a89e8c;
  text-align: right;
}
.sheet-body {
  flex: 1;
  min-height: 0;
  font-size: 12px;
  line-height: 1.5;
  overflow-x: hidden;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: #c9c0aa transparent;
}

/* ===========================================================================
 * Everything below reaches ENGINE-AUTHORED HTML: the sheet's markup comes from
 * `source/scenes/status_new.scene.dry` through `renderView` -> Prose's
 * `v-html`, and (from Task B) from widget SFCs mounted inside it. Scoped CSS
 * does not reach either, hence `:deep()` on every rule — the scope attribute
 * lands on `.sheet-body`, which is this component's own element, and the
 * selector inside `:deep()` matches anywhere beneath it. Keeping the grammar
 * in this one scoped block is deliberate: no global `.brief-*` namespace, and
 * the Brief cannot leak style onto the desk/newspaper surfaces.
 *
 * `status_new.scene.dry` is rendered by the DESK ONLY (the old shell renders
 * the separate `status.scene.dry`), so the class names it authors are
 * legitimately Desk-specific. See that file's header.
 *
 * Hexes: `var(--…)` where tokens.css has the exact value, otherwise the
 * design-canvas literal, transcribed — same convention as Clipboard.vue.
 * ======================================================================== */

/* --- paragraph rhythm ----------------------------------------------------
 * Dendry wraps every content line in <p>, so `<p><div class="brief-card">…`
 * is what the sheets emit; the HTML parser hoists the block out and leaves
 * empty <p> shells on both sides of it. They are invisible but they carry the
 * UA's 1em margins. Both rules apply to every <p> the sheet body has ever
 * contained (engine prose, nothing else), which all carried the same UA
 * default — this is a uniform replacement, not a property added to elements
 * that never had one. */
.sheet-body :deep(p) {
  margin: 0 0 8px;
}
.sheet-body :deep(p:empty) {
  display: none;
}

/* --- section labels (§1 chrome) -----------------------------------------
 * The sheets' `=` headings compile to <h1> (verified against renderView's
 * real output). They are labels, not headings: 800 11px HK, ls .14em,
 * #a89e8c. h2/h3 are styled alongside so a future `==` reads the same; only
 * `=` is used today. */
.sheet-body :deep(h1),
.sheet-body :deep(h2),
.sheet-body :deep(h3) {
  font-family: var(--font-title);
  font-weight: 800;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #a89e8c;
  margin: 14px 0 6px;
}

/* --- shape 1/5 row: dotted leader, key left, value right -----------------
 * Key 600 15px HK #3a342c · leader 1.5px dotted #c6bda8 · value 700
 * Newsreader. EXACTLY TWO children, always: the leader is drawn by the first
 * child's ::after eating the slack, which is what keeps the value hard right
 * without a table. (With one child, :first-child and :last-child both match.) */
.sheet-body :deep(.brief-row) {
  display: flex;
  align-items: baseline;
  margin: 0 0 3px;
}
.sheet-body :deep(.brief-row > :first-child) {
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  align-items: baseline;
  white-space: nowrap;
  font-family: var(--font-title);
  font-weight: 600;
  font-size: 15px;
  color: #3a342c;
}
.sheet-body :deep(.brief-row > :first-child)::after {
  content: '';
  flex: 1 1 auto;
  align-self: center;
  position: relative;
  top: 0.22em; /* centre of the line box -> just under the text baseline */
  min-width: 14px;
  margin: 0 7px;
  border-bottom: 1.5px dotted #c6bda8;
}
.sheet-body :deep(.brief-row > :last-child) {
  flex: 0 0 auto;
  text-align: right;
  font-family: var(--font-news);
  font-weight: 700;
  font-size: 15px;
  color: #2e2a22;
}

/* --- the framed resource box (§2 frame 11) -------------------------------
 * `#f6f2e6` on `#e2d8bd`. Children, in order: <strong> = the figure, and a
 * final <span> = the small-caps label (with an optional <em> second line). */
.sheet-body :deep(.brief-boxes) {
  display: flex;
  align-items: stretch;
  gap: 10px;
  margin: 0 0 10px;
}
.sheet-body :deep(.brief-boxes > .brief-box) {
  flex: 1 1 0;
  min-width: 0;
  margin: 0;
}
.sheet-body :deep(.brief-box) {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 12px;
  margin: 0 0 10px;
  background: #f6f2e6;
  border: 1px solid #e2d8bd;
  border-radius: 3px;
}
.sheet-body :deep(.brief-box > strong) {
  flex: 0 0 auto;
  font-family: var(--font-news);
  font-weight: 800;
  font-size: 19px;
  line-height: 1.1;
  color: #2e2a22;
  white-space: nowrap;
}
.sheet-body :deep(.brief-box > span:last-child) {
  font-family: var(--font-title);
  font-weight: 700;
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  line-height: 1.35;
  color: #6b655a;
}
.sheet-body :deep(.brief-box em) {
  display: block;
  font-style: normal;
  font-weight: 400;
  font-size: 10.5px;
  letter-spacing: 0.02em;
  text-transform: none;
  color: #a89e8c;
}

/* --- the dashed ghost: "when you govern" / not yet adopted ---------------
 * §2 frames 11, 12 and 14 all draw the not-yet state as a dashed box with no
 * fill, its figure greyed out. */
.sheet-body :deep(.brief-ghost) {
  background: transparent;
  border-style: dashed;
  border-color: #d8cfb6;
}
.sheet-body :deep(.brief-ghost > strong) {
  color: #b8b0a0;
}

/* --- shape 5: the institution index card --------------------------------
 * `#fdfcf8`, border `#e0d9c8`, soft shadow. The Generalitat and the Gobierno
 * use THIS ONE FRAME so the eye compares them — that is the design's point;
 * there is no per-institution variant and none may be added.
 * `.brief-card-head` holds exactly two children: name, then place label. */
.sheet-body :deep(.brief-card) {
  background: #fdfcf8;
  border: 1px solid #e0d9c8;
  border-radius: 3px;
  box-shadow: 0 1px 3px rgba(60, 45, 20, 0.07);
  padding: 9px 12px 10px;
  margin: 0 0 6px;
}
.sheet-body :deep(.brief-card-head) {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 5px;
}
.sheet-body :deep(.brief-card-head > :first-child) {
  font-family: var(--font-news);
  font-variant: small-caps;
  font-weight: 700;
  font-size: 14.5px;
  letter-spacing: 0.02em;
  color: #2e2a22;
}
.sheet-body :deep(.brief-card-head > :last-child) {
  flex: 0 0 auto;
  font-family: var(--font-title);
  font-weight: 600;
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #a89e8c;
}

/* --- shape 8: prose ------------------------------------------------------
 * Typewriter body, 400 14.5px/1.55, TWO LINES MAXIMUM and only ever at the
 * top of a sheet. The clamp is the rule made mechanical: a third line is
 * newspaper copy, not Brief copy. */
.sheet-body :deep(.brief-note) {
  font-family: var(--font-typed);
  font-weight: 400;
  font-size: 14.5px;
  line-height: 1.55;
  color: #6b655a;
  margin: 0 0 10px;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  overflow: hidden;
}

/* --- shape 2B: the rubber stamp -----------------------------------------
 * 1.5px border, rotated ~-3°, letter-spaced caps, coloured by band token.
 *
 * The chrome goes on whichever element CARRIES the token, so `currentColor`
 * is already the band's ink and the border needs no second colour table:
 *   - content wraps a qdisplay's own output, so the token is on the inner
 *     `<span class="q-band" data-band=…>`  ->  `.brief-stamp > .q-band`;
 *   - a widget builds its own stamp from `useBand()` and puts the token on
 *     the element itself  ->  `.brief-stamp[data-band]`.
 * A bare `.brief-stamp` with neither is intentionally un-stamped: a stamp
 * with no band has no ink to draw itself in. */
.sheet-body :deep(.brief-stamp[data-band]),
.sheet-body :deep(.brief-stamp > .q-band) {
  display: inline-block;
  border: 1.5px solid currentColor;
  border-radius: 2px;
  padding: 1px 6px 2px;
  transform: rotate(-3deg);
  font-family: var(--font-title);
  font-weight: 700;
  font-size: 9.5px;
  line-height: 1.35;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  white-space: nowrap;
}
/* The stamp line: label · stamp, centred (§2 frame 11's MADRID ⇄ BARCELONA). */
.sheet-body :deep(.brief-stamp-line) {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin: 8px 0;
}
.sheet-body :deep(.brief-stamp-line > span:first-child) {
  font-family: var(--font-title);
  font-weight: 600;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #a89e8c;
}

/* --- the band ink ramp ---------------------------------------------------
 * ONE Desk ramp, keyed on `data-band` alone. The old shell keys on
 * `[data-scale][data-band]` because its four ladder scales each had their own
 * palette and it must stay pixel-identical; the Desk deliberately does not
 * (LEARNINGS 2026-07-22: "the Desk ignores data-scale and inks one consistent
 * ramp"). This is also what gives `.brief-stamp` its border: `currentColor`.
 *
 * ANCHORS ARE THE DESIGN'S: bad red `#b03030` and good green `#3f8f3f` (§1
 * shape 1), party gold `#a9821f` (§1 shape 6 / `--accent-gold`), slate
 * `#4a5b6a` (`--accent-slate`), grey `#8a8273` (§1 shape 4). The intermediate
 * rungs are a straight green->gold->red interpolation between them — the
 * design names endpoints, not seven hexes. Cross-check: §2 frame 14 says debt
 * MEDIOCRE is gold, and `mediocre` is rung 4 of 7, i.e. exactly the gold
 * anchor.
 *
 * NB `#b03030` is the literal tokens.css records as replaced by
 * `--paper-rule-ink` under the red-reservation rule. It is used here as the
 * design's own signal red for the Brief's bad-end ink, the same way
 * Clipboard.vue uses `--accent-red` for the active tab — flagging it rather
 * than silently repainting the design. */

/* the shared very_low…very_high ladder: social_dissent, cat_spa_relations,
   independence_movement (7 rungs) and dissent (5 of the same tokens). */
.sheet-body :deep([data-band='very_low']) { color: #3f8f3f; }
.sheet-body :deep([data-band='low']) { color: #628b34; }
.sheet-body :deep([data-band='medium_low']) { color: #86862a; }
.sheet-body :deep([data-band='medium']) { color: #a9821f; }
.sheet-body :deep([data-band='medium_high']) { color: #ab6725; }
.sheet-body :deep([data-band='high']) { color: #ad4b2a; }
.sheet-body :deep([data-band='very_high']) { color: #b03030; }

/* public_debt_qualifier — same ramp, excellent (good) -> terrible (bad). */
.sheet-body :deep([data-band='excellent']) { color: #3f8f3f; }
.sheet-body :deep([data-band='good']) { color: #628b34; }
.sheet-body :deep([data-band='fair']) { color: #86862a; }
.sheet-body :deep([data-band='mediocre']) { color: #a9821f; }
.sheet-body :deep([data-band='poor']) { color: #ab6725; }
.sheet-body :deep([data-band='very_poor']) { color: #ad4b2a; }
.sheet-body :deep([data-band='terrible']) { color: #b03030; }

/* politics_trust — the same ramp INVERTED: distrust is the bad end. */
.sheet-body :deep([data-band='completely_distrustful']) { color: #b03030; }
.sheet-body :deep([data-band='distrustful']) { color: #ad4b2a; }
.sheet-body :deep([data-band='skeptical']) { color: #ab6725; }
.sheet-body :deep([data-band='willing']) { color: #86862a; }
.sheet-body :deep([data-band='trusting']) { color: #628b34; }
.sheet-body :deep([data-band='blindly_trusting']) { color: #3f8f3f; }

/* control's ladder — §2 frame 15 names these outright: NONE red / LIMITED
   gold / PARTIAL slate / FULL (= complete) green. `disputed` is a fifth rung
   the design's four-notch caption does not cover; it takes the darker
   `--accent-red` so it reads as a harder failure than plain NONE. */
.sheet-body :deep([data-band='none']) { color: #b03030; }
.sheet-body :deep([data-band='limited']) { color: var(--accent-gold); }
.sheet-body :deep([data-band='partial']) { color: var(--accent-slate); }
.sheet-body :deep([data-band='complete']) { color: #3f8f3f; }
.sheet-body :deep([data-band='disputed']) { color: var(--accent-red); }

/* international_opinion — §2 frame 15: WATCHING slate, UNAWARE grey. */
.sheet-body :deep([data-band='hostile']) { color: #b03030; }
.sheet-body :deep([data-band='unaware']) { color: #8a8273; }
.sheet-body :deep([data-band='watching']) { color: var(--accent-slate); }
.sheet-body :deep([data-band='sympathetic']) { color: #628b34; }
.sheet-body :deep([data-band='supportive']) { color: #3f8f3f; }

/* roadmap — §2 frame 12 names only UNSET (grey). The three adopted lines are
   not a good/bad scale, so they read as escalation: negotiated / pressure /
   rupture. Derived, not transcribed. */
.sheet-body :deep([data-band='unset']) { color: #8a8273; }
.sheet-body :deep([data-band='vote']) { color: var(--accent-slate); }
.sheet-body :deep([data-band='negotiate']) { color: var(--accent-gold); }
.sheet-body :deep([data-band='unilateral']) { color: #b03030; }

/* relationships carries no data-band of its own — useBand() slugs its word
   into one (`warm`, `cold`, `very_friendly`, …). §2 frame 13's bench stamps:
   WARM/FRIENDLY green, COLD slate, HOSTILE red, YOU neutral ink. `hostile` is
   shared with international_opinion and `neutral` with politics_trust; both
   mean the same thing on both scales, which is what makes one ramp possible. */
.sheet-body :deep([data-band='frigid']) { color: #3c4c5a; }
.sheet-body :deep([data-band='cold']) { color: var(--accent-slate); }
.sheet-body :deep([data-band='cool']) { color: #6b7c8a; }
.sheet-body :deep([data-band='neutral']) { color: #a9821f; }
.sheet-body :deep([data-band='warm']) { color: #628b34; }
.sheet-body :deep([data-band='friendly']) { color: #3f8f3f; }
.sheet-body :deep([data-band='very_friendly']) { color: #2f7a33; }
/* Not a qdisplay band: the design stamps your own bench YOU, in neutral ink. */
.sheet-body :deep([data-band='you']) { color: #6b655a; }

/* international_opinion is the one qdisplay that emits a <p>; keep it from
   picking up the paragraph rhythm above when a widget nests it in a row. */
.sheet-body :deep(p.q-band) {
  margin: 0;
  font-style: italic;
}
</style>
