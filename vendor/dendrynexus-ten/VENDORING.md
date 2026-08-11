# Vendored `dendrynexus-ten`

This directory is a **vendored, in-tree copy** of the `dendrynexus` narrative engine,
detached from npm so it can be modified directly as part of _Route to Ítaca_. It is
distributed under this repo as the fork `dendrynexus-ten`.

- **Upstream:** https://github.com/aucchen/dendrynexus
- **Vendored from commit:** `49e259416940b3537ac992c3d33a99c6f5e5b209` (`49e2594`)
- **Vendored on:** 2026-07-04
- **Fork renamed to `dendrynexus-ten`:** 2026-07-08

## How it's wired

`package.json` (repo root) consumes this via a local dependency:

```json
"dendrynexus-ten": "file:vendor/dendrynexus-ten"
```

`npm install` symlinks this folder into `node_modules/dendrynexus-ten`, so edits here are
**live** — no reinstall needed. The CLI `bin` (`dendrynexus-ten`) stays linked, so
`npm run dendrynexus-ten make-html -- --pretty` builds the game as before. The engine is
browserified into `out/html/core.js` (generated/gitignored) at build time.

## Diffing against / rebasing on upstream

The recorded base commit lets a future extraction still track upstream. Example:

```sh
# in a scratch clone of this vendored source
git remote add upstream https://github.com/aucchen/dendrynexus.git
git fetch upstream
git diff 49e2594 upstream/master   # see what changed upstream since we forked
```

## Promoting this back into its own repo

Vendoring is not a one-way door. To extract this folder (with its history) into a
standalone repo later:

```sh
git subtree split --prefix=vendor/dendrynexus-ten -b dendrynexus-ten-split
# then push dendrynexus-ten-split to a new empty repo
```

(or `git filter-repo --path vendor/dendrynexus-ten` for a cleaner rewrite).

## Local changes (not upstream)

Kept deliberately minimal — touch only the lines that must change, never a
whole-file reformat, so this list stays the complete upstream diff.

- **`lib/parsers/info.js`, `lib/parsers/validators.js`,
  `lib/parsers/compiler.js`, and `lib/engine.js`** (2026-08-11) — game
  manifests may declare a validated, lowercase `storage-id` and a two- or
  three-component `version` with an optional hyphenated tag. The nested
  runtime `game.info` manifest now carries those fields plus `ifid`.
  Achievement persistence uses `<storageId>:achievements` when the field is
  present, with a title-based fallback only for third-party games that omit
  it. There is deliberately no read fallback or migration from the old title
  key when a storage id exists.

- **`lib/persistence.js`, `lib/ui/browser.js`, and stock HTML templates**
  (2026-08-11) — both shells share manifest-scoped, versioned save envelopes,
  positional two-save autos, and one-based manual slot names. The old shell
  exposes manual 1…8, has no quicksave, and stores its settings separately at
  `<storageId>:settings-old`. Canonical-envelope imports only; corrupt and
  unsupported records remain manageable without being loadable.

- **`lib/ui/browser.js` + `lib/ui/save-label.js`** (2026-07-26) — the old
  shell sanitizes the scene-id line when populating save slots: it removes
  `post_event`, drops nested suffixes after `.`, replaces `_` with spaces, and
  title-cases the result. Stored timestamps and save blobs stay unchanged, so
  existing saves improve at display time without a migration.
- **`lib/engine.js`** (2026-07-20) — two additive changes:
  1. **Loud, findable swallow.** The three scene-code swallow sites
     (`runActions`, `runPredicate`, `runExpression`) still swallow (a broken
     block must not crash the game) but now report via a shared `logCodeError`
     helper — `console.error` naming the exact scene, the phase, and the
     offending source line, instead of an anonymous `console.warn`/`console.log`
     with no scene id. `runActions`/`_runActions` gained a `phase` param, threaded
     from the five call sites (`'on-arrival'`, `'on-departure'`, `'on-display'`,
     `'on-arrival (call)'`). Purely diagnostic; no behaviour change, still
     swallows. Fixes the "faulty JS in a scene block silently stops running"
     DX gap (a half-run on-arrival was previously invisible without a console).
     To name the failing **statement** rather than the
     block — pointing at a 556-line `on-arrival` is barely better than not
     pointing at all. `logCodeError` now resolves the throw back to the exact
     line of the block's own source and prints it with ±1 line of context and
     a caret, then drops the fixed engine/jQuery frames below it (frames
     _above_ are kept: that is where a `G.*` call actually broke). Four new
     module-private helpers — `findCodeFrame`, `BODY_LINE_OFFSET`, `excerpt`,
     `describeError`. Two non-obvious points:
     - The `new Function` wrapper's line offset is **probed at load**, not
       hardcoded: a throwaway block built through `makeFunctionFromSource`
       itself throws on source line 1 and the reported line gives the offset
       (2 on V8 and SpiderMonkey). The wrapper's shape is a runtime detail,
       not a spec guarantee.
     - The generated-frame regex is **anchored** on `<anonymous>`/`Function`
       because V8 embeds the _defining_ location in the same frame
       (`at eval (eval at makeFunctionFromSource (…:57:14), <anonymous>:83:28)`)
       and Windows paths carry a drive-letter colon — a bare `/:(\d+):(\d+)/`
       resolves every error to the same wrong line.
       Still swallows, still no behaviour change; the whole formatter is wrapped
       in its own try/catch that degrades to the previous flat message, since it
       runs inside a catch handler and must never be the thing that throws.
       Covered by `ui/tests/engine.code-errors.test.ts` (7 cases).
  2. **`state.lastPlayedCardQ`** — `playCard` now shallow-snapshots
     `Object.assign({}, this.state.qualities)` **before** running the played
     card's on-arrival (added to the initial state as `null`). A generic
     mechanism like `lastPlayedCard` itself; the engine attaches no meaning to
     it. Consumed by `source/scenes/easy_discard.scene.dry` to revert the
     card's own cooldown timer on "return to hand" (the engine's hand re-filter
     — `displayChoices` → `__filterViewable` — otherwise drops the returned card
     the instant its own view-if cooldown fails).
- **`lib/parsers/compiler.js`** (`compileGame`) — routes any
  `source/data/*.json` file to a generic registry keyed by its basename,
  attached at `game.json.data.<basename>` (e.g. `source/data/glossary.json` →
  `game.data.glossary`). Absent when there are no `data/*.json` files.
  Malformed registry JSON fails the compile with an error naming the file,
  rather than being silently skipped. Additive top-level key; the old UI's
  `convertJSONToGame` ignores unknown keys.
- **`lib/cli/utils.js`** (`fetchContent`) — added a `CONTENT_PATTERN` read
  filter (`/(\.dry|[\\/]data[\\/][^\\/]+\.json)$/`) so only actual game
  content is read and `.toString()`'d into memory. Previously `fetchContent`
  walked `source/` with no pattern at all, reading and stringifying every
  file it found (including binaries) before `compileGame` silently skipped
  unrecognised ones. Prerequisite for `source/lib/` and `source/img/`
  existing alongside compiled content without every compile slurping them.
- **`lib/parsers/scene.js` + `lib/parsers/compiler.js`** (2026-07-19) — the
  `card-parlament` role renamed to **`card-parliament`** in the role enum and
  the role→mechanics map. "Parlament" is Route to Ítaca's chamber; the role
  vocabulary is engine surface shared by any game, so it uses the generic
  English word. One string, no behaviour change; `source/` and `ui/` renamed in
  the same pass.
- **`lib/engine.js`** (2026-07-13, `engine.setGameLib(lib)`) — compiled
  `$code` gains a third parameter, `G`: `makeFunctionFromSource` now does
  `new Function('state', 'Q', 'G', source)` (was `'state', 'Q'`), and the
  three call sites (`runActions`, `runPredicate`, `runExpression`) pass
  `context.gameLib` as that third argument. `DendryEngine` gets a `gameLib`
  instance property (default `{}`, never `undefined`) and
  `setGameLib(lib)` to install it — same shape as `setLocale`: a UI hands
  the engine the game's own code (`source/lib/*`), and neither UI knows what
  is in it. `beginGame()` prints one `console.warn` if `setGameLib` was
  never called, so a UI that forgets it fails loudly instead of reproducing
  the swallowed-`TypeError` bug this API exists to kill (content calling
  `G.engineTick(Q)` instead of the browser-only `window.engineTick(Q)`).
  Purely additive: content that never mentions `G` is unaffected. See
  `lib/engine.js` around lines 50-105 (`makeFunctionFromSource`/
  `runActions`/`runPredicate`/`runExpression`), ~318-340 (constructor +
  `setGameLib`), and `beginGame`'s first statement.
- **`deck-gov` / `deck-party` / `deck-parliament` role variants**: `lib/parsers/scene.js` role enum + `lib/parsers/compiler.js`
  `ROLE_TO_MECHANIC` (all derive `isDeck`, mirroring the `card-*` family). Lets a deck
  scene carry its paper skin so no UI needs a deck-id table. Generic vocabulary — no
  game names.
- **`status` role** (2026-07-22) — `lib/parsers/scene.js` role enum only; no
  `ROLE_TO_MECHANIC` entry, because like `info-tab` it is presentational and
  implies no mechanical boolean. Marks the scene whose **options declare a tab
  set** — the same "the hub scene's options declare what is on the surface"
  rule `desk` already uses for its trays, so a UI finds the tab list by role
  instead of hardcoding one. Generic vocabulary: the engine learns that a
  tabbed info surface can exist, never what this game's tabs are.
