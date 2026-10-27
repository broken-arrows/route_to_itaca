# DendryNexus-Ten

In-tree **vendored fork** of [DendryNexus](https://github.com/aucchen/dendrynexus), a
StoryNexus-style card-game superset of [Dendry](https://github.com/idmillington/dendry),
maintained locally for _Route to Ítaca_. The upstream base commit and the
`git subtree` sync recipe are recorded in [`VENDORING.md`](./VENDORING.md). MIT-licensed
(see `LICENSE`).

Edits here are **live**: the engine is a `file:` dependency, so changes flow into the game
on the next build with no reinstall.

## What `-ten` adds on top of DendryNexus

- **`choice-image` optional scene/choice attribute**: an image URL passed through as
  `choiceImage` in choice display records (a choice override takes precedence over
  its destination scene). Classic reuses `.card` and `.card-img` for illustrated
  choices, with the title and subtitle alongside. Rows use one column unless there
  is room for two. Text-only choices follow in the ordinary `.choices` list.
  Hiding portraits returns all choices to that ordinary list. This is independent of
  `card-image` and `face-image`. Other renderers may ignore it.
- **`role` scene attribute**: a compiler-validated enum naming which UI surface a scene
  renders on: `desk`, `deck`, `card`, `pinned-action`, `newspaper`, `event`, `info-tab`,
  `pause-item`, `main-menu-item`, `library-item`, `ending`, `default`. Additive and
  presentation-only (tags still drive deck draw-pools). `default`/inheritance is resolved
  by the UI adapter at runtime; the compiler only validates the enum.
- **`info` manifest → `game.json`**: the compiler emits a whitelisted `info` block
  (`title`, `author`, `ifid`, `version`, `languages`) so the runtime can read game-level metadata.
  `languages: en ca` in `info.dry` declares the locale set. Browser persistence uses
  `dnt:<ifid>` without a separate `storage-id`. (Also fixes a Windows-only bug
  where `info.dry` was silently skipped by a forward-slash-only path check.)
- **Runtime i18n overlay**: `engine.setLocale(locale, catalog)` installs a translation
  catalog keyed by English templates, with named inserts, nested recipes and
  select/plural entries. Engine-owned presentation recipes capture context while
  qualities and saved fallback prose remain English. Locale refresh redraws
  existing output without replaying actions or consuming randomness. See
  [LOCALIZATION.md](LOCALIZATION.md) for the authoring and catalog contract.
- **`on-display`-on-load fix**: `onDisplay` now re-fires when a saved state is loaded,
  before choices are recompiled, matching normal scene entry.

## Build commands

- `dendrynexus-ten compile`: compile the authored game into `out/game.json`.
- `dendrynexus-ten make-html`: build the engine's classic browser shell. Add
  `--publish-game-lib` when that shell loads `source/lib/`; this requires
  `source/lib/index.js` and copies the directory to `out/html/lib/`.
  Add `--publish-game-assets` to validate and publish `source/img/` and declared
  locales' runtime `content.json`/`ui.json` files. Missing required inputs fail;
  untranslated entries warn and keep English fallback. Reports are excluded.
- `dendrynexus-ten audit`: check compiled scene code for browser globals and,
  when supplied by `dendrynexus.audit.mjs`, validate widget and derivation names.
- `make-html --unminified` leaves `core.js` readable. `--pretty` is an older alias;
  the default build minifies it with Terser.

This CLI does not build a project's separate Vue/Vite UI. In Route to Ítaca,
`npm run build:classic` combines the old UI build, library/asset publication and
the project audit. `npm run audit` runs the engine's compiled-content audit and
runtime asset validation together. `npm run package` builds both UIs and stages
them locally; these project commands are defined in the repository root.

## Core features (from DendryNexus)

### Hands

A scene with `is-hand: true` is presented as a hand: its choices are shown as the decks,
the hand cards, and the pinned cards.

### Decks

A scene with `is-deck: true` (no text, a set of tag choices) is a draw pile. Drawing picks
a random available scene from the deck's potential choices. `card-image` sets the deck image.

### Cards

Scenes with `is-card: true` (or `is-pinned-card: true`) are cards — otherwise normal scenes,
and can lead to a chain of non-card scenes. Returning to the hand scene is done manually at
the end of a chain. `card-image` sets the card image.

### Stat checks

Combine `check-quality:` with `broad-difficulty:`/`narrow-difficulty:` and
`check-success-go-to:` / `check-failure-go-to:`. Selecting the scene rolls the check and
transitions accordingly; the scene's text is shown before the result.

## Debugging

Legacy (jQuery) UI: engine state is at `dendryUI.dendryEngine.state`, qualities at
`dendryUI.dendryEngine.state.qualities`. The new Vue app (`ui/`) drives the engine
headlessly via `ui/src/engine/`.
