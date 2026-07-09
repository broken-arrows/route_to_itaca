# DendryNexus-Ten

In-tree **vendored fork** of [DendryNexus](https://github.com/aucchen/dendrynexus), a
StoryNexus-style card-game superset of [Dendry](https://github.com/idmillington/dendry),
maintained locally for _Route to Ítaca_. The upstream base commit and the
`git subtree` sync recipe are recorded in [`VENDORING.md`](./VENDORING.md). MIT-licensed
(see `LICENSE`).

Edits here are **live**: the engine is a `file:` dependency, so changes flow into the game
on the next build with no reinstall.

## What `-ten` adds on top of DendryNexus

- **`role` scene attribute**: a compiler-validated enum naming which UI surface a scene
  renders on: `desk`, `deck`, `card`, `pinned-action`, `newspaper`, `event`, `info-tab`,
  `pause-item`, `main-menu-item`, `library-item`, `ending`, `default`. Additive and
  presentation-only (tags still drive deck draw-pools). `default`/inheritance is resolved
  by the UI adapter at runtime; the compiler only validates the enum.
- **`info` manifest → `game.json`**: the compiler emits a whitelisted `info` block
  (`title`, `author`, `languages`) so the runtime can read game-level metadata.
  `languages: en ca` in `info.dry` declares the locale set. (Also fixes a Windows-only bug
  where `info.dry` was silently skipped by a forward-slash-only path check.)
- **Runtime i18n overlay**: `engine.setLocale(locale, catalog)` installs a translation
  catalog keyed by the English source string; the content renderer substitutes whole
  plain-string runs at display time. Inert unless a catalog is set (no locale ⇒ English),
  so `game.json` stays English and existing consumers are unaffected. The
  placeholder/conditional transform is deferred.
- **`on-display`-on-load fix**: `onDisplay` now re-fires when a saved state is loaded,
  before choices are recompiled, matching normal scene entry.

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
