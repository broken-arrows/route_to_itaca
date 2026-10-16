# Vendored `dendrynexus-ten`

This directory is a **vendored, in-tree copy** of the `dendrynexus` narrative engine,
detached from npm so it can be modified directly as part of _Route to Ítaca_. It is
distributed under this repo as the fork `dendrynexus-ten`.

- **Upstream:** https://github.com/aucchen/dendrynexus
- **Vendored from commit:** `49e259416940b3537ac992c3d33a99c6f5e5b209` (`49e2594`)
- **Vendored on:** 2026-07-04
- **Fork renamed to `dendrynexus-ten`:** 2026-07-08

## How it's wired

- **Explicit HTML renderers (2026-10-16):** `lib/ui/content/html.js` exports
  `createRenderer({ formatText })`, returning `convert` and `convertLine`.
  Instances own their callback and work without `window`. The module-level
  conversion functions now produce unformatted HTML; `window.displayText` is
  no longer consulted (intentional API break, no fallback). Migrate custom
  shells by calling `ui.setTextFormatter(formatText)` in `dendryModifyUI`, before
  rendering. Both project UIs and all four shipped HTML templates are migrated.
  The new UI owns one renderer per adapter, shared with its capture UI. Text
  callbacks retain their existing fragment order/lifetime; raw `magic` HTML
  still bypasses them. This is formatting, not HTML sanitization. Direct widget
  glossary paths remain separate. Regressions: `html.renderer.test.ts`,
  `html.templates.test.ts`, and `legacy.coalition-tooltip.test.ts` in `ui/tests`.

- **Formatted qdisplay capture (2026-10-14):** inline node arrays returned by
  qdisplays are captured as content recipes even when they have no dynamic
  dependencies. Ordinary expression arrays still use native coercion. This fixes
  italic text becoming comma-separated `[object Object]` in monthly quotations.
  Regression: `ui/tests/phase6b.runtime.test.ts`, including locale refresh.

- **Runtime asset publication (2026-10-12):** `make-html --publish-game-assets`
  validates literal local image references, required art/library/catalog inputs
  and exact filename casing, then replaces only generated `img/` and `locales/`
  directories. Both UIs use `source/img`; only declared locales' `content.json`
  and `ui.json` files ship. Missing translations warn and retain English fallback.
  `lib/cli/game-assets.js` is shared by the old UI build, Vite and local packaging;
  its CommonJS implementation retains the engine's Node 16.7 compatibility.
  The root `npm run audit` combines this validation with the engine's existing
  compiled-content audit; `build:classic` and local packaging use that command.

- **Saved localization recipes (2026-10-11):** `lib/localization.js` builds the
  same authored units for runtime and extraction. English qualities and saved
  prose retain their original meaning; `state.presentation` carries captured
  translation recipes. Engine-owned `G` helpers track generated text, and a pure
  presentation refresh translates existing pages/cards without replaying actions
  or advancing randomness. `lib/localization-extract.js` and `extract-strings`
  merge catalogs and report uncertain display paths. See [LOCALIZATION.md](LOCALIZATION.md)
  for authoring, grammar, compatibility and review instructions.

- **Scene-relative face images (2026-10-11):** `lib/engine.js` retains the most
  recent face image and its paragraph offset in `currentContentFaceImage`, with
  a matching special-scene backup. Save/load and special returns forward the
  offset as an optional third `displayContent` argument. Old UI implementations
  may ignore it; gameplay qualities and prose are unchanged. This lets Desk place
  an image with its own scene text after earlier prose on the same page.

`package.json` (repo root) consumes this via a local dependency:

```json
"dendrynexus-ten": "file:vendor/dendrynexus-ten"
```

`npm install` symlinks this folder into `node_modules/dendrynexus-ten`, so edits here are
**live** — no reinstall needed. The CLI `bin` (`dendrynexus-ten`) stays linked, so
`npm run build:classic` builds the classic UI. The engine is
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
