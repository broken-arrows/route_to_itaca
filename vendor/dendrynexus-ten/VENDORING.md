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
