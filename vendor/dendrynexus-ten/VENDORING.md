# Vendored `dendrynexus-ten`

In-tree fork of [dendrynexus](https://github.com/aucchen/dendrynexus), based on
`49e259416940b3537ac992c3d33a99c6f5e5b209` (vendored 2026-07-04).

## Development

- Edit this directory, not `node_modules`. The local `file:` dependency links it
  into both packages; no reinstall is needed. Restart Vite with `--force` if its
  CommonJS prebundle still serves old exports.
- `npm run build:classic` rebuilds the browser engine, publishes game libraries
  and assets, and audits content. `npm run package` builds and checks both UIs
  locally. Never delete `out/html`: it also contains hand-maintained shell files.
- Keep Classic and the new UI compatible with shared gameplay and saves.
  Project-specific audit rules belong in `dendrynexus.audit.mjs`.

## Fork contracts

- **Roles:** derive compiled mechanical flags. This project's source audit rejects
  `is-card`, `is-pinned-card`, `is-hand` and `is-deck`; general parser overrides
  and the generated flags remain supported.
- **HTML:** `html.createRenderer({ formatText })` owns its formatter. Plain exports
  never consult `window.displayText` (intentional API break). Classic installs
  `ui.setTextFormatter(fn)` before rendering; new-UI adapters own their renderer.
  Preserve fragment order; raw magic and direct widget formatting stay separate.
- **Localization:** English state plus saved presentation recipes; refresh must
  not replay actions or consume RNG. Formatted qdisplay arrays are content recipes.
  See [LOCALIZATION.md](LOCALIZATION.md) for the authoring/runtime contract.
- **Face images:** saved paragraph offsets accompany `displayContent` as its
  optional third argument, including load and special-scene returns.
- **Assets:** `make-html --publish-game-lib --publish-game-assets` publishes from
  `source/lib`, `source/img` and runtime locale catalogs. Missing required assets
  fail; untranslated entries warn and fall back to English. Shared asset tooling
  retains Node 16.7 compatibility; local UI packaging requires the newer repo Node.

Use Git history for individual changes. For upstream comparison, use the base
commit above in a scratch upstream clone. To extract this fork with history:
`git subtree split --prefix=vendor/dendrynexus-ten -b dendrynexus-ten-split`.
