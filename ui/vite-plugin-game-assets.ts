/* =============================================================================
 * vite-plugin-game-assets — serve the GAME's static art to the Vue app.
 * =============================================================================
 *
 * THE PROBLEM
 * Compiled content references art by a path that is relative to the rendering
 * UI's web root: `card-image: img/erc/araerc.jpg` (198 files, ~101 MB). The OLD
 * UI's web root IS `out/html/`, so `img/...` just resolves. The Vue app's root
 * is `ui/` in dev and `/desk/` in prod, where no `img/` exists — so every image
 * 404s and `HandCard`'s `@error` handler falls back to the striped placeholder.
 * The placeholder was working perfectly, on assets nobody had wired up.
 *
 * THE FIX (deliberately the MINIMAL one — see WHY below)
 * Point the Vue app at the art where it already lives. One copy of every file,
 * no duplication, no drift:
 *   - dev:   a middleware serves /img/* straight off `out/html/img`
 *   - build: the tree is copied into `dist/img`, so a built app is standalone
 *            (`npm run build && npm run preview` renders real art, and CI's
 *            `out/html/desk/` assembly picks it up with no extra step).
 * The relative paths inside `game.json` never change — each UI simply serves an
 * `img/` at its own root. They were always UI-agnostic.
 *
 * WHY NOT MOVE THE FILES TO `source/img/` NOW (asked + decided 2026-07-13)
 * That IS the correct end state — and it is viable: the compiler's walk is
 * `pattern = pattern || /\.dry$/` (`lib/cli/utils.js:59`), so it ignores
 * non-.dry files and art can live beside the scenes that reference it. But
 * doing it now means changing the FROZEN old-UI build path (out/html/img would
 * become build output) and moving 193 tracked files / 101 MB mid-phase, for
 * zero functional gain over this. And it is the SAME relocation problem as
 * `src/game-bindings.ts`'s `cat_engine.js`.
 *
 * >>> PHASE 6 (the swap) MUST relocate BOTH, together, in one tested change:
 * >>> the macro simulation AND this art tree move to a neutral home, with the
 * >>> old UI's build and this plugin both repointed. `out/html/` dies there.
 * >>> Recorded in `docs/design/desk_ui_plan.md`, phase 6.
 * ============================================================================= */
import { createReadStream, existsSync, statSync, cpSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';

const here = path.dirname(fileURLToPath(import.meta.url));

/** The game's art tree. Single source of truth, shared with the old UI. */
export const GAME_IMG_DIR = path.resolve(here, '..', 'out', 'html', 'img');
/** New game-owned assets added before the phase-6 bulk art relocation. */
export const SOURCE_IMG_DIR = path.resolve(here, '..', 'source', 'img');

/**
 * The game's i18n catalogs — `source/locales/<loc>/{ui,content}.json` (§3.4 of
 * `docs/superpowers/specs/2026-07-13-content-ui-decoupling-and-the-brief-design.md`).
 *
 * Deliberate deviation from the engine plan: the compiler does NOT copy these
 * next to `game.json`. The Desk is the ONLY consumer (the old shell is
 * English-only and never reads a catalog), and this plugin already owns
 * exactly this job for `game.json`/`img/` — serving `source/locales/**`
 * straight off disk is the same mechanism, one fewer engine change, and it
 * moves cleanly when the old shell dies at phase 6.
 *
 * `ui/src/i18n.ts` fetches `/locales/<loc>/ui.json` at boot and deep-merges it
 * OVER the bundled `ui/src/locales/<loc>.json` defaults (the game wins on any
 * key collision) — `ui/src/locales` is a fallback layer, not a fence. A
 * missing catalog is NOT an error: i18n.ts treats a 404 as "no override
 * shipped" and the UI's defaults simply stand.
 */
export const GAME_LOCALES_DIR = path.resolve(here, '..', 'source', 'locales');

/**
 * The compiled game, straight from the compiler's output.
 *
 * This used to be a HAND-COPIED `ui/public/game.en.json` (gitignored), which
 * meant the dev server silently served a STALE game: edit a `.dry`, run
 * `dendrynexus-ten compile`, reload — and see the old content, because nobody
 * re-ran the copy. It cost a real debugging session (a `role:` added to
 * `post_event.events_choice` "did not work"; in fact the browser never saw it —
 * the served artifact had 159 roled scenes, the fresh compile 161).
 *
 * Serving `out/game.json` directly makes a stale artifact structurally
 * impossible: `compile` IS the deploy. `ui/public/game.en.json` must NOT exist —
 * Vite's publicDir would shadow this middleware and the footgun would be back.
 */
export const GAME_JSON = path.resolve(here, '..', 'out', 'game.json');

/** The URL the app fetches (`GameView`/`DebugPage`: `${BASE_URL}game.en.json`). */
const GAME_JSON_URL = '/game.en.json';

const MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

export function gameAssets(): Plugin {
  return {
    name: 'rti-game-assets',

    configureServer(server) {
      // Serve the compiled game from the compiler's own output, so `compile` is
      // the only step between a `.dry` edit and the browser. No stale copy.
      server.middlewares.use(GAME_JSON_URL, (req, res) => {
        if (!existsSync(GAME_JSON)) {
          res.statusCode = 404;
          return res.end(
            `${GAME_JSON} not found — run \`npm run dendrynexus-ten compile\` from the repo root.`,
          );
        }
        res.setHeader('Content-Type', 'application/json');
        createReadStream(GAME_JSON).pipe(res);
      });

      server.middlewares.use('/img', (req, res) => {
        // `req.url` is already stripped of the /img mount point.
        const rel = decodeURIComponent((req.url ?? '').split('?')[0]);
        const sourceFile = path.join(SOURCE_IMG_DIR, rel);
        const legacyFile = path.join(GAME_IMG_DIR, rel);
        const file =
          sourceFile.startsWith(SOURCE_IMG_DIR + path.sep) &&
          existsSync(sourceFile) &&
          statSync(sourceFile).isFile()
            ? sourceFile
            : legacyFile;

        // Containment check: never serve outside the art tree, whatever the
        // request says (`..%2f` and friends decode to a traversal above).
        if (
          !sourceFile.startsWith(SOURCE_IMG_DIR + path.sep) ||
          !legacyFile.startsWith(GAME_IMG_DIR + path.sep)
        ) {
          res.statusCode = 403;
          return res.end('Forbidden');
        }

        // 404 a missing image rather than `next()`-ing it. Falling through hands
        // the request to Vite's SPA fallback, which answers `200 text/html` — so
        // a typo'd art path would render the index page INTO an <img>, and the
        // card's placeholder fallback would fire for a baffling reason. Under
        // /img, a miss is a miss.
        if (!existsSync(file) || !statSync(file).isFile()) {
          res.statusCode = 404;
          return res.end('Not found');
        }

        res.setHeader(
          'Content-Type',
          MIME[path.extname(file).toLowerCase()] ?? 'application/octet-stream',
        );
        createReadStream(file).pipe(res);
      });

      // Serve the game's locale catalogs straight off `source/locales/**`.
      // Same shape as `/img` above, same reasoning, same trap: a MISS MUST
      // 404, never `next()`. Falling through hands the request to Vite's SPA
      // fallback, which answers `200 text/html` — `ui/src/i18n.ts` would then
      // JSON.parse the index page and die on a baffling parse error instead of
      // cleanly treating "no override for this locale" as the unauthored,
      // default-case state it actually is.
      server.middlewares.use('/locales', (req, res) => {
        const rel = decodeURIComponent((req.url ?? '').split('?')[0]);
        const file = path.join(GAME_LOCALES_DIR, rel);

        if (!file.startsWith(GAME_LOCALES_DIR + path.sep)) {
          res.statusCode = 403;
          return res.end('Forbidden');
        }

        if (!existsSync(file) || !statSync(file).isFile()) {
          res.statusCode = 404;
          return res.end('Not found');
        }

        res.setHeader('Content-Type', 'application/json');
        createReadStream(file).pipe(res);
      });
    },

    // Make a built app self-contained: `dist/game.en.json` + `dist/img/**` mirror
    // the compiler output, so `vite preview` and CI's `/desk/` deploy are the
    // real game with real art.
    closeBundle() {
      const dist = path.resolve(here, 'dist');

      if (existsSync(GAME_JSON)) {
        cpSync(GAME_JSON, path.join(dist, 'game.en.json'));
      } else {
        this.warn(`${GAME_JSON} not found — the built app will have no game to load`);
      }

      if (existsSync(GAME_IMG_DIR)) {
        cpSync(GAME_IMG_DIR, path.join(dist, 'img'), { recursive: true });
      } else {
        this.warn(`game art not found at ${GAME_IMG_DIR} — built app will show placeholders`);
      }
      // Overlay the assets already moved into the game-owned tree (currently
      // the polls map). Existing out/html/img remains the bulk source until
      // phase 6; this avoids breaking every card image to land one new asset.
      if (existsSync(SOURCE_IMG_DIR)) {
        cpSync(SOURCE_IMG_DIR, path.join(dist, 'img'), { recursive: true });
      }

      if (existsSync(GAME_LOCALES_DIR)) {
        cpSync(GAME_LOCALES_DIR, path.join(dist, 'locales'), { recursive: true });
      } else {
        this.warn(
          `no locale catalogs found at ${GAME_LOCALES_DIR} — built app will use only ui/'s own defaults`,
        );
      }
    },
  };
}
