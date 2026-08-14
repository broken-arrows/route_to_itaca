import { createReadStream, existsSync, statSync, cpSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';

const here = path.dirname(fileURLToPath(import.meta.url));

/** The game's art tree. Single source of truth, shared with the old UI. */
export const GAME_IMG_DIR = path.resolve(here, '..', 'out', 'html', 'img');
/** New game-owned assets added before the phase-6 bulk art relocation. */
export const SOURCE_IMG_DIR = path.resolve(here, '..', 'source', 'img');

export const GAME_LOCALES_DIR = path.resolve(here, '..', 'source', 'locales');

export const GAME_JSON = path.resolve(here, '..', 'out', 'game.json');

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
