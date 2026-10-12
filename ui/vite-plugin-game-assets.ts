import { createReadStream, existsSync, statSync, copyFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';

const here = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(here, '..');
export const GAME_IMG_DIR = path.join(projectDir, 'source', 'img');
export const GAME_LOCALES_DIR = path.join(projectDir, 'source', 'locales');
export const GAME_JSON = path.join(projectDir, 'out', 'game.json');

interface AssetValidation {
  locales: string[];
  imageCount: number;
  referenceCount: number;
  warnings: string[];
}
const { validateGameAssets, publishRuntimeAssets } = createRequire(import.meta.url)(
  '../vendor/dendrynexus-ten/lib/cli/game-assets.js',
) as {
  validateGameAssets(root: string, options: { extraSources: string[] }): AssetValidation;
  publishRuntimeAssets(root: string, dest: string, validation: AssetValidation): void;
};

const MIME: Record<string, string> = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.gif': 'image/gif', '.webp': 'image/webp', '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon', '.json': 'application/json',
};

export function gameAssets(): Plugin {
  let outputDir = path.join(here, 'dist');
  let building = false;
  let validation: AssetValidation | undefined;
  return {
    name: 'rti-game-assets',
    configResolved(config) {
      building = config.command === 'build';
      outputDir = path.resolve(config.root, config.build.outDir);
    },
    buildStart() {
      if (!building) return;
      validation = validateGameAssets(projectDir, {
        extraSources: [path.join(projectDir, 'out', 'html'), path.join(here, 'src')],
      });
      validation.warnings.forEach(warning => this.warn(warning));
    },
    configureServer(server) {
      // A miss must return 404 rather than letting Vite's SPA fallback return
      // HTML as an image or catalog. Reports are never runtime locale files.
      server.middlewares.use('/game.en.json', (_req, res) => {
        if (!existsSync(GAME_JSON)) {
          res.statusCode = 404;
          return res.end('Compiled game missing; run npm run build:classic from the repo root.');
        }
        res.setHeader('Content-Type', 'application/json');
        createReadStream(GAME_JSON).pipe(res);
      });
      for (const [mount, directory] of [['/img', GAME_IMG_DIR], ['/locales', GAME_LOCALES_DIR]] as const) {
        server.middlewares.use(mount, (req, res) => {
          let rel: string;
          try { rel = decodeURIComponent((req.url ?? '').split('?')[0]); }
          catch { res.statusCode = 400; return res.end('Invalid URL'); }
          const file = path.resolve(directory, '.' + rel);
          if (!file.startsWith(directory + path.sep)) {
            res.statusCode = 403;
            return res.end('Forbidden');
          }
          if (mount === '/locales' && !/^\/[A-Za-z][\w-]*\/(content|ui)\.json$/.test(rel)) {
            res.statusCode = 404;
            return res.end('Not found');
          }
          if (!existsSync(file) || !statSync(file).isFile()) {
            res.statusCode = 404;
            return res.end('Not found');
          }
          res.setHeader('Content-Type', MIME[path.extname(file).toLowerCase()] ?? 'application/octet-stream');
          createReadStream(file).pipe(res);
        });
      }
    },
    closeBundle() {
      if (!building || !validation) return;
      copyFileSync(GAME_JSON, path.join(outputDir, 'game.en.json'));
      publishRuntimeAssets(projectDir, outputDir, validation);
    },
  };
}
