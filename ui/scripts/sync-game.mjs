import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '../../out/game.json');
const dest = resolve(here, '../public/game.en.json');

if (!existsSync(src)) {
  console.error(
    'out/game.json not found. Build it first from the repo root:\n' +
    '  npm run dendrynexus make-html'
  );
  process.exit(1);
}
mkdirSync(dirname(dest), { recursive: true });
copyFileSync(src, dest);
console.log('Copied out/game.json -> ui/public/game.en.json');
