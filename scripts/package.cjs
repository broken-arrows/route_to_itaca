'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const { listFiles } = require('../vendor/dendrynexus-ten/lib/cli/game-assets');
const root = path.resolve(__dirname, '..');
const engineCLI = path.join(root, 'vendor', 'dendrynexus-ten', 'lib', 'cli', 'main.js');
const stage = path.join(root, 'out', 'release');

function run(script, args, cwd = root) {
  const result = spawnSync(process.execPath, [script, ...args], { cwd, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error('Build failed: ' + path.relative(root, script) + ' ' + args.join(' '));
}

function verifyLinks(webRoot) {
  const files = listFiles(webRoot);
  const names = new Set(files.map(file => path.relative(webRoot, file).split(path.sep).join('/')));
  for (const file of files.filter(file => /\.(html|css)$/.test(file))) {
    const text = fs.readFileSync(file, 'utf8');
    const refs = [...text.matchAll(/(?:src|href)\s*=\s*["']([^"']+)["']|url\(\s*["']?([^"')]+)["']?\s*\)/gi)]
      .map(match => match[1] || match[2]);
    for (const ref of refs) {
      if (/^(?:[a-z][\w+.-]*:|\/\/|#)/i.test(ref)) continue;
      const clean = decodeURIComponent(ref.split(/[?#]/)[0]);
      if (!clean) continue;
      if (clean.startsWith('/')) throw new Error('Non-relative local asset in ' + file + ': ' + ref);
      const resolved = path.relative(webRoot, path.resolve(path.dirname(file), clean)).split(path.sep).join('/');
      if (!names.has(resolved)) throw new Error('Missing staged asset: ' + ref + ' referenced by ' + file);
    }
  }
}

try {
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (!((major === 20 && minor >= 19) || (major === 22 && minor >= 12) || major > 22)) {
    throw new Error('Packaging requires Node ^20.19.0 || >=22.12.0; use the local baseline in .node-version.');
  }
  // Force one shared compilation, then build both consumers from that artifact.
  // The same publication flags and audit are used by build:classic and existing CI.
  run(engineCLI, ['make-html', '--force', '--publish-game-lib', '--publish-game-assets']);
  run(path.join(root, 'scripts', 'audit.cjs'), []);
  run(path.join(root, 'ui', 'node_modules', 'vite', 'bin', 'vite.js'), ['build'], path.join(root, 'ui'));

  // Never clean out/html: it contains hand-maintained scripts and fonts.
  const owned = path.relative(root, path.resolve(stage));
  if (owned !== path.join('out', 'release')) throw new Error('Unexpected staging directory: ' + stage);
  fs.rmSync(stage, { recursive: true, force: true });
  fs.cpSync(path.join(root, 'ui', 'dist'), stage, { recursive: true });
  const oldUI = path.join(stage, 'classic');
  fs.mkdirSync(oldUI);
  // Exclude historical output folders from the old UI's runtime package.
  for (const name of fs.readdirSync(path.join(root, 'out', 'html'))) {
    const source = path.join(root, 'out', 'html', name);
    if (fs.statSync(source).isDirectory() ? ['fonts', 'img', 'lib', 'locales'].includes(name) : /\.(html|css|js)$/.test(name)) {
      fs.cpSync(source, path.join(oldUI, name), { recursive: true });
    }
  }
  fs.copyFileSync(path.join(root, 'out', 'game.json'), path.join(oldUI, 'game.json'));
  if (!fs.readFileSync(path.join(stage, 'game.en.json')).equals(fs.readFileSync(path.join(oldUI, 'game.json')))) {
    throw new Error('The staged UIs do not share the same compiled game');
  }
  verifyLinks(oldUI);
  verifyLinks(stage);
  console.log('Local package ready: out/release/ (new UI), out/release/classic/ (old UI).');
} catch (err) {
  console.error(err.message);
  process.exitCode = 1;
}
