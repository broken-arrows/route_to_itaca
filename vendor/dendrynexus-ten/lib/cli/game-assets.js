'use strict';

const fs = require('fs');
const path = require('path');

function readJSON(file) {
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
    throw new Error('Missing required input: ' + file);
  }
  try {
    const value = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('expected a JSON object');
    }
    return value;
  } catch (err) {
    throw new Error('Invalid JSON in ' + file + ': ' + err.message);
  }
}

function listFiles(dir) {
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    throw new Error('Missing required directory: ' + dir);
  }
  return fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))
    .flatMap(entry => entry.isDirectory()
      ? listFiles(path.join(dir, entry.name))
      : [path.join(dir, entry.name)]);
}

function relative(root, file) {
  return path.relative(root, file).split(path.sep).join('/');
}

function inspectImages(value, location, references, dynamic) {
  if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      inspectImages(child, location + '.' + key, references, dynamic);
    }
    return;
  }
  if (typeof value !== 'string') return;
  // Remote URLs may contain /img/ too; they are outside local packaging.
  value = value.replace(/(?:[a-z][\w+.-]*:)?\/\/[^\s'"`<>]+/gi, '');
  // Literal authored properties, HTML attributes, CSS URLs and JS strings.
  // Expressions such as img/parties/${logo} cannot be resolved statically.
  if (/img\/[^\r\n'"`<>]*\$\{|img\/[^\r\n'"`<>]*\{!/.test(value)) dynamic.add(location);
  const matches = value.matchAll(/\bimg\/[^\r\n'"`<>\s{}()\\]+\.(?:png|jpe?g|svg|gif|webp|ico)\b/gi);
  const quoted = value.matchAll(/["'`](img\/[^\r\n"'`{}<>]+\.(?:png|jpe?g|svg|gif|webp|ico))["'`]/gi);
  const refs = [...matches].map(match => match[0]).concat([...quoted].map(match => match[1]));
  if (/^img\/[^\r\n{}]+\.(?:png|jpe?g|svg|gif|webp|ico)$/i.test(value)) refs.push(value);
  for (const ref of refs) {
    if (!references.has(ref)) references.set(ref, new Set());
    references.get(ref).add(location);
  }
}

function flattenUI(value, prefix = '', result = {}) {
  for (const [key, child] of Object.entries(value)) {
    const name = prefix ? prefix + '.' + key : key;
    if (child && typeof child === 'object' && !Array.isArray(child)) flattenUI(child, name, result);
    else result[name] = child;
  }
  return result;
}

function validateGameAssets(projectDir, options = {}) {
  const root = path.resolve(projectDir);
  const game = readJSON(path.join(root, 'out', 'game.json'));
  const entry = path.join(root, 'source', 'lib', 'index.js');
  if (!fs.existsSync(entry) || !fs.statSync(entry).isFile()) {
    throw new Error('Missing required game library: ' + entry);
  }
  const artDir = path.join(root, 'source', 'img');
  const art = listFiles(artDir);
  if (!art.length) throw new Error('Empty required art tree: ' + artDir);
  // Exact relative filenames are important even on Windows: published servers
  // may be case-sensitive. existsSync alone would miss case errors locally.
  const artNames = new Set(art.map(file => 'img/' + relative(artDir, file)));
  const references = new Map();
  const dynamic = new Set();
  inspectImages(game, 'out/game.json', references, dynamic);
  const sources = [path.join(root, 'source', 'lib'), ...(options.extraSources || [])];
  for (const source of sources) {
    const files = fs.statSync(source).isDirectory() ? listFiles(source) : [source];
    for (const file of files) {
      if (!/\.(?:js|ts|vue|html|css)$/.test(file) || path.basename(file) === 'core.js') continue;
      const text = fs.readFileSync(file, 'utf8');
      text.split(/\r?\n/).forEach((line, index) =>
        inspectImages(line, relative(root, file) + ':' + (index + 1), references, dynamic));
    }
  }
  const missing = [...references].filter(([ref]) => !artNames.has(ref));
  if (missing.length) {
    throw new Error('Missing local images (' + missing.length + '):\n' + missing
      .map(([ref, uses]) => '  ' + ref + ' — ' + [...uses].slice(0, 3).join(', ')).join('\n'));
  }

  const languages = game.info && game.info.languages;
  const locales = Array.isArray(languages) ? languages : typeof languages === 'string' ? languages.split(/\s+/) : ['en'];
  if (!locales.length || locales.some(locale => !/^[a-zA-Z][\w-]*$/.test(locale))) {
    throw new Error('Invalid required languages in out/game.json');
  }
  const declarations = game.data && game.data.localization || {};
  const nameCatalogs = declarations.nameCatalogs || {};
  for (const file of Object.values(nameCatalogs)) if (!/^[a-z][a-z_-]*\.json$/.test(file)) throw new Error('Invalid naming catalog path: ' + file);
  const catalogs = locales.map(locale => {
    const dir = path.join(root, 'source', 'locales', locale);
    for (const [namespace, file] of Object.entries(nameCatalogs)) {
      const names = readJSON(path.join(dir, file));
      require('../localization').mergeNames({}, { [namespace]: names });
      for (const [id, forms] of Object.entries(names)) for (const form of (declarations.nameForms || {})[namespace] || []) {
        if (typeof forms[form] !== 'string') throw new Error(locale + ': missing name form ' + id + '/' + form);
      }
    }
    return { locale, content: readJSON(path.join(dir, 'content.json')), ui: readJSON(path.join(dir, 'ui.json')) };
  });
  const english = catalogs.find(catalog => catalog.locale === 'en');
  if (!english) throw new Error('Missing required English runtime catalogs');
  const warnings = [];
  for (const catalog of catalogs) {
    if (catalog.locale === 'en') continue;
    const missingContent = Object.keys(english.content).filter(key => catalog.content[key] == null || catalog.content[key] === '');
    const englishUI = flattenUI(english.ui);
    const localizedUI = flattenUI(catalog.ui);
    const missingUI = Object.keys(englishUI).filter(key => localizedUI[key] == null || localizedUI[key] === '');
    if (missingContent.length || missingUI.length) {
      warnings.push(catalog.locale + ': ' + missingContent.length + ' untranslated content entries, ' + missingUI.length +
        ' untranslated UI entries; English fallback retained. Edit source/locales/' + catalog.locale +
        '/{content,ui}.json (source keys: source/locales/en/{content,ui}.json).');
    }
  }
  if (dynamic.size) warnings.push(dynamic.size + ' dynamic image references require runtime checks; e.g. ' + [...dynamic].slice(0, 3).join(', '));
  return { game, locales, imageCount: art.length, referenceCount: references.size, warnings };
}

function replaceGeneratedDirectory(projectDir, source, target) {
  const root = path.resolve(projectDir);
  const destination = path.resolve(target);
  const owned = path.relative(root, destination);
  if (!owned || owned.startsWith('..') || path.isAbsolute(owned)) {
    throw new Error('Generated output must stay inside project: ' + destination);
  }
  const sourcePath = path.resolve(source);
  if (destination === sourcePath || destination.startsWith(sourcePath + path.sep) || sourcePath.startsWith(destination + path.sep)) {
    throw new Error('Generated output overlaps canonical source: ' + destination);
  }
  fs.rmSync(destination, { recursive: true, force: true });
  fs.cpSync(source, destination, { recursive: true });
}

function publishRuntimeAssets(projectDir, destDir, validation) {
  const root = path.resolve(projectDir);
  const checked = validation || validateGameAssets(root);
  replaceGeneratedDirectory(root, path.join(root, 'source', 'img'), path.join(destDir, 'img'));
  const localesDir = path.resolve(destDir, 'locales');
  const owned = path.relative(root, localesDir);
  if (!owned || owned.startsWith('..') || path.isAbsolute(owned)) throw new Error('Invalid generated locale output: ' + localesDir);
  fs.rmSync(localesDir, { recursive: true, force: true });
  for (const locale of checked.locales) {
    fs.mkdirSync(path.join(localesDir, locale), { recursive: true });
    for (const name of ['content.json', 'ui.json'].concat(Object.values(checked.game.data && checked.game.data.localization && checked.game.data.localization.nameCatalogs || {}))) {
      fs.copyFileSync(path.join(root, 'source', 'locales', locale, name), path.join(localesDir, locale, name));
    }
  }
  return checked;
}

module.exports = { validateGameAssets, publishRuntimeAssets, listFiles, replaceGeneratedDirectory };
