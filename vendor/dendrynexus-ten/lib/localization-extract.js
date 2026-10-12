'use strict';

// Source inspection only: never execute authored JavaScript to discover text.
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var acorn = require('acorn');
var dry = require('./parsers/dry');
var contentParser = require('./parsers/content');
var qdParser = require('./parsers/qdcontent');
var codec = require('./localization');

var TEXT_FIELDS = ['title', 'subtitle', 'unavailableSubtitle', 'content'];
var HELPERS = ['text', 'setText', 'getText', 'copyText', 'updateText', 'joinText', 'markupText', 'literalText', 'rawText'];
var REVIEW_STATUSES = ['non-translatable', 'migration-pending', 'analysis-deferred'];
var normalizePath = function(value) { return value.replace(/\\/g, '/'); };
var hash = function(value) {
  return crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex').slice(0, 24);
};
var unique = function(values) { return Array.from(new Set(values)).sort(); };
var parse = function(parser, args) {
  return new Promise(function(resolve, reject) {
    parser.apply(null, args.concat(function(err, result) {
      if (err) reject(err); else resolve(result);
    }));
  });
};
var stringValue = function(node) {
  if (node && node.type === 'Literal' && typeof node.value === 'string') return node.value;
  if (node && node.type === 'TemplateLiteral' && !node.expressions.length) return node.quasis[0].value.cooked;
  return null;
};
var propertyName = function(node) {
  if (!node) return null;
  if (!node.computed && node.property.type === 'Identifier') return node.property.name;
  return stringValue(node.property);
};
var qualityName = function(node) {
  return node && node.type === 'MemberExpression' && node.object.type === 'Identifier' &&
    node.object.name === 'Q' ? propertyName(node) : null;
};
var isQuality = function(node) {
  return node && node.type === 'MemberExpression' && node.object.type === 'Identifier' && node.object.name === 'Q';
};
var callName = function(node) {
  if (!node || node.type !== 'CallExpression') return null;
  if (node.callee.type === 'Identifier') return node.callee.name;
  if (node.callee.type === 'MemberExpression') return propertyName(node.callee);
  return null;
};
var isTextHelper = function(node) {
  return node && node.type === 'CallExpression' && node.callee.type === 'MemberExpression' &&
    node.callee.object.type === 'Identifier' && node.callee.object.name === 'G' && HELPERS.includes(callName(node));
};
var walk = function(node, visit, parent) {
  if (!node || typeof node !== 'object') return;
  if (typeof node.type === 'string') visit(node, parent);
  Object.keys(node).forEach(function(key) {
    if (key === 'loc') return;
    var value = node[key];
    if (Array.isArray(value)) value.forEach(function(child) { walk(child, visit, node); });
    else if (value && typeof value === 'object') walk(value, visit, node);
  });
};
var namesIn = function(node) {
  var names = [];
  walk(node, function(value) { var name = qualityName(value); if (name) names.push(name); });
  return unique(names);
};
var identifiersIn = function(node) {
  var names = [];
  walk(node, function(value, parent) {
    if (value.type !== 'Identifier' || value.name === 'Q' || value.name === 'G') return;
    if (parent && parent.type === 'MemberExpression' && parent.property === value && !parent.computed) return;
    if (parent && parent.type === 'Property' && parent.key === value && !parent.computed) return;
    names.push(value.name);
  });
  return unique(names);
};
var literalStrings = function(node) {
  var values = [];
  walk(node, function(value) { var text = stringValue(value); if (text !== null) values.push(text); });
  return unique(values);
};
var sourceLocation = function(file, owner, field, line, column) {
  return { file: normalizePath(file), owner: owner, field: field, line: line || 1, column: column || 1 };
};

// Passing files makes discovery independently testable without changing catalogs.
// Files are {name: project-relative path, contents: string}.
async function extractFiles(files, options) {
  if (Number(acorn.version.split('.')[0]) < 8) throw new Error('Localization discovery requires the declared Acorn 8 dependency.');
  options = options || {};
  var declarations = options.declarations || {};
  var grammarBindings = declarations.grammarBindings || {};
  var confirmed = new Map();
  var findings = [];
  var limitations = [];
  var errors = [];
  var sinks = new Map();
  var writes = [];
  var jsBlocks = [];
  var internal = new Set(declarations.literalValues || []);
  var helperReturns = new Map();
  var registryFiles = new Map();
  var findingOrdinals = new Map();
  var addError = function(location, message) { errors.push({ location: location, message: message }); };
  var limit = function(location, expression, reason) {
    limitations.push({ location: location, expression: expression, reason: reason });
  };
  var addSink = function(name, location, expression) {
    if (!name) return;
    var uses = sinks.get(name) || [];
    if (!uses.some(function(use) { return JSON.stringify(use) === JSON.stringify({ location: location, expression: expression }); })) {
      uses.push({ location: location, expression: expression });
      sinks.set(name, uses);
    }
  };
  var addUnit = function(unit, location, kind) {
    if (!unit || typeof unit.key !== 'string' || !unit.key.trim()) return;
    var parameters = unique((unit.parameters || unit.params || []).concat(unit.context || []));
    var validation = codec.validateTemplate(unit.key, parameters);
    validation.forEach(function(message) { addError(location, message); });
    var current = confirmed.get(unit.key);
    if (!current) {
      current = { key: unit.key, placeholders: unique(unit.placeholders || []), parameters: parameters,
        context: unique(unit.context || []), protectedMarkup: unit.protectedMarkup || [], sources: [], schemas: [] };
      confirmed.set(unit.key, current);
    } else {
      current.parameters = unique(current.parameters.concat(parameters));
      current.context = unique(current.context.concat(unit.context || []));
    }
    var reference = Object.assign({ kind: kind }, location);
    if (!current.schemas.some(function(schema) { return JSON.stringify(schema.parameters) === JSON.stringify(parameters); })) {
      current.schemas.push({ parameters: parameters, source: reference });
    }
    if (!current.sources.some(function(source) { return JSON.stringify(source) === JSON.stringify(reference); })) current.sources.push(reference);
  };
  var compileField = async function(text, location, paragraphs) {
    if (typeof text !== 'string' || !text.trim()) return;
    try {
      var compiled = await parse(contentParser.compile, [text, paragraphs]);
      codec.collectUnits(compiled, { grammarBindings: grammarBindings }).forEach(function(unit) { addUnit(unit, location, 'authored'); });
      var deps = compiled && compiled.stateDependencies || [];
      deps.forEach(function(dep) {
        if (dep.type !== 'insert') return;
        var expression = dep.source || dep.fn && dep.fn.source || '';
        var name = dep.quality;
        if (!name && dep.language === 'logic' && /^\s*[A-Za-z_][\w]*\s*$/.test(expression)) name = expression.trim();
        if (!name && !dep.source) {
          var match = /^return\s+\(Q\[['"]([\w]+)['"]\]\s*\|\|\s*0\);?\s*$/.exec(expression.trim()) || /^return\s+Q\.([\w]+);?\s*$/.exec(expression.trim());
          if (match) name = match[1];
        }
        if (name) addSink(name, location, expression);
        else limit(location, expression, 'Insert expression is not a statically resolved quality; review its display-producing dependencies.');
      });
    } catch (error) { addError(location, 'Content extraction failed: ' + error.message); }
  };
  var addMagic = function(file, owner, field, text, baseLine) {
    if (typeof text !== 'string') return;
    var pattern = /\{!([\s\S]*?)!\}/g;
    var match;
    while ((match = pattern.exec(text))) {
      // DRY magic also wraps literal HTML; that is handled by the shared codec.
      if (/^\s*</.test(match[1])) continue;
      var prefix = text.slice(0, match.index + 2);
      jsBlocks.push({ file: file, owner: owner, field: field, source: match[1],
        baseLine: baseLine + (prefix.match(/\n/g) || []).length, baseColumn: 1 });
    }
  };

  var sortedFiles = files.slice().sort(function(a, b) { return a.name.localeCompare(b.name); });
  for (var file of sortedFiles) {
    var normalized = normalizePath(file.name);
    if (/\/(?:locales|img)\//.test('/' + normalized)) continue;
    if (/\/(?:data)\/[^/]+\.json$/.test('/' + normalized)) {
      try { registryFiles.set(path.basename(normalized, '.json'), { file: normalized, value: JSON.parse(file.contents) }); }
      catch (error) { addError(sourceLocation(normalized, 'registry', 'json'), error.message); }
      continue;
    }
    if (/\.js$/.test(normalized)) {
      jsBlocks.push({ file: normalized, owner: path.basename(normalized), field: 'javascript', source: file.contents, baseLine: 1, baseColumn: 1 });
      continue;
    }
    if (!/\.(scene|qdisplay|quality)\.dry$/.test(normalized)) continue;
    var parsed;
    try { parsed = await parse(dry.parseFromContent, [path.normalize(file.name), file.contents]); }
    catch (error) { addError(sourceLocation(normalized, 'file', 'dry'), error.message); continue; }
    internal.add(parsed.id);
    var sections = [parsed].concat(parsed.sections || []);
    for (var section of sections) {
      internal.add(section.id);
      var owner = section.id;
      var metadata = section.$metadata || {};
      for (var field of Object.keys(section)) {
        if (field === '$metadata' || typeof section[field] !== 'string') continue;
        var fieldMeta = metadata[field] || metadata;
        var location = sourceLocation(normalized, owner, field, fieldMeta.$line || 1);
        addMagic(normalized, owner, field, section[field], location.line);
        if (TEXT_FIELDS.includes(field)) {
          if (parsed.type === 'qdisplay' && field === 'content') {
            try {
              var ranges = await parse(qdParser.compile, [section[field]]);
              for (var range of ranges) if (range.output) codec.collectUnits(range.output, { grammarBindings: grammarBindings }).forEach(function(unit) {
                addUnit(unit, location, 'qdisplay');
              });
            } catch (error) { addError(location, 'Qdisplay extraction failed: ' + error.message); }
          } else await compileField(section[field], location, field === 'content');
        }
      }
      for (var option of section.options || []) {
        if (option.id) internal.add(option.id.replace(/^[@#]/, ''));
        for (var optField of TEXT_FIELDS) {
          var optionMeta = option.$metadata && (option.$metadata[optField] || option.$metadata) || {};
          var optionLocation = sourceLocation(normalized, owner + '->' + option.id, optField, optionMeta.$line || metadata.$line || 1);
          await compileField(option[optField], optionLocation, false);
          addMagic(normalized, owner + '->' + option.id, optField, option[optField], optionLocation.line);
        }
      }
    }
  }

  // Registry fields are declared, not inferred from every string in JSON.
  for (var declaration of declarations.registryFields || []) {
    var registry = registryFiles.get(declaration.registry);
    if (!registry) { addError(sourceLocation('source/data/localization.json', 'declarations', 'registryFields'), 'Unknown registry: ' + declaration.registry); continue; }
    for (var selector of declaration.paths || []) {
      var select = function(value, segments, actual) {
        if (!segments.length) {
          if (typeof value !== 'string') return;
          var location = sourceLocation(registry.file, declaration.registry, actual.join('.'));
          addUnit({ key: value, parameters: declaration.parameters || [], placeholders: [], context: [] }, location, 'registry');
          return;
        }
        if (!value || typeof value !== 'object') return;
        var next = segments[0];
        if (next === '*') Object.keys(value).forEach(function(key) { select(value[key], segments.slice(1), actual.concat(key)); });
        else if (Object.prototype.hasOwnProperty.call(value, next)) select(value[next], segments.slice(1), actual.concat(next));
      };
      select(registry.value, selector.split('.'), []);
    }
  }
  (declarations.displayQualities || []).forEach(function(name) {
    addSink(name, sourceLocation('source/data/localization.json', 'declarations', 'displayQualities'), name);
  });
  Object.keys(grammarBindings).forEach(function(name) { internal.add(grammarBindings[name]); });

  var parsedBlocks = [];
  for (let block of jsBlocks) {
    try {
      var ast = acorn.parse(block.source, { ecmaVersion: 'latest', locations: true, sourceType: 'script', allowReturnOutsideFunction: true });
      var localDefinitions = new Map();
      var blockLocation = function(node, owner) {
        return sourceLocation(block.file, owner || block.owner, block.field,
          block.baseLine + node.loc.start.line - 1, node.loc.start.column + block.baseColumn);
      };
      walk(ast, function(node) {
        if (node.type === 'VariableDeclarator' && node.id.type === 'Identifier' && node.init) localDefinitions.set(node.id.name, node.init);
        if (node.type === 'AssignmentExpression' && node.left.type === 'Identifier') localDefinitions.set(node.left.name, node.right);
        if (node.type === 'FunctionDeclaration' && node.id) helperReturns.set(node.id.name, { node: node, block: block, location: blockLocation(node, node.id.name) });
      });
      parsedBlocks.push({ block: block, ast: ast, localDefinitions: localDefinitions, location: blockLocation });
    } catch (error) { limit(sourceLocation(block.file, block.owner, block.field, block.baseLine), '', 'JavaScript analysis could not parse this block: ' + error.message); }
  }
  var expand = function(node, definitions, seen) {
    seen = seen || new Set();
    var expanded = [node];
    identifiersIn(node).forEach(function(name) {
      if (!seen.has(name) && definitions.has(name)) {
        seen.add(name);
        expanded = expanded.concat(expand(definitions.get(name), definitions, seen));
      }
    });
    return expanded;
  };
  var find = function(location, quality, kind, expression, reason, displayUses, related) {
    var semantic = [location.file, location.owner, location.field, quality || '', kind];
    var ordinalKey = JSON.stringify(semantic);
    var ordinal = (findingOrdinals.get(ordinalKey) || 0) + 1;
    findingOrdinals.set(ordinalKey, ordinal);
    var uses = displayUses || [];
    var fingerprint = hash({ expression: expression.replace(/\s+/g, ' ').trim(), reason: reason,
      uses: uses.map(function(use) { return { file: use.location.file, owner: use.location.owner, field: use.location.field, expression: use.expression }; }), related: related || [] });
    findings.push({ id: hash(semantic.concat(ordinal)), fingerprint: fingerprint, location: location,
      quality: quality || null, kind: kind, expression: expression, reason: reason, displayUses: uses, related: related || [] });
  };
  for (var parsedBlock of parsedBlocks) {
    var currentBlock = parsedBlock.block;
    var source = currentBlock.source;
    var definitions = parsedBlock.localDefinitions;
    var locationOf = parsedBlock.location;
    walk(parsedBlock.ast, function(node) {
      if (node.type === 'AssignmentExpression' && isQuality(node.left)) {
        var name = qualityName(node.left);
        var values = expand(node.right, definitions);
        var reads = unique(values.flatMap(namesIn));
        writes.push({ quality: name, node: node, values: values, reads: reads, location: locationOf(node),
          expression: source.slice(node.start, node.end), explicit: isTextHelper(node.right) });
        if (!name) limit(locationOf(node), source.slice(node.start, node.end), 'Dynamic quality write cannot be linked reliably to display uses.');
      }
      if (node.type !== 'CallExpression') return;
      var method = callName(node);
      if (isTextHelper(node)) {
        if (method === 'getText') {
          var readQuality = stringValue(node.arguments[1]);
          if (readQuality) addSink(readQuality, locationOf(node), source.slice(node.start, node.end));
        }
        if (method === 'copyText') {
          var destination = stringValue(node.arguments[1]);
          var copied = stringValue(node.arguments[2]);
          if (copied) addSink(copied, locationOf(node), source.slice(node.start, node.end));
          if (!destination || !copied) limit(locationOf(node), source.slice(node.start, node.end), 'Dynamic text-copy quality cannot be resolved statically.');
        }
        if (method === 'setText' || method === 'text') {
          var keyIndex = method === 'setText' ? 2 : 0;
          var key = stringValue(node.arguments[keyIndex]);
          if (key !== null) {
            var params = node.arguments[keyIndex + 1];
            var names = [];
            if (params && params.type === 'ObjectExpression') params.properties.forEach(function(prop) {
              if (prop.type === 'SpreadElement') { limit(locationOf(prop), source.slice(prop.start, prop.end), 'Template parameter spread requires manual schema review.'); return; }
              var propName = prop.computed ? stringValue(prop.key) : prop.key.name || stringValue(prop.key);
              if (propName) names.push(propName);
            });
            else if (params) limit(locationOf(params), source.slice(params.start, params.end), 'Template parameter names are not a literal object; review the schema.');
            // The codec validates author templates and all catalog selector entries.
            var unit = { key: key, parameters: names, context: [], placeholders: names };
            addUnit(unit, locationOf(node), 'javascript-template');
          } else if (!node.arguments[keyIndex] || !isTextHelper(node.arguments[keyIndex])) {
            limit(locationOf(node), source.slice(node.start, node.end), 'Explicit text recipe uses a dynamic template; possible keys require manual review.');
          }
        }
      }
      for (var sinkDeclaration of declarations.displaySinks || []) {
        if (sinkDeclaration.helper !== method) continue;
        for (var argumentIndex of sinkDeclaration.arguments || []) {
          var argument = node.arguments[argumentIndex];
          if (!argument) continue;
          namesIn(argument).forEach(function(name) { addSink(name, locationOf(argument), source.slice(argument.start, argument.end)); });
          if (!isTextHelper(argument) && literalStrings(argument).some(function(value) { return value.trim() && !internal.has(value); })) {
            find(locationOf(argument), null, 'unmarked-display-argument', source.slice(argument.start, argument.end),
              'Unmarked text flows directly into declared display helper ' + method + '.', [{ location: locationOf(node), expression: source.slice(node.start, node.end) }]);
          }
        }
      }
    });
  }
  // Follow quality copies and statically resolved local aliases backwards from sinks.
  var changed = true;
  while (changed) {
    changed = false;
    for (var write of writes) {
      if (!write.quality || !sinks.has(write.quality)) continue;
      for (var read of write.reads) if (!sinks.has(read)) {
        addSink(read, write.location, write.expression);
        changed = true;
      }
    }
  }
  for (var write of writes) {
    if (!write.quality || !sinks.has(write.quality) || write.explicit) continue;
    var strings = unique(write.values.flatMap(literalStrings)).filter(function(value) {
      return value.trim() && !internal.has(value) && !/^img\//.test(value) && !/^[\w]+(?:_[\w]+)+$/.test(value);
    });
    var calls = [];
    write.values.forEach(function(value) { walk(value, function(node) {
      if (node.type === 'CallExpression' && !isTextHelper(node)) calls.push(callName(node));
    }); });
    var rhs = write.node.right;
    var kind = qualityName(rhs) ? 'unmarked-copy' :
      calls.some(function(name) { return name === 'replace' || name === 'replaceAll'; }) ? 'unmarked-replacement' :
      calls.includes('join') ? 'unmarked-join' :
      write.node.operator !== '=' || rhs.type === 'BinaryExpression' || rhs.type === 'TemplateLiteral' ? 'unmarked-composition' :
      strings.length ? 'unmarked-display-write' : 'unresolved-display-write';
    if (!strings.length && !write.reads.length && !calls.length) continue;
    var related = unique(calls).filter(function(name) { return helperReturns.has(name); }).map(function(name) {
      var helper = helperReturns.get(name);
      return { helper: name, location: helper.location, reason: 'Statically named helper body available; return/control-flow effects are not fully traced.' };
    });
    find(write.location, write.quality, kind, write.expression,
      'Quality reaches a display use without a recipe-preserving text operation.', sinks.get(write.quality), related);
  }

  var units = Array.from(confirmed.values()).sort(function(a, b) { return a.key.localeCompare(b.key); });
  var catalogs = {};
  var catalogReports = {};
  Object.keys(options.catalogs || {}).sort().forEach(function(locale) {
    var original = options.catalogs[locale];
    var merged = Object.assign({}, original);
    var added = [];
    var untranslated = [];
    units.forEach(function(unit) {
      if (!Object.prototype.hasOwnProperty.call(merged, unit.key)) {
        merged[unit.key] = locale === 'en' ? unit.key : null;
        added.push(unit.key);
      }
      var entry = merged[unit.key];
      if (entry === null || (locale !== 'en' && entry === unit.key)) untranslated.push(unit.key);
      else (unit.schemas || [{ parameters: unit.parameters }]).forEach(function(schema) {
        codec.validateEntry(unit.key, entry, schema.parameters).forEach(function(message) {
          addError(sourceLocation('source/locales/' + locale + '/content.json', locale, unit.key),
            message + (schema.source ? ' (source: ' + schema.source.file + ':' + schema.source.line + ')' : ''));
        });
      });
    });
    catalogs[locale] = merged;
    catalogReports[locale] = { newKeys: added, untranslatedKeys: untranslated,
      orphanedKeys: Object.keys(original).filter(function(key) { return !confirmed.has(key); }).sort() };
  });
  var reviews = options.dispositions && options.dispositions.findings || {};
  var prior = options.previousReport && options.previousReport.findings || [];
  var priorById = new Map(prior.map(function(finding) { return [finding.id, finding]; }));
  findings.forEach(function(finding) {
    var review = reviews[finding.id];
    var previous = priorById.get(finding.id);
    finding.change = !previous ? 'new' : previous.fingerprint !== finding.fingerprint ? 'changed' :
      previous.location.line !== finding.location.line || previous.location.column !== finding.location.column ? 'moved' : 'unchanged';
    if (review) {
      if (!REVIEW_STATUSES.includes(review.status) || typeof review.reason !== 'string' || !review.reason.trim()) {
        addError(finding.location, 'Invalid finding disposition: status and a nonempty reason are required.');
        finding.review = { status: 'needs-review' };
      } else finding.review = Object.assign({}, review, { stale: review.fingerprint !== finding.fingerprint });
    } else finding.review = { status: 'needs-review' };
  });
  var currentIds = new Set(findings.map(function(finding) { return finding.id; }));
  return { version: 1, units: units, findings: findings, limitations: limitations, errors: errors,
    catalogs: catalogs, catalogReports: catalogReports,
    absentFindings: prior.filter(function(finding) { return !currentIds.has(finding.id); }),
    analysisScope: { sourceFiles: sortedFiles.length, displayQualities: sinks.size,
      statement: 'Best-effort static analysis. Dynamic property access, implicit string copying/composition, helper control flow and unregistered display paths are not exhaustive.' } };
}

function readJSON(file, fallback) {
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : fallback;
}
async function extractProject(project, options) {
  options = options || {};
  project = path.resolve(project || '.');
  var source = path.join(project, 'source');
  var files = [];
  var read = function(directory) {
    fs.readdirSync(directory, { withFileTypes: true }).sort(function(a, b) { return a.name.localeCompare(b.name); }).forEach(function(entry) {
      if (entry.name === 'locales' || entry.name === 'img' || entry.name.startsWith('.')) return;
      var file = path.join(directory, entry.name);
      if (entry.isDirectory()) read(file);
      else if (entry.isFile() && /\.(dry|js|json)$/.test(entry.name)) files.push({ name: path.relative(project, file), contents: fs.readFileSync(file, 'utf8') });
    });
  };
  read(source);
  var localesDir = path.join(source, 'locales');
  var catalogs = {};
  if (fs.existsSync(localesDir)) fs.readdirSync(localesDir, { withFileTypes: true }).forEach(function(entry) {
    var file = path.join(localesDir, entry.name, 'content.json');
    if (entry.isDirectory() && !entry.name.startsWith('_') && fs.existsSync(file)) catalogs[entry.name] = readJSON(file, {});
  });
  if (!Object.keys(catalogs).length) catalogs.en = {};
  var reportPath = path.resolve(project, options.report || 'source/locales/_reports/extraction.json');
  var dispositionPath = path.resolve(project, options.review || 'source/locales/_reports/dispositions.json');
  var result = await extractFiles(files, {
    declarations: options.declarations || readJSON(path.join(source, 'data', 'localization.json'), {}),
    catalogs: catalogs, dispositions: readJSON(dispositionPath, { findings: {} }), previousReport: readJSON(reportPath, null)
  });
  if (!options.dryRun) {
    // Validation failures leave the source catalogs/review/report untouched.
    if (result.errors.length) return result;
    Object.keys(result.catalogs).forEach(function(locale) {
      var file = path.join(localesDir, locale, 'content.json');
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, JSON.stringify(result.catalogs[locale], null, 2) + '\n');
    });
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    var report = Object.assign({}, result);
    delete report.catalogs;
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
    if (!fs.existsSync(dispositionPath)) {
      fs.mkdirSync(path.dirname(dispositionPath), { recursive: true });
      fs.writeFileSync(dispositionPath, JSON.stringify({ version: 1, findings: {} }, null, 2) + '\n');
    }
  }
  return result;
}

module.exports = { extractFiles: extractFiles, extractProject: extractProject, reviewStatuses: REVIEW_STATUSES };
