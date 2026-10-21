/* Shared, JSON-only localization recipes and authored-content units. */
'use strict';

var own = function (object, key) { return Object.prototype.hasOwnProperty.call(object, key); };
var clone = function (value) {
  if (value === undefined) return undefined;
  return JSON.parse(JSON.stringify(value));
};
var placeholder = /\{([A-Za-z_][\w]*)(?::([a-z][a-z_]*)(?::(cap))?)?\}/g;
var nameKey = function(namespace, id, form) { return '@name/' + namespace + '/' + id + '/' + form; };
var nameText = function(namespace, id, fallback) {
  return { kind: 'name', namespace: String(namespace), id: String(id).toLowerCase(), value: String(fallback === undefined ? id : fallback) };
};
// Names are plain text. Formatting and glossary markup belong to the renderer.
var mergeNames = function(catalog, namespaces) {
  var result = Object.assign({}, catalog || {});
  Object.keys(namespaces || {}).forEach(function(namespace) {
    Object.keys(namespaces[namespace] || {}).forEach(function(id) {
      var forms = namespaces[namespace][id];
      if (!forms || typeof forms !== 'object' || typeof forms.bare !== 'string') throw new Error('Name requires a bare form: ' + id);
      Object.keys(forms).forEach(function(form) {
        if (!/^[a-z][a-z_]*$/.test(form) || typeof forms[form] !== 'string' || /[<>{}]/.test(forms[form])) throw new Error('Invalid name form: ' + id + '/' + form);
        result[nameKey(namespace, id, form)] = forms[form];
      });
    });
  });
  return result;
};
var nameParameters = function(params) {
  var result = {};
  Object.keys(params || {}).forEach(function(key) { if (params[key] && params[key].kind === 'name') result[key] = params[key].namespace; });
  return result;
};
var namesIn = function (key) {
  var names = [];
  String(key).replace(placeholder, function (_, name) {
    if (names.indexOf(name) === -1) names.push(name);
    return _;
  });
  return names;
};
var markupIn = function (key) { return String(key).match(/<[^>]*>/g) || []; };
var validateTemplate = function (key, parameters, named) {
  var errors = [];
  if (typeof key !== 'string') return ['Template must be a string.'];
  var allowed = Array.isArray(parameters) ? parameters : Object.keys(parameters || {});
  namesIn(key).forEach(function (name) {
    if (allowed.indexOf(name) === -1) errors.push('Unknown placeholder {' + name + '}.');
  });
  String(key).replace(placeholder, function(_, name, form) {
    if (form && named && !own(named, name)) errors.push('Formatted placeholder requires a named reference: ' + name + '.');
    return _;
  });
  // JSON widget attributes and CSS inside protected HTML are literal structure.
  // Their braces are not translation-placeholder syntax.
  var visible = key.replace(/<[^>]*>/g, '');
  if (/[{}]/.test(visible.replace(placeholder, ''))) errors.push('Malformed placeholder.');
  return errors;
};
var validateTranslation = function (entry, parameters, named) {
  var allowed = Array.isArray(parameters) ? parameters : Object.keys(parameters || {});
  if (typeof entry === 'string') return validateTemplate(entry, allowed, named);
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return ['Catalog entry must be a string or select/plural entry.'];
  var selector = entry.select || entry.plural;
  var errors = [];
  if (!!entry.select === !!entry.plural || typeof selector !== 'string') errors.push('Exactly one select/plural parameter is required.');
  if (allowed.indexOf(selector) === -1) errors.push('Unknown selector ' + selector + '.');
  if (!entry.cases || typeof entry.cases !== 'object' || !own(entry.cases, 'other')) errors.push('Selector requires an other case.');
  else Object.keys(entry.cases).forEach(function (name) {
    if (entry.plural && !/^(?:=\d+(?:\.\d+)?|zero|one|two|few|many|other)$/.test(name)) errors.push('Unknown plural case ' + name + '.');
    errors = errors.concat(validateTranslation(entry.cases[name], allowed, named));
  });
  return errors;
};
var validateNameForms = function(entry, named, forms) {
  var errors = [];
  if (typeof entry === 'string') entry.replace(placeholder, function(_, name, form) {
    var supported = forms && forms[(named || {})[name]];
    if (form && supported && supported.indexOf(form) === -1) errors.push('Unknown name form ' + form + ' for ' + name + '.');
    return _;
  });
  else if (entry && entry.cases) Object.keys(entry.cases).forEach(function(key) { errors = errors.concat(validateNameForms(entry.cases[key], named, forms)); });
  return errors;
};
var validateEntry = function (key, entry, parameters, named) {
  var errors = validateTranslation(entry, parameters, named);
  var required = namesIn(key);
  var protectedMarkup = markupIn(key).slice().sort();
  var visit = function (value, selected) {
    selected = selected || [];
    if (typeof value === 'string') {
      required.forEach(function (name) {
        if (namesIn(value).indexOf(name) === -1 && selected.indexOf(name) === -1) errors.push('Missing placeholder {' + name + '}.');
      });
      if (JSON.stringify(markupIn(value).slice().sort()) !== JSON.stringify(protectedMarkup)) errors.push('Protected HTML markup changed.');
    } else if (value && value.cases) Object.keys(value.cases).forEach(function (name) {
      visit(value.cases[name], selected.concat(value.select || value.plural));
    });
  };
  visit(entry);
  return errors;
};
var literal = function (value) { return { kind: 'literal', value: value == null ? '' : String(value) }; };
var raw = function (html) { return { kind: 'raw', html: String(html == null ? '' : html) }; };
var text = function (key, params) {
  params = clone(params || {});
  var errors = validateTemplate(key, Object.keys(params), nameParameters(params));
  if (/\{\w+:/.test(key.replace(/<[^>]*>/g, ''))) errors.push('Grammatical forms belong in translations, not source keys.');
  if (errors.length) throw new Error(errors.join(' '));
  return { kind: 'text', key: key, params: params };
};
var joinText = function (parts, separator) {
  return { kind: 'join', parts: clone(parts || []), separator: clone(separator === undefined ? '' : separator) };
};
var markup = function (tag, attrs, content) {
  if (!/^[A-Za-z][\w-]*$/.test(tag)) throw new Error('Invalid markup tag.');
  Object.keys(attrs || {}).forEach(function (name) {
    if (!/^[A-Za-z_:][\w:.-]*$/.test(name)) throw new Error('Invalid markup attribute.');
  });
  return { kind: 'markup', tag: tag, attrs: clone(attrs || {}), content: clone(content == null ? literal('') : content) };
};
var escapeAttribute = function (value) {
  return String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
};
var selectEntry = function (entry, params, locale) {
  while (entry && typeof entry === 'object') {
    var value = params[entry.select || entry.plural];
    if (value && typeof value === 'object') value = render(value, 'en', null);
    var choice = String(value);
    if (entry.plural) {
      var numeric = Number(value);
      choice = '=' + numeric;
      if (!own(entry.cases, choice)) choice = new Intl.PluralRules(locale || 'en').select(numeric);
    }
    entry = own(entry.cases, choice) ? entry.cases[choice] : entry.cases.other;
  }
  return entry;
};
var render = function (recipe, locale, catalog) {
  if (recipe == null) return '';
  if (typeof recipe !== 'object') return String(recipe);
  switch (recipe.kind) {
    // Unformatted inserts retain their exact source value, including IDs used
    // in CSS/data attributes. Canonical display spelling is explicit (:bare).
    case 'name': return recipe.value;
    case 'literal': return recipe.value;
    case 'raw': return recipe.html;
    case 'join': return recipe.parts.map(function (part) { return render(part, locale, catalog); }).join(render(recipe.separator, locale, catalog));
    case 'markup':
      var attrs = Object.keys(recipe.attrs).map(function (name) {
        return ' ' + name + '="' + escapeAttribute(render(recipe.attrs[name], locale, catalog)) + '"';
      }).join('');
      return '<' + recipe.tag + attrs + '>' + render(recipe.content, locale, catalog) + '</' + recipe.tag + '>';
    case 'content': return require('./ui/content/html').convertLine(renderContent(recipe, locale, catalog));
    case 'text':
      var entry = catalog && own(catalog, recipe.key) ? catalog[recipe.key] : recipe.key;
      // A malformed external catalog must never execute code or break gameplay.
      if (entry == null || validateEntry(recipe.key, entry, Object.keys(recipe.params || {}), nameParameters(recipe.params)).length) entry = recipe.key;
      var template = selectEntry(entry, recipe.params || {}, locale);
      var missingForm = false;
      var output = String(template).replace(placeholder, function (_, name, form, capital) {
        var param = recipe.params[name];
        if (!form) return render(param, locale, catalog);
        var key = param && param.kind === 'name' ? nameKey(param.namespace, param.id, form) : '';
        if (!catalog || !own(catalog, key) || typeof catalog[key] !== 'string') { missingForm = true; return ''; }
        var value = catalog[key];
        return capital ? value.charAt(0).toLocaleUpperCase(locale) + value.slice(1) : value;
      });
      // Older saves/plain parameters and unknown names retain the exact English
      // sentence instead of silently dropping a required article/preposition.
      return missingForm ? render(text(recipe.key, recipe.params), 'en', null) : output;
    default: throw new Error('Unknown text recipe kind: ' + recipe.kind);
  }
};
var validateRecipe = function (recipe) {
  var errors = [];
  if (!recipe || typeof recipe !== 'object') return ['Recipe must be an object.'];
  if (recipe.kind === 'name') {
    if (typeof recipe.namespace !== 'string' || typeof recipe.id !== 'string' || typeof recipe.value !== 'string') errors.push('Invalid named reference.');
  } else if (recipe.kind === 'text') {
    errors = validateTemplate(recipe.key, Object.keys(recipe.params || {}), nameParameters(recipe.params));
    if (typeof recipe.key === 'string' && /\{\w+:/.test(recipe.key.replace(/<[^>]*>/g, ''))) errors.push('Grammatical forms belong in translations, not source keys.');
    Object.keys(recipe.params || {}).forEach(function (name) {
      if (recipe.params[name] && typeof recipe.params[name] === 'object') errors = errors.concat(validateRecipe(recipe.params[name]));
    });
  } else if (recipe.kind === 'join') {
    if (!Array.isArray(recipe.parts)) errors.push('Join parts must be an array.');
    else recipe.parts.forEach(function (part) { if (part && typeof part === 'object') errors = errors.concat(validateRecipe(part)); });
  } else if (recipe.kind === 'markup') {
    try { markup(recipe.tag, recipe.attrs, recipe.content); } catch (error) { errors.push(error.message); }
    if (recipe.content && typeof recipe.content === 'object') errors = errors.concat(validateRecipe(recipe.content));
    Object.keys(recipe.attrs || {}).forEach(function (name) {
      if (recipe.attrs[name] && typeof recipe.attrs[name] === 'object') errors = errors.concat(validateRecipe(recipe.attrs[name]));
    });
  } else if (recipe.kind === 'content') {
    if (!Array.isArray(recipe.nodes)) errors.push('Content nodes must be an array.');
  } else if (recipe.kind !== 'literal' && recipe.kind !== 'raw') errors.push('Unknown recipe kind.');
  return errors;
};

var paragraphTypes = { paragraph: true, heading: true, quotation: true, attribution: true, hrule: true };
var formatTags = { 'emphasis-1': 'em', 'emphasis-2': 'strong', 'emphasis-3': 'code', hidden: 'span' };
var parameterName = function (dependency, index) {
  return dependency && dependency.quality ? dependency.quality.replace(/[^\w]/g, '_') : 'value' + (index + 1);
};

// Both tooling and runtime walk the same compiled content. Tooling supplies no
// resolver; runtime supplies one whose values were evaluated exactly once.
var build = function (content, options) {
  options = options || {};
  var units = [];
  var dependencies = content && content.stateDependencies || [];
  var values = options.values || [];
  var bindings = options.grammarBindings || {};
  var context = {};
  var contextNames = [];
  dependencies.forEach(function (dependency) {
    var binding = dependency.quality && bindings[dependency.quality];
    if (!binding) return;
    (Array.isArray(binding) ? binding : [binding]).forEach(function (name) {
      if (contextNames.indexOf(name) === -1) contextNames.push(name);
      var captured = options.captureContext ? options.captureContext(name) : '';
      context[name] = captured === undefined ? null : captured;
    });
  });
  var addUnit = function (key, params) {
    // Pure formatting/widget placeholders are technical structure, not prose.
    if (key.replace(/\{[\w]+\}/g, '').replace(/<[^>]*>/g, '').trim()) {
      units.push({ key: key, placeholders: namesIn(key), params: Object.keys(params), parameters: Object.keys(params), context: contextNames.slice(), nameParameters: nameParameters(params), protectedMarkup: markupIn(key) });
    }
    return text(key, params);
  };
  var htmlRecipe = function (html) {
    var params = Object.assign({}, context), number = 0;
    var key = String(html).replace(/<[^>]*>/g, function (tag) {
      var name = 'markup' + (++number);
      params[name] = raw(tag);
      return '{' + name + '}';
    });
    return addUnit(key, params);
  };
  var run = function (input) {
    var parts = Array.isArray(input) ? input : [input];
    var params = Object.assign({}, context), counts = {}, clause = 0, emphasis = 0, technical = 0;
    var nameFor = function (base) {
      counts[base] = (counts[base] || 0) + 1;
      return base + (counts[base] === 1 ? '' : '_' + counts[base]);
    };
    var key = parts.map(function (part) {
      if (typeof part === 'string') return part;
      if (part == null) return '';
      var name, value;
      if (part.type === 'insert') {
        name = nameFor(parameterName(dependencies[part.insert], part.insert));
        value = values[part.insert];
        var dependency = dependencies[part.insert];
        var namespace = dependency && !dependency.qdisplay && (options.nameBindings || {})[dependency.quality];
        params[name] = namespace ? nameText(namespace, value && value.kind === 'literal' ? value.value : (value === undefined ? '' : value)) : (value === undefined ? literal('') : clone(value));
        return '{' + name + '}';
      }
      if (part.type === 'conditional') {
        name = nameFor('clause' + (++clause));
        var branch = run(part.content);
        params[name] = values[part.predicate] ? branch : literal('');
        return '{' + name + '}';
      }
      if (formatTags[part.type]) {
        name = nameFor('emphasis' + (++emphasis));
        params[name] = run(part.content);
        var tag = formatTags[part.type];
        return '<' + tag + (part.type === 'hidden' ? ' class="hidden"' : '') + '>{' + name + '}</' + tag + '>';
      }
      if (part.type === 'line-break') return '<br>';
      name = nameFor('markup' + (++technical));
      params[name] = part.type === 'magic' ? htmlRecipe(Array.isArray(part.content) ? part.content.join('') : part.content) : raw('');
      return '{' + name + '}';
    }).join('');
    return addUnit(key, params);
  };
  var body = content && typeof content === 'object' && !Array.isArray(content) && content.content !== undefined
    ? (content.type ? content : content.content) : content;
  var list = Array.isArray(body) ? body : [body];
  var nodes;
  if (list.some(function (node) { return node && typeof node === 'object' && paragraphTypes[node.type]; })) {
    nodes = list.map(function (node) {
      if (!node || typeof node !== 'object') return { type: 'paragraph', content: run(node) };
      if (node.type === 'hrule') return { type: 'hrule' };
      if (node.type === 'magic') return { type: 'magic', content: htmlRecipe(node.content) };
      return { type: node.type, content: run(node.content) };
    });
  } else if (body && body.type === 'magic' && options.useParas) {
    nodes = [{ type: 'magic', content: htmlRecipe(body.content) }];
  } else {
    var inline = run(list);
    nodes = options.useParas ? [{ type: 'paragraph', content: inline }] : [inline];
  }
  return { snapshot: { kind: 'content', nodes: nodes }, units: units };
};
var collectUnits = function (content, options) { return build(content, options).units; };
var captureContent = function (content, options) {
  options = Object.assign({}, options || {});
  var dependencies = content && content.stateDependencies || [];
  options.values = dependencies.map(function (dependency, index) {
    return options.resolveDependency ? options.resolveDependency(dependency, index) : undefined;
  });
  return build(content, options).snapshot;
};
var renderContent = function (snapshot, locale, catalog) {
  return (snapshot.nodes || []).map(function (node) {
    if (node.kind) return render(node, locale, catalog);
    var copy = Object.assign({}, node);
    if (node.content !== undefined) copy.content = render(node.content, locale, catalog);
    return copy;
  });
};

module.exports = {
  validateNameForms: validateNameForms, nameText: nameText, mergeNames: mergeNames, nameParameters: nameParameters,
  clone: clone, text: text, literal: literal, raw: raw, markup: markup,
  joinText: joinText, render: render, collectUnits: collectUnits, buildUnits: collectUnits,
  captureContent: captureContent, renderContent: renderContent,
  validateTemplate: validateTemplate, validateTranslation: validateTranslation,
  validateEntry: validateEntry, validateRecipe: validateRecipe,
  namesIn: namesIn
};
