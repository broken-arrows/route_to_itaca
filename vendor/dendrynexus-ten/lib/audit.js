'use strict';

var async = require('async');
var dry = require('./parsers/dry');

// Source-only policy: compiled mechanical fields may be derived from roles,
// so inspecting game.json cannot distinguish authored legacy declarations.
function auditSourceFiles(files, options, callback) {
  var deprecated = Object.entries((options || {}).deprecatedSceneProperties || {});
  var violations = [];
  if (!deprecated.length) return callback(null, violations);

  async.eachSeries(files.filter(function(file) { return /\.scene\.dry$/.test(file.name); }), function(file, done) {
    dry.parseFromContent(file.name, file.contents, function(err, scene) {
      if (err) return done(err);
      [scene].concat(scene.sections || []).forEach(function(section) {
        deprecated.forEach(function(entry) {
          var property = dry.convertPropertyNameToCamelCase(entry[0]);
          if (section[property] !== undefined) {
            violations.push({
              sceneId: section.id,
              property: entry[0],
              location: dry.propertyFileAndLine(section, property),
              replacement: entry[1]
            });
          }
        });
      });
      done();
    });
  }, function(err) { callback(err, violations); });
}

// Inspect compiled content rather than source files: $code is the JavaScript
// the runtime will execute, and content markers can be transformed by parsing.
var BROWSER_GLOBAL = /\b(window|document|globalThis)\.|(^|[^.\w])d3\./;

function auditCompiledGame(game, options) {
  options = options || {};
  var widgetNames = options.widgetNames && new Set(options.widgetNames);
  var deriveNames = options.deriveNames && new Set(options.deriveNames);
  var allowedGlobals = new Set(options.allowBrowserGlobals || []);
  var widgetsUsed = new Set();
  var derivesUsed = new Set();
  var violations = [];
  var scenes = game.scenes || {};

  function walk(sceneId, node) {
    if (typeof node === 'string') {
      for (var widget of node.matchAll(/data-widget=["']([\w-]+)["']/g)) {
        widgetsUsed.add(widget[1]);
      }
      for (var escaped of node.matchAll(/&quot;deriveFrom&quot;\s*:\s*&quot;([\w-]+)&quot;/g)) {
        derivesUsed.add(escaped[1]);
      }
      for (var plain of node.matchAll(/"deriveFrom"\s*:\s*"([\w-]+)"/g)) {
        derivesUsed.add(plain[1]);
      }
      return;
    }
    if (Array.isArray(node)) {
      node.forEach(function(child) { walk(sceneId, child); });
      return;
    }
    if (!node || typeof node !== 'object') return;

    var top = sceneId.split('.')[0];
    if (typeof node.$code === 'string' &&
        !allowedGlobals.has(sceneId) && !allowedGlobals.has(top)) {
      var hit = node.$code.match(BROWSER_GLOBAL);
      if (hit) {
        violations.push({sceneId: sceneId, hit: hit[0], source: node.$code.trim().slice(0, 120)});
      }
    }
    Object.values(node).forEach(function(child) { walk(sceneId, child); });
  }

  Object.keys(scenes).forEach(function(id) { walk(id, scenes[id]); });
  return {
    sceneCount: Object.keys(scenes).length,
    widgetCount: widgetsUsed.size,
    deriveCount: derivesUsed.size,
    violations: violations,
    unknownWidgets: widgetNames ? [...widgetsUsed].filter(function(name) { return !widgetNames.has(name); }) : [],
    unknownDerives: deriveNames ? [...derivesUsed].filter(function(name) { return !deriveNames.has(name); }) : [],
  };
}

module.exports = {auditCompiledGame: auditCompiledGame, auditSourceFiles: auditSourceFiles};
