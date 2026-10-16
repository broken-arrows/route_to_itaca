'use strict';

var fs = require('fs');
var path = require('path');
var pathToFileURL = require('url').pathToFileURL;
var utils = require('../utils');
var auditCompiledGame = require('../../audit').auditCompiledGame;
var auditSourceFiles = require('../../audit').auditSourceFiles;

var cmdAudit = new utils.Command('audit');
cmdAudit.createArgumentParser = function(subparsers) {
  var parser = subparsers.addParser(this.name, {
    help: 'Audit compiled content and configured source authoring rules.'
  });
  parser.addArgument(['project'], {
    nargs: '?',
    help: 'The project to audit (default: the current directory).'
  });
  parser.addArgument(['-c', '--config'], {
    help: 'Project audit config module (default: dendrynexus.audit.mjs, if present).'
  });
};
cmdAudit.run = function(args, callback) {
  var project = path.resolve(args.project || '.');
  var compiled = path.join(project, 'out', 'game.json');
  var configPath = path.resolve(project, args.config || 'dendrynexus.audit.mjs');
  if (!fs.existsSync(compiled)) {
    return callback(new Error('Compiled game not found: ' + compiled));
  }
  if (args.config && !fs.existsSync(configPath)) {
    return callback(new Error('Audit config not found: ' + configPath));
  }
  var configPromise = fs.existsSync(configPath)
    ? import(pathToFileURL(configPath).href).then(function(module) { return module.default || module; })
    : Promise.resolve({});
  configPromise.then(function(config) {
    ['widgetNames', 'deriveNames', 'allowBrowserGlobals'].forEach(function(key) {
      if (config[key] !== undefined && !Array.isArray(config[key])) {
        throw new TypeError('Audit config ' + key + ' must be an array');
      }
    });
    var deprecated = config.deprecatedSceneProperties;
    if (deprecated !== undefined && (!deprecated || typeof deprecated !== 'object' ||
        Array.isArray(deprecated) || Object.values(deprecated).some(function(value) { return typeof value !== 'string'; }))) {
      throw new TypeError('Audit config deprecatedSceneProperties must map property names to replacement guidance');
    }
    var result = auditCompiledGame(JSON.parse(fs.readFileSync(compiled, 'utf8')), config);
    result.violations.forEach(function(v) {
      console.error('Browser global in ' + v.sceneId + ' (' + v.hit + '): ' + v.source);
    });
    result.unknownWidgets.forEach(function(name) { console.error('Unknown widget: ' + name); });
    result.unknownDerives.forEach(function(name) { console.error('Unknown derivation: ' + name); });
    if (result.violations.length || result.unknownWidgets.length || result.unknownDerives.length) {
      throw new Error('Game audit failed');
    }
    function finish() {
      console.log('Game audit clean (' + result.sceneCount + ' scenes, ' +
        result.widgetCount + ' widgets, ' + result.deriveCount + ' derivations)');
      callback(null);
    }
    if (!deprecated || !Object.keys(deprecated).length) return finish();
    // Read current source even when game.json is already up to date. Derived
    // compiled flags must not hide reintroduced authoring declarations.
    utils.fetchContent(path.join(project, 'source'), function(err, files) {
      if (err) return callback(err);
      auditSourceFiles(files, config, function(err, violations) {
        if (err) return callback(err);
        violations.forEach(function(v) {
          console.error(v.location + ': Deprecated scene property "' + v.property + '" in ' +
            v.sceneId + '. ' + v.replacement);
        });
        if (violations.length) return callback(new Error('Source authoring audit failed'));
        finish();
      });
    });
  }).catch(callback);
};

module.exports = {cmd: cmdAudit};
