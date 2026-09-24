'use strict';

var fs = require('fs');
var path = require('path');
var pathToFileURL = require('url').pathToFileURL;
var utils = require('../utils');
var auditCompiledGame = require('../../audit').auditCompiledGame;

var cmdAudit = new utils.Command('audit');
cmdAudit.createArgumentParser = function(subparsers) {
  var parser = subparsers.addParser(this.name, {
    help: 'Audit a compiled game for browser globals and unknown content markers.'
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
    var result = auditCompiledGame(JSON.parse(fs.readFileSync(compiled, 'utf8')), config);
    result.violations.forEach(function(v) {
      console.error('Browser global in ' + v.sceneId + ' (' + v.hit + '): ' + v.source);
    });
    result.unknownWidgets.forEach(function(name) { console.error('Unknown widget: ' + name); });
    result.unknownDerives.forEach(function(name) { console.error('Unknown derivation: ' + name); });
    if (result.violations.length || result.unknownWidgets.length || result.unknownDerives.length) {
      throw new Error('Game audit failed');
    }
    console.log('Game audit clean (' + result.sceneCount + ' scenes, ' +
      result.widgetCount + ' widgets, ' + result.deriveCount + ' derivations)');
    callback(null);
  }).catch(callback);
};

module.exports = {cmd: cmdAudit};
