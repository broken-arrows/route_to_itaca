'use strict';

var utils = require('../utils');
var extractor = require('../../localization-extract');
var readline = require('readline');

async function confirmPrune(catalogReports, ask, warn) {
  warn('WARNING: --prune permanently removes the following orphaned content catalog keys.');
  warn('Extraction is best effort; older saves may still use these translations. UI catalogs are untouched.');
  var total = 0;
  Object.keys(catalogReports).sort().forEach(function(locale) {
    var keys = catalogReports[locale].orphanedKeys;
    total += keys.length;
    if (!keys.length) return;
    warn('source/locales/' + locale + '/content.json: ' + keys.length + ' keys');
    keys.forEach(function(key) { warn('  ' + JSON.stringify(key)); });
  });
  if (!total) warn('No orphaned content keys found; catalog regeneration still requires confirmation.');
  if ((await ask('Type PRUNE to continue: ')).trim() !== 'PRUNE') return false;
  return (await ask('Type DELETE to confirm again: ')).trim() === 'DELETE';
}

function promptPrune(catalogReports) {
  if (!process.stdin.isTTY) return confirmPrune(catalogReports, function() {
    console.error('Pruning requires an interactive terminal. Use --prune --dry-run to preview.');
    return Promise.resolve('');
  }, console.error);
  var prompt = readline.createInterface({ input: process.stdin, output: process.stderr });
  var ask = function(message) {
    return new Promise(function(resolve) {
      var answered = false;
      var onClose = function() { if (!answered) resolve(''); };
      prompt.once('close', onClose);
      prompt.question(message, function(answer) {
        answered = true;
        prompt.removeListener('close', onClose);
        resolve(answer);
      });
    });
  };
  return confirmPrune(catalogReports, ask, console.error).finally(function() { prompt.close(); });
}

var cmd = new utils.Command('extract-strings');
cmd.createArgumentParser = function(subparsers) {
  var parser = subparsers.addParser(this.name, {
    help: 'Extract confirmed translation templates and report uncertain display text.',
    description: 'Add missing content keys and emit a source/discovery report. ' +
      'Existing translations, orphaned keys and reviewed dispositions are retained by default. ' +
      'Uncertain discovery findings are advisory; malformed explicit templates/catalog entries fail validation.'
  });
  parser.addArgument(['project'], { nargs: '?', help: 'Project directory (default: current directory).' });
  parser.addArgument(['--dry-run'], { action: 'storeTrue', defaultValue: false, help: 'Inspect and validate without writing any files.' });
  parser.addArgument(['--prune'], { action: 'storeTrue', defaultValue: false,
    help: 'Remove orphaned content keys after two interactive confirmations; combine with --dry-run to preview.' });
  parser.addArgument(['--report'], { help: 'Project-relative report path (default: source/locales/_reports/extraction.json).' });
  parser.addArgument(['--review'], { help: 'Persisted disposition path (default: source/locales/_reports/dispositions.json).' });
  parser.addArgument(['--json'], { action: 'storeTrue', defaultValue: false, help: 'Print the complete report as JSON, excluding merged catalog previews.' });
};
cmd.run = function(args, callback) {
  utils.getProjectPath(args.project, function(err, project) {
    if (err) return callback(err);
    extractor.extractProject(project, { dryRun: args.dry_run, prune: args.prune,
      confirmPrune: args.prune && !args.dry_run ? promptPrune : undefined,
      report: args.report, review: args.review }).then(function(result) {
      if (args.json) {
        var report = Object.assign({}, result);
        delete report.catalogs;
        console.log(JSON.stringify(report, null, 2));
      } else {
        console.log('Localization: ' + result.units.length + ' confirmed keys, ' + result.findings.length +
          ' advisory findings, ' + result.limitations.length + ' analysis limitations, ' + result.errors.length + ' validation errors.');
        Object.keys(result.catalogReports).forEach(function(locale) {
          var catalog = result.catalogReports[locale];
          var fate = 'retained';
          if (args.prune && !result.errors.length) {
            fate = args.dry_run ? 'would be removed' : result.pruneCancelled ? 'retained (prune cancelled)' : 'removed';
          }
          console.log(locale + ': ' + catalog.newKeys.length + ' new, ' + catalog.untranslatedKeys.length +
            ' untranslated, ' + catalog.orphanedKeys.length + ' orphaned keys ' + fate + '.');
          if (args.prune && args.dry_run) catalog.orphanedKeys.forEach(function(key) {
            console.log('  ' + JSON.stringify(key));
          });
        });
        console.log(args.dry_run ? 'Dry run: no files written.' : result.errors.length ? 'Validation failed: no files written.' :
          result.pruneCancelled ? 'Prune cancelled: no files written.' :
          'Catalogs and extraction report written; existing review dispositions retained.');
      }
      if (result.errors.length) {
        if (!args.json) result.errors.slice(0, 20).forEach(function(error) {
          console.error(error.location.file + ':' + error.location.line + ': ' + error.message);
        });
        callback(new Error('Localization validation failed (' + result.errors.length + ' errors).'));
      } else if (result.pruneCancelled) callback(new Error('Prune cancelled; no files written.'));
      else callback(null);
    }).catch(callback);
  });
};

module.exports = { cmd: cmd, confirmPrune: confirmPrune };
