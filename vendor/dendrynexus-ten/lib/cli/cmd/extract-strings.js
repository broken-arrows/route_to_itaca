'use strict';

var utils = require('../utils');
var extractor = require('../../localization-extract');

var cmd = new utils.Command('extract-strings');
cmd.createArgumentParser = function(subparsers) {
  var parser = subparsers.addParser(this.name, {
    help: 'Extract confirmed translation templates and report uncertain display text.',
    description: 'Non-destructively add missing content keys and emit a source/discovery report. ' +
      'Existing translations, orphaned keys and reviewed dispositions are retained. ' +
      'Uncertain discovery findings are advisory; malformed explicit templates/catalog entries fail validation.'
  });
  parser.addArgument(['project'], { nargs: '?', help: 'Project directory (default: current directory).' });
  parser.addArgument(['--dry-run'], { action: 'storeTrue', defaultValue: false, help: 'Inspect and validate without writing any files.' });
  parser.addArgument(['--report'], { help: 'Project-relative report path (default: source/locales/_reports/extraction.json).' });
  parser.addArgument(['--review'], { help: 'Persisted disposition path (default: source/locales/_reports/dispositions.json).' });
  parser.addArgument(['--json'], { action: 'storeTrue', defaultValue: false, help: 'Print the complete report as JSON, excluding merged catalog previews.' });
};
cmd.run = function(args, callback) {
  utils.getProjectPath(args.project, function(err, project) {
    if (err) return callback(err);
    extractor.extractProject(project, { dryRun: args.dry_run, report: args.report, review: args.review }).then(function(result) {
      if (args.json) {
        var report = Object.assign({}, result);
        delete report.catalogs;
        console.log(JSON.stringify(report, null, 2));
      } else {
        console.log('Localization: ' + result.units.length + ' confirmed keys, ' + result.findings.length +
          ' advisory findings, ' + result.limitations.length + ' analysis limitations, ' + result.errors.length + ' validation errors.');
        Object.keys(result.catalogReports).forEach(function(locale) {
          var catalog = result.catalogReports[locale];
          console.log(locale + ': ' + catalog.newKeys.length + ' new, ' + catalog.untranslatedKeys.length +
            ' untranslated, ' + catalog.orphanedKeys.length + ' orphaned keys retained.');
        });
        console.log(args.dry_run ? 'Dry run: no files written.' : result.errors.length ? 'Validation failed: no files written.' :
          'Catalogs and extraction report written; existing review dispositions retained.');
      }
      if (result.errors.length) {
        if (!args.json) result.errors.slice(0, 20).forEach(function(error) {
          console.error(error.location.file + ':' + error.location.line + ': ' + error.message);
        });
        callback(new Error('Localization validation failed (' + result.errors.length + ' errors).'));
      } else callback(null);
    }).catch(callback);
  });
};

module.exports = { cmd: cmd };
