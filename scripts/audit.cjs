'use strict';

const path = require('path');
const { cmd: auditCommand } = require('../vendor/dendrynexus-ten/lib/cli/cmd/audit');
const { validateGameAssets } = require('../vendor/dendrynexus-ten/lib/cli/game-assets');
const root = path.resolve(__dirname, '..');
auditCommand.run({ project: root }, err => {
  if (err) {
    console.error(err.message);
    process.exitCode = 1;
    return;
  }
  try {
    const checked = validateGameAssets(root, { extraSources: [path.join(root, 'out', 'html'), path.join(root, 'ui', 'src')] });
    checked.warnings.forEach(warning => console.warn('Asset warning: ' + warning));
    console.log('Assets checked: ' + checked.imageCount + ' files, ' + checked.referenceCount + ' literal references.');
  } catch (err) {
    console.error(err.message);
    process.exitCode = 1;
  }
});
