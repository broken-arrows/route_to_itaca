'use strict';

function formatSaveSceneName(scene) {
  var label = String(scene || '')
    .replace(/post_event/gi, '')
    .replace(/^[._\s-]+|[._\s-]+$/g, '')
    .split('.')[0]
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (!label) {
    return 'Between events';
  }

  return label.replace(/\b\w/g, function(letter) {
    return letter.toUpperCase();
  });
}

function formatSaveTimestamp(timestamp) {
  var lines = String(timestamp || '').split('\n');
  lines[0] = formatSaveSceneName(lines[0]);
  return lines.join('\n');
}

module.exports = {
  formatSaveSceneName: formatSaveSceneName,
  formatSaveTimestamp: formatSaveTimestamp,
};
