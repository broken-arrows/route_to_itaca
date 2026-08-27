/* =============================================================================
 * Semantic government-tooltip markup shared by authored content.
 *
 * The game owns the meaning (institution, formal cabinet parties, formation
 * arithmetic and visible label); each UI owns the popover presentation. Party
 * names that should receive ordinary glossary treatment belong OUTSIDE the
 * returned span.
 * ========================================================================== */
(function () {
  'use strict';

  var INSTITUTIONS = {
    generalitat: 'generalitat-coalition',
    gobierno: 'gobierno-coalition',
    ajuntament: 'ajuntament-coalition'
  };

  function escapeAttribute(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function governmentTooltip(institution, parties, summary, labelHtml) {
    var className = INSTITUTIONS[institution];
    if (!className) return String(labelHtml || '');

    var members = Array.isArray(parties)
      ? parties.filter(function (party) { return typeof party === 'string' && party; })
      : [];
    return '<span class="' + className + '" data-parties="' +
      escapeAttribute(members.join(' ')) + '" data-summary="' +
      escapeAttribute(summary || '') + '">' + String(labelHtml || '') + '</span>';
  }

  var api = { governmentTooltip: governmentTooltip };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    window.RTI_GOVERNMENT = api;
  }
})();
