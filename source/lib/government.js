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

  /** Keep translated summaries separate from technical coalition attributes. */
  function governmentTooltipText(institution, parties, summary, labelHtml) {
    var className = INSTITUTIONS[institution];
    if (!className) return labelHtml || '';
    var members = Array.isArray(parties)
      ? parties.filter(function (party) { return typeof party === 'string' && party; })
      : [];
    return {
      kind: 'markup', tag: 'span',
      attrs: { class: className, 'data-parties': members.join(' '), 'data-summary': summary || '' },
      content: labelHtml || ''
    };
  }

  // Older authored coalition spans carry a rendered English summary in an HTML
  // attribute. Translate its two sentence shapes at the tooltip boundary; this
  // needs no saved qualities and leaves the party/seat arithmetic in the scene.
  function localizeGovernmentSummary(summary, translate) {
    if (typeof summary !== 'string' || typeof translate !== 'function') return summary || '';
    var match = /^\s*(Majority|Minority) government\s*-\s*(\d+)\s*\/\s*(\d+)\s*$/.exec(summary);
    if (match) {
      var key = match[1] + ' government - {seats}/{majority}';
      var params = { seats: match[2], majority: match[3] };
      var translated = translate(key, params);
      return translated === key || translated === match[1] + ' government - ' + match[2] + '/' + match[3]
        ? summary : translated;
    }
    if (summary.trim() === 'Majority government') {
      return translate('Majority government') || summary;
    }
    return summary;
  }

  /** Rebrand captured display parameters without changing translation keys. */
  function rebrandGovernmentText(value, from, to, exact) {
    if (typeof value === 'string') {
      if (exact) return value.replaceAll(from, to);
      return value.replaceAll(from.toUpperCase(), to.toUpperCase()).replaceAll(from, to);
    }
    if (!value || typeof value !== 'object') return value;
    if (Array.isArray(value)) return value.map(function (child) { return rebrandGovernmentText(child, from, to, exact); });
    var result = {};
    Object.keys(value).forEach(function (key) {
      result[key] = value.kind === 'text' && key === 'key'
        ? value[key] : rebrandGovernmentText(value[key], from, to, exact);
    });
    return result;
  }

  var api = {
    governmentTooltip: governmentTooltip,
    governmentTooltipText: governmentTooltipText,
    localizeGovernmentSummary: localizeGovernmentSummary,
    rebrandGovernmentText: rebrandGovernmentText
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    window.RTI_GOVERNMENT = api;
  }
})();
