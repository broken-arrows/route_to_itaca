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

  // Pure projection used when checking readiness and starting the selected motion.
  function spanishCensureOffer(Q, visits, kind) {
    var chamber = visits['congreso_election.post_election'] || 0;
    var S = function (p) { return Math.max(0, Number(Q[p + '_congreso_s']) || 0); };
    var left = S('sumar') ? 'sumar' : S('up') ? 'up' : 'podemos';
    var vehicle = Q.player_party;
    if (Q.jxsi_in_congreso && Q[vehicle + '_in_jxsi']) vehicle = 'jxsi';
    else if (Q.jxcat_in_congreso && Q[vehicle + '_in_jxcat']) vehicle = 'jxcat';
    var eligible = Q.next_spanish_election_time - Q.time <= 36 && !Q.spa_caretaker_gov
        && ['pp', 'cs', 'vox'].includes(Q.spanish_pm_party)
        && !(Q.spanish_coalition || []).includes(left) && !(Q.spanish_coalition || []).includes('psoe');
    var corruption = Number(Q['corruption_' + Q.spanish_pm_party]) || 0;
    var relations = Math.min(Number(Q.psc_relations) || 0, Number(Q[Q.parlament_current_icv + '_relations']) || 0);
    var territorial = Number(Q.cat_spa_relations) || 0;
    var catFamily = ['ciu', 'cdc', 'dl', 'pdcat', 'jxsi', 'jxcat', 'junts'];
    var roadmap = Number(Q[Q.parlament_current_ciu + '_roadmap']) || 0;
    var parties = Array.from(new Set(Q.congreso_parties || []));
    var total = Object.values(Q.congreso_seats || {}).reduce(function (n, s) { return n + Number(s); }, 0) || 350;
    var majority = Math.floor(total / 2) + 1;
        var party = kind === 'psoe' ? 'psoe' : left;
        var candidate = Q[party + '_leader'];
        var offer = { kind: kind, party: party, candidate: candidate, gender: Q[party + '_leader_gender'] || 'm',
            chamber: chamber, incumbent: Q.spanish_pm, incumbentParty: Q.spanish_pm_party,
            total: total, majority: majority, left: left, player: vehicle, playerSeats: S(vehicle),
            votes: [], mediated: false, ciuPersuaded: false, mediationCost: 2, mediationTarget: 0,
            mediationParty: kind === 'psoe' ? left : 'psoe', noReferendum: false };
        var socialists = 0;
        if (kind === 'podemos') {
            if (Q.psoe_leader === 'Pedro Sánchez' && territorial >= 40 && (corruption >= 40 || S(left) > S('psoe'))) socialists = S('psoe');
            if (Q.psoe_leader === '???') socialists = Math.floor(S('psoe') * 0.2);
            if (Q.psoe_leader === 'Pedro Sánchez' && relations >= 65 && territorial >= 30) offer.mediationTarget = S('psoe');
            if (Q.psoe_leader === 'Eduardo Madina' && relations >= 75 && territorial >= 50 && corruption >= 50 && S(left) >= S('psoe')) {
                offer.mediationTarget = S('psoe');
                offer.noReferendum = true;
            }
            if (Q.psoe_leader === '???' && relations >= 65 && territorial >= 30) {
                var stretch = relations >= 80 && territorial >= 45 && corruption >= 50;
                offer.mediationTarget = Math.floor(S('psoe') * (stretch ? 0.6 : 0.4));
                offer.mediationCost = stretch ? 3 : 2;
            }
        } else if (relations >= 65 && territorial >= 30 && Q.psoe_leader !== 'Susana Díaz') {
            offer.mediationTarget = S(left);
        }
        offer.ciuCanPersuade = roadmap <= 2 && territorial >= 25
            && !(kind === 'psoe' && Q.psoe_leader === 'Susana Díaz' && territorial < 50);
        offer.ciuRelations = Number(Q[Q.parlament_current_ciu + '_relations']) || 0;
        for (var p of parties) {
            var seats = S(p);
            if (!seats || p === vehicle) continue;
            var yes = 0, no = 0;
            if (p === party) yes = seats;
            else if (kind === 'podemos' && p === 'psoe') {
                yes = socialists;
                if (Q.psoe_leader === 'Susana Díaz') no = seats;
            } else if (kind === 'psoe' && p === left) {
                if (Q.psoe_leader === 'Pedro Sánchez' || (Q.psoe_leader === 'Eduardo Madina' && territorial >= 35)
                    || (Q.psoe_leader === 'Susana Díaz' && territorial >= 50 && corruption >= 50)) yes = seats;
            } else if (['iu', 'mpais', 'compromis', 'mes', 'gbai', 'bng', 'te', 'prc', 'psc', 'ehbildu', 'amaiur'].includes(p)) yes = seats;
            else if (p === 'erc' && (kind === 'podemos' || (Q.erc_roadmap || 0) < 3)) yes = seats;
            else if (['cup', 'fr'].includes(p) && kind === 'podemos') yes = seats;
            else if (catFamily.includes(p)) {
                if (kind === 'psoe' && roadmap <= 2 && Q.psoe_leader !== 'Susana Díaz') yes = seats;
                else if (!offer.ciuCanPersuade) no = seats;
            } else if (['pp', 'cs', 'vox', 'upn', 'nsuma', 'fac', 'upyd'].includes(p)) no = seats;
            offer.votes.push({ party: p, seats: seats, yes: yes, no: no, catRight: catFamily.includes(p) });
        }
        // PNV needs a viable candidate and, for Podemos, socialist cooperation.
        offer.pnvCompatible = kind === 'podemos' || Q.psoe_leader !== 'Susana Díaz' || territorial >= 50;
        var potential = offer.playerSeats;
        for (var row of offer.votes) {
            if (row.party === 'pnv') continue;
            potential += row.party === offer.mediationParty ? Math.max(row.yes, offer.mediationTarget)
                : row.catRight && offer.ciuCanPersuade ? row.seats : row.yes;
        }
        if (offer.pnvCompatible && (kind === 'psoe' || Math.max(socialists, offer.mediationTarget) > 0)) potential += S('pnv');
        offer.ready = eligible && S(party) >= 60 && candidate && candidate !== '???' && !(Q.censure_attempted || {})[kind]
            && (corruption >= 50 || potential >= majority || (kind === 'podemos' && Q.censure_podemos_trigger));
        return offer;
  }

  // Select one challenger before pending-event discovery. Votes are the last
  // election's national popular-vote shares, using each offer's live carrier.
  function spanishCensurePending(Q, visits) {
    if (Q.censure_current) return '';
    var psoe = spanishCensureOffer(Q, visits, 'psoe');
    var podemos = spanishCensureOffer(Q, visits, 'podemos');
    if (!psoe.ready) return podemos.ready ? 'podemos' : '';
    if (!podemos.ready) return 'psoe';
    var seats = Number(Q[podemos.party + '_congreso_s']) - Number(Q.psoe_congreso_s);
    if (seats !== 0) return seats > 0 ? 'podemos' : 'psoe';
    var votes = (Number(Q[podemos.party + '_congreso_pv']) || 0) - (Number(Q.psoe_congreso_pv) || 0);
    return votes > 0 ? 'podemos' : 'psoe';
  }

  var api = {
    spanishCensurePending: spanishCensurePending,
    spanishCensureOffer: spanishCensureOffer,
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
