/* =============================================================================
 * source/lib/brief.js — the Brief's row derivations.
 * =============================================================================
 * PURE. Every function is `Q -> rows[]` and writes NOTHING. That is the whole
 * point: getExportableState() returns state wholesale, so anything assigned to
 * Q is serialized into every save AND restored stale. These are views, not
 * state — they are rebuilt on every render instead.
 *
 * Reached from content as a widget marker, never called by content directly:
 *     <div data-widget="roster-rows" data-props='{"deriveFrom":"benches"}'></div>
 *
 * ROW CONTRACT (spec §4.1): every row carries its SCALAR SUMMARY as first-class
 * fields — `label`, `value`, `stamp`, `subtitle` — and any series/geometry as
 * ADDITIVE fields. The Desk draws the rich shape; the old shell's generic
 * fallback renders label + value and ignores the rest. Do not invert this.
 *
 * A row never carries a classified band; it carries the RAW VALUE plus the
 * id of the qdisplay (source/qdisplays/*.dry) that classifies it. This file
 * only ever gets `Q` — it cannot reach the engine, so it must not compute a
 * band itself. A `stamp`-bearing field is always paired with a sibling
 * `<field>Display` naming the qdisplay; the renderer calls
 * adapter.qdisplay(value, qdisplayId) to band it. This keeps every threshold
 * written in exactly one place instead of duplicated into JS wherever a row
 * needs one — follow this for every row type this file adds later.
 *
 * Labels are English and live in the LABELS tables below, resolved at render
 * time through engine.translate() (English-as-key, same catalog as scene
 * prose). Keep every display string in a declared table — the deferred i18n
 * extraction tool has to find them.
 * ========================================================================== */
(function () {
  'use strict';

  // Shared party walk: the visible set and its order, computed ONCE so every
  // party-shaped row set agrees. Seats desc, ties by Q.parties order.
  function seatedParties(Q) {
    var parties = Q.parties || [];
    var out = [];
    for (var i = 0; i < parties.length; i++) {
      var p = parties[i];
      var seats = Q[p + '_parlament_s'] || 0;
      if (seats > 0) out.push({ id: p, seats: seats, order: i });
    }
    out.sort(function (a, b) {
      return b.seats - a.seats || a.order - b.order;
    });
    return out;
  }

  function majorityOf(Q) {
    return Math.floor((Q.parlament_size || 0) / 2) + 1;
  }

  // Display name per party id. The glossary supplies colour + logo; it does NOT
  // supply a short chamber label, so this table does.
  var PARTY_LABELS = {
    ciu: 'CiU', erc: 'ERC', psc: 'PSC', ppc: 'PPC', icv: 'ICV-EUiA',
    cs: 'Cs', cup: 'CUP', si: 'SI', cdc: 'CDC', unio: 'UDC', dl: 'DL',
    jxsi: 'JxSí', junts: 'Junts', jxcat: 'JxCat', pdcat: 'PDeCAT',
    csqp: 'CSQP', cecp: 'CECP', ecp: 'ECP', vox: 'VOX', fnc: 'FNC', pxc: 'PxC'
  };

  function partyLabel(id) {
    return PARTY_LABELS[id] || id.toUpperCase();
  }

  function benches(Q) {
    var seated = seatedParties(Q);
    var rows = [];
    for (var i = 0; i < seated.length; i++) {
      var id = seated[i].id;
      var isPlayer = id === Q.player_party;
      var leader = Q[id + '_leader'] || '';
      var ideology = Q[id + '_ideology'] || '';
      rows.push({
        id: id,
        label: partyLabel(id),
        value: seated[i].seats,
        // The raw relation value (0-100 scale, see root.scene.dry's
        // `Q[party]_relations` assignments) — NOT a band word. Null for your
        // own bench (you have no relationship with yourself — the design
        // stamps it YOU instead) and null when the game hasn't assigned a
        // relation yet (undefined), uniformly: null always means "no stamp
        // to draw", never a third "unset" state. `stampDisplay` names the
        // qdisplay that turns this number into a band; see the ROW CONTRACT
        // note above the file header for why brief.js doesn't do it itself.
        stamp: (isPlayer || Q[id + '_relations'] === undefined)
          ? null : Q[id + '_relations'],
        stampDisplay: 'relationships',
        subtitle: leader && ideology ? leader + ' — ' + ideology : String(leader || ''),
        isPlayer: isPlayer
      });
    }
    return rows;
  }

  function composition(Q) {
    var seated = seatedParties(Q);
    var max = 0;
    for (var i = 0; i < seated.length; i++) {
      if (seated[i].seats > max) max = seated[i].seats;
    }
    var majority = majorityOf(Q);
    var rows = [];
    for (var j = 0; j < seated.length; j++) {
      rows.push({
        id: seated[j].id,
        label: partyLabel(seated[j].id),
        value: seated[j].seats,
        // Relative to the LARGEST party, reproducing today's bar scaling
        // (status.scene.dry used Q.parlament_max_seats for exactly this).
        share: max > 0 ? seated[j].seats / max : 0,
        majority: majority
      });
    }
    return rows;
  }

  // -- The Govern roster -------------------------------------------------
  // id -> [label, Q name key, Q party key]. Order is the design's
  // (brief-frames.md §2 frame 13) and matches today's status.government.
  var CABINET = [
    ['president',     'President',          'president',          'president_party'],
    ['vicepresident', 'Vice-president',     'vicepresident',      'vicepresident_party'],
    ['economy',       'Economy',            'economy_minister',   'economy_minister_party'],
    ['health',        'Health',             'health_minister',    'health_minister_party'],
    ['education',     'Education',          'education_minister', 'education_minister_party'],
    ['interior',      'Interior',           'interior_minister',  'interior_minister_party'],
    ['foreign',       'Foreign affairs',    'foreign_minister',   'foreign_minister_party'],
    ['bnl',           'Business + labour',  'bnl_minister',       'bnl_minister_party'],
    ['territory',     'Territory',          'territory_minister', 'territory_minister_party']
  ];

  function cabinet(Q) {
    var rows = [];
    for (var i = 0; i < CABINET.length; i++) {
      var e = CABINET[i];
      rows.push({
        id: e[0],
        label: e[1],
        value: String(Q[e[2]] === undefined ? '' : Q[e[2]]),
        // A minister's name is not a scale — nothing classifies it.
        valueDisplay: null,
        party: Q[e[3]] === undefined ? null : String(Q[e[3]])
      });
    }
    return rows;
  }

  // -- State control -----------------------------------------------------
  // The Q value IS the rung (0..4), so a pip renderer needs nothing else; the
  // WORD comes from control.qdisplay.dry via the row's valueDisplay. No band
  // table here — see the ROW CONTRACT rule in this file's header.
  var CONTROL = [
    ['airports',       'Airports',       'airports_control'],
    ['railways',       'Railways',       'railways_control'],
    ['ports',          'Ports',          'ports_control'],
    ['borders',        'Borders',        'borders_control'],
    ['security',       'Security',       'security_control'],
    ['communications', 'Communications', 'communications_control'],
    ['taxation',       'Taxation',       'tax_control']
  ];

  function control(Q) {
    var rows = [];
    for (var i = 0; i < CONTROL.length; i++) {
      var e = CONTROL[i];
      rows.push({
        id: e[0],
        label: e[1],
        value: Q[e[2]] || 0,
        valueDisplay: 'control',
        party: null
      });
    }
    return rows;
  }

  // -- The chancelleries -------------------------------------------------
  var CHANCELLERIES = [
    ['eu',     'European Union', 'international_eu_opinion',     'img/flags/eu.svg'],
    ['usa',    'United States',  'international_usa_opinion',    'img/flags/usa.svg'],
    ['russia', 'Russia',         'international_russia_opinion', 'img/flags/russia.svg'],
    ['china',  'P.R. China',     'international_china_opinion',  'img/flags/prc.svg']
  ];

  function chancelleries(Q) {
    var rows = [];
    for (var i = 0; i < CHANCELLERIES.length; i++) {
      var e = CHANCELLERIES[i];
      var v = Q[e[2]];
      var n = typeof v === 'number' ? v : 0;
      if (n < -1) n = -1;
      if (n > 3) n = 3;
      rows.push({
        id: e[0],
        label: e[1],
        value: n,
        // international_opinion.qdisplay.dry turns this into the stance line.
        // No band table here — see the ROW CONTRACT rule.
        valueDisplay: 'international_opinion',
        party: null,
        flag: e[3]
      });
    }
    return rows;
  }

  var api = {
    brief: {
      benches: benches,
      composition: composition,
      cabinet: cabinet,
      control: control,
      chancelleries: chancelleries
    },
    // Exported for unit tests and for the sheets that need the same ordering.
    _seatedParties: seatedParties
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    window.RTI_BRIEF = api;
  }
})();
