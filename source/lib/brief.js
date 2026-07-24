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
  // The Q value IS the rung (0..4); the visible WORD comes from
  // control.qdisplay.dry via the row's valueDisplay. No band
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

  // -- Party factions ----------------------------------------------------
  var FACTIONS = {
    erc: [
      ['left',      'Left',             'erc_left'],
      ['core',      'Republican Core',  'erc_core'],
      ['pragmatic', 'Pragmatics',       'erc_pragmatic'],
      ['civic',     'Civil Orgs.',      'erc_civic']
    ],
    cup: [
      ['endavant',    'Endavant',     'cup_endavant'],
      ['poblelliure', 'Poble Lliure', 'cup_poblelliure'],
      ['thirdsector', 'Third Sector', 'cup_thirdsector']
    ]
  };

  function factions(Q) {
    var set = FACTIONS[Q.player_party] || [];
    var rows = [];
    for (var i = 0; i < set.length; i++) {
      rows.push({
        id: set[i][0],
        label: set[i][1],
        strength: Q[set[i][2] + '_strength'] || 0,
        dissent: Q[set[i][2] + '_dissent'] || 0,
        // Both scales are classified by the SAME qdisplay today — the legacy
        // content renders each as `[+ erc_left_strength : dissent +]`. Named
        // separately anyway so one can move without touching the other.
        strengthDisplay: 'dissent',
        dissentDisplay: 'dissent'
      });
    }
    return rows;
  }

  // -- The street --------------------------------------------------------
  // All three are 0..100 scales, so `share` is just value/100. Each has its
  // OWN qdisplay, which is what supplies the word — no local ladder here (see
  // the ROW CONTRACT rule). [id, label, Q key, qdisplay id]
  var STREET = [
    ['social_dissent', 'Social dissent',  'social_dissent',        'social_dissent'],
    ['independence',   'Indep. momentum', 'independence_movement', 'independence_movement'],
    ['trust',          'Trust in gov.',   'independence_trust',    'politics_trust']
  ];

  function street(Q) {
    var rows = [];
    for (var i = 0; i < STREET.length; i++) {
      var v = Q[STREET[i][2]];
      var n = typeof v === 'number' ? v : 0;
      if (n < 0) n = 0;
      if (n > 100) n = 100;
      rows.push({
        id: STREET[i][0],
        label: STREET[i][1],
        value: n,
        valueDisplay: STREET[i][3],
        share: n / 100
      });
    }
    return rows;
  }

  // -- Economy trails ----------------------------------------------------
  // [id, label, record field, unit, risingIsGood]
  var TRAILS = [
    ['gdp',          'GDP growth',           'gdp_growth',          '%', true],
    ['unemployment', 'Unemployment',         'unemployment',        '%', false],
    ['surplus',      'Generalitat surplus',  'generalitat_surplus', '%', true]
  ];

  function seriesFrom(records, field) {
    var out = [];
    if (!records || !records.length) return out;
    for (var i = 0; i < records.length; i++) {
      var v = records[i] && records[i][field];
      if (typeof v === 'number') out.push(v);
    }
    return out;
  }

  function trails(Q) {
    var records = Q.economic_records || [];
    var rows = [];
    for (var i = 0; i < TRAILS.length; i++) {
      var e = TRAILS[i];
      var series = seriesFrom(records, e[2]);
      // Current value: prefer the live quality, fall back to the series tail —
      // at game start the record array can still be empty.
      var live = Q[e[2]];
      var current = typeof live === 'number'
        ? live
        : (series.length ? series[series.length - 1] : 0);
      // Records are written after a turn resolves, so their tail can lag the
      // live scalar currently printed in the row. Add that scalar to the VIEW
      // when needed; do not mutate Q or duplicate a matching recorded point.
      if (typeof live === 'number' &&
          (!series.length || series[series.length - 1] !== live)) {
        series.push(live);
      }
      // Direction from the SERIES, never from Q.<x>_change: those are already
      // rendered HTML (<img src="img/arrowup.png">, root.scene.dry:96-101).
      var dir = 'flat';
      if (series.length >= 2) {
        var delta = series[series.length - 1] - series[series.length - 2];
        dir = delta > 0 ? 'up' : (delta < 0 ? 'down' : 'flat');
      }
      rows.push({
        id: e[0],
        label: e[1],
        value: Math.round(current * 10) / 10,
        unit: e[3],
        dir: dir,
        // With no movement yet, colour the CURRENT scalar rather than
        // declaring every start-of-run value good. Otherwise a fresh game
        // inks -3.1% GDP, 22.5% unemployment and -2.3% surplus all green.
        // For the two "higher is better" metrics, non-negative is healthy;
        // for unemployment ("lower is better"), only non-positive is.
        good: dir === 'flat' ? (e[4] ? current >= 0 : current <= 0)
          : ((dir === 'up') === e[4]),
        series: series
      });
    }
    return rows;
  }

  // -- The player's own chamber standing ---------------------------------
  // Replaces the four mutually-exclusive hand-styled bars the old OVERVIEW
  // carried, one per coalition arrangement (erc alone / erc-or-cup in JxSí /
  // erc-or-cup in JxCat / cup alone). Unlike `composition`, which is every
  // party's seats, this is only YOURS — the first tab's at-a-glance answer to
  // "how big am I".
  //
  // Precedence is JxSí, then JxCat, then your own party. The old content's
  // last branch tested only `player_party == "cup"` with NO coalition guard,
  // so a CUP player inside JxSí matched two branches and rendered TWO bars.
  // Returning a single row fixes that incidentally.
  //
  // Coverage is EVERY player party, not just the two the old content handled:
  // its four branches covered erc and cup only, so a CiU player (root.scene.dry
  // assigns player_party = "ciu") simply had no bar. Falling through to the
  // player's own id fixes that and needs no new rule — the coalition branches
  // above are erc/cup-specific because `*_in_jxsi`/`*_in_jxcat` only exist for
  // those two; there are no `ciu_in_*` flags in content.
  function standing(Q) {
    var player = Q.player_party;
    if (!player) { return []; }
    var id;
    if ((player === 'erc' && Q.erc_in_jxsi) || (player === 'cup' && Q.cup_in_jxsi)) {
      id = 'jxsi';
    } else if ((player === 'erc' && Q.erc_in_jxcat) || (player === 'cup' && Q.cup_in_jxcat)) {
      id = 'jxcat';
    } else {
      id = player;
    }
    var total = Q.parlament_size || 0;
    var value = Q[id + '_parlament_s'] || 0;
    return [{
      id: id,
      label: partyLabel(id),
      value: value,
      total: total,
      // A seat count is not a classified scale — no qdisplay bands it.
      valueDisplay: null,
      share: total > 0 ? value / total : 0
    }];
  }

  // -- Polls ---------------------------------------------------------------
  // The old renderer's actual Q schema, centralized here so the Desk never
  // grows a second interpretation of polling arithmetic.
  var PROVINCES = ['barcelona', 'tarragona', 'lleida', 'girona'];
  var PROVINCE_LABELS = {
    barcelona: 'Barcelona', tarragona: 'Tarragona',
    lleida: 'Lleida', girona: 'Girona'
  };
  var DEMOGRAPHIC_LABELS = {
    buss: 'Business', ind: 'Ind. workers', middle: 'Middle class',
    young: 'Young', retired: 'Retired', rural: 'Rural',
    unemployed: 'Unemployed'
  };

  function pollParties(Q) {
    var parties = Q.parties || [];
    var out = [];
    for (var i = 0; i < parties.length; i++) {
      if (parties[i] !== 'abstain') out.push(parties[i]);
    }
    return out;
  }

  function provinceVotes(Q, province) {
    var demos = Q.parlament_demographics || [];
    var parties = pollParties(Q);
    var totals = {};
    var population = 0;
    for (var i = 0; i < demos.length; i++) {
      var pop = Number(Q['parlament_' + province + '_' + demos[i] + '_pop']) || 0;
      population += pop;
      for (var j = 0; j < parties.length; j++) {
        var support = Number(Q[
          parties[j] + '_parlament_' + province + '_' + demos[i] + '_support'
        ]) || 0;
        totals[parties[j]] = (totals[parties[j]] || 0) + support * pop;
      }
    }
    return { totals: totals, population: population };
  }

  function provinces(Q) {
    var rows = [];
    for (var i = 0; i < PROVINCES.length; i++) {
      var id = PROVINCES[i];
      var votes = provinceVotes(Q, id);
      var winner = '';
      var best = 0;
      Object.keys(votes.totals).forEach(function (party) {
        if (votes.totals[party] > best) {
          winner = party;
          best = votes.totals[party];
        }
      });
      rows.push({
        id: id,
        label: PROVINCE_LABELS[id],
        value: winner,
        party: winner || null,
        population: votes.population,
        seats: (Q.parlament_seats && Q.parlament_seats[id]) || 0
      });
    }
    return rows;
  }

  function crosstab(Q) {
    var rows = [];
    var demos = Q.parlament_demographics || [];
    var parties = pollParties(Q);
    for (var p = 0; p < PROVINCES.length; p++) {
      var province = PROVINCES[p];
      for (var d = 0; d < demos.length; d++) {
        var demo = demos[d];
        var population = Number(Q['parlament_' + province + '_' + demo + '_pop']) || 0;
        if (population <= 0) continue;
        var cells = [];
        for (var i = 0; i < parties.length; i++) {
          var value = Number(Q[
            parties[i] + '_parlament_' + province + '_' + demo + '_support'
          ]) || 0;
          if (value > 0) {
            cells.push({ id: parties[i], label: partyLabel(parties[i]), value: value });
          }
        }
        rows.push({
          id: province + '_' + demo,
          label: DEMOGRAPHIC_LABELS[demo] || demo,
          value: population,
          province: province,
          cells: cells,
          playerParty: Q.player_party || null
        });
      }
    }
    return rows;
  }

  function seatProjection(Q) {
    var rows = [];
    var parties = pollParties(Q);
    for (var p = 0; p < PROVINCES.length; p++) {
      var province = PROVINCES[p];
      var votes = provinceVotes(Q, province).totals;
      var totalVotes = Object.keys(votes).reduce(function (sum, party) {
        return sum + votes[party];
      }, 0);
      var eligible = parties.filter(function (party) {
        return totalVotes > 0 && (votes[party] || 0) / totalVotes >= 0.03;
      });
      var allocation = {};
      var seats = (Q.parlament_seats && Q.parlament_seats[province]) || 0;
      for (var s = 0; s < seats; s++) {
        var winner = '';
        var quotient = 0;
        for (var i = 0; i < eligible.length; i++) {
          var party = eligible[i];
          var q = (votes[party] || 0) / ((allocation[party] || 0) + 1);
          if (q > quotient) {
            quotient = q;
            winner = party;
          }
        }
        if (!winner) break;
        allocation[winner] = (allocation[winner] || 0) + 1;
      }
      var maxSeats = Object.keys(allocation).reduce(function (max, party) {
        return Math.max(max, allocation[party]);
      }, 0);
      eligible.forEach(function (party) {
        if (!allocation[party]) return;
        rows.push({
          id: province + '_' + party,
          label: partyLabel(party),
          value: allocation[party],
          province: province,
          party: party,
          share: maxSeats > 0 ? allocation[party] / maxSeats : 0
        });
      });
    }
    return rows;
  }

  var api = {
    brief: {
      benches: benches,
      composition: composition,
      standing: standing,
      cabinet: cabinet,
      control: control,
      chancelleries: chancelleries,
      factions: factions,
      street: street,
      trails: trails,
      crosstab: crosstab,
      seatProjection: seatProjection,
      provinces: provinces
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
