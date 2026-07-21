/**
 * cat_coalitions.js
 * Route to Ítaca — Coalition visualisation helper
 *
 * ── Usage ────────────────────────────────────────────────────────────────────
 *
 *   The config is a Q object, not a browser global — content computes it,
 *   the widget protocol's marker names the Q key, out/html/widgets.js reads
 *   it (see docs/design/desk_ui_plan.md §6 "the widget protocol").
 *
 *   on-arrival:
 *   {!
 *     Q.parlament_coalition_config = {
 *       totalSeats:     135,
 *       majoritySeats:  Q.parlament_s_majority || 68,
 *       majorityMarkAt: 0.6,       // 0–1, where the majority line sits in the bar
 *       seatsKey:       "{party}_parlament_s",
 *       partyAliases: {
 *         "ciu_alias": Q.parlament_current_ciu,
 *         "icv_alias": Q.parlament_current_icv
 *       },
 *       coalitions: [
 *         {
 *           label:    "Sovereignty Government",
 *           seatsVar: "parlament_coalition_ciu_erc_cup_s",
 *           members: [
 *             { party: "ciu_alias", type: "government", label: "CiU" },
 *             { party: "erc",       type: "government", label: "ERC" },
 *             { party: "cup",       type: "abstain",    label: "CUP" }
 *           ]
 *         }
 *       ]
 *     };
 *   !}
 *
 *   scene body:
 *   {!<div id="parlament-coalition-widget" data-widget="coalitions"
 *        data-props='{"configFrom":"parlament_coalition_config"}'></div>!}
 *
 *   out/html/widgets.js dispatches "coalitions" to:
 *   initCatCoalitions(el.id, Q[props.configFrom], Q);
 *
 * ── Member types ─────────────────────────────────────────────────────────────
 *   "government"  solid bar — party IS in government
 *   "support"     solid bar, 60% opacity — gives assent/yes but outside govt
 *   "abstain"     striped bar — tolerates; prevents a blocking no
 *   Default: "government"
 *
 * ── Aggregate "bucket" members ───────────────────────────────────────────────
 *   A member may stand for a whole bloc of minor parties rather than one party.
 *   Give it an explicit `seats` (the bloc total) and `color` (a CSS colour, e.g.
 *   "#b5485d" or "var(--erc)") so it can paint without a per-party CSS var; the
 *   `label` is the bloc name. Used by the Congreso widget, where individual
 *   kingmaker minors are bucketed (plurinational left / pact-pragmatists / …).
 *
 * ── Visibility ───────────────────────────────────────────────────────────────
 *   Every entry names a `seatsVar`: a precomputed coalition-total quality (e.g.
 *   "parlament_coalition_ciu_erc_s"). The row renders iff
 *     Q[seatsVar] > 0
 *   For a coalition that also has a clickable option, that is the SAME variable
 *   the option scene reads in its `view-if`, so bar and option can never
 *   disagree. The total is 0 when the coalition is structurally inapplicable (a
 *   party absent, the wrong who-leads ordering, or — for display-only landscape
 *   bars — the scene's own near-miss gate). The members list is then used ONLY
 *   to draw the coloured per-party segments + count.
 *
 *   There is deliberately no `condition`/`alwaysShow`/`nearMissThreshold` here.
 *   Those were the old escape hatch for landscape bars with no option, and they
 *   took `function(s){…}` closures in Q — which JSON.stringify drops on save, so
 *   a game resumed on a coalition scene silently lost every such bar. Both
 *   scenes now precompute a named quality instead (see parlament_coalition's
 *   `barSeats`), which keeps Q data-only as the widget protocol requires and is
 *   what lets the Desk render this at all.
 */

(function (global) {
  "use strict";

  // ── helpers ────────────────────────────────────────────────────────────────

  function getPartyColor(party) {
    var val = getComputedStyle(document.documentElement)
      .getPropertyValue("--" + party)
      .trim();
    return val || "#888";
  }

  function makeSeatsResolver(config, Q) {
    var aliases = config.partyAliases || {};
    var template = config.seatsKey || "{party}_s";
    return function (partyKey) {
      var realKey = aliases[partyKey] || partyKey;
      var qKey = template.replace("{party}", realKey);
      return Q && Q[qKey] ? +Q[qKey] : 0;
    };
  }

  // ── styles (injected once) ─────────────────────────────────────────────────

  var _stylesInjected = false;
  function injectStyles() {
    if (_stylesInjected) return;
    _stylesInjected = true;
    var el = document.createElement("style");
    el.textContent = [
      ".cv-wrap{width:100%}",

      /* one row = label line + bar line stacked */
      ".cv-entry{margin-bottom:.9em}",

      /* label row: [ name + members ............ ] [count]
         the name+members block (cv-label-main) takes the remaining width and
         WRAPS onto as many lines as it needs; the count stays pinned top-right. */
      ".cv-label-row{display:flex;align-items:baseline;gap:.5em;margin-bottom:.3em}",
      ".cv-label-main{flex:1 1 auto;min-width:0;line-height:1.5}",
      ".cv-entry-name{color:var(--text-color,#ddd);font-weight:bold;margin-right:.35em}",
      ".cv-member{white-space:nowrap}" /* keep e.g. \"EAJ-PNV (abst.)\" intact; only the \" + \" joins break */,
      ".cv-entry-count{flex:0 0 auto;white-space:nowrap;align-self:baseline}",
      ".cv-count-ok{color:var(--success-color,#4caf50)}",
      ".cv-count-warn{color:var(--warning-color,#ff9800)}",
      ".cv-count-info{color:var(--info-color,#2196f3)}",
      ".cv-count-muted{color:var(--muted-color,#888)}",

      /* bar track */
      ".cv-track{position:relative;height:2em;border-radius:3px;",
      "background:var(--bar-track-bg,rgba(128,128,128,.15));overflow:visible}",

      /* segments sit inside a flex fill div */
      ".cv-fill{position:absolute;left:0;top:0;height:100%;width:100%;display:flex;",
      "border-radius:3px;overflow:hidden}",
      ".cv-seg{height:100%;flex-shrink:0;min-width:1px; display:flex;align-items:center;justify-content:center;color:#fff;font-weight:bold;}",
      ".cv-seg-support{opacity:.6}",
      ".cv-seg-abstain{position:relative}",
      ".cv-seg-abstain::after{content:'';position:absolute;inset:0;",
      "background:repeating-linear-gradient(-45deg,",
      "transparent 0,transparent 3px,rgba(0,0,0,.3) 3px,rgba(0,0,0,.3) 6px)}",

      /* majority line */
      ".cv-line{position:absolute;top:-3px;bottom:-3px;width:2px;",
      "background:var(--text-color,#ccc);z-index:2}",
      ".cv-line-label{position:absolute;top:-1.5em;left:50%;transform:translateX(-50%);",
      "font-size:.8em;white-space:nowrap;color:var(--muted-color,#888)}",
    ].join("");
    document.head.appendChild(el);
  }

  // ── render one coalition entry ─────────────────────────────────────────────

  function renderEntry(def, config, seats, Q) {
    // ── visibility gate (single source of truth) ────────────────────────────
    // Visibility is driven entirely by the precomputed coalition-total quality
    // the entry names. For a coalition with a clickable option that is the SAME
    // variable the option scene reads in its `view-if`, so the two can never
    // disagree; for a display-only landscape bar the scene has already folded
    // its own condition and near-miss gate into the total. Either way `> 0`
    // means "this coalition is on the table", exactly mirroring `view-if`
    // (which shows the option — greyed — regardless of distance to majority).
    if (!(Q && +Q[def.seatsVar] > 0)) return null;

    var majority = +config.majoritySeats;
    var markAt = config.majorityMarkAt != null ? +config.majorityMarkAt : 0.55;
    var aliases = config.partyAliases || {};

    // resolve members
    var members = (def.members || []).map(function (m) {
      var realKey = aliases[m.party] || m.party;
      return {
        party: realKey,
        label: m.label != null ? m.label : realKey,
        seats: m.seats != null ? +m.seats : seats(m.party),
        type: m.type || "government",
        // optional explicit segment colour — lets an aggregate "bucket" member
        // (e.g. a whole bloc of minor parties) paint without a per-party CSS var.
        color: m.color != null ? m.color : null,
      };
    });

    // Consistent reading order: government → support → abstain. The scene emits
    // courted partners in court order (an abstaining tolerator can precede a later
    // active supporter), but supports should always read/paint before abstains.
    // Stable sort (ES2019+) keeps the original order within each type.
    var TYPE_RANK = { government: 0, support: 1, abstain: 2 };
    members.sort(function (a, b) {
      return (TYPE_RANK[a.type] || 0) - (TYPE_RANK[b.type] || 0);
    });

    // tally
    var govSeats = 0,
      supportSeats = 0,
      abstainSeats = 0;
    members.forEach(function (m) {
      if (m.type === "government") govSeats += m.seats;
      if (m.type === "support") supportSeats += m.seats;
      if (m.type === "abstain") abstainSeats += m.seats;
    });
    var yesSeats = govSeats + supportSeats;
    var effectiveSeats = Math.floor(yesSeats + abstainSeats / 2);

    // status
    var countClass, countText;
    if (govSeats >= majority) {
      countClass = "cv-count-ok";
      countText = `<strong> ${govSeats} </strong> / ${majority}`;
    } else if (yesSeats >= majority) {
      countClass = "cv-count-ok";
      countText = `<strong> ${yesSeats} </strong> / ${majority}`;
    } else if (abstainSeats > 0 && effectiveSeats >= majority) {
      countClass = "cv-count-info";
      countText = `${yesSeats} / ${majority} (tolerated)`;
    } else {
      countClass = "cv-count-muted";
      countText = `${yesSeats} / <strong>${majority}</strong>`;
    }

    // ── DOM construction ───────────────────────────────────────────────────

    var entry = document.createElement("div");
    entry.className = "cv-entry";

    // label row
    var labelRow = document.createElement("div");
    labelRow.className = "cv-label-row";

    // name + members share one flex column that wraps across as many lines as it
    // needs; each member token is kept whole (so "EAJ-PNV (abst.)" never splits)
    // while the " + " joins stay breakable.
    var main = document.createElement("div");
    main.className = "cv-label-main";

    var name = document.createElement("span");
    name.className = "cv-entry-name";
    name.innerHTML = window.applyWholesome(def.label) + ":" || "";

    var memberSpan = document.createElement("span");
    memberSpan.className = "cv-entry-members";
    memberSpan.innerHTML = members
      .filter(function (m) {
        return m.seats > 0;
      })
      .map(function (m) {
        var suffix =
          m.type === "support"
            ? " <span style='opacity:.6'>(support)</span>"
            : m.type === "abstain"
              ? " <span style='opacity:.6'>(abst.)</span>"
              : "";
        var partyLabel;
        if (
          config.seatsKey.includes("congreso") &&
          (m.label == "cs" || m.label == "Cs")
        ) {
          partyLabel = "csspa";
        } else if (config.seatsKey.includes("congreso") && m.label == "up") {
          partyLabel = "UP";
        } else {
          partyLabel = m.label;
        }
        return (
          "<span class='cv-member'>" +
          window.applyWholesome(partyLabel) +
          suffix +
          "</span>"
        );
      })
      .join(" + ");

    var countSpan = document.createElement("span");
    countSpan.className = "cv-entry-count " + countClass;
    countSpan.innerHTML = countText;

    main.appendChild(name);
    main.appendChild(memberSpan);
    labelRow.appendChild(main);
    labelRow.appendChild(countSpan);

    // bar
    // Scale: `majority` seats maps to `markAt` fraction of bar width.
    // Segments beyond markAt are allowed to overflow up to 100%.
    var scale = markAt / majority; // track-fraction per seat

    var track = document.createElement("div");
    track.className = "cv-track";

    var fill = document.createElement("div");
    fill.className = "cv-fill";

    var runPct = 0;
    members.forEach(function (m) {
      if (m.seats <= 0) return;
      var pct = Math.min(m.seats * scale * 100, 100 - runPct);
      runPct += pct;
      var seg = document.createElement("div");
      seg.className =
        "cv-seg" +
        (m.type === "support" ? " cv-seg-support" : "") +
        (m.type === "abstain" ? " cv-seg-abstain" : "");
      seg.style.width = pct.toFixed(3) + "%";
      var segColor = m.color || getPartyColor(m.party);
      seg.style.background = segColor;
      // abstain stripe needs the color for the ::after pseudo-element too
      if (m.type === "abstain") {
        seg.style.setProperty("--seg-color", segColor);
      }
      seg.title = m.label + ": " + m.seats + " seats";
      seg.textContent = m.seats;
      fill.appendChild(seg);
    });

    // majority line
    var line = document.createElement("div");
    line.className = "cv-line";
    line.style.left = (markAt * 100).toFixed(3) + "%";

    // Label to display the needed majority seats. Currently disabled.
    // var lineLabel = document.createElement("span");
    // lineLabel.className = "cv-line-label";
    // lineLabel.textContent = majority;

    // line.appendChild(lineLabel);
    track.appendChild(fill);
    track.appendChild(line);

    entry.appendChild(labelRow);
    entry.appendChild(track);
    return entry;
  }

  // ── main render ────────────────────────────────────────────────────────────

  function render(config, Q) {
    if (!config) return document.createDocumentFragment();
    injectStyles();
    var seats = makeSeatsResolver(config, Q);
    var wrap = document.createElement("div");
    wrap.className = "cv-wrap";
    (config.coalitions || []).forEach(function (def) {
      var el = renderEntry(def, config, seats, Q);
      if (el) wrap.appendChild(el);
    });
    return wrap;
  }

  function initCatCoalitions(target, config, Q) {
    var el =
      typeof target === "string" ? document.getElementById(target) : target;
    if (!el || !config) return;
    el.innerHTML = "";
    el.appendChild(render(config, Q));
  }

  global.CatCoalitionViz = { render: render };
  global.initCatCoalitions = initCatCoalitions;
})(typeof window !== "undefined" ? window : this);
