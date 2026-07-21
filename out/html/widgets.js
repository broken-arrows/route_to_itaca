/**
 * widgets.js — the OLD (jQuery) shell's widget host.
 *
 * Content declares `<div id="LEGACY_ID" data-widget="NAME" data-props='JSON'>`
 * (the widget protocol — docs/design/desk_ui_plan.md §6 / the widget-protocol
 * spec). `mountWidgets(root, Q)` scans `root` for `[data-widget]` and
 * dispatches to this shell's EXISTING d3 renderers by name — the renderers
 * themselves (cat_polls.js, cat_coalitions.js, cat_maps.js) are UNCHANGED.
 *
 * This is also the single call `game.js` now makes instead of the five
 * duplicated call-sequences (initCataloniaPolls / initCatLocalMap /
 * initCongresoMap / initCatCoalitions / initCongresoPartyTour) that used to
 * be pasted, unconditionally, into onNewPage / updateSidebar / changeTab /
 * onDisplayContent / onload — each `init*` early-returns via
 * `getElementById` when its target div isn't on the current page, so calling
 * all of them on every page was always a bunch of silent no-ops; that part
 * of the shape is preserved here, just centralised.
 *
 * A few widget divs predate the protocol and are not yet dual-marked with
 * `data-widget` (out of Task 6's scope — see
 * .superpowers/sdd/p25-task-6-report.md): the wide poll map
 * (#cat-polls-widget-wide), the local results map (#catalonia-local-map),
 * the Congreso map (#congreso-map-widget) and the Congreso party tour
 * (#congreso-party-tour-widget). They are still mounted here, by their
 * legacy id, so this file remains the ONE place old-shell pages wire up a
 * visualisation — consolidating what used to be scattered across game.js —
 * without silently dropping functionality this task didn't ask it to change.
 */
(function () {
  "use strict";

  function readDataProps(el) {
    var raw = el.getAttribute("data-props");
    if (!raw) return {};
    try {
      return JSON.parse(raw);
    } catch (err) {
      console.warn(
        'widget "' + (el.id || el.getAttribute("data-widget")) + '": invalid data-props JSON',
        err,
      );
      return {};
    }
  }

  // Dispatch table for names declared via `data-widget`. Each handler owns
  // reading whatever data-props it needs; the shell's renderers themselves
  // are called exactly as they always were (by id + Q [+ config]).
  var HANDLERS = {
    hemicycle: function (el, Q) {
      // Phase 2.5 Task 7: the seat-arc rendering call itself — the second
      // half of what used to be parlament_card.scene.dry's on-display block
      // (docs/design/LEARNINGS.md, 2026-07-17). Content now only computes
      // the view-model (Q[configFrom] = {seats, majority}, data-only, no
      // function closures); this is the ONLY place left that still calls
      // d3.parliament — that plugin (d3-parliament.js) is unchanged.
      var props = readDataProps(el);
      var config = typeof props.configFrom === "string" ? Q[props.configFrom] : null;
      if (!config || !Array.isArray(config.seats) || !config.seats.length) return;

      // d3.parliament appends <g>/<circle> children via d3's
      // namespace-INHERITING element creator, which only yields real SVG
      // elements when the node it appends into is itself in the SVG
      // namespace. The widget-protocol marker is a plain <div> (matching
      // every other widget marker in this file), so a real <svg> child is
      // created here once — exactly the element the old on-display block's
      // own `<svg id="parlament" style="width: 500px; height: 250px;">`
      // markup used to provide directly. Idempotent: d3.parliament's own
      // enter/update/exit join tolerates being called again on the same
      // <svg>, so re-running mountWidgets() (onNewPage/updateSidebar/
      // changeTab/onDisplayContent/onload, per this file's header) is safe.
      var svg = el.querySelector("svg");
      if (!svg) {
        svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        // No id: the parent div already has id="parlament"; the render call uses
        // this node reference directly, so a duplicate id would be invalid HTML.
        svg.style.width = "500px";
        svg.style.height = "250px";
        el.appendChild(svg);
      }

      // Reconstruct the {id, seats} shape d3.parliament's own internals key
      // off (seatClasses/tooltip lookups read d.party.id — see
      // d3-parliament.js) from the minimal Q view-model. `id` == `party`:
      // content writes both `party` and `colour` from the same token
      // (parlament_card.scene.dry), the same token the .seat.<id> CSS
      // classes in game.css already resolve to a fill colour, so dropping
      // straight to `.id` here reproduces today's colouring unchanged.
      var data = config.seats.map(function (s) {
        return { id: s.party, seats: s.seats };
      });

      // Ported verbatim from the old on-display block, including its
      // harmless no-op `.height()` call (the vendored d3-parliament.js's
      // height setter never assigns its own internal variable — real render
      // height is always width/2, computed inside the plugin itself) and
      // the fact that only the SVG's CSS *height* gets adjusted on narrow
      // screens, never its CSS width. Not fixed here: d3-parliament.js
      // stays byte-unchanged, out of this task's territory.
      var width = 500;
      var height = 500;
      var contentEl = document.getElementById("content");
      var screenWidth = contentEl ? contentEl.offsetWidth : width;
      if (screenWidth < width - 50) {
        width = screenWidth - 50;
        height = width;
        svg.style.height = screenWidth / 2 + "px";
      }
      var parliament = d3.parliament();
      parliament.width(width).height(height).innerRadiusCoef(0.4);
      parliament.enter.fromCenter(true).smallToBig(false);
      parliament.exit.toCenter(false).bigToSmall(false);
      parliament.highlightedParty(Q.player_party);
      d3.select(svg).datum(data).call(parliament);
    },
    "poll-map": function (el, Q) {
      initCataloniaPolls(el.id, Q, el.id === "cat-polls-widget-wide");
    },
    "achievement-gallery": function (el, Q) {
      // Phase 2.5 Task 8: the registry (source/data/achievements.json ->
      // game.json.data.achievements) replaces the 13 hand-written HTML
      // blocks that used to live directly in game_over.scene.dry's
      // @achievements section. `data-props`'s "scope" picks WHICH quality
      // prefix marks a row unlocked: "ever" reads Q.achievement_* (the
      // engine's cross-save set, pre-seeded at boot from localStorage);
      // "playthrough" reads Q.game_achievement_* (reset every new game).
      // Regenerates the SAME markup game.css already styles
      // (.achievement--unlocked/--locked, .achievement-image/-title/-stars/
      // -body/-description) so the gallery is visually identical to before.
      var props = readDataProps(el);
      var scope = props.scope === "playthrough" ? "playthrough" : "ever";
      var prefix = scope === "playthrough" ? "game_achievement_" : "achievement_";
      var reg =
        ((window.dendryUI.game.data || {}).achievements || {}).achievements ||
        [];
      if (!reg.length) return;

      el.innerHTML = reg
        .map(function (a) {
          var state = Q[prefix + a.id] ? "unlocked" : "locked";
          var stars = "";
          for (var i = 0; i < 5; i++) {
            stars +=
              '<span class="' +
              (i < a.stars ? "star--filled" : "star--empty") +
              '">★</span>';
          }
          return (
            '<div class="achievement achievement--' +
            state +
            '" style="display:flex">' +
            '<div class="achievement-image achievement-image--' +
            state +
            '">' +
            '<img src="' +
            a.image +
            '" style="width:100%;height:100%;object-fit:cover;">' +
            "</div>" +
            '<div class="achievement-body">' +
            '<div class="achievement-title achievement-title--' +
            state +
            '">' +
            "<span> " +
            a.name +
            " </span>" +
            '<div class="achievement-stars">' +
            stars +
            "</div>" +
            "</div>" +
            '<div class="achievement-description achievement-description--' +
            state +
            '">' +
            a.description +
            "</div>" +
            "</div>" +
            "</div>"
          );
        })
        .join("");
    },
    coalitions: function (el, Q) {
      // The old channel was a browser global content pushed the coalition
      // view-model through (docs/design/LEARNINGS.md, 2026-07-13). The
      // widget protocol's fix: content writes an ordinary Q object, and the
      // marker names the key via `configFrom` — read it straight off Q.
      var props = readDataProps(el);
      var config = typeof props.configFrom === "string" ? Q[props.configFrom] : null;
      initCatCoalitions(el.id, config, Q);
    },
  };

  // Legacy (not-yet-declared) widget ids — see header comment.
  var LEGACY_IDS = [
    function (Q) {
      initCataloniaPolls("cat-polls-widget-wide", Q, true);
    },
    function (Q) {
      initCatLocalMap("catalonia-local-map", Q);
    },
    function (Q) {
      initCongresoMap("congreso-map-widget", Q);
    },
    function (Q) {
      initCongresoPartyTour("congreso-party-tour-widget", Q);
    },
  ];

  // A widget the shell has no bespoke handler for still gets its data resolved
  // here; Part 2 of phase 3b adds the generic fallback RENDERER that consumes
  // it. Until then an unhandled marker stays an empty div, exactly as before.
  function resolveProps(el, Q) {
    var props = readDataProps(el);
    if (typeof props.configFrom === "string") {
      var cfg = Q[props.configFrom];
      delete props.configFrom;
      if (cfg) {
        for (var k in cfg) {
          if (Object.prototype.hasOwnProperty.call(cfg, k)) props[k] = cfg[k];
        }
      }
    }
    if (typeof props.deriveFrom === "string") {
      var lib = window.RTI_GAME_LIB || {};
      var fn = lib.brief && lib.brief[props.deriveFrom];
      delete props.deriveFrom;
      props.rows = typeof fn === "function" ? fn(Q) : [];
    }
    return props;
  }
  window.resolveWidgetProps = resolveProps;

  window.mountWidgets = function mountWidgets(root, Q) {
    if (!root || !Q) return;
    var marked = root.querySelectorAll("[data-widget]");
    for (var i = 0; i < marked.length; i++) {
      var el = marked[i];
      var handler = HANDLERS[el.getAttribute("data-widget")];
      if (!handler) continue;
      handler(el, Q);
    }
    for (var j = 0; j < LEGACY_IDS.length; j++) {
      LEGACY_IDS[j](Q);
    }
  };
})();
