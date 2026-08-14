/**
 * widgets.js — the OLD (jQuery) shell's widget host.
 *
 * Content declares `<div id="LEGACY_ID" data-widget="NAME" data-props='JSON'>`
 * `mountWidgets(root, Q)` scans `root` for `[data-widget]` and
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
 * `data-widget`: the wide poll map
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
        'widget "' +
          (el.id || el.getAttribute("data-widget")) +
          '": invalid data-props JSON',
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
      var props = readDataProps(el);
      var config =
        typeof props.configFrom === "string" ? Q[props.configFrom] : null;
      if (!config || !Array.isArray(config.seats) || !config.seats.length)
        return;

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
    "chamber-vote": function (el, Q) {
      var props = resolveProps(el, Q);
      var outcomes = Array.isArray(props.outcomes) ? props.outcomes : [];
      outcomes = outcomes.filter(function (outcome) {
        return outcome && Number(outcome.votes) > 0;
      });

      el.innerHTML = "";
      if (!outcomes.length) return;

      var wrap = document.createElement("div");
      wrap.className = "chamber-vote";
      var totalVotes = outcomes.reduce(function (total, outcome) {
        return total + Math.max(0, Number(outcome.votes) || 0);
      }, 0);
      var abstainIndex = outcomes.findIndex(function (outcome) {
        return outcome.kind === "abstain";
      });
      if (totalVotes > 0 && abstainIndex >= 0) {
        var votesBeforeAbstain = outcomes
          .slice(0, abstainIndex)
          .reduce(function (total, outcome) {
            return total + Math.max(0, Number(outcome.votes) || 0);
          }, 0);
        var abstainVotes = Math.max(
          0,
          Number(outcomes[abstainIndex].votes) || 0,
        );
        var abstainCenter =
          (votesBeforeAbstain + abstainVotes / 2) / totalVotes;
        var equalColumnCenter = (abstainIndex + 0.5) / outcomes.length;
        var abstainShift =
          (abstainCenter - equalColumnCenter) * outcomes.length * 100;
        wrap.style.setProperty(
          "--chamber-vote-abstain-shift",
          String(abstainShift) + "%",
        );
      }
      var labels = document.createElement("div");
      labels.className = "chamber-vote__labels";
      var bar = document.createElement("div");
      bar.className = "chamber-vote__bar";
      if (totalVotes > 0) {
        var majorityVotes = Math.floor(totalVotes / 2) + 1;
        bar.style.setProperty(
          "--chamber-vote-majority-left",
          String((majorityVotes / totalVotes) * 100) + "%",
        );
        bar.setAttribute("data-majority", String(majorityVotes));
        bar.title = "Majority: " + majorityVotes + " yes votes";
      }
      var breakdowns = document.createElement("div");
      breakdowns.className = "chamber-vote__breakdowns";
      var hasBreakdowns = false;

      outcomes.forEach(function (outcome) {
        var kind =
          outcome.kind === "yes" ||
          outcome.kind === "abstain" ||
          outcome.kind === "no"
            ? outcome.kind
            : "abstain";
        var votes = Math.max(0, Number(outcome.votes) || 0);
        var label = document.createElement("div");
        label.className = "chamber-vote__label chamber-vote__label--" + kind;
        label.textContent = String(outcome.label || "");
        labels.appendChild(label);

        var segment = document.createElement("div");
        segment.className =
          "chamber-vote__outcome chamber-vote__outcome--" + kind;
        segment.style.flexGrow = String(votes);
        segment.setAttribute(
          "aria-label",
          String(outcome.label || "") + ": " + votes + " votes",
        );
        segment.textContent = String(votes);
        bar.appendChild(segment);

        var breakdown = document.createElement("div");
        breakdown.className =
          "chamber-vote__breakdown chamber-vote__breakdown--" + kind;
        if (Array.isArray(outcome.parties) && outcome.parties.length) {
          hasBreakdowns = true;
          var parties = document.createElement("ul");
          parties.className = "chamber-vote__parties";
          outcome.parties.forEach(function (party) {
            if (!party || !party.label) return;
            var item = document.createElement("li");
            var partyName = String(party.label);
            item.innerHTML =
              typeof window.applyWholesome === "function"
                ? window.applyWholesome(partyName)
                : partyName;
            if (party.count !== undefined && party.count !== null) {
              item.appendChild(
                document.createTextNode(" (" + String(party.count) + ")"),
              );
            }
            parties.appendChild(item);
          });
          breakdown.appendChild(parties);
        }
        breakdowns.appendChild(breakdown);
      });

      wrap.appendChild(labels);
      wrap.appendChild(bar);
      if (hasBreakdowns) wrap.appendChild(breakdowns);
      el.appendChild(wrap);
    },
    "law-grid": function (el, Q) {
      var gameLib =
        window.dendryUI &&
        window.dendryUI.dendryEngine &&
        window.dendryUI.dendryEngine.gameLib;
      var buildRows = gameLib && gameLib.getLawsForUI;
      var laws = typeof buildRows === "function" ? buildRows(Q) : [];

      el.innerHTML = "";
      var visibleLaws = Array.isArray(laws)
        ? laws.filter(function (law) {
            return law && law.status !== "expired" && law.icon;
          })
        : [];
      if (!visibleLaws.length) {
        var empty = document.createElement("p");
        empty.className = "law-grid__empty";
        empty.textContent = "No laws passed yet";
        el.appendChild(empty);
        return;
      }

      var statusLabels = {
        active: "Active",
        repealed:
          "Repealed: this law has been scaled down by the Constitutional Court.",
        disputed:
          "Disputed: the Constitutional Court and the Parlament have varying interpretations of this law's status.",
        struck_down:
          "Struck down: the Constitutional Court has ruled against the totality of this law.",
        imposed:
          "Imposed: this is a top-down law that has not been voted by the Parlament.",
      };
      var grid = document.createElement("div");
      grid.className = "law-grid";

      visibleLaws.forEach(function (law, index) {
        var status = statusLabels[law.status] ? law.status : "repealed";
        var label =
          String(law.title || law.id || "Law") +
          " — " +
          (statusLabels[law.status] || String(law.status || "Unknown"));
        var icon = document.createElement("div");
        icon.className = "law-grid__law law-grid__law--" + status;
        icon.setAttribute("role", "img");
        icon.setAttribute("tabindex", "0");
        icon.setAttribute("aria-label", label);

        var image = document.createElement("img");
        image.src = law.icon;
        image.alt = "";
        icon.appendChild(image);

        var tooltip = document.createElement("span");
        tooltip.className = "law-grid__tooltip";
        tooltip.id = "law-grid-tooltip-" + index;
        tooltip.setAttribute("role", "tooltip");
        var tooltipTitle = document.createElement("span");
        tooltipTitle.className = "law-grid__tooltip-title";
        tooltipTitle.textContent = String(law.title || law.id || "Law");
        var tooltipStatus = document.createElement("span");
        tooltipStatus.className = "law-grid__tooltip-status";
        tooltipStatus.textContent =
          statusLabels[law.status] || String(law.status || "Unknown");
        tooltip.appendChild(tooltipTitle);
        tooltip.appendChild(tooltipStatus);
        icon.setAttribute("aria-describedby", tooltip.id);
        icon.appendChild(tooltip);
        grid.appendChild(icon);
      });

      if (grid.childNodes.length) el.appendChild(grid);
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
      var prefix =
        scope === "playthrough" ? "game_achievement_" : "achievement_";
      var reg =
        ((window.dendryUI.game.data || {}).achievements || {}).achievements ||
        [];
      if (scope === "playthrough") {
        reg = reg.filter(function (a) {
          return !!Q[prefix + a.id];
        });
      }
      if (!reg.length) {
        el.innerHTML = "";
        return;
      }

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
      var props = readDataProps(el);
      var config =
        typeof props.configFrom === "string" ? Q[props.configFrom] : null;
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

  // Shared explicit handlers resolve their data here. Desk-only Brief widgets
  // live in status_new.scene.dry, which this frozen shell never renders, so an
  // unknown marker deliberately remains untouched rather than growing a
  // generic fallback solely for a UI that is scheduled for retirement.
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
