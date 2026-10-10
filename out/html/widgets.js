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
 * Older compiled content can still contain unmarked map or party-tour divs.
 * The legacy-id fallback below mounts those only when they have no marker,
 * so marked content is rendered once through the named handler.
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

  // Parties which can carry a Catalan independence roadmap. Successor parties
  // keep separate entries because their parliamentary seats and roadmap state
  // move between different Q keys over the course of the game.
  var ROADMAP_PARTIES = [
    { key: "ciu", label: "CiU", logo: "img/parties/logo_ciu.png" },
    { key: "cdc", label: "CDC", logo: "img/parties/logo_cdc.png" },
    { key: "dl", label: "DL", logo: "img/parties/logo_dl.png" },
    { key: "pdcat", label: "PDeCAT", logo: "img/parties/logo_pdecat.svg" },
    { key: "jxsi", label: "JxSí", logo: "img/parties/logo_jxsi.png" },
    { key: "jxcat", label: "JxCat", logo: "img/parties/logo_jxcat.svg" },
    { key: "junts", label: "Junts", logo: "img/parties/logo_junts.png" },
    { key: "erc", label: "ERC", logo: "img/parties/logo_erc.png" },
    { key: "cup", label: "CUP", logo: "img/parties/logo_cup.svg" },
    { key: "fnc", label: "FNC", logo: "img/parties/logo_fnc.png" },
    { key: "unio", label: "UDC", logo: "img/parties/logo_unio.png" },
  ];

  var ROADMAP_LEVELS = [
    {
      label: "No roadmap",
      description: "",
    },
    {
      label: "Vote on agreement",
      description:
        "Focus is put on negotiating a referendum on independence with the Gobierno.",
    },
    {
      label: "Tactical Confrontation",
      description:
        "A delicate balance between escalating tensions and openness to dialogue, hoping to force the hand of the Gobierno and the EU.",
    },
    {
      label: "Unilateralism",
      description:
        "Independence is to be achieved by exercising sovereignty, not asking for permission.",
    },
  ];

  // Dispatch table for names declared via `data-widget`. Each handler owns
  // reading whatever data-props it needs; the shell's renderers themselves
  // are called exactly as they always were (by id + Q [+ config]).
  var HANDLERS = {
    "builder-playground": function (el, Q) {
      var config = resolveProps(el, Q);
      var engine = window.dendryUI && window.dendryUI.dendryEngine;
      if (!engine || !config.rows) return;
      el.replaceChildren();
      var jointRows = config.rows.filter(function (row) {
        return (row.isJoint || row.slot === "joint") && !row.hidden;
      });
      var hasJoint = jointRows.length > 0;
      var groups = jointRows
        .filter(function (row) {
          return row.seats > 0;
        })
        .map(function (row) {
          return {
            slot: row.slot,
            start: row.position || 1,
            end:
              Math.max.apply(
                null,
                [row.position || 1].concat(
                  config.rows
                    .filter(function (member) {
                      return (
                        member.member && (member.group || "joint") === row.slot
                      );
                    })
                    .map(function (member) {
                      return member.position || 1;
                    }),
                ),
              ) + 1,
          };
        });
      el.classList.add("builder-playground");
      el.classList.toggle("has-joint", hasJoint);
      el.style.cssText = "";
      groups.forEach(function (group) {
        var brace = document.createElement("div");
        brace.className = "builder-brace";
        brace.setAttribute("aria-hidden", "true");
        brace.dataset.group = group.slot;
        brace.style.gridRow = group.start + " / " + group.end;
        el.appendChild(brace);
      });
      function refresh(changed, remount) {
        if (!changed) return;
        engine.ui.removeChoices();
        engine.choiceCache = engine._compileChoices(engine.getCurrentScene());
        engine.displayChoices();
        if (remount) HANDLERS["builder-playground"](el, Q);
      }
      config.rows
        .filter(function (row) {
          return !row.hidden;
        })
        .forEach(function (row) {
          var entry = document.createElement("div");
          entry.className = "builder-party";
          var isJoint = row.isJoint || row.slot === "joint";
          entry.classList.toggle("is-joint", !!isJoint);
          entry.dataset.party = row.id;
          entry.style.minWidth = "0";
          if (row.column !== undefined) {
            entry.style.gridColumn =
              row.column + (hasJoint ? (row.column === 0 ? 1 : 2) : 0);
            var start = row.position || 1;
            var end =
              Math.max.apply(
                null,
                [start].concat(
                  config.rows
                    .filter(function (member) {
                      return (
                        member.member &&
                        !member.hidden &&
                        (member.group || "joint") === row.slot
                      );
                    })
                    .map(function (member) {
                      return member.position || 1;
                    }),
                ),
              ) + 1;
            entry.style.gridRow = isJoint ? start + " / " + end : row.position;
          }
          var content = document.createElement("div");
          content.className = "builder-content";
          var label = document.createElement("div");
          label.className = "builder-name";
          if (row.canJoin) {
            var checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.className = "builder-checkbox";
            checkbox.checked = !!row.member;
            checkbox.setAttribute(
              "aria-label",
              "Include " + row.name + " in the joint list",
            );
            checkbox.onchange = function () {
              refresh(
                engine.gameLib.toggleCoalitionBuilderMember(Q, row.slot),
                true,
              );
            };
            label.appendChild(checkbox);
          }
          function arrow(direction) {
            var button = document.createElement("button");
            button.type = "button";
            button.textContent = direction < 0 ? "‹" : "›";
            button.setAttribute(
              "aria-label",
              (direction < 0 ? "Previous" : "Next") + " party for " + row.name,
            );
            button.className =
              direction < 0 ? "builder-previous" : "builder-next";
            button.onclick = function () {
              refresh(
                engine.gameLib.switchCoalitionBuilderParty(
                  Q,
                  row.slot,
                  direction,
                ),
                true,
              );
            };
            return button;
          }
          var switches = row.variants && row.variants.length > 1;
          if (switches) label.appendChild(arrow(-1));
          var name = document.createElement("span");
          name.className = "builder-label";
          if (typeof window.applyWholesome === "function")
            name.innerHTML = window.applyWholesome(row.name);
          else name.textContent = row.name;
          label.appendChild(name);
          if (switches) label.appendChild(arrow(1));
          var controls = document.createElement("div");
          controls.className = "builder-controls";
          var minus = document.createElement("button");
          var input = document.createElement("input");
          var plus = document.createElement("button");
          minus.textContent = "−";
          plus.textContent = "+";
          minus.type = plus.type = "button";
          minus.setAttribute("aria-label", "Remove a seat from " + row.name);
          plus.setAttribute("aria-label", "Add a seat to " + row.name);
          input.setAttribute("aria-label", row.name + " seats");
          input.type = "text";
          input.inputMode = "numeric";
          input.pattern = "[0-9]*";
          function paint() {
            entry.classList.toggle("is-muted", row.seats === 0 || !!row.member);
            input.value = row.seats;
            input.disabled = !!row.member;
            minus.disabled = row.member || row.seats <= 0;
            plus.disabled =
              row.member || row.seats >= Q.playground_coalitions_needed_seats;
            [minus, input, plus].forEach(function (control) {
              control.style.opacity = control.disabled ? ".35" : "1";
            });
          }
          function edit(seats) {
            refresh(
              engine.gameLib.editCoalitionBuilder(Q, row.id, seats),
              row.isJoint || row.slot === "joint",
            );
            paint();
          }
          minus.onclick = function () {
            edit(row.seats - 1);
          };
          plus.onclick = function () {
            edit(row.seats + 1);
          };
          input.onchange = function () {
            edit(input.value === "" ? NaN : Number(input.value));
          };
          paint();
          controls.append(minus, input, plus);
          content.append(label, controls);
          if (row.isJoint || row.slot === "joint") {
            var members = document.createElement("div");
            members.className = "builder-members";
            config.rows
              .filter(function (party) {
                return (
                  party.member &&
                  party.slot !== "si" &&
                  (party.group || "joint") === row.slot
                );
              })
              .forEach(function (party) {
                var member = document.createElement("div");
                if (typeof window.applyWholesome === "function")
                  member.innerHTML = window.applyWholesome(party.name);
                else member.textContent = party.name;
                members.appendChild(member);
              });
            content.appendChild(members);
          }
          entry.appendChild(content);
          el.appendChild(entry);
        });
    },
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
      parliament.enter
        .fromCenter(Boolean(props.animate))
        .smallToBig(Boolean(props.animate));
      parliament.exit.toCenter(false).bigToSmall(false);
      parliament.highlightedParty(
        typeof config.highlightedParty === "string"
          ? config.highlightedParty
          : Q.player_party,
      );
      d3.select(svg).datum(data).call(parliament);

      var majority = Number(config.majority) || 0;
      var majorityLabel = el.querySelector(".hemicycle-majority");
      if (majority > 0) {
        if (!majorityLabel) {
          majorityLabel = document.createElement("div");
          majorityLabel.className = "hemicycle-majority";
          el.appendChild(majorityLabel);
        }
        majorityLabel.textContent = "Majority: " + majority + " seats";
      } else if (majorityLabel) {
        majorityLabel.remove();
      }
    },
    "chamber-vote": function (el, Q) {
      var props = resolveProps(el, Q);
      var outcomes = Array.isArray(props.outcomes) ? props.outcomes : [];
      outcomes = outcomes
        .filter(function (outcome) {
          return outcome && Number(outcome.votes) > 0;
        })
        .sort(function (a, b) {
          var order = { yes: 0, abstain: 1, "not-present": 2, no: 3 };
          return order[a.kind] - order[b.kind];
        });

      el.innerHTML = "";
      if (!outcomes.length) return;

      var wrap = document.createElement("div");
      wrap.className = "chamber-vote";
      var totalVotes = outcomes.reduce(function (total, outcome) {
        return total + Math.max(0, Number(outcome.votes) || 0);
      }, 0);
      var votesBefore = 0;
      outcomes.forEach(function (outcome, index) {
        var votes = Math.max(0, Number(outcome.votes) || 0);
        if (outcome.kind === "abstain" || outcome.kind === "not-present") {
          var segmentCenter = (votesBefore + votes / 2) / totalVotes;
          var columnCenter = (index + 0.5) / outcomes.length;
          var shift = (segmentCenter - columnCenter) * outcomes.length * 100;
          wrap.style.setProperty(
            "--chamber-vote-" + outcome.kind + "-shift",
            String(shift) + "%",
          );
        }
        votesBefore += votes;
      });
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
          outcome.kind === "not-present" ||
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
            return law && law.icon;
          })
        : [];
      if (!visibleLaws.length) {
        var empty = document.createElement("p");
        empty.className = "law-grid__empty";
        empty.textContent = "No laws passed yet";
        el.appendChild(empty);
        return;
      }

      var validColours = { gray: true, orange: true, red: true, green: true };
      var grid = document.createElement("div");
      grid.className = "law-grid";

      visibleLaws.forEach(function (law, index) {
        var colour = Object.prototype.hasOwnProperty.call(
          validColours,
          law.colour,
        )
          ? law.colour
          : "gray";
        var description = String(law.description || "");
        var label = String(law.title || law.id || "Law");
        if (description) label += " — " + description;
        var icon = document.createElement("div");
        icon.className = "law-grid__law law-grid__law--" + colour;
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
        var tooltipDescription = document.createElement("span");
        tooltipDescription.className = "law-grid__tooltip-description";
        tooltipDescription.textContent = description;
        tooltip.appendChild(tooltipTitle);
        tooltip.appendChild(tooltipDescription);
        icon.setAttribute("aria-describedby", tooltip.id);
        icon.appendChild(tooltip);
        grid.appendChild(icon);
      });

      if (grid.childNodes.length) el.appendChild(grid);
    },
    roadmaps: function (el, Q) {
      var partiesByLevel = ROADMAP_LEVELS.map(function () {
        return [];
      });

      ROADMAP_PARTIES.forEach(function (party) {
        if (!(Number(Q[party.key + "_parlament_s"]) > 0)) return;

        var roadmap = Number(Q[party.key + "_roadmap"]);
        if (!Number.isFinite(roadmap)) roadmap = 0;
        roadmap = Math.max(0, Math.min(3, Math.round(roadmap)));
        partiesByLevel[roadmap].push(party);
      });

      el.innerHTML = "";

      var spectrum = document.createElement("div");
      spectrum.className = "roadmaps-widget__spectrum";
      spectrum.setAttribute("role", "list");
      spectrum.setAttribute("aria-label", "Party roadmaps to independence");

      [3, 2, 1, 0].forEach(function (level) {
        var position = ROADMAP_LEVELS[level];
        var column = document.createElement("div");
        column.className = "roadmaps-widget__level";
        column.setAttribute("data-roadmap", String(level));

        var parties = document.createElement("div");
        parties.className = "roadmaps-widget__parties";

        partiesByLevel[level].forEach(function (party) {
          var item = document.createElement("div");
          item.className = "roadmaps-widget__party";
          item.setAttribute("role", "listitem");
          item.setAttribute("aria-label", party.label + ": " + position.label);

          var logo = document.createElement("img");
          logo.className = "roadmaps-widget__logo";
          logo.src = party.logo;
          logo.alt = "";

          var name = document.createElement("span");
          name.className = "roadmaps-widget__name";
          name.innerHTML =
            typeof window.applyWholesome === "function"
              ? window.applyWholesome(party.label)
              : party.label;

          item.appendChild(logo);
          item.appendChild(name);
          parties.appendChild(item);
        });

        var levelLabel = document.createElement("div");
        levelLabel.className = "roadmaps-widget__label";
        var levelTitle = document.createElement("strong");
        levelTitle.textContent = position.label;
        var levelDescription = document.createElement("span");
        levelDescription.textContent = position.description;
        levelLabel.appendChild(levelTitle);
        levelLabel.appendChild(levelDescription);

        column.appendChild(parties);
        column.appendChild(levelLabel);
        spectrum.appendChild(column);
      });

      el.appendChild(spectrum);
    },
    "poll-map": function (el, Q) {
      initCataloniaPolls(el.id, Q, el.id === "cat-polls-widget-wide");
    },
    "parlament-results-map": function (el, Q) {
      initCataloniaPolls(el.id, Q, true);
    },
    "local-results-map": function (el, Q) {
      initCatLocalMap(el.id, Q);
    },
    "congreso-results-map": function (el, Q) {
      initCongresoMap(el.id, Q);
    },
    "congreso-party-tour": function (el, Q) {
      initCongresoPartyTour(el.id, Q);
    },
    "achievement-gallery": function (el, Q) {
      var props = readDataProps(el);
      var scope = props.scope === "playthrough" ? "playthrough" : "ever";
      var prefix =
        scope === "playthrough" ? "game_achievement_" : "achievement_";
      var reg =
        ((window.dendryUI.game.data || {}).achievements || {}).achievements ||
        [];
      var total = reg.filter(function (a) {
        return !a.hidden;
      }).length;
      var ordinary = reg.filter(function (a) {
        return !a.hidden && !!Q[prefix + a.id];
      }).length;
      var hidden = reg.filter(function (a) {
        return a.hidden && !!Q[prefix + a.id];
      }).length;
      var percentage = total
        ? Math.round(((ordinary + hidden) / total) * 1000) / 10
        : null;
      var percentageLabel = percentage === null ? "—%" : percentage + "%";
      var fill = Math.min(percentage === null ? 0 : percentage, 100);
      var count = ordinary + (hidden ? "+" + hidden : "") + " / " + total;
      var summary =
        '<div class="achievement-completion">' +
        '<span class="achievement-percentage">' +
        percentageLabel +
        "</span>" +
        '<div class="achievement-progress" role="progressbar" aria-label="' +
        count +
        '" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' +
        fill +
        '" aria-valuetext="' +
        percentageLabel +
        " — " +
        count +
        '">' +
        '<div class="achievement-progress-fill" style="width:' +
        fill +
        '%"></div></div>' +
        '<span class="achievement-count">' +
        count +
        "</span></div>";

      el.innerHTML =
        summary +
        reg
          .filter(function (a) {
            return !!Q[prefix + a.id] || (scope !== "playthrough" && !a.hidden);
          })
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

  // Preserve rendering of older compiled content with unmarked placeholders.
  var LEGACY_IDS = [
    { id: "cat-polls-widget-wide", handler: "parlament-results-map" },
    { id: "catalonia-local-map", handler: "local-results-map" },
    { id: "congreso-map-widget", handler: "congreso-results-map" },
    { id: "congreso-party-tour-widget", handler: "congreso-party-tour" },
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
      var legacy = LEGACY_IDS[j];
      var unmarked = document.getElementById(legacy.id);
      if (unmarked && !unmarked.hasAttribute("data-widget")) {
        HANDLERS[legacy.handler](unmarked, Q);
      }
    }
  };
})();
