(function () {
  var game;
  var ui;

  var DateOptions = {
    hour: "numeric",
    minute: "numeric",
    second: "numeric",
    year: "numeric",
    month: "short",
    day: "numeric",
  };

  var main = function (dendryUI) {
    ui = dendryUI;
    game = ui.game;

    dendryUI.dendryEngine.setGameLib(window.RTI_GAME_LIB);

    // I want these settings on by default, but the engine does not have a way to set them and I don't fancy touching that sht.
    if (!window.dendryUI.dendryEngine.touchedSettings) {
      window.dendryUI.animate = true;
      window.dendryUI.show_portraits = true;
      window.dendryUI.dark_mode = false;
      window.dendryUI.saveSettings();
    }
  };

  var TITLE =
    "Route to Ítaca - An Alternate History" +
    "_" +
    "BrokenArrow, modded from Autumn Chen's work";

  window.showStats = function () {
    if (window.dendryUI.dendryEngine.state.sceneId.startsWith("library")) {
      window.dendryUI.dendryEngine.goToScene("backSpecialScene");
    } else {
      window.dendryUI.dendryEngine.goToScene("library");
    }
    addTooltipEventListeners();
  };

  window.showOptions = function () {
    var save_element = document.getElementById("options");
    window.populateOptions();
    window.dendryUI.dendryEngine.touchedSettings = true;
    save_element.style.display = "block";
    if (!save_element.onclick) {
      save_element.onclick = function (evt) {
        var target = evt.target;
        var save_element = document.getElementById("options");
        if (target == save_element) {
          window.hideOptions();
        }
      };
    }
  };

  window.hideOptions = function () {
    var save_element = document.getElementById("options");
    save_element.style.display = "none";
  };

  window.disableBg = function () {
    window.dendryUI.disable_bg = true;
    document.body.style.backgroundImage = "none";
    window.dendryUI.saveSettings();
  };

  window.enableBg = function () {
    window.dendryUI.disable_bg = false;
    window.dendryUI.setBg(window.dendryUI.dendryEngine.state.bg);
    window.dendryUI.saveSettings();
  };

  window.disableAnimate = function () {
    window.dendryUI.animate = false;
    window.dendryUI.saveSettings();
  };

  window.enableAnimate = function () {
    window.dendryUI.animate = true;
    window.dendryUI.saveSettings();
  };

  window.disableAnimateBg = function () {
    window.dendryUI.animate_bg = false;
    window.dendryUI.saveSettings();
  };

  window.enableAnimateBg = function () {
    window.dendryUI.animate_bg = true;
    window.dendryUI.saveSettings();
  };

  window.disableAudio = function () {
    window.dendryUI.toggle_audio(false);
    window.dendryUI.saveSettings();
  };

  window.enableAudio = function () {
    window.dendryUI.toggle_audio(true);
    window.dendryUI.saveSettings();
  };

  window.enableImages = function () {
    window.dendryUI.show_portraits = true;
    window.dendryUI.saveSettings();
  };

  window.disableImages = function () {
    window.dendryUI.show_portraits = false;
    window.dendryUI.saveSettings();
  };

  window.enableLightMode = function () {
    window.dendryUI.dark_mode = false;
    document.body.classList.remove("dark-mode");
    window.dendryUI.saveSettings();
  };
  window.enableDarkMode = function () {
    window.dendryUI.dark_mode = true;
    document.body.classList.add("dark-mode");
    window.dendryUI.saveSettings();
  };

  // populates the checkboxes in the options view
  window.populateOptions = function () {
    var disable_bg = window.dendryUI.disable_bg;
    var animate = window.dendryUI.animate;
    var disable_audio = true; // TODO: enable on adding music!
    // var disable_audio = window.dendryUI.disable_audio;
    var show_portraits = window.dendryUI.show_portraits;
    if (disable_bg) {
      $("#backgrounds_no")[0].checked = true;
    } else {
      $("#backgrounds_yes")[0].checked = true;
    }
    if (animate) {
      $("#animate_yes")[0].checked = true;
    } else {
      $("#animate_no")[0].checked = true;
    }
    if (disable_audio) {
      $("#audio_no")[0].checked = true;
    } else {
      $("#audio_yes")[0].checked = true;
    }
    if (show_portraits) {
      $("#images_yes")[0].checked = true;
    } else {
      $("#images_no")[0].checked = true;
    }
    if (window.dendryUI.dark_mode) {
      $("#dark_mode")[0].checked = true;
    } else {
      $("#light_mode")[0].checked = true;
    }
  };

  // This function allows you to modify the text before it's displayed.
  // E.g. wrapping chat-like messages in spans.
  window.displayText = function (text) {
    return applyWholesome(text);
  };

  // The glossary used to be two hand-parallel tables (tooltipList/colourList)
  // baked into this file's web root as data.js. It is now compiled game data,
  // reachable as game.json.data.glossary, see source/data/glossary.json and
  // tools/harvest-glossary.mjs (which explains, in its header comment, why a
  // couple of ids like "icv"/"icv_euia" and "omnium"/"omnium_cultural" stayed
  // split rather than merging into one term).
  function glossary() {
    return (
      (window.dendryUI.game.data && window.dendryUI.game.data.glossary) || {
        terms: [],
      }
    );
  }

  // Registry colours are bare tokens ("ciu") or, where the old palette never
  // had a CSS var for a party (CUP, ETA, ANC, \u00D2mnium\u2026), a raw hex kept
  // verbatim by the harvester. Only the token form needs the var(--x) wrap.
  function cssColour(token) {
    return token.startsWith("#") ? token : `var(--${token})`;
  }

  // Word matching, kept in lockstep with the Desk's ui/src/glossary/mark.ts:
  // - metachars in match words are escaped ("NA+" used to be a live regex
  //   operator here);
  // - the boundary is a Unicode lookaround pair, not `\b` — `\b` is ASCII-only,
  //   so words starting/ending with accented letters or punctuation
  //   ("Àngel Ros", "JxSí", "BComú", "¡TE!") silently never matched;
  // - longest match word wins when one is a prefix of another;
  // - the compiled regex/word map is cached per glossary terms array (this runs
  //   on every rendered text run via window.displayText).
  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  var _wholesomeTerms = null;
  var _wholesomeCompiled = null;
  function compileWholesome(terms) {
    if (_wholesomeTerms === terms) return _wholesomeCompiled;
    const byWord = {};
    const allWords = [];
    terms.forEach((t) => {
      t.match.forEach((w) => {
        byWord[w] = t;
        allWords.push(w);
      });
    });
    _wholesomeTerms = terms;
    _wholesomeCompiled =
      allWords.length === 0
        ? null
        : {
            byWord: byWord,
            regex: new RegExp(
              `(?<![\\p{L}\\p{N}_])(${allWords
                .sort((a, b) => b.length - a.length)
                .map(escapeRegex)
                .join("|")})(?![\\p{L}\\p{N}_])`,
              "gu",
            ),
          };
    return _wholesomeCompiled;
  }

  function applyWholesome(str) {
    const compiled = compileWholesome(glossary().terms);
    if (!compiled) return str;
    const byWord = compiled.byWord;
    const regex = compiled.regex;

    return str.replace(
      /(<(?:span|strong)[^>]*>.*?<\/(?:span|strong)>|<[^>]+>|[^<]+)/g,
      (segment) => {
        if (segment.startsWith("<")) return segment;

        return segment.replace(regex, (match) => {
          const t = byWord[match];

          // skip if preceded by zero-width space
          const zwspIndex = segment.indexOf("\u200B" + match);
          if (zwspIndex !== -1) {
            return match;
          }

          // find if just before the match there was "--"
          const matchStart = segment.lastIndexOf(
            "--",
            segment.indexOf(match) - 2,
          );
          if (matchStart !== -1 && matchStart === segment.indexOf(match) - 2) {
            return match;
          }

          const textColor = t.colour ? cssColour(t.colour) : "inherit";
          const style = t.bold ? "font-weight: bold;" : "";
          const innerText = match;

          if (t.tooltip) {
            // Lightweight trigger only. The tooltip body is built on demand at
            // hover time against the shared singleton (see renderTipContent and
            // addTooltipEventListeners); data-term keys into glossary().terms.
            const displayText = t.display || innerText;
            return `<span class='mytooltip' style='--mytooltip-color:${textColor}; ${style}' data-term='${t.id}'>${displayText}</span>`;
          } else if (t.colour) {
            return `<span style='color: ${textColor}; ${style}'>${t.display || innerText}</span>`;
          }

          return match;
        });
      },
    );
  }

  window.applyWholesome = applyWholesome;
  window.glossary = glossary;

  // This function allows you to do something in response to signals.
  window.handleSignal = function (signal, event, scene_id) {};
  window._achievementsSeen = {};

  function checkAchievements() {
    var q = window.dendryUI.dendryEngine.state.qualities;
    var reg =
      ((window.dendryUI.game.data || {}).achievements || {}).achievements || [];
    reg.forEach(function (a) {
      var key = "achievement_" + a.id;
      if (q[key] && !window._achievementsSeen[key]) {
        window._achievementsSeen[key] = true;
        window.achievementNotif(a.name, a.image, a.stars);
      }
    });
  }

  // This function runs on a new page
  window.onNewPage = function () {
    var scene = window.dendryUI.dendryEngine.state.sceneId;
    console.log("New page: " + scene);
    // Widget protocol host (out/html/widgets.js): scans for [data-widget]
    // and dispatches to the renderers above by name — replaces what used to
    // be this exact five-call sequence, duplicated across five call sites
    // (onNewPage / updateSidebar / changeTab / onDisplayContent / onload).

    mountWidgets(document, dendryUI.dendryEngine.state.qualities);
    addTooltipEventListeners();
    window.syncTabLocks();

    if (window.justLoaded) {
      var q0 = window.dendryUI.dendryEngine.state.qualities;
      for (var key in q0) {
        if (q0[key] && key.indexOf("achievement_") === 0) {
          window._achievementsSeen[key] = true;
        }
      }
    } else {
      checkAchievements();
    }
    if (scene != "root" && !window.justLoaded) {
      window.dendryUI.autosave();
    }
    if (window.justLoaded) {
      window.justLoaded = false;
    }
  };

  // tabbed browsing
  //
  // This shell renders `source/scenes/status.scene.dry` — its own five-tab
  // status scene, restored 2026-07-23 when the Brief's sheets moved to
  // `status_new.scene.dry` for the Desk alone. The `_runActions(scene.onArrival)`
  // below is load-bearing HERE and only here: this scene's on-arrival computes
  // `spanish_seal_img` and `unemployment_dataviz`, which its own markup then
  // substitutes. Rendering-by-mutation is the pattern the Desk migration exists
  // to kill, and `status_new`'s sheets have no on-arrival at all — but this
  // shell is frozen and dies at phase 6, so it keeps the mechanism it shipped
  // with rather than acquiring a divergent one.
  window.updateSidebar = function () {
    $("#qualities").empty();
    var scene = dendryUI.game.scenes[window.statusTab];
    dendryUI.dendryEngine._runActions(scene.onArrival);
    var displayContent = dendryUI.dendryEngine._makeDisplayContent(
      scene.content,
      true,
    );
    $("#qualities").append(dendryUI.contentToHTML.convert(displayContent));
    // Widget protocol host (out/html/widgets.js): scans for [data-widget]
    // and dispatches to the renderers above by name — replaces what used to
    // be this exact five-call sequence, duplicated across five call sites
    // (onNewPage / updateSidebar / changeTab / onDisplayContent / onload).

    mountWidgets(document, dendryUI.dendryEngine.state.qualities);
    addTooltipEventListeners();
  };

  window.changeTab = function (newTab, tabId) {
    var tabButton = document.getElementById(tabId);
    // NOT the `disabled` attribute: disabled buttons suppress pointer events,
    // so the CSS `:hover::after` tooltip (game.css) would never fire and the
    // player would get no explanation at all. A class keeps hover alive.
    if (tabButton.classList.contains("locked")) return;
    var tabButtons = document.getElementsByClassName("tab_button");
    for (var i = 0; i < tabButtons.length; i++) {
      tabButtons[i].className = tabButtons[i].className.replace(" active", "");
    }
    tabButton.className += " active";
    window.statusTab = newTab;
    window.updateSidebar();
  };

  // Unavailable actions explain themselves in place, via the same `.locked`
  // class + `data-tooltip` pattern, instead of interrupting with window.alert.
  // Both dims live on something that does NOT own the ::after tooltip (the tab
  // icon; the link's own colour) — `opacity` on the tooltip's own element makes
  // the explanation unreadable. See game.css.
  window.syncTabLocks = function () {
    var state = dendryUI.dendryEngine.state;

    // Historical mode has no poll data.
    var polls = document.getElementById("polls_tab");
    if (polls) {
      var pollsLocked = !!state.qualities.historical_mode;
      polls.classList.toggle("locked", pollsLocked);
      polls.setAttribute(
        "data-tooltip",
        pollsLocked
          ? "Polls are not available in historical mode"
          : "Opinion Polls",
      );
    }

    // Saving/loading. `state.disableSaves` is the engine's own flag (set by
    // historical mode), and it is what BrowserUserInterface.showSaveSlots
    // alerts on — read the same source rather than re-deriving the rule here.
    var save = document.getElementById("save_link");
    if (save) {
      var savesLocked = !!state.disableSaves;
      save.classList.toggle("locked", savesLocked);
      save.setAttribute(
        "data-tooltip",
        savesLocked
          ? "Saving and loading are disabled in historical mode"
          : "Save/Load",
      );
    }
  };

  // Intercepts BEFORE dendryUI.showSaveSlots so its window.alert never fires:
  // the disabled state is already visible on the link itself.
  window.openSaveSlots = function () {
    if (dendryUI.dendryEngine.state.disableSaves) return;
    dendryUI.showSaveSlots();
  };

  window.onDisplayContent = function () {
    window.updateSidebar();
    // Widget protocol host (out/html/widgets.js): scans for [data-widget]
    // and dispatches to the renderers above by name — replaces what used to
    // be this exact five-call sequence, duplicated across five call sites
    // (onNewPage / updateSidebar / changeTab / onDisplayContent / onload).

    addTooltipEventListeners();
    // BrowserUI calls this hook while the engine is still inside
    // displaySceneContent(), immediately BEFORE scene.onDisplay runs. Defer
    // content widgets one task so JSON-safe presentation models can be
    // reconstructed by on-display on both normal entry and save restoration.
    setTimeout(function () {
      mountWidgets(document, dendryUI.dendryEngine.state.qualities);
    }, 0);
  };

  window._achievementStack = [];
  window._achievementLabel = null;

  function updateLabel() {
    if (!window._achievementLabel) return;
    const n = window._achievementStack.length;
    window._achievementLabel.textContent =
      n === 1 ? "Achievement Unlocked" : "Achievements Unlocked";
  }

  function getOrCreateLabel() {
    if (!window._achievementLabel) {
      const label = document.createElement("div");
      label.style.cssText = `
      position: fixed; right: 24px; z-index: 9999;
      background: var(--card-bg-color);
      border-radius: 8px; padding: 8px 14px;
      font-family: 'UbuntuCustom', system-ui, -apple-system, BlinkMacSystemFont, Ubuntu, Cantarell, sans-serif;
      font-size: .72em; font-weight: 700; letter-spacing: .14em;
      color: var(--text-color); text-transform: uppercase;
      width: fit-content; white-space: nowrap;
      transition: bottom .35s cubic-bezier(.16,1,.3,1), transform .45s cubic-bezier(.16,1,.3,1);
      transform: translateX(0);
    `;
      document.body.appendChild(label);
      window._achievementLabel = label;
    }
    return window._achievementLabel;
  }

  function restack() {
    const GAP = 12;
    let bottom = 24;
    for (let i = window._achievementStack.length - 1; i >= 0; i--) {
      window._achievementStack[i].style.bottom = bottom + "px";
      bottom += window._achievementStack[i].offsetHeight + GAP;
    }
    if (window._achievementLabel) {
      window._achievementLabel.style.bottom = bottom + "px";
    }
  }

  window.achievementNotif = function (title, img, stars) {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const beat = 60 / 180;
    [
      { freq: 293.66, delay: 0, dur: beat * 1.5 },
      { freq: 329.63, delay: beat * 1.5, dur: beat * 0.5 },
      { freq: 349.23, delay: beat * 2, dur: beat * 2.0 },
    ].forEach(({ freq, delay, dur }) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "sine";
      osc.frequency.value = freq;
      const t = ctx.currentTime + delay;
      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(0.25, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, t + dur);
      osc.start(t);
      osc.stop(t + dur);
    });

    const el = document.createElement("div");
    el.className = "achievement achievement--unlocked";
    el.style.cssText = `
    position: fixed; right: 24px; z-index: 9999;
    flex-direction: column; align-items: stretch;
    gap: 10px; padding: 12px 14px;
    border-top: none; border-radius: 8px;
    background: var(--card-bg-color);
    box-shadow:
      0 0 0 1px rgba(220,150,0,.4),
      0 0 14px rgba(220,150,0,.5),
      0 0 28px rgba(220,150,0,.2);
    width: 18em;
    transform: translateX(calc(100% + 32px));
    transition: transform .45s cubic-bezier(.16,1,.3,1), bottom .35s cubic-bezier(.16,1,.3,1);
  `;

    el.innerHTML = `
    <div style="display: flex; align-items: center; gap: 12px;">
      <div class="achievement-image achievement-image--unlocked" style="flex: 1 0 0; height: auto; aspect-ratio: 3/2; box-shadow: none;">
        <img src="${img}" style="width:100%;height:100%;object-fit:cover;">
      </div>
      <div class="achievement-body" style="flex: 2 0 0; min-width: 0;">
        <div class="achievement-title achievement-title--unlocked" style="display: block; margin-bottom: 6px;">${title}</div>
        <div class="achievement-stars">
          ${Array.from(
            { length: 5 },
            (_, i) =>
              `<span class="${i < stars ? "star--filled" : "star--empty"}">★</span>`,
          ).join("")}
        </div>
      </div>
    </div>
  `;

    document.body.appendChild(el);
    window._achievementStack.push(el);
    getOrCreateLabel();
    updateLabel();
    restack();

    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        el.style.transform = "translateX(0)";
      }),
    );

    setTimeout(() => {
      window._achievementStack = window._achievementStack.filter(
        (e) => e !== el,
      );

      // if last card, slide label out together
      if (window._achievementStack.length === 0 && window._achievementLabel) {
        window._achievementLabel.style.transform =
          "translateX(calc(100% + 32px))";
        window._achievementLabel.addEventListener(
          "transitionend",
          () => {
            window._achievementLabel?.remove();
            window._achievementLabel = null;
          },
          { once: true },
        );
      } else {
        updateLabel();
        restack();
      }

      el.style.transform = "translateX(calc(100% + 32px))";
      el.addEventListener(
        "transitionend",
        () => {
          el.remove();
        },
        { once: true },
      );
    }, 4500);
  };

  window.justLoaded = true;
  window.statusTab = "status";
  window.dendryModifyUI = main;
  console.log("Modifying stats: see dendryUI.dendryEngine.state.qualities");

  window.onload = function () {
    window.dendryUI.loadSettings({ show_portraits: false });
    if (window.dendryUI.dark_mode) {
      document.body.classList.add("dark-mode");
    }
    window.pinnedCardsDescription =
      "Advisor cards - actions are only usable once per 6 months.";

    // Widget protocol host (out/html/widgets.js): scans for [data-widget]
    // and dispatches to the renderers above by name — replaces what used to
    // be this exact five-call sequence, duplicated across five call sites
    // (onNewPage / updateSidebar / changeTab / onDisplayContent / onload).
    mountWidgets(document, dendryUI.dendryEngine.state.qualities);
    addTooltipEventListeners();
    window.syncTabLocks();
  };

  // ---- Tooltips: one shared, viewport-aware floating element ----
  // A single .mytooltiptext node lives on <body>; every .mytooltip trigger
  // reuses it. Content is built on demand so it always reflects the current
  // qualities, and listeners are delegated + attached exactly once, so repeated
  // renders never pile up handlers.
  let _tipEl = null; // the shared tooltip node
  let _tipAnchor = null; // trigger the tooltip is currently shown for
  let _tipInited = false;
  let _tipPinned = false; // true once a click pins the tooltip open (sticky)

  // Builds the inner HTML of the tooltip for a given glossary term. This is
  // the exact body applyWholesome used to bake inline, just parameterised on
  // the live qualities and the matched label. `term.tooltip` is the compiled
  // registry shape (title/subtitle/img/infoDesc/q); allegiances (if any) come
  // from the game lib (window.RTI_GAME_LIB.allegiances, source/lib/allegiances.js),
  // keyed by the same term id, as {colour, label, note?} entries this UI wraps.
  function renderTipContent(term, qualities, label) {
    const tooltip = term.tooltip;
    const q = tooltip.q || {};
    const imgHtml = tooltip.img
      ? `<img src='${tooltip.img}' alt='${label} image'/>`
      : "";
    const subText = tooltip.subtitle
      ? `<br/><span class='mytooltip-sub-text'>${tooltip.subtitle}</span>`
      : "";
    let ledBy = "";
    let ideology = "";
    let allegiances = "";
    if (q.ledBy && qualities.hasOwnProperty(q.ledBy)) {
      ledBy = `<br/>Leader: <span class='mytooltip-ledby'>${qualities[q.ledBy]}</span>`;
    } else if (tooltip.infoDesc) {
      ledBy = `<br/>${tooltip.infoDesc}`;
    }
    if (q.ideology) {
      ideology = `<br/><span class='mytooltip-ideology'>${
        qualities.hasOwnProperty(q.ideology)
          ? qualities[q.ideology]
          : q.ideology
      }</span>`;
    }
    const allegianceMap =
      (window.RTI_GAME_LIB && window.RTI_GAME_LIB.allegiances) || {};
    const allegiancesFn = allegianceMap[term.id];
    if (allegiancesFn) {
      const al = allegiancesFn(qualities);
      if (al && al.length) {
        allegiances = `<br/><span class='mytooltip-allegiances'>`;
        allegiances += al.length > 1 ? "Allegiances: " : "Allegiance: ";
        // Entries are presentation-neutral {colour, label, note?}; THIS UI wraps
        // them (colour token -> CSS via cssColour, "(note)" as trailing text).
        allegiances += al
          .map((e) => {
            const span = `<span style='color: ${cssColour(e.colour)}'>${e.label}</span>`;
            return e.note ? `${span} (${e.note})` : span;
          })
          .join(", ");
        allegiances += `</span>`;
      }
    }
    return `<span class='mytooltip-content'>${imgHtml}<span class='mytooltip-text'><span class='mytooltip-main-text'>${tooltip.title}</span>${subText}${ledBy}${ideology}${allegiances}</span></span>`;
  }

  // Position the singleton over an anchor, clamped/flipped to stay on screen.
  function positionTip(anchor) {
    const tip = _tipEl;
    const margin = 8; // min gap from any viewport edge
    const gap = 10; // gap between the word and the tooltip (room for the arrow)

    tip.classList.remove("below"); // measure in the default (above) state
    const a = anchor.getBoundingClientRect();
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    const vw = document.documentElement.clientWidth;

    // Horizontal: centre over the anchor, then clamp inside the viewport.
    let left = a.left + a.width / 2 - tw / 2;
    left = Math.max(margin, Math.min(left, vw - tw - margin));

    // Vertical: prefer above; flip below if it would clip the top edge.
    let top = a.top - th - gap;
    let below = false;
    if (top < margin) {
      top = a.bottom + gap;
      below = true;
    }
    tip.classList.toggle("below", below);

    tip.style.left = left + "px";
    tip.style.top = top + "px";

    // Keep the arrow pointing at the anchor centre after clamping.
    let arrowLeft = a.left + a.width / 2 - left;
    arrowLeft = Math.max(14, Math.min(arrowLeft, tw - 14));
    tip.style.setProperty("--arrow-left", arrowLeft + "px");
  }

  function showTipFor(anchor) {
    const termId = anchor.getAttribute("data-term");
    const term = glossary().terms.find((t) => t.id === termId);
    if (!term || !term.tooltip) return;
    const qualities = window.dendryUI.dendryEngine.state.qualities;
    _tipEl.innerHTML = renderTipContent(term, qualities, anchor.textContent);
    // The trigger no longer wraps the tooltip, so carry its colour across.
    _tipEl.style.setProperty(
      "--mytooltip-color",
      anchor.style.getPropertyValue("--mytooltip-color") || "inherit",
    );
    positionTip(anchor);
    _tipEl.classList.add("visible");
    _tipAnchor = anchor;

    // First-ever hover: the logo may not have loaded yet, so the height we just
    // measured is too short and the tooltip lands low (over the word). Re-place
    // it once the image loads. Cached on later hovers (complete === true), so
    // this is a no-op then.
    _tipEl.querySelectorAll("img").forEach((img) => {
      if (!img.complete) {
        img.addEventListener(
          "load",
          function () {
            if (_tipAnchor === anchor) positionTip(anchor);
          },
          { once: true },
        );
      }
    });
  }

  function hideTip() {
    if (!_tipEl) return;
    _tipEl.classList.remove("visible");
    _tipAnchor = null;
    _tipPinned = false;
  }

  // Idempotent: builds the singleton and wires the delegated listeners once.
  // The existing per-render call sites are kept (harmless after the first run)
  // so behaviour stays identical regardless of init order.
  function addTooltipEventListeners() {
    if (_tipInited) return;
    _tipInited = true;

    _tipEl = document.createElement("div");
    _tipEl.className = "mytooltiptext";
    document.body.appendChild(_tipEl);

    // Whether the device has a real hover (mouse). On touch-only devices a tap
    // toggles; on hover devices hover owns show/hide and a click must not close.
    const canHover = window.matchMedia
      ? window.matchMedia("(hover: hover)").matches
      : true;

    // Desktop hover. Gated so a synthetic tap-generated mouseover on touch
    // devices can't fight the click toggle below.
    document.addEventListener("mouseover", function (e) {
      if (!canHover || _tipPinned) return; // pinned tooltip ignores hover
      const t = e.target.closest(".mytooltip");
      if (t && t !== _tipAnchor) showTipFor(t);
    });
    document.addEventListener("mouseout", function (e) {
      if (!canHover || _tipPinned) return; // don't hide a pinned tooltip
      const t = e.target.closest(".mytooltip");
      if (t && t === _tipAnchor) hideTip();
    });

    // Click: tap-toggle on touch, click-outside to dismiss everywhere. On hover
    // devices a click over a trigger is a no-op (hover already shows it, so a
    // click must not close it).
    document.addEventListener("click", function (e) {
      const t = e.target.closest(".mytooltip");
      if (t) {
        e.stopPropagation();
        if (!canHover) {
          // Touch: a tap toggles the tooltip.
          if (_tipAnchor === t && _tipEl.classList.contains("visible")) {
            hideTip();
          } else {
            showTipFor(t);
          }
        } else {
          // Hover device: a click pins the tooltip open (sticky). Clicking the
          // already-pinned trigger unpins it; clicking a different trigger moves
          // the pin there.
          if (_tipPinned && _tipAnchor === t) {
            hideTip(); // also clears _tipPinned
          } else {
            showTipFor(t); // (re)anchor to the clicked word
            _tipPinned = true;
          }
        }
        return;
      }
      if (!e.target.closest(".mytooltiptext")) hideTip();
    });

    // Esc dismisses the tooltip (handy for a pinned/sticky one).
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && _tipEl.classList.contains("visible")) hideTip();
    });

    // The tooltip is position:fixed, so keep it glued to its anchor while the
    // page scrolls or resizes underneath it.
    window.addEventListener(
      "scroll",
      function () {
        if (_tipAnchor) positionTip(_tipAnchor);
      },
      true,
    );
    window.addEventListener("resize", function () {
      if (_tipAnchor) positionTip(_tipAnchor);
    });
  }
})();
