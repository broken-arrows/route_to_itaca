(function () {
  'use strict';
  var catalogs = {}, pending = {}, generation = 0, locale = 'en';
  function load(language) {
    if (!pending[language]) pending[language] = Promise.all(['ui', 'content', 'parties'].map(function (kind) {
      return fetch('locales/' + language + '/' + kind + '.json').then(function (response) {
        if (!response.ok) throw new Error('Missing ' + language + ' ' + kind + ' catalog');
        return response.json();
      });
    })).then(function (values) {
      return catalogs[language] = { ui: values[0], content: values[1], parties: values[2] };
    }).catch(function (error) { delete pending[language]; throw error; });
    return pending[language];
  }
  function lookup(language, key) {
    return key.split('.').reduce(function (value, part) { return value && value[part]; }, (catalogs[language] || {}).ui);
  }
  function t(key, fallback, params) {
    var value = lookup(locale, key) || lookup('en', key) || fallback || key;
    return String(value).replace(/\{(\w+)\}/g, function (match, name) {
      return params && Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : match;
    });
  }
  // Classic's English labels are stable lookup keys within its own namespace.
  function text(english, params) {
    var selected = ((catalogs[locale] || {}).ui || {}).classic || {};
    var base = ((catalogs.en || {}).ui || {}).classic || {};
    var value = (selected.messages || {})[english] || (base.messages || {})[english] || english;
    return String(value).replace(/\{(\w+)\}/g, function (match, name) {
      return params && Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : match;
    });
  }
  function content(value, params) {
    var engine = window.dendryUI && window.dendryUI.dendryEngine;
    return engine ? engine.translate(value, params) : value;
  }
  function apply(root) {
    (root || document).querySelectorAll('[data-classic-text]').forEach(function (el) {
      el.textContent = text(el.getAttribute('data-classic-text'));
    });
    ['aria-label', 'title', 'alt', 'data-tooltip'].forEach(function (attribute) {
      (root || document).querySelectorAll('[data-classic-' + attribute + ']').forEach(function (el) {
        el.setAttribute(attribute, text(el.getAttribute('data-classic-' + attribute)));
      });
    });
  }
  function syncLanguageControl() {
    var radio = document.getElementById('classic-language-' + locale);
    if (radio) radio.checked = true;
    var switchButton = document.getElementById('menu-language-switch');
    if (switchButton) {
      var target = locale === 'en' ? 'ca' : 'en';
      switchButton.value = target;
      switchButton.lang = target;
      switchButton.textContent = target === 'ca' ? 'També en Català!' : 'Also in English';
    }
  }
  async function setLocale(language, options) {
    options = options || {};
    if (language !== 'en' && language !== 'ca') language = 'en';
    var request = ++generation;
    try { await Promise.all([load('en'), load(language)]); }
    catch (error) {
      if (request !== generation) return false;
      console.warn('Language catalog unavailable; retaining current language.', error);
      syncLanguageControl();
      return false;
    }
    if (request !== generation) return false;
    locale = language;
    var ui = window.dendryUI, engine = ui.dendryEngine;
    engine.setLocale(language, catalogs[language].content, { party: catalogs[language].parties });
    document.documentElement.lang = language;
    document.title = text('Route to Ítaca - An Alternate History');
    try { localStorage.setItem(ui.settings_key + ':language', language); } catch (_) {}
    syncLanguageControl();
    apply();
    if (!options.initial && engine.state && engine.state.sceneId) {
      ui.refreshingLocale = true;
      var animate = ui.animate;
      ui.animate = false;
      try {
        if (window.hideClassicTooltip) window.hideClassicTooltip();
        engine.refreshPresentation();
        if (engine.state.gameOver) ui.displayGameOver();
        if (window.syncTabLocks) window.syncTabLocks();
        if (document.getElementById('save')) ui.populateSaveSlots(ui.max_slots, 2);
      } finally { ui.animate = animate; ui.refreshingLocale = false; }
    }
    return true;
  }
  function init(ui) {
    var saved = 'en';
    try { saved = localStorage.getItem(ui.settings_key + ':language') || 'en'; } catch (_) {}
    return setLocale(saved, { initial: true });
  }
  window.ClassicI18n = { t: t, text: text, content: content, apply: apply, setLocale: setLocale, init: init,
    get locale() { return locale; } };
})();
