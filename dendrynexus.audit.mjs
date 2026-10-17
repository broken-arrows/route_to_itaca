import { WIDGET_NAMES, DERIVE_NAMES } from './ui/src/components/viz/widget-names.mjs';

export default {
  widgetNames: WIDGET_NAMES,
  deriveNames: DERIVE_NAMES,
  deprecatedSceneProperties: {
    'is-card': 'Use role: card, card-party, card-gov or card-parliament.',
    'is-pinned-card': 'Use role: pinned-action, pinned-advisor or pinned-parliament.',
    'is-hand': 'Use role: desk.',
    'is-deck': 'Use role: deck, deck-party, deck-gov or deck-parliament.',
  },
  // Root's origin check. Exceptions apply only
  // to browser-global code; every scene still gets its markers checked.
  allowBrowserGlobals: [
    'root',
  ],
};
