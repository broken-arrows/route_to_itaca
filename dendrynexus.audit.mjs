import { WIDGET_NAMES, DERIVE_NAMES } from './ui/src/components/viz/widget-names.mjs';

export default {
  widgetNames: WIDGET_NAMES,
  deriveNames: DERIVE_NAMES,
  // Unmigrated authored scenes and root's origin check. Exceptions apply only
  // to browser-global code; every scene still gets its markers checked.
  allowBrowserGlobals: [
    'parlament_election',
    'congreso_election',
    'election_simulation',
    'root',
  ],
};
