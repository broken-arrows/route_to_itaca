// The glossary LOOK, shared by Prose.vue and GlossaryTerm.vue. mark.ts (the
// DATA/marker) never touches colour or presentation on purpose — this is
// where a token ("ciu") or a raw hex ("#555555", for parties the old
// palette never had a var for) becomes an actual CSS colour, exactly the
// convention `out/html/game.js`'s cssColour() used. `allegiancesFor` is the
// other half of a person tooltip's body (§4 of this task's brief): the
// game's own Q-branching logic (source/lib/allegiances.js), reached through
// gameLib exactly as the old shell reaches it (window.RTI_GAME_LIB.allegiances).
import { computed } from 'vue';
import { useGameStore } from '../stores/game';
import { gameLib } from '../game-bindings';
import type { GlossaryTerm } from './mark';
// Relative import, not a bare specifier — TypeScript's resolver picks up the
// co-located source/lib/index.d.ts sibling automatically for any importer
// (see docs/design/LEARNINGS.md 2026-07-14 #1; game-bindings.ts's GameLib
// import is the other example of this same pattern, one directory shallower).
import type { AllegianceEntry } from '../../../source/lib/index.js';

export function useGlossary() {
  const game = useGameStore();

  const byId = computed(() => {
    const m = new Map<string, GlossaryTerm>();
    for (const t of game.glossary) m.set(t.id, t);
    return m;
  });

  function termFor(id: string | undefined | null): GlossaryTerm | undefined {
    if (!id) return undefined;
    return byId.value.get(id);
  }

  // Token -> `var(--x)`; a raw hex passes through verbatim. Never throws on
  // an unknown/absent colour — callers get undefined and skip the style.
  function colourValue(colour?: string): string | undefined {
    if (!colour) return undefined;
    return colour.startsWith('#') ? colour : `var(--${colour})`;
  }

  // Q-conditional party history for a person term id, or [] if this term has
  // none (most glossary terms aren't people). Never throws on a missing fn.
  function allegiancesFor(id: string): AllegianceEntry[] {
    const fn = gameLib.allegiances?.[id];
    return fn ? fn(game.q) : [];
  }

  return { termFor, colourValue, allegiancesFor };
}
