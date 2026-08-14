import { computed } from 'vue';
import { useGameStore } from '../stores/game';
import { gameLib } from '../game-bindings';
import type { GlossaryTerm } from './mark';
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
