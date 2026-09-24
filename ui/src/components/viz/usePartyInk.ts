import { useGameStore } from '../../stores/game';
import type { GlossaryTerm } from '../../glossary/mark';

const NEUTRAL = '#8a8273';

function cssColour(token?: string): string {
  if (!token) return NEUTRAL;
  return token.startsWith('#') ? token : `var(--${token})`;
}

/**
 * Resolve party ids through the field that actually drives glossary matching.
 * Entry ids are not equivalent (`jxsi` lives under glossary id `jxs_`).
 */
export function usePartyTerm(): (party?: string | null) => GlossaryTerm | undefined {
  const game = useGameStore();
  return (party?: string | null) => {
    if (!party) return undefined;
    const key = party.toLowerCase();
    return game.glossary.find((entry) =>
      entry.match.some((match) => match.toLowerCase() === key),
    );
  };
}

export function usePartyInk(): (party?: string | null) => string {
  const termFor = usePartyTerm();
  return (party?: string | null) => cssColour(termFor(party)?.colour);
}
