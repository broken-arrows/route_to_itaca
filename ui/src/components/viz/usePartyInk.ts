import { useGameStore } from '../../stores/game';

const NEUTRAL = '#8a8273';

function cssColour(token?: string): string {
  if (!token) return NEUTRAL;
  return token.startsWith('#') ? token : `var(--${token})`;
}

/**
 * Resolve party ids through the field that actually drives glossary matching.
 * Entry ids are not equivalent (`jxsi` lives under glossary id `jxs_`).
 */
export function usePartyInk(): (party?: string | null) => string {
  const game = useGameStore();
  return (party?: string | null) => {
    if (!party) return NEUTRAL;
    const key = party.toLowerCase();
    const term = game.glossary.find((entry) =>
      entry.match.some((match) => match.toLowerCase() === key),
    );
    return cssColour(term?.colour);
  };
}
