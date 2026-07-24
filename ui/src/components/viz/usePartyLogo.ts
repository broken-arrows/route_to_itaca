import { useGameStore } from '../../stores/game';

export function usePartyLogo() {
  const game = useGameStore();
  return (party: string): string | null => {
    // The glossary already owns the party -> artwork mapping. Resolve by the
    // behaviour-driving match field (entry ids are not party ids), then make
    // its game-relative path absolute against this document's deployed base.
    const key = party.toLowerCase();
    const term = game.glossary.find((entry) =>
      entry.match.some((match) => match.toLowerCase() === key),
    );
    const path = term?.tooltip?.img;
    return path ? new URL(path, document.baseURI).href : null;
  };
}
