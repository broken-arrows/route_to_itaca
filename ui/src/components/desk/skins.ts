export interface Skin {
  key: 'neutral' | 'gov' | 'party' | 'parliament';
  bg: string;
  bd: string;
}

const SKINS: Record<Skin['key'], Skin> = {
  neutral: { key: 'neutral', bg: '#fdfcf8', bd: '#e0d9c8' },
  gov: { key: 'gov', bg: '#f4f1e6', bd: '#c9bfa4' },
  party: { key: 'party', bg: '#e3d3a8', bd: '#c2ad72' },
  parliament: { key: 'parliament', bg: '#f6f4ec', bd: '#4a5b6a' },
};

export function skinFor(role?: string): Skin {
  if (role === 'card-gov' || role === 'deck-gov') return SKINS.gov;
  if (role === 'card-party' || role === 'deck-party') return SKINS.party;
  if (role === 'card-parliament' || role === 'deck-parliament') return SKINS.parliament;
  return SKINS.neutral;
}
