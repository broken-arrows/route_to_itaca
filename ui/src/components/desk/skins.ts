// Desk paper skins — spec: docs/design/reference/prototype-draw-to-dossier-NOTES.md
// ("Card anatomy") + docs/design/desk_ui_plan.md §5.3 (tag->stationery table,
// superseded here by the role-driven version). Exact hex values are the
// component-brief contract (.superpowers/sdd/p2-task-6-brief.md) — do not
// retune without re-checking it; the `parliament` pair happens to already
// match tokens.css's --paper-2/--accent-slate, kept as literals here so this
// module has zero runtime dependency on the CSS custom-property cascade
// (skins.ts is plain TS, imported by every desk component for its color
// values directly, not just for a CSS class name).
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

// Role -> desk paper skin. A HandCard's resolved role is `card-*` (the card
// scene's own role); an InTray's is the deck scene's own compiled role,
// `deck-*` (Task 1's engine-side addition) — both route to the same paper.
// Never throws: any role outside those six (including undefined, empty
// string, plain `deck` (the misfiled-folder fallback, e.g. `debug_deck`), or
// an unrecognized value) falls back to neutral paper — the "misfiled folder"
// fallback the plan calls for when a role/tag is unmapped.
export function skinFor(role?: string): Skin {
  if (role === 'card-gov' || role === 'deck-gov') return SKINS.gov;
  if (role === 'card-party' || role === 'deck-party') return SKINS.party;
  if (role === 'card-parliament' || role === 'deck-parliament') return SKINS.parliament;
  return SKINS.neutral;
}
