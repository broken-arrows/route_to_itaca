// The Brief tab list. ONE function on purpose: where the list comes from is
// the deliberately-open §5.3 checkpoint (native sheets vs `role: info-tab`
// scenes declaring themselves as desk-scene options) — see
// docs/superpowers/specs/2026-07-13-content-ui-decoupling-and-the-brief-design.md.
// Until the checkpoint, the list is static here and labels are game chrome
// (source/locales/<loc>/ui.json `brief.tab.*`). Phase 3b replaces the body of
// briefTabs(), not its callers.
export const BRIEF_W = 474; // 470/1500 of the design canvas, scaled to the 1512 stage

export interface BriefTab {
  key: string; // i18n key suffix under brief.tab.*
  gold?: boolean; // LIBRARY treatment (reading, not action)
}

export function briefTabs(): BriefTab[] {
  return [
    { key: 'overview' },
    { key: 'party' },
    { key: 'chamber' },
    { key: 'economy' },
    { key: 'world' },
    { key: 'polls' },
    { key: 'library', gold: true },
  ];
}
