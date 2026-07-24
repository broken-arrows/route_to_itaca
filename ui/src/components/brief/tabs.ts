export interface BriefTab {
  id: string;      // scene id, or '' for the inert LIBRARY tab
  label: string;
  gold?: boolean;
  inert?: boolean;
}

/**
 * The tab rail: the game's own tab scenes, plus the inert LIBRARY tab.
 *
 * §5.3 CHECKPOINT RESOLVED (2026-07-21): the tab set is content — the hub
 * scene's options — not a list in `ui/`. LIBRARY is appended here and NOT in
 * content because it is an engine scene (is-special, sub-scenes, return-nav),
 * built in phase 5; until then it renders visible and inert.
 */
export function briefTabs(scenes: { id: string; title: string }[]): BriefTab[] {
  return [
    ...scenes.map((s) => ({ id: s.id, label: s.title })),
    { id: '', label: '▤ LIBRARY', gold: true, inert: true },
  ];
}
