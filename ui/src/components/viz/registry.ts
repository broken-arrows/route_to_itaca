import type { Component } from 'vue';
import { WIDGET_NAMES } from './widget-names.mjs';
import Hemicycle from './Hemicycle.vue';
import AchievementGallery from './AchievementGallery.vue';
import LevelBars from './LevelBars.vue';
import TensionRows from './TensionRows.vue';
import SeatBars from './SeatBars.vue';
import RosterRows from './RosterRows.vue';
import LeaderRows from './LeaderRows.vue';
import Trail from './Trail.vue';
import PollMap from './PollMap.vue';
import ChamberVote from './ChamberVote.vue';
import Coalitions from './Coalitions.vue';

export type WidgetName = (typeof WIDGET_NAMES)[number];

// A partial Record here because old-shell-only names such as `law-grid` do not
// have a Desk component. What IS a typecheck error is registering
// a component under a key
// that isn't a WidgetName at all — see widget.registry.test.ts's "WIDGETS
// keys ⊆ WIDGET_NAMES" guard for the runtime equivalent against real content.
export const WIDGETS: Partial<Record<WidgetName, Component>> = {
  hemicycle: Hemicycle,
  'achievement-gallery': AchievementGallery,
  'level-bars': LevelBars,
  'tension-rows': TensionRows,
  'seat-bars': SeatBars,
  'roster-rows': RosterRows,
  'leader-rows': LeaderRows,
  trail: Trail,
  'poll-map': PollMap,
  'chamber-vote': ChamberVote,
  coalitions: Coalitions,
};
