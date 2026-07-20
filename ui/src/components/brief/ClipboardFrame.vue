<script setup lang="ts">
// Inert clipboard frame — phase 3a places the Brief's furniture; phase 3b
// fills it. Geometry/colours: docs/design/reference/desk-frames.md §2.
// Single root node on purpose (fragment roots break attrs fallthrough and
// VTU assertions — LEARNINGS 2026-07-17).
import { useI18n } from 'vue-i18n';
import { briefTabs } from './tabs';

const { t } = useI18n();
const tabs = briefTabs();
</script>

<template>
  <div class="clipboard-frame" aria-hidden="true">
    <div class="sheet">
      <span class="clip"></span>
    </div>
    <!-- Tabs are inert until 3b; OVERVIEW carries the canvas's active state
         (frame 1 draws the resting Brief on its OVERVIEW sheet) so the rail
         reads as the design does, not as seven equal blanks. -->
    <div class="tab-rail">
      <span
        v-for="tab in tabs"
        :key="tab.key"
        class="tab"
        :class="{ 'tab-gold': tab.gold, 'tab-active': tab.key === 'overview' }"
        data-test="brief-tab"
        >{{ t(`brief.tab.${tab.key}`) }}</span
      >
    </div>
  </div>
</template>

<style scoped>
/* Clipboard-specific hexes are design-canvas literals (desk-frames.md §2),
   kept literal like skins.ts does; token vars used where an exact token
   exists. */
.clipboard-frame {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 474px;
  background: var(--paper-3);
  border-right: 1px solid #d9d2c4;
  padding: 18px 28px 18px 18px;
}
.sheet {
  position: relative;
  height: 100%;
  background: var(--paper-0);
  border: 1px solid #e0d9c8;
  box-shadow: 0 3px 12px rgba(60, 45, 20, 0.14);
}
.clip {
  position: absolute;
  left: 50%;
  top: -8px;
  transform: translateX(-50%);
  width: 104px;
  height: 19px;
  background: #c9c0aa;
  border: 1px solid #a89e8c;
  border-radius: 6px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.18);
}
/* The canvas's `right:-13px` is measured from the SHEET (its right edge sits
   at 474 - 28px padding = 446), not from the padded column — so the rail's
   right edge belongs at 446 + 13 = 459, i.e. `right: 15px` of this frame.
   The old `right: -13px` floated the tabs 15px off the sheet, detached over
   the column padding (visible in the 2026-07-20 live comparison). */
.tab-rail {
  position: absolute;
  right: 15px;
  top: 110px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 4;
}
.tab {
  writing-mode: vertical-rl;
  padding: 9px 4px;
  background: var(--paper-1);
  color: #6b655a;
  border: 1px solid #d9d2c4;
  border-left: none;
  border-radius: 0 6px 6px 0;
  font-family: var(--font-title);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.12em;
  user-select: none;
}
.tab-gold {
  background: #f6edd2;
  color: #7c6120;
  border: 1.5px solid #d8c58a;
  border-left: none;
}
/* Active state per the canvas's mkTabs (desk-frames §2). #8c1d1d here is
   the design's own Brief-navigation treatment, recorded in desk-frames.md —
   flagged against the red-reservation rule in the 2026-07-20 session notes. */
.tab-active {
  background: var(--accent-red);
  color: #fff;
  border-color: var(--accent-red);
  box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.18);
}
</style>
