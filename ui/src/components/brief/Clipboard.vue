<script setup lang="ts">
// Live clipboard — phase 3b Task 9. Replaces the inert ClipboardFrame
// (phase 3a): the tab rail now comes from the hub scene's own options
// (via useBriefStore/adapter.tabScenes, never a hardcoded scene id), and each
// click swaps the rendered sheet with no scene transition, no choice
// compilation, no autosave, no Q writes (renderView is out of band).
// Geometry/colours ported verbatim from ClipboardFrame.vue (tuned against the
// live design on 2026-07-20) — see the .clipboard-frame/.tab-rail styles
// below. Single root node on purpose (fragment roots break attrs fallthrough
// and VTU assertions — LEARNINGS 2026-07-17).
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { useBriefStore } from '../../stores/brief';
import { briefTabs } from './tabs';
import BriefSheet from './BriefSheet.vue';

const { t } = useI18n();
const brief = useBriefStore();

const tabs = computed(() => briefTabs(brief.tabs));
const activeTitle = computed(
  () => brief.tabs.find((s) => s.id === brief.activeTab)?.title ?? '',
);
// Context labels are this game's chrome, keyed by the sheet's short name.
const contextKey = computed(() => (brief.activeTab ?? '').split('.').pop() ?? '');
const context = computed(() => t(`brief.context.${contextKey.value}`, ''));
</script>

<template>
  <div class="clipboard-frame">
    <BriefSheet :title="activeTitle" :context="context" :html="brief.activeHtml" />
    <div class="tab-rail">
      <button
        v-for="tab in tabs"
        :key="tab.id || 'library'"
        class="tab"
        :class="{
          'tab-gold': tab.gold,
          'tab-active': tab.id === brief.activeTab,
          'tab-inert': tab.inert,
        }"
        data-test="brief-tab"
        :disabled="tab.inert"
        @click="tab.inert ? null : brief.select(tab.id)"
      >{{ tab.label }}</button>
    </div>
  </div>
</template>

<style scoped>
/* Clipboard-specific hexes are design-canvas literals (desk-frames.md §2),
   kept literal like skins.ts does; token vars used where an exact token
   exists. Ported verbatim from ClipboardFrame.vue — do not re-derive. */
.clipboard-frame {
  position: relative;
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  background: var(--paper-3);
  border-right: 1px solid #d9d2c4;
  padding: clamp(12px, 1.15vw, 22px) clamp(24px, 1.8vw, 34px)
    clamp(12px, 1.15vw, 22px) clamp(12px, 1.15vw, 22px);
}
/* The canvas's `right:-13px` is measured from the SHEET (its right edge sits
   at 474 - 28px padding = 446), not from the padded column — so the rail's
   right edge belongs at 446 + 13 = 459, i.e. `right: 15px` of this frame.
   The old `right: -13px` floated the tabs 15px off the sheet, detached over
   the column padding (visible in the 2026-07-20 live comparison). */
.tab-rail {
  position: absolute;
  right: clamp(11px, 0.95vw, 18px);
  top: clamp(88px, 12.8vh, 122px);
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 1;
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
.tab { cursor: pointer; }
.tab-inert { cursor: default; }
/* The paper is the top layer: tab roots tuck underneath its right edge.
   Clipboard itself remains outside the dossier dim, so lowering the rail
   inside this local stacking context does not dim the Brief. */
.tab-rail { z-index: 1; }
</style>
