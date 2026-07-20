<script setup lang="ts">
// OUT tray: bottom-right, 210×104 tray base + resolved slip. Geometry/
// colours: docs/design/reference/desk-frames.md §3 "Out tray" (same
// tray-base recipe as the in-trays: 2px #c3b893, radius 10, inset shadow).
// Unlike InTray (reused for 3 different kinds), this component is always
// the same single tray, so it self-labels with the fixed desk.tray.out
// chrome key.
import { useI18n } from 'vue-i18n';
import Prose from '../Prose.vue';

defineProps<{
  entry: { title: string } | null;
}>();

const { t } = useI18n();
</script>

<template>
  <div class="out-tray" :class="{ 'is-empty': !entry }">
    <p class="tray-label">{{ t('desk.tray.out') }}</p>
    <div class="tray-well">
      <p v-if="!entry" class="tray-note">{{ t('desk.out.empty') }}</p>
      <div v-else class="slip" data-test="out-entry">
        <!-- entry.title = the resolved card's title (stores/desk.ts's
             outTray, set from openCard.title) — same already-marked
             convertLine/window.displayText output as OpenDossier's cover
             title; same fix, same reason (see the longer comment there). -->
        <p class="slip-title"><Prose tag="span" :html="entry.title" /></p>
        <span class="stamp" aria-hidden="true">{{ t('desk.out.resolved') }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Tray hexes are design-canvas literals (desk-frames.md §3). */
.out-tray {
  width: 210px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
/* Label above-left — desk-frames §3: 800 8.5px letter-spacing, #8a7a58. */
.tray-label {
  margin: 0;
  font-family: var(--font-title);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #8a7a58;
}
.tray-well {
  position: relative;
  height: 104px;
  border: 2px solid #c3b893;
  border-radius: 10px;
  background: rgba(250, 249, 245, 0.4);
  box-shadow: inset 0 3px 8px rgba(60, 45, 20, 0.14);
  padding: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.tray-note {
  margin: 0;
  font-family: var(--font-typed);
  font-size: 11px;
  color: var(--ink-0);
  opacity: 0.55;
  text-align: center;
}
/* Resolved slip: manila note, slight counter-rotation (desk-frames §3:
   #e6d8b2 on #cbb87e, rotate(-1.6deg)). */
.slip {
  position: relative;
  width: 100%;
  height: 100%;
  background: #e6d8b2;
  border: 1px solid #cbb87e;
  border-radius: 3px;
  transform: rotate(-1.6deg);
  box-shadow: 0 2px 6px rgba(60, 45, 20, 0.18);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 4px 8px;
  overflow: hidden;
}
.slip-title {
  margin: 0;
  font-family: var(--font-title);
  font-size: 11.5px;
  font-weight: 700;
  text-align: center;
  color: #6b655a;
}
.stamp {
  /* Ink, not red: an OUT-tray entry can be a resolved government or party
     dossier too, not only a Parlament/world one — red stays reserved for
     Parlament/world surfaces per the binding style rule. */
  position: absolute;
  bottom: 5px;
  right: 4px;
  transform: rotate(-8deg);
  font-family: var(--font-typed);
  font-size: 9.5px;
  letter-spacing: 0.08em;
  color: var(--ink-0);
  border: 1.5px solid var(--ink-0);
  border-radius: 3px;
  padding: 1px 6px;
  opacity: 0.7;
}
</style>
