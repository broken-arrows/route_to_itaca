<script setup lang="ts">
// OUT tray: bottom-right, 210x100, same inset-well styling as the in-trays.
// Spec: prototype-draw-to-dossier-NOTES.md "Other desk objects". Unlike
// InTray (reused for 3 different kinds), this component is always the same
// single tray, so it self-labels with the fixed desk.tray.out chrome key.
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
.out-tray {
  width: 210px;
  height: 100px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.tray-label {
  margin: 0;
  font-family: var(--font-typed);
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--ink-0);
  opacity: 0.75;
}
.tray-well {
  position: relative;
  flex: 1;
  border-radius: 4px;
  background: var(--paper-3);
  box-shadow: inset 0 3px 8px rgba(46, 42, 34, 0.28);
  padding: 8px;
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
.slip {
  position: relative;
  width: 100%;
  height: 100%;
  background: var(--accent-gold);
  background: color-mix(in srgb, var(--accent-gold) 25%, var(--paper-0));
  border: 1px solid var(--accent-gold);
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 4px 8px;
  overflow: hidden;
}
.slip-title {
  margin: 0;
  font-family: var(--font-title);
  font-size: 12px;
  text-align: center;
  color: var(--ink-0);
}
.stamp {
  /* Ink, not red: an OUT-tray entry can be a resolved government or party
     dossier too, not only a Parlament/world one — red stays reserved for
     Parlament/world surfaces per the binding style rule. */
  position: absolute;
  top: 8px;
  right: -14px;
  transform: rotate(-18deg);
  font-family: var(--font-typed);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--ink-0);
  border: 1.5px solid var(--ink-0);
  border-radius: 3px;
  padding: 1px 6px;
  opacity: 0.7;
}
</style>
